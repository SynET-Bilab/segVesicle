import os
import re
import json
import numpy as np
import mrcfile

from tqdm import tqdm
from segVesicle.bin.util.structures import VesicleList, Surface
from tifffile.tifffile import imwrite
from segVesicle.bin.morph import density_fit_2d


def normalize_scale(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    max_value = img.max()
    min_value = img.min()
    img_norm = (255 * (img - min_value) / (max_value - min_value)).astype(np.uint8)
    return img_norm



def distance_calc(
    json_path,
    mod_path,
    xml_output_path,
    print_func,
    fit_2d: bool = False,
    mrc_path: str = None,
    img_output_path: str = None,
):
    try:
        ori_pix_size = 1.714
        resampled_pix_size = 1.714
        ratio = resampled_pix_size / ori_pix_size

        if not os.path.exists(json_path):
            print_func(f"JSON file path does not exist: {json_path}")
            return
        if not os.path.exists(mod_path):
            print_func(f"Mod file path does not exist: {mod_path}")
            return
        if fit_2d and (not mrc_path or not os.path.exists(mrc_path)):
            print_func(f"MRC file path does not exist: {mrc_path}")
            fit_2d = False

        with open(json_path, 'r') as f:
            data = json.load(f)

        vesicles = data.get('vesicles', [])
        if not vesicles:
            print_func("No 'vesicles' data found in the JSON file.")
            return

        center_list = []
        for vesicle in vesicles:
            center = ratio * np.asarray(vesicle.get('center', [0, 0, 0]))
            center_list.append(center)
        center_list = np.asarray(center_list)
        center_list = center_list[:, [2, 1, 0]]  # zyx to xyz

        vl = VesicleList(ori_pix_size)
        vl.fromCenterList(center_list)
        surface = Surface()
        
        '''
        temporary use
        '''
        if 'premembrane.mod' in mod_path:
            surface.from_model_use_imod_mesh(mod_path)
            print('use manual membrane')
        else:
            surface.from_model_auto_segment(mod_path, objNum=2)

        # 定义2d囊泡需要保留的属性，用于后续删除不需要的属性
        attributes_to_keep = [
            '_vesicleId', '_type', '_center', '_radius',
            '_center2D', '_radius2D', '_rotation2D',
            '_distance', '_distance2D', '_projectionPoint', '_projectionPoint2D'
        ]

        for i, vesicle in tqdm(enumerate(vl), total=len(vl), desc="Processing vesicles", dynamic_ncols=True):
            ves_data = vesicles[i]
            
            # 1. 提取并设置名称和 ID
            name = ves_data.get('name', f'vesicle_{i}')  # 默认名称为 'vesicle_i'，避免重复默认 'vesicle_0'
            match = re.search(r'\d+', name)
            vesicle_id = int(match.group()) if match else i  # 如果未找到数字，则使用索引作为 ID
            vesicle.setId(vesicle_id)
            
            # 2. 设置缩放后的半径
            radii = np.asarray(ves_data.get('radii', [0]), dtype=float)[::-1] * ratio
            vesicle.setRadius(np.mean(radii))
            vesicle.setRadius3D(radii)
            
            # 3. 设置缩放后的中心点
            center = np.asarray(ves_data.get('center', [0, 0, 0]), dtype=float)[[2, 1, 0]] * ratio
            vesicle.setCenter(center)
            
            # 4. 设置方向向量(update: do L^2(E) here and never change evecs later)
            evecs = np.asarray(ves_data.get('evecs', [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), dtype=float)
            B = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).reshape((3, 3))
            evecs_final = B@evecs@B
            vesicle._evecs = evecs_final
            
            # 5. 设置类型
            vesicle.setType('vesicle')
            
            # * 6. for 2D section parralleled to xy plane of 3D vesicle
            _, radius2D, eigvecs = vesicle.ellipse_in_plane()
            vesicle.setRadius2D(np.asarray(radius2D, dtype=float) * ratio)
            vesicle.setRotation2D(np.arctan2(eigvecs[0, 1], eigvecs[0, 0]) - np.pi / 2)
            
            if np.array_equal(vesicle._evecs[2], [0.0, 0.0, 1.0]):
                print(f"Correcting 2D vesicle with ID: {vesicle.getId()}")
                
                # 7.1 使用 Radius3D 的 r1 和 r2 更新 Radius2D
                if hasattr(vesicle, '_radius3D') and len(vesicle.getRadius3D()) >= 2:
                    r1, r2 = vesicle.getRadius3D()[1], vesicle.getRadius3D()[0]
                    vesicle.setRadius2D([r1, r2])
                    print(f"Updated Radius2D to r1: {r1}, r2: {r2}")
                    vesicle.setRadius((r1+r2)/2)
                else:
                    print(f"Warning: Vesicle ID {vesicle.getId()} lacks valid Radius3D data. Skipping Radius2D update.")
                    continue  # 如果 Radius3D 无效，跳过后续步骤
                
                # 7.2 计算 Rotation2D
                X, Y = vesicle._evecs[0, 1], vesicle._evecs[1, 1]  #TODO: need to test
                phi = np.arctan2(Y, X) - np.pi / 2  # to same definition of phi: vesicle._rotation2D
                vesicle.setRotation2D(phi)
                print(f"Computed Rotation2D for vesicle ID {vesicle.getId()}: phi = {phi} radians")
        
        # 8. 移除2d囊泡不需要的属性
        for i, vesicle in tqdm(enumerate(vl), total=len(vl), desc="Processing vesicles", dynamic_ncols=True):
            if np.array_equal(vesicle._evecs[2], [0.0, 0.0, 1.0]):
                current_attrs = list(vars(vesicle).keys())
                for attr in current_attrs:
                    if attr not in attributes_to_keep:
                        delattr(vesicle, attr)
                print(f"Removed unwanted attributes for vesicle ID {vesicle.getId()}")
        
        # 9. Calculate the distance to the surface
        if 'premembrane.mod' in mod_path:
            vl.distance_to_surface(surface, 3600, 'sparse')
        else:
            vl.distance_to_surface(surface, 3600, 'dense')
        
        if fit_2d:
            # copy from bin/fitradius2D.py, which is difficult to import directly
            # maybe need to functionalize bin/fitradius2D.py later
            img_path = img_output_path
            if img_path is None:
                img_path = os.path.join(os.path.dirname(xml_output_path), 'images')
            if os.path.exists(img_path):
                if os.path.exists(img_path + '_bak'):
                    os.system('rm -r {}'.format(img_path + '_bak'))
                os.system('mv {} {}'.format(img_path, img_path + '_bak'))
            if not os.path.exists(img_path):
                os.makedirs(img_path)

            with mrcfile.open(mrc_path) as mrc:
                mrc_data = mrc.data.astype(np.float32)

            padwidth = 100
            margin = 15
            mean_value = mrc_data.mean()
            data_pad = np.pad(mrc_data, padwidth, mode='constant', constant_values=mean_value)

            for i in tqdm(range(len(vl))):
                center = np.round(vl[i].getCenter()).astype(np.uint16)
                radius = vl[i].getRadius().mean()
                x_init, y_init, z_init = center
                z_range = range(z_init - 1, z_init + 2)
                r_ma = 0

                for z in z_range:
                    center_z = np.array([z, y_init, x_init])
                    #  'list' object has no attribute 'mean'
                    center_fit, evecs_fit, radii_fit, ccf = density_fit_2d(mrc_data, center_z, radius)
                    if (radii_fit is not None) and (ccf >= 0.6):
                        r_z = 0.5 * (radii_fit[1] + radii_fit[2])
                        if r_z > r_ma:
                            r_ma = r_z
                            X, Y = evecs_fit[1, 2], evecs_fit[1, 1]
                            phi = np.arctan2(Y, X) - np.pi/2
                            vl[i].setCenter2D(center_fit[[2, 1, 0]] + np.array([1, 1, 1]))  # zyx to xyz
                            vl[i].setRadius2D(np.array([radii_fit[1], radii_fit[2]]))
                            vl[i].setRotation2D(phi)

                radius_new = vl[i].getRadius2D().max()
                fit_vesicle = vl[i].sample_on_vesicle(360)
                shift = np.array([
                    vl[i]._center2D[0] - radius_new - margin,
                    vl[i]._center2D[1] - radius_new - margin,
                    fit_vesicle[:, 2].mean()
                ])
                fit_vesicle_shift = np.round(fit_vesicle - shift).astype(np.uint16)  # local coordinate, xyz, and z=0

                img = data_pad[
                    np.round(vl[i]._center2D[2] + padwidth - 1).astype(np.uint16),
                    np.round(vl[i]._center2D[1] + padwidth - radius_new - margin - 1).astype(np.uint16): np.round(vl[i]._center2D[1] + padwidth + radius_new + margin - 1).astype(np.uint16),
                    np.round(vl[i]._center2D[0] + padwidth - radius_new - margin - 1).astype(np.uint16): np.round(vl[i]._center2D[0] + padwidth + radius_new + margin - 1).astype(np.uint16)
                ]  # xml from 1, but array from 0
                img_norm = normalize_scale(img)
                out = np.array([img_norm] * 3)

                out[0, np.round(radius_new + margin).astype(np.uint16), np.round(radius_new + margin).astype(np.uint16)] = 255  # center
                out[0, fit_vesicle_shift[:, 1], fit_vesicle_shift[:, 0]] = 255  # ellipse, RGB:(255,0,0), in red
                out[1, fit_vesicle_shift[:, 1], fit_vesicle_shift[:, 0]] = 0
                out[2, fit_vesicle_shift[:, 1], fit_vesicle_shift[:, 0]] = 0
                if r_ma == 0:
                    out[1, np.round(radius_new + margin).astype(np.uint16), np.round(radius_new + margin).astype(np.uint16)] = 255
                    out[0, fit_vesicle_shift[:, 1], fit_vesicle_shift[:, 0]] = 0
                    out[1, fit_vesicle_shift[:, 1], fit_vesicle_shift[:, 0]] = 255
                    out[2, fit_vesicle_shift[:, 1], fit_vesicle_shift[:, 0]] = 0

                imwrite(os.path.join(img_path, '{}.tif'.format(vl[i].getId())), out, photometric='rgb')

            if 'premembrane.mod' in mod_path:
                vl.distance_to_surface(surface, precision=360, mode='sparse', vesicle_mode='fitradius2D')
            else:
                vl.distance_to_surface(surface, precision=360, mode='dense', vesicle_mode='fitradius2D')
        vl.toXMLFile(xml_output_path)
        print_func(f"XML file successfully generated at: {xml_output_path}")

    except Exception as e:
        print_func(f"Distance calculation failed: {str(e)}")


distance_calc('/share/data/CryoET_Data/zhenhang/software_test/segVesicle_test/test_fitradius2D_fix2D_SV_cannot_use/pp95_vesicle_test.json', 
              '/share/data/CryoET_Data/zhenhang/software_test/segVesicle_test/test_fitradius2D_fix2D_SV_cannot_use/membrane/pp95.mod',
              '/share/data/CryoET_Data/zhenhang/software_test/segVesicle_test/test_fitradius2D_fix2D_SV_cannot_use/pp95_vesicle_test1.xml',
              print,
              fit_2d=False,
              mrc_path='/share/data/CryoET_Data/zhenhang/software_test/segVesicle_test/test_fitradius2D_fix2D_SV_cannot_use/pp95_wbp_corrected.mrc',
              img_output_path='/share/data/CryoET_Data/zhenhang/software_test/segVesicle_test/test_fitradius2D_fix2D_SV_cannot_use/images')