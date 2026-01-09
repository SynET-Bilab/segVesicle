import os
import re
import json
import numpy as np

from tqdm import tqdm
from util.structures import VesicleList, Surface



def distance_calc(json_path, mod_path, xml_output_path, print_func):
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
            '_center2D', '_radius2D', '_rotation2D'
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
            vesicle._rotation2D = np.arctan2(eigvecs[0, 1], eigvecs[0, 0]) - np.pi / 2
            
            # 7. 若为 2D 膜囊泡，进行校正 (update: use [1,0,0] instead of [0,0,1])
            if np.array_equal(vesicle._evecs[0], [1.0, 0.0, 0.0]):
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
                
                
                # 7.3 计算 Rotation2D
                X, Y = vesicle._evecs[2, 1], vesicle._evecs[1, 1]
                phi = np.arctan2(Y, X) - np.pi / 2  # to same definition of phi: vesicle._rotation2D
                vesicle.setRotation2D(phi)
                print(f"Computed Rotation2D for vesicle ID {vesicle.getId()}: phi = {phi} radians")
                
                # 7.4 移除2d囊泡不需要的属性
                current_attrs = list(vars(vesicle).keys())
                for attr in current_attrs:
                    if attr not in attributes_to_keep:
                        delattr(vesicle, attr)
                print(f"Removed unwanted attributes for vesicle ID {vesicle.getId()}")

        # Calculate the distance to the surface
        if 'premembrane.mod' in mod_path:
            vl.distance_to_surface(surface, 3600, 'sparse')
        else:
            vl.distance_to_surface(surface, 3600, 'dense')
        
        vl.toXMLFile(xml_output_path)
        print_func(f"XML file successfully generated at: {xml_output_path}")

    except Exception as e:
        print_func(f"Distance calculation failed: {str(e)}")
