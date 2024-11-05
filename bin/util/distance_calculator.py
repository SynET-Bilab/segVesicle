import os
import re
import json
from tqdm import tqdm  # 用于显示进度条
import numpy as np
from util.structures import VesicleList, Surface  # Assuming these modules are available

def distance_calc(json_path, mod_path, xml_output_path, print_func):
    try:
        ori_pix_size = 1.714
        resampled_pix_size = 1.714
        ratio = resampled_pix_size / ori_pix_size

        # Check if json_path and mod_path exist
        if not os.path.exists(json_path):
            print_func(f"JSON file path does not exist: {json_path}")
            return
        
        if not os.path.exists(mod_path):
            print_func(f"Mod file path does not exist: {mod_path}")
            return

        # Read JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        vesicles = data.get('vesicles', [])
        if not vesicles:
            # Log the error if 'vesicles' data is missing in the JSON
            print_func("No 'vesicles' data found in the JSON file.")
            return

        center_list = []
        for vesicle in vesicles:
            center = ratio * np.asarray(vesicle.get('center', [0, 0, 0]))
            center_list.append(center)
        center_list = np.asarray(center_list)
        
        # Convert zyx to xyz
        center_list = center_list[:, [2, 1, 0]]

        # Initialize VesicleList
        vl = VesicleList(ori_pix_size)
        vl.fromCenterList(center_list)

        # Initialize Surface and read the .mod file
        surface = Surface()
        surface.from_model_auto_segment(mod_path, objNum=2)

        # 定义2d囊泡需要保留的属性，用于后续删除不需要的属性
        attributes_to_keep = [
            '_vesicleId', '_type', '_center', '_radius', 
            '_center2D', '_radius2D', '_rotation2D'
        ]

        # 主循环，使用 tqdm 显示进度
        for i, vesicle in tqdm(enumerate(vl), total=len(vl), desc="Processing vesicles", dynamic_ncols=True):
            ves_data = vesicles[i]
            
            # 1. 提取并设置名称和 ID
            name = ves_data.get('name', f'vesicle_{i}')  # 默认名称为 'vesicle_i'，避免重复默认 'vesicle_0'
            match = re.search(r'\d+', name)
            vesicle_id = int(match.group()) if match else i  # 如果未找到数字，则使用索引作为 ID
            vesicle.setId(vesicle_id)  # 设置 Vesicle ID
            
            # 2. 提取并处理 radii
            radii = ves_data.get('radii', [0])
            radii_np = np.asarray(radii, dtype=float)  # 转换为 numpy 数组
            radii_scaled = radii_np * ratio
            # radii_xyz = radii_scaled[[2, 1, 0]]  # 转换 zyx 到 xyz
            radii_xyz = radii_scaled
            vesicle.setRadius(np.mean(radii_xyz))       # 设置平均半径
            vesicle.setRadius3D(radii_xyz)              # 设置 3D 半径
            
            # 3. 提取并处理 center
            center = ves_data.get('center', [0, 0, 0])
            center_np = np.asarray(center, dtype=float)
            center_scaled = center_np * ratio
            # center_xyz = center_scaled[[2, 1, 0]]  # 转换 zyx 到 xyz
            center_xyz = center_scaled
            vesicle.setCenter(center_xyz)          # 设置中心点
            
            # 4. 提取并处理 evecs
            evecs = ves_data.get('evecs', [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            evecs_np = np.asarray(evecs, dtype=float)
            # evecs_np = evecs_np[::-1,:]
            # 转换位置，从 [[z1, z2, z3], [y1, y2, y3], [x1, x2, x3]] 到 [[z1, y1, x1], [z2, y2, x2], [z3, y3, x3]]
            evecs_xyz = evecs_np.T
            vesicle._evecs = evecs_xyz             # 设置方向向量
            
            # 5. 设置类型
            vesicle.setType('vesicle')
            
            # *6. 计算并设置 2D 半径和旋转角度
            _, radius2D, eigvecs = vesicle.ellipse_in_plane()
            radius2D_scaled = np.asarray(radius2D, dtype=float) * ratio
            vesicle.setRadius2D(radius2D_scaled)    # 设置 2D 半径
            # 计算旋转角度
            vesicle._rotation2D = np.arctan2(eigvecs[0, 1], eigvecs[0, 0]) - np.pi / 2
            # vesicle._rotation2D = np.arctan2(eigvecs[0, 1], eigvecs[0, 0])
            
            # *7. 检查是否为 2D 膜囊，并进行校正
            # 通过检查第一个 evec 是否等于 [0.0, 0.0, 1.0] 来判断是否为 2D
            if hasattr(vesicle, '_evecs') and np.array_equal(vesicle._evecs[0], [0.0, 0.0, 1.0]):
                print(f"Correcting 2D vesicle with ID: {vesicle.getId()}")
                
                # 7.1 使用 Radius3D 的 r1 和 r2 更新 Radius2D
                if hasattr(vesicle, '_radius3D') and len(vesicle.getRadius3D()) >= 2:
                    r1, r2 = vesicle.getRadius3D()[0], vesicle.getRadius3D()[1]
                    vesicle.setRadius2D([r1, r2])
                    print(f"Updated Radius2D to r1: {r1}, r2: {r2}")
                else:
                    print(f"Warning: Vesicle ID {vesicle.getId()} lacks valid Radius3D data. Skipping Radius2D update.")
                    continue  # 如果 Radius3D 无效，跳过后续步骤
                
                # 7.2 修改 Evecs
                if hasattr(vesicle, '_evecs') and vesicle._evecs.shape == (3, 3):
                    evecs = vesicle._evecs.copy()
                    original_evec = evecs[1].copy()
                    # 修改第二个 evec
                    evecs[1, 0] = evecs[1, 1]  # X = Y
                    evecs[1, 1] = evecs[1, 2]  # Y = Z
                    evecs[1, 2] = 0.0           # Z = 0
                    vesicle.setEvecs(evecs)
                    print(f"Modified Evecs for vesicle ID {vesicle.getId()}:")
                    print(f"Original Evecs[1]: {original_evec}")
                    print(f"New Evecs[1]: {evecs[1]}")
                else:
                    print(f"Warning: Vesicle ID {vesicle.getId()} lacks valid Evecs data. Skipping Evecs modification.")
                    continue  # 如果 Evecs 无效，跳过后续步骤
                
                # 7.3 计算 Rotation2D
                X, Y = vesicle._evecs[1, 0], vesicle._evecs[1, 1]
                phi = np.arctan2(X, Y)
                vesicle.setRotation2D(phi)
                print(f"Computed Rotation2D for vesicle ID {vesicle.getId()}: phi = {phi} radians")
                
                # 7.4 移除不需要的属性
                current_attrs = list(vars(vesicle).keys())
                for attr in current_attrs:
                    if attr not in attributes_to_keep:
                        delattr(vesicle, attr)
                print(f"Removed unwanted attributes for vesicle ID {vesicle.getId()}")

        # Calculate the distance to the surface
        vl.distance_to_surface(surface, 3600, 'dense')

        # Output the XML file
        vl.toXMLFile(xml_output_path)

        # Notify user of success
        print_func(f"XML file successfully generated at: {xml_output_path}")

    except Exception as e:
        # Handle exceptions and notify user of error
        print_func(f"Distance calculation failed: {str(e)}")
