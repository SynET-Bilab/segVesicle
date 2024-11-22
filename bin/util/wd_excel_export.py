import os
import xml.etree.ElementTree as ET
from PyQt5 import QtWidgets
import pandas as pd

def export_wd_excel(tomo_path_and_stage, print_func):
    xml_path = tomo_path_and_stage.filter_xml_path  # 使用新的XML路径
    if not os.path.exists(xml_path):
        print_func(f"XML 文件不存在: {xml_path}")
        return

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print_func(f"解析 XML 文件失败: {e}")
        return

    current_pixel_size_str = root.attrib.get("pixelSize")
    if current_pixel_size_str is None:
        print_func("XML 文件缺少 pixelSize 属性。")
        return
    try:
        current_pixel_size_nm = float(current_pixel_size_str)
        if current_pixel_size_nm <= 0:
            raise ValueError("pixelSize 必须是一个正数。")
    except ValueError as e:
        print_func(f"pixelSize 属性无效: {e}")
        return

    scale = current_pixel_size_nm  # 直接使用 current_pixel_size_nm 作为缩放因子
    vesicles_data = []

    for vesicle in root.findall('Vesicle'):
        type_element = vesicle.find('Type')
        # 只保留 'Type' 元素中 't' 属性为 'others' 的囊泡
        if type_element is not None and type_element.attrib.get('t') == 'others':
            vesicle_data = {}
            vesicle_data['vesicleId'] = vesicle.attrib.get('vesicleId', '')

            # 处理坐标类元素
            for elem_name in ['Center', 'Center2D', 'Center3D', 'ProjectionPoint']:
                elem = vesicle.find(elem_name)
                if elem is not None:
                    for coord in ['X', 'Y', 'Z']:
                        val = elem.attrib.get(coord)
                        if val is not None:
                            try:
                                scaled_val = float(val) * scale
                                vesicle_data[f'{elem_name}_{coord}'] = scaled_val
                            except ValueError:
                                vesicle_data[f'{elem_name}_{coord}'] = None
                        else:
                            vesicle_data[f'{elem_name}_{coord}'] = None
                else:
                    for coord in ['X', 'Y', 'Z']:
                        vesicle_data[f'{elem_name}_{coord}'] = None

            # 处理半径类元素
            # Radius 仅有属性 'r'
            radius = vesicle.find('Radius')
            if radius is not None:
                r_val = radius.attrib.get('r')
                if r_val is not None:
                    try:
                        scaled_r_val = float(r_val) * scale
                        vesicle_data['Radius_r'] = scaled_r_val
                    except ValueError:
                        vesicle_data['Radius_r'] = None
                else:
                    vesicle_data['Radius_r'] = None
            else:
                vesicle_data['Radius_r'] = None

            # Radius2D 仅有属性 'r1' 和 'r2'
            radius2d = vesicle.find('Radius2D')
            if radius2d is not None:
                for r_attr in ['r1', 'r2']:
                    r_val = radius2d.attrib.get(r_attr)
                    if r_val is not None:
                        try:
                            scaled_r_val = float(r_val) * scale
                            vesicle_data[f'Radius2D_{r_attr}'] = scaled_r_val
                        except ValueError:
                            vesicle_data[f'Radius2D_{r_attr}'] = None
                    else:
                        vesicle_data[f'Radius2D_{r_attr}'] = None
            else:
                vesicle_data['Radius2D_r1'] = None
                vesicle_data['Radius2D_r2'] = None

            # Radius3D 仅有属性 'r1', 'r2', 'r3'
            radius3d = vesicle.find('Radius3D')
            if radius3d is not None:
                for r_attr in ['r1', 'r2', 'r3']:
                    r_val = radius3d.attrib.get(r_attr)
                    if r_val is not None:
                        try:
                            scaled_r_val = float(r_val) * scale
                            vesicle_data[f'Radius3D_{r_attr}'] = scaled_r_val
                        except ValueError:
                            vesicle_data[f'Radius3D_{r_attr}'] = None
                    else:
                        vesicle_data[f'Radius3D_{r_attr}'] = None
            else:
                vesicle_data['Radius3D_r1'] = None
                vesicle_data['Radius3D_r2'] = None
                vesicle_data['Radius3D_r3'] = None

            # 处理 Distance 元素
            distance = vesicle.find('Distance')
            if distance is not None:
                d_val = distance.attrib.get('d')
                if d_val is not None:
                    try:
                        scaled_d_val = float(d_val) * scale
                        vesicle_data['Distance_d'] = scaled_d_val
                    except ValueError:
                        vesicle_data['Distance_d'] = None
                else:
                    vesicle_data['Distance_d'] = None
            else:
                vesicle_data['Distance_d'] = None

            vesicles_data.append(vesicle_data)

    if not vesicles_data:
        print_func("未找到 Type 't' == 'others' 的囊泡。")
        return

    # 创建DataFrame并导出到Excel
    df = pd.DataFrame(vesicles_data)
    excel_path = tomo_path_and_stage.weidong_excel_path
    try:
        df.to_excel(excel_path, index=False)
        print_func(f"Excel 文件已成功保存到 {excel_path}。")
    except Exception as e:
        print_func(f"保存 Excel 文件失败: {e}")
        return
