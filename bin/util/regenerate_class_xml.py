#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import xml.etree.ElementTree as ET
import copy
import sys

def parse_xml(file_path):
    """
    解析XML文件，返回pixelSize和一个以vesicleId为键的囊泡字典。
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    pixel_size = root.get('pixelSize')
    vesicles = {}
    for vesicle in root.findall('Vesicle'):
        vesicle_id_str = vesicle.get('vesicleId')
        if vesicle_id_str is None:
            continue
        try:
            vesicle_id = int(vesicle_id_str)
            vesicles[vesicle_id] = vesicle
        except ValueError:
            print(f"Invalid vesicleId '{vesicle_id_str}' in file {file_path}", file=sys.stderr)
            continue

    return pixel_size, vesicles

def determine_new_type(type_filter, type_ori_filter, type_ori_class):
    """
    根据规则确定新的Type属性值。
    """
    if type_filter == 'others' and type_ori_filter == 'vesicle':
        return 'others'
    elif type_filter == 'vesicle' and type_ori_filter == 'others':
        return 'vesicle'
    elif type_filter == 'others' and type_ori_filter == 'others':
        return type_ori_class
    elif type_filter == 'vesicle' and type_ori_filter == 'vesicle':
        return 'vesicle'
    else:
        # 默认情况，可以根据需要调整
        return 'vesicle'

def indent(elem, level=0):
    """
    为XML元素添加缩进，使输出更美观。
    """
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level+1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    return elem

def main():
    parser = argparse.ArgumentParser(description='生成新的class XML文件。')
    parser.add_argument('filter_xml_path', help='filter XML 文件路径')
    parser.add_argument('class_xml_path', help='输出的class XML 文件路径')
    parser.add_argument('ori_filter_xml_path', help='原始filter XML 文件路径')
    parser.add_argument('ori_class_xml_path', help='原始class XML 文件路径')

    args = parser.parse_args()

    # 解析所有XML文件
    pixel_size_filter, vesicles_filter = parse_xml(args.filter_xml_path)
    pixel_size_ori_filter, vesicles_ori_filter = parse_xml(args.ori_filter_xml_path)
    pixel_size_ori_class, vesicles_ori_class = parse_xml(args.ori_class_xml_path)

    # 检查pixelSize是否一致
    if not (pixel_size_filter == pixel_size_ori_filter == pixel_size_ori_class):
        print("Error: All XML files must have the same pixelSize.", file=sys.stderr)
        sys.exit(1)

    pixel_size = pixel_size_filter  # 任意选择一个，因为它们相同

    # 找到filter_xml_path中的最大vesicleId
    if not vesicles_filter:
        print("Error: filter_xml_path中没有囊泡信息。", file=sys.stderr)
        sys.exit(1)

    max_id = max(vesicles_filter.keys())

    # 创建新的VesicleList根元素
    new_root = ET.Element('VesicleList', {'pixelSize': pixel_size})

    # 遍历ori_class_xml_path中的所有囊泡
    for vesicle_id in sorted(vesicles_ori_class.keys()):
        if vesicle_id <= max_id:
            ves_filter = vesicles_filter.get(vesicle_id)
            ves_ori_filter = vesicles_ori_filter.get(vesicle_id)
            ves_ori_class = vesicles_ori_class.get(vesicle_id)

            if ves_filter is None:
                print(f"Warning: vesicleId {vesicle_id} 在 filter_xml_path 中缺失，跳过。", file=sys.stderr)
                continue
            if ves_ori_filter is None:
                print(f"Warning: vesicleId {vesicle_id} 在 ori_filter_xml_path 中缺失，跳过。", file=sys.stderr)
                continue
            if ves_ori_class is None:
                print(f"Warning: vesicleId {vesicle_id} 在 ori_class_xml_path 中缺失，跳过。", file=sys.stderr)
                continue

            # 深拷贝filter中的囊泡以避免修改原始树
            new_vesicle = copy.deepcopy(ves_filter)
            new_vesicle.set('vesicleId', str(vesicle_id))  # 确保vesicleId为字符串

            # 获取Type信息
            type_filter_elem = ves_filter.find('Type')
            type_ori_filter_elem = ves_ori_filter.find('Type')
            type_ori_class_elem = ves_ori_class.find('Type')

            if type_filter_elem is None:
                print(f"Warning: vesicleId {vesicle_id} 在 filter_xml_path 中缺少 Type 元素，默认设置为 'vesicle'。", file=sys.stderr)
                type_filter = 'vesicle'
            else:
                type_filter = type_filter_elem.get('t', 'vesicle')

            if type_ori_filter_elem is None:
                print(f"Warning: vesicleId {vesicle_id} 在 ori_filter_xml_path 中缺少 Type 元素，默认设置为 'vesicle'。", file=sys.stderr)
                type_ori_filter = 'vesicle'
            else:
                type_ori_filter = type_ori_filter_elem.get('t', 'vesicle')

            if type_ori_class_elem is None:
                print(f"Warning: vesicleId {vesicle_id} 在 ori_class_xml_path 中缺少 Type 元素，默认设置为 'vesicle'。", file=sys.stderr)
                type_ori_class = 'vesicle'
            else:
                type_ori_class = type_ori_class_elem.get('t', 'vesicle')

            # 确定新的Type
            new_type = determine_new_type(type_filter, type_ori_filter, type_ori_class)

            # 更新Type元素
            type_elem = new_vesicle.find('Type')
            if type_elem is not None:
                type_elem.set('t', new_type)
            else:
                ET.SubElement(new_vesicle, 'Type', {'t': new_type})

            # 将修改后的囊泡添加到新的根元素
            new_root.append(new_vesicle)

        else:
            # 使用ori_class_xml_path中的囊泡，进行深拷贝
            ves_ori_class = vesicles_ori_class.get(vesicle_id)
            if ves_ori_class is None:
                print(f"Warning: vesicleId {vesicle_id} 在 ori_class_xml_path 中缺失，跳过。", file=sys.stderr)
                continue
            new_root.append(copy.deepcopy(ves_ori_class))

    # 美化XML缩进
    indent(new_root)

    # 写入新的XML文件
    tree = ET.ElementTree(new_root)
    try:
        tree.write(args.class_xml_path, encoding='utf-8', xml_declaration=False)
        print(f"新XML文件已生成: {args.class_xml_path}")
    except IOError as e:
        print(f"Error writing to file {args.class_xml_path}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
