#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import xml.etree.ElementTree as ET
import json
import os

def main():
    if len(sys.argv) != 5:
        print("Usage: python vesicle_type_difference.py filter_xml_path ori_filter_xml_path current_path tomo_name")
        sys.exit(1)

    filter_xml_path = sys.argv[1]
    ori_filter_xml_path = sys.argv[2]
    current_path = sys.argv[3]
    tomo_name = sys.argv[4]

    # 检查文件是否存在
    if not os.path.isfile(filter_xml_path):
        print(f"文件不存在：{filter_xml_path}")
        sys.exit(1)

    if not os.path.isfile(ori_filter_xml_path):
        print(f"文件不存在：{ori_filter_xml_path}")
        sys.exit(1)

    # 解析 XML 文件
    tree_filter = ET.parse(filter_xml_path)
    root_filter = tree_filter.getroot()

    tree_ori = ET.parse(ori_filter_xml_path)
    root_ori = tree_ori.getroot()

    # 创建字典以快速查找
    vesicles_filter = {}
    vesicles_ori = {}

    for vesicle in root_filter.findall('.//Vesicle'):
        vesicleId = vesicle.get('vesicleId')
        type_element = vesicle.find('Type')
        if type_element is not None:
            vesicles_filter[vesicleId] = type_element.get('t')
        else:
            vesicles_filter[vesicleId] = None

    for vesicle in root_ori.findall('.//Vesicle'):
        vesicleId = vesicle.get('vesicleId')
        type_element = vesicle.find('Type')
        if type_element is not None:
            vesicles_ori[vesicleId] = type_element.get('t')
        else:
            vesicles_ori[vesicleId] = None

    # 比较类型并收集符合条件的囊泡 ID
    vesicle_ids = []

    for vesicleId in vesicles_filter.keys():
        type_filter = vesicles_filter[vesicleId]
        type_ori = vesicles_ori.get(vesicleId)
        if type_filter == 'others' and type_ori == 'vesicle':
            vesicle_ids.append(vesicleId)

    # 将结果保存为 JSON，包含 tomo_name 和对应的 ID 列表
    output_data = {
        tomo_name: vesicle_ids
    }

    output_json_path = os.path.join(current_path, 'vesicle_ids.json')

    # 检查 JSON 文件是否已存在
    if os.path.isfile(output_json_path):
        # 加载现有数据
        with open(output_json_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    # 更新数据
    existing_data[tomo_name] = vesicle_ids

    # 保存更新后的数据
    with open(output_json_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

if __name__ == '__main__':
    main()
