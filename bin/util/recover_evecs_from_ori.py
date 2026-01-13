#!/usr/bin/env python3
"""
please use function recover_evecs_from_ori
    inputs: ori_xml_path, aim_xml_path
    this funtion will use ori_xml information to replace Evecs/Distance/ProjectionPoint in aim_xml
    the original aim xml will be renamed to xxx_bak.xml
"""
import os
import copy
from lxml import etree
# import fire


def _backup_xml(xml_path: str) -> str:
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    if xml_path.lower().endswith(".xml"):
        backup_path = xml_path[:-4] + "_bak.xml"
    else:
        backup_path = xml_path + "_bak.xml"

    if os.path.exists(backup_path):
        os.remove(backup_path)

    os.replace(xml_path, backup_path)
    return backup_path


def _load_xml_root(xml_path: str) -> etree._Element:
    with open(xml_path, "r") as f:
        xml_str = f.read()
    return etree.fromstring(xml_str)


def _collect_recovery_by_id(ori_root: etree._Element) -> dict:
    recovery_by_id = {}
    for vesicle in ori_root.xpath("Vesicle"):
        vesicle_id = vesicle.get("vesicleId")
        if vesicle_id is None:
            continue

        evecs = vesicle.xpath("Evecs")
        distance = vesicle.find("Distance")
        projection = vesicle.find("ProjectionPoint")

        if not (evecs or distance is not None or projection is not None):
            continue

        recovery_by_id[vesicle_id] = {
            "evecs": [copy.deepcopy(e) for e in evecs],
            "distance": copy.deepcopy(distance) if distance is not None else None,
            "projection": copy.deepcopy(projection) if projection is not None else None,
        }
    return recovery_by_id


def _replace_recovery_items(filter_root: etree._Element, recovery_by_id: dict) -> int:
    replaced = 0

    for vesicle in filter_root.xpath("Vesicle"):
        vesicle_id = vesicle.get("vesicleId")
        if vesicle_id not in recovery_by_id:
            continue

        recovery = recovery_by_id[vesicle_id]

        # 1. 移除原有 Evecs / Distance / ProjectionPoint
        for evec in vesicle.xpath("Evecs"):
            vesicle.remove(evec)
        for node in vesicle.xpath("Distance | ProjectionPoint"):
            vesicle.remove(node)

        # 2. 找插入位置：Rotation2D 之后，没有则追加
        rotation2d = vesicle.find("Rotation2D")
        if rotation2d is not None:
            insert_index = vesicle.index(rotation2d) + 1
        else:
            insert_index = len(vesicle)

        # 3. 插入 Evecs / Distance / ProjectionPoint，并修复缩进
        for evec in recovery["evecs"]:
            e = copy.deepcopy(evec)
            e.tail = "\n    "
            vesicle.insert(insert_index, e)
            insert_index += 1

        if recovery["distance"] is not None:
            d = copy.deepcopy(recovery["distance"])
            d.tail = "\n    "
            vesicle.insert(insert_index, d)
            insert_index += 1

        if recovery["projection"] is not None:
            p = copy.deepcopy(recovery["projection"])
            p.tail = "\n    "
            vesicle.insert(insert_index, p)
            insert_index += 1

        replaced += 1

    return replaced



def recover_evecs_from_ori(ori_xml_path: str, aim_xml_path: str) -> None:
    """
    Replace Evecs/Distance/ProjectionPoint in aim_xml_path with those from
    ori_xml_path, by vesicleId. The original aim xml will be renamed to
    xxx_bak.xml.
    """
    ori_root = _load_xml_root(ori_xml_path)
    filter_root = _load_xml_root(aim_xml_path)

    if not filter_root.xpath("Vesicle"):
        return

    recovery_by_id = _collect_recovery_by_id(ori_root)
    if not recovery_by_id:
        raise ValueError("No Evecs/Distance/ProjectionPoint found in ori xml.")

    replaced = _replace_recovery_items(filter_root, recovery_by_id)
    if replaced == 0:
        raise ValueError("No matching vesicleId found to replace items.")

    _backup_xml(aim_xml_path)

    xml_str = etree.tostring(filter_root, pretty_print=True, encoding="utf-8").decode("utf-8")
    with open(aim_xml_path, "w") as f:
        f.write(xml_str)


def main() -> None:
    # from util.make_L_for_ori_evecs import make_L_for_ori_evecs
    ori_xml_path = '/media/liushuo/data1/data/synapse_seg/Testing_data/fix_xml/pp95_ori.xml'
    # make_L_for_ori_evecs(ori_xml_path)
    aim_xml_path = '/media/liushuo/data1/data/synapse_seg/Testing_data/fix_xml/pp95_omega.xml'
    recover_evecs_from_ori(ori_xml_path, aim_xml_path)


if __name__ == "__main__":
    # fire.Fire(main)
    main()
