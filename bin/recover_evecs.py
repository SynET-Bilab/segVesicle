#!/usr/bin/env python3
"""
Recover Evecs/Distance/ProjectionPoint from *_ori.xml to target xml files.

Usage:
    python bin/recover_evecs.py                # use current working directory
    python bin/recover_evecs.py /path/to/work  # use specified working directory

The script reads segVesicle_QCheckBox_state.json under the working directory,
finds tomos checked as true, and recovers:
    *_filter.xml
    *_vesicle_class.xml
using the corresponding *_ori.xml.
"""
import os
import json
import fire

from util.recover_evecs_from_ori import recover_evecs_from_ori


def _get_checkbox_tomos(json_path: str) -> list:
    if not os.path.exists(json_path):
        print(f"segVesicle_QCheckBox_state.json not found: {json_path}")
        return []

    with open(json_path, "r") as f:
        state = json.load(f)

    return [tomo_name for tomo_name, checked in state.items() if checked]


def _recover_for_tomo(current_path: str, tomo_name: str) -> None:
    base_tomo_name = tomo_name.split("-1")[0] if "-1" in tomo_name else tomo_name
    vesicle_dir = os.path.join(current_path, tomo_name, "ves_seg", "vesicle_analysis")

    ori_xml_path = os.path.join(vesicle_dir, f"{base_tomo_name}_ori.xml")
    if not os.path.exists(ori_xml_path):
        print(f"ori xml not found, skip: {ori_xml_path}")
        return

    target_xmls = [
        os.path.join(vesicle_dir, f"{base_tomo_name}_filter.xml"),
        os.path.join(vesicle_dir, f"{base_tomo_name}_vesicle_class.xml"),
    ]

    for target_xml in target_xmls:
        if not os.path.exists(target_xml):
            print(f"target xml not found, skip: {target_xml}")
            continue
        try:
            recover_evecs_from_ori(ori_xml_path, target_xml)
            print(f"recovered: {target_xml}")
        except Exception as exc:
            print(f"failed to recover {target_xml}: {exc}")


def main(current_path: str = ".") -> None:
    current_path = os.path.abspath(current_path)
    json_path = os.path.join(current_path, "segVesicle_QCheckBox_state.json")

    active_tomos = _get_checkbox_tomos(json_path)
    if not active_tomos:
        print("no checked tomo found")
        return

    for tomo_name in active_tomos:
        _recover_for_tomo(current_path, tomo_name)


if __name__ == "__main__":
    fire.Fire(main)
