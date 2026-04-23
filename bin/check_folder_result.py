#!/usr/bin/env python3
"""
Check per-tomo consistency among:
1) *_vesicle.json
2) *_label_vesicle.mrc
3) *_vesicle_class.xml

Selection rule:
- tomo is checked in segVesicle_QCheckBox_state.json
- tomo is not broken in segVesicle_heart_broken.json
"""

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Sequence, Tuple

import mrcfile
import numpy as np


VESICLE_NAME_RE = re.compile(r"^vesicle_(\d+)$")


def natural_sort_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def to_base_tomo_name(tomo_name: str) -> str:
    return tomo_name.split("-1")[0] if "-1" in tomo_name else tomo_name


def round_index(value: float) -> int:
    return int(np.rint(float(value)))


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_selected_tomos(current_path: str) -> Tuple[List[str], List[str]]:
    state_file = os.path.join(current_path, "segVesicle_QCheckBox_state.json")
    heart_state_file = os.path.join(current_path, "segVesicle_heart_broken.json")

    notes: List[str] = []
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"missing state file: {state_file}")

    checkbox_states = load_json(state_file)
    heart_states: Dict[str, bool] = {}
    if os.path.exists(heart_state_file):
        heart_states = load_json(heart_state_file)
    else:
        notes.append(f"heart state file not found, treat all as not broken: {heart_state_file}")

    selected: List[str] = []
    for tomo_name, checked in checkbox_states.items():
        if bool(checked) and not bool(heart_states.get(tomo_name, False)):
            selected.append(tomo_name)

    selected.sort(key=natural_sort_key)
    return selected, notes


def parse_json_vesicles(json_file_path: str) -> Tuple[List[Tuple[int, Tuple[float, float, float]]], List[str]]:
    issues: List[str] = []
    data = load_json(json_file_path)
    vesicles = data.get("vesicles", [])
    if not isinstance(vesicles, list):
        return [], [f"json format error: 'vesicles' should be a list in {json_file_path}"]

    records: List[Tuple[int, Tuple[float, float, float]]] = []
    seen_ids = set()
    for idx, ves in enumerate(vesicles, start=1):
        if not isinstance(ves, dict):
            issues.append(f"json entry #{idx} is not an object")
            continue

        name = ves.get("name", "")
        center = ves.get("center", None)

        match = VESICLE_NAME_RE.match(str(name))
        if not match:
            issues.append(f"json entry #{idx} invalid name: {name!r}")
            continue

        vesicle_id = int(match.group(1))
        if vesicle_id in seen_ids:
            issues.append(f"json duplicate vesicle id: {vesicle_id}")
            continue
        seen_ids.add(vesicle_id)

        if not isinstance(center, Sequence) or len(center) < 3:
            issues.append(f"json vesicle_{vesicle_id} invalid center: {center!r}")
            continue
        try:
            z = float(center[0])
            y = float(center[1])
            x = float(center[2])
        except (TypeError, ValueError):
            issues.append(f"json vesicle_{vesicle_id} center not numeric: {center!r}")
            continue

        records.append((vesicle_id, (z, y, x)))

    return records, issues


def parse_xml_vesicles(xml_file_path: str) -> Tuple[List[Tuple[int, Tuple[float, float, float]]], List[str]]:
    issues: List[str] = []
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        return [], [f"xml parse error: {xml_file_path}: {exc}"]

    records: List[Tuple[int, Tuple[float, float, float]]] = []
    seen_ids = set()
    for ves in root.findall("Vesicle"):
        type_element = ves.find("Type")
        if type_element is not None and type_element.attrib.get("t") == "pit":
            continue

        vesicle_id_text = ves.attrib.get("vesicleId")
        if vesicle_id_text is None:
            issues.append("xml vesicle missing vesicleId")
            continue
        try:
            vesicle_id = int(vesicle_id_text)
        except ValueError:
            issues.append(f"xml invalid vesicleId: {vesicle_id_text!r}")
            continue

        if vesicle_id in seen_ids:
            issues.append(f"xml duplicate vesicle id: {vesicle_id}")
            continue
        seen_ids.add(vesicle_id)

        center = ves.find("Center")
        if center is None:
            issues.append(f"xml vesicle_{vesicle_id} missing <Center>")
            continue

        try:
            x = float(center.attrib["X"])
            y = float(center.attrib["Y"])
            z = float(center.attrib["Z"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"xml vesicle_{vesicle_id} invalid Center attrs: {center.attrib}")
            continue

        records.append((vesicle_id, (z, y, x)))

    return records, issues


def get_label_value(label_data: np.ndarray, z: int, y: int, x: int) -> Tuple[int, str]:
    if not (0 <= z < label_data.shape[0] and 0 <= y < label_data.shape[1] and 0 <= x < label_data.shape[2]):
        return 0, f"center out of bounds: (z={z}, y={y}, x={x}), label shape={tuple(label_data.shape)}"
    return int(label_data[z, y, x]), ""


def check_records_against_label(
    records: List[Tuple[int, Tuple[float, float, float]]],
    label_data: np.ndarray,
    source_name: str,
) -> List[str]:
    issues: List[str] = []
    for vesicle_id, (z, y, x) in records:
        zi = round_index(z)
        yi = round_index(y)
        xi = round_index(x)
        label_value, err = get_label_value(label_data, zi, yi, xi)
        if err:
            issues.append(f"{source_name} vesicle_{vesicle_id}: {err}")
            continue
        if label_value != vesicle_id:
            issues.append(
                f"{source_name} vesicle_{vesicle_id}: label[{zi},{yi},{xi}]={label_value}, expected {vesicle_id}"
            )
    return issues


def check_one_tomo(current_path: str, tomo_name: str) -> Dict[str, object]:
    base_tomo_name = to_base_tomo_name(tomo_name)
    class_xml_path = os.path.join(
        current_path,
        tomo_name,
        "ves_seg",
        "vesicle_analysis",
        f"{base_tomo_name}_vesicle_class.xml",
    )
    label_path = os.path.join(current_path, tomo_name, "ves_seg", f"{base_tomo_name}_label_vesicle.mrc")
    json_file_path = os.path.join(current_path, tomo_name, "ves_seg", f"{base_tomo_name}_vesicle.json")

    issues: List[str] = []
    for path in [json_file_path, label_path, class_xml_path]:
        if not os.path.exists(path):
            issues.append(f"missing file: {path}")

    if issues:
        return {
            "tomo_name": tomo_name,
            "ok": False,
            "issues": issues,
        }

    try:
        with mrcfile.open(label_path, permissive=True) as mrc:
            label_data = np.asarray(mrc.data)
    except Exception as exc:
        return {
            "tomo_name": tomo_name,
            "ok": False,
            "issues": [f"failed to open label mrc: {label_path}: {exc}"],
        }

    if label_data.ndim != 3:
        return {
            "tomo_name": tomo_name,
            "ok": False,
            "issues": [f"label mrc is not 3D: {label_path}, ndim={label_data.ndim}"],
        }

    json_records, json_parse_issues = parse_json_vesicles(json_file_path)
    xml_records, xml_parse_issues = parse_xml_vesicles(class_xml_path)
    issues.extend(json_parse_issues)
    issues.extend(xml_parse_issues)

    issues.extend(check_records_against_label(json_records, label_data, "json"))
    issues.extend(check_records_against_label(xml_records, label_data, "xml"))

    return {
        "tomo_name": tomo_name,
        "ok": len(issues) == 0,
        "issues": issues,
        "num_json_records": len(json_records),
        "num_xml_records_non_pit": len(xml_records),
    }


def run(current_path: str, max_issues: int, only_failed: bool) -> int:
    current_path = os.path.abspath(current_path)
    print(f"current path: {current_path}")

    try:
        selected_tomos, notes = get_selected_tomos(current_path)
    except Exception as exc:
        print(f"failed to load tomo selection: {exc}")
        return 1

    for note in notes:
        print(f"note: {note}")

    if not selected_tomos:
        print("no tomo selected (checked and not broken).")
        return 0

    print(f"selected tomo count: {len(selected_tomos)}")

    results = []
    for tomo_name in selected_tomos:
        result = check_one_tomo(current_path, tomo_name)
        results.append(result)

        if result["ok"]:
            if not only_failed:
                print(
                    f"[PASS] {tomo_name} "
                    f"(json={result.get('num_json_records', 0)}, xml_non_pit={result.get('num_xml_records_non_pit', 0)})"
                )
            continue

        print(f"[FAIL] {tomo_name}: {len(result['issues'])} issue(s)")
        for issue in result["issues"][:max_issues]:
            print(f"  - {issue}")
        if len(result["issues"]) > max_issues:
            remain = len(result["issues"]) - max_issues
            print(f"  - ... and {remain} more issue(s)")

    failed = [r for r in results if not r["ok"]]
    passed = len(results) - len(failed)
    print("")
    print(f"summary: total={len(results)}, pass={passed}, fail={len(failed)}")

    if failed:
        print("failed tomos:")
        for r in failed:
            print(f"  - {r['tomo_name']}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check consistency for selected tomos: json center id, xml center id, and label id."
    )
    parser.add_argument(
        "current_path",
        nargs="?",
        default=".",
        help="Working directory containing segVesicle_QCheckBox_state.json (default: current dir).",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=20,
        help="Max issue lines shown per failed tomo.",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Only print failed tomo lines (plus summary).",
    )
    args = parser.parse_args()

    return run(args.current_path, max_issues=args.max_issues, only_failed=args.only_failed)


if __name__ == "__main__":
    raise SystemExit(main())
