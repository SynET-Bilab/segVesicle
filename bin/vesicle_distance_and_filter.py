import argparse

from util.distance_calculator import distance_calc
from window.distance_filter_window import filter_vesicles_and_extract

def vesicle_distance_and_filter(json_path, mod_path, xml_output_path, filter_xml_path, distance_nm, isonet_tomo_path):
    try:
        distance_calc(json_path, mod_path, xml_output_path, print)
        filter_vesicles_and_extract(
            xml_path=xml_output_path,
            filter_xml_path=filter_xml_path,
            distance_nm=distance_nm,
            isonet_tomo_path=isonet_tomo_path,
            print_func=print,
        )

    except Exception as e:
        print(f"Vesicle distance calculation and filtering failed: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Vesicle Distance Calculation and Filtering with Cropping")
    parser.add_argument('--json_path', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--mod_path', type=str, required=True, help="Path to the input .mod file")
    parser.add_argument('--xml_output_path', type=str, required=True, help="Path to the output XML file")
    parser.add_argument('--filter_xml_path', type=str, required=True, help="Path to the filtered XML file")
    parser.add_argument('--distance_nm', type=float, required=True, help="Distance threshold for filtering vesicles (in nm)")
    parser.add_argument('--isonet_tomo_path', type=str, required=True, help="Path to the input Tomogram MRC file")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    vesicle_distance_and_filter(
        json_path=args.json_path,
        mod_path=args.mod_path,
        xml_output_path=args.xml_output_path,
        filter_xml_path=args.filter_xml_path,
        distance_nm=args.distance_nm,
        isonet_tomo_path=args.isonet_tomo_path
    )
