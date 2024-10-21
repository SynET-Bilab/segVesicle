import os
import json
import argparse
import re
import numpy as np
import xml.etree.ElementTree as ET
import mrcfile
from util.structures import VesicleList, Surface  # Assuming these modules are available

def get_patch_around_point(data, z, y, x, size=128):
    """
    Extracts a patch of a specified size from 3D data, handles boundaries, and fills missing parts with 0.
    
    Parameters:
        data (np.ndarray): 3D image data (z, y, x).
        z, y, x (int): Coordinates of the center point.
        size (int): Size of the patch along each axis.
    
    Returns:
        patch (np.ndarray): Extracted patch.
    """
    half_size = size // 2
    patch = np.zeros((size, size, size), dtype=data.dtype)
    
    # Define the start and end coordinates for the patch
    z_min = z - half_size
    z_max = z + half_size
    y_min = y - half_size
    y_max = y + half_size
    x_min = x - half_size
    x_max = x + half_size

    # Dimensions of the image data
    z_dim, y_dim, x_dim = data.shape

    # Compute the overlap between the patch and the image boundaries and determine the position in the patch
    # Z-axis
    if z_min < 0:
        patch_z_start = -z_min
        z_min = 0
    else:
        patch_z_start = 0
    if z_max > z_dim:
        patch_z_end = size - (z_max - z_dim)
        z_max = z_dim
    else:
        patch_z_end = z_max - (z - half_size)

    # Y-axis
    if y_min < 0:
        patch_y_start = -y_min
        y_min = 0
    else:
        patch_y_start = 0
    if y_max > y_dim:
        patch_y_end = size - (y_max - y_dim)
        y_max = y_dim
    else:
        patch_y_end = y_max - (y - half_size)

    # X-axis
    if x_min < 0:
        patch_x_start = -x_min
        x_min = 0
    else:
        patch_x_start = 0
    if x_max > x_dim:
        patch_x_end = size - (x_max - x_dim)
        x_max = x_dim
    else:
        patch_x_end = x_max - (x - half_size)

    # Compute the valid region in the image data
    data_z_start = z_min
    data_z_end = z_max
    data_y_start = y_min
    data_y_end = y_max
    data_x_start = x_min
    data_x_end = x_max

    # Compute the placement position in the patch
    patch_z_end = patch_z_start + (data_z_end - data_z_start)
    patch_y_end = patch_y_start + (data_y_end - data_y_start)
    patch_x_end = patch_x_start + (data_x_end - data_x_start)

    # Fill the valid data into the patch
    patch[patch_z_start:patch_z_end, patch_y_start:patch_y_end, patch_x_start:patch_x_end] = \
        data[data_z_start:data_z_end, data_y_start:data_y_end, data_x_start:data_x_end]
    
    return patch

def vesicle_distance_and_filter(json_path, mod_path, xml_output_path, filter_xml_path, distance_nm, isonet_tomo_path):
    try:
        ori_pix_size = 1.714  # nm
        resampled_pix_size = 1.714  # nm
        ratio = resampled_pix_size / ori_pix_size

        # Check if the paths exist
        if not os.path.exists(json_path):
            print(f"JSON file path does not exist: {json_path}")
            return
        
        if not os.path.exists(mod_path):
            print(f"Mod file path does not exist: {mod_path}")
            return

        if not os.path.exists(isonet_tomo_path):
            print(f"Tomogram file path does not exist: {isonet_tomo_path}")
            return

        # Read JSON file
        with open(json_path, 'r') as f:
            data_json = json.load(f)

        vesicles = data_json.get('vesicles', [])
        if not vesicles:
            print("No 'vesicles' data found in the JSON file.")
            return

        center_list = []
        for vesicle in vesicles:
            center = ratio * np.asarray(vesicle.get('center', [0, 0, 0]))
            center_list.append(center)
        center_list = np.asarray(center_list)
        
        # Convert z, y, x to x, y, z
        center_list = center_list[:, [2, 1, 0]]

        # Initialize VesicleList
        vl = VesicleList(ori_pix_size)
        vl.fromCenterList(center_list)

        # Initialize Surface and read .mod file
        surface = Surface()
        surface.from_model_auto_segment(mod_path, objNum=2)

        # Process each vesicle
        for i, vesicle in enumerate(vl):
            ves_data = vesicles[i]
            
            name = ves_data.get('name', 'vesicle_0')  # Default to 'vesicle_0' if name not found
            vesicle_id = int(re.search(r'\d+', name).group())  # Extract the number from the name
            vesicle.setId(vesicle_id)  # Set the vesicle ID
    
            radii = ves_data.get('radii', [0])
            center = ves_data.get('center', [0, 0, 0])
            directions = ves_data.get('evecs', [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            center_scaled = ratio * np.asarray(center)
            center_xyz = center_scaled[[2, 1, 0]]  # Convert z, y, x to x, y, z

            vesicle.setRadius(np.mean(ratio * np.asarray(radii)))  # Set average radius
            vesicle.setRadius3D(ratio * np.asarray(radii))         # Set 3D radius
            vesicle.setCenter(center_xyz)                          # Set center point
            vesicle.setType('vesicle')                             # Set type
            vesicle._evecs = np.asarray(directions)                # Set direction vectors

            _, radius2D, eigvecs = vesicle.ellipse_in_plane()
            vesicle.setRadius2D(ratio * np.asarray(radius2D))      # Set 2D radius
            vesicle._rotation2D = np.arctan2(eigvecs[0, 1], eigvecs[0, 0]) - np.pi / 2  # Set 2D rotation angle

        # Calculate distance to the surface
        vl.distance_to_surface(surface, 3600, 'dense')

        # Save XML file
        vl.toXMLFile(xml_output_path)
        print(f"Distance calculation XML successfully generated: {xml_output_path}")

        # Filter vesicles
        filter_vesicles(xml_output_path, filter_xml_path, distance_nm)

        # Load and save Tomogram data
        with mrcfile.open(isonet_tomo_path, "r") as tomo:
            tomo_data = tomo.data.copy()

        # Set voxel size to 17.14 Å (1.714 nm)
        voxel_size = 17.14  # Å

        # Create or rebuild the extractRRP_3D folder
        extract_folder = os.path.join(os.path.dirname(xml_output_path), 'extractRRP_3D')
        if os.path.exists(extract_folder):
            # Delete existing folder
            import shutil
            shutil.rmtree(extract_folder)
        os.makedirs(extract_folder, exist_ok=True)
        print(f"Created folder: {extract_folder}")

        # Read the filtered XML file
        tree = ET.parse(filter_xml_path)
        root = tree.getroot()
        filtered_vesicles = root.findall('Vesicle')

        # Only process vesicles where t=='others'
        for vesicle in filtered_vesicles:
            type_elem = vesicle.find('Type')
            if type_elem is not None and type_elem.attrib.get('t') == 'others':
                vesicle_id = vesicle.attrib['vesicleId']
                center = vesicle.find('Center')
                x = float(center.attrib['X'])
                y = float(center.attrib['Y'])
                z = float(center.attrib['Z'])

                # Convert coordinates to integers
                x = int(round(x))
                y = int(round(y))
                z = int(round(z))

                # Extract patch
                patch = get_patch_around_point(tomo_data, z, y, x, size=128)

                # Save the patch as .mrc file
                output_filename = f'vesicle_{vesicle_id}.mrc'
                output_path = os.path.join(extract_folder, output_filename)
                with mrcfile.new(output_path, overwrite=True) as output_mrc:
                    output_mrc.set_data(patch.astype(np.float32))
                    output_mrc.voxel_size = voxel_size
                print(f"Saved cropped image: {output_path}")

        print("All filtered vesicle patches have been saved.")

    except Exception as e:
        print(f"Vesicle distance calculation and filtering failed: {str(e)}")

def filter_vesicles(xml_path, filter_xml_path, distance_nm):
    try:
        # Parse the original XML data
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Default pixel size if not found in the XML
        pixel_size = float(root.attrib.get('pixelSize', 1.714))  # nm

        # Filter vesicles based on distance
        for vesicle in root.findall('Vesicle'):
            distance = float(vesicle.find('Distance').attrib['d'])  # Assuming 'd' is the distance
            if distance * pixel_size < distance_nm:
                # If distance is less than the threshold, change Type to 'others'
                type_elem = vesicle.find('Type')
                if type_elem is not None and type_elem.attrib.get('t') == 'vesicle':
                    type_elem.set('t', 'others')

        # Save the filtered XML
        tree.write(filter_xml_path, encoding='utf-8', xml_declaration=True)
        print(f"Filtered vesicle XML successfully saved: {filter_xml_path}")

    except Exception as e:
        print(f"Vesicle filtering failed: {str(e)}")

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
