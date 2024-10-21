import os
import re
import json
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
            center_xyz = center_scaled[[2, 1, 0]]  # Convert zyx to xyz

            vesicle.setRadius(np.mean(ratio * np.asarray(radii)))  # Set average radius
            vesicle.setRadius3D(ratio * np.asarray(radii))         # Set 3D radius
            vesicle.setCenter(center_xyz)                          # Set center point
            vesicle.setType('vesicle')                             # Set type
            vesicle._evecs = np.asarray(directions)                # Set direction vectors

            _, radius2D, eigvecs = vesicle.ellipse_in_plane()
            vesicle.setRadius2D(ratio * np.asarray(radius2D))      # Set 2D radius
            vesicle._rotation2D = np.arctan2(eigvecs[0, 1], eigvecs[0, 0]) - np.pi / 2  # Set 2D rotation angle

        # Calculate the distance to the surface
        vl.distance_to_surface(surface, 3600, 'dense')

        # Output the XML file
        vl.toXMLFile(xml_output_path)

        # Notify user of success
        print_func(f"XML file successfully generated at: {xml_output_path}")

    except Exception as e:
        # Handle exceptions and notify user of error
        print_func(f"Distance calculation failed: {str(e)}")
