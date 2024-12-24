# @Auther: Liu Shuo

import os
import tempfile
import subprocess
import mrcfile
import pandas as pd
import numpy as np

def get_tomo(path):
    """
    Load a 3D MRC file as a numpy array.

    Parameters:
    - path: str
        Path to the MRC file.

    Returns:
    - data: ndarray
        The 3D data loaded from the MRC file.
    """
    with mrcfile.open(path) as mrc:
        data = mrc.data
    return data
#save_points_as_mod(points, object_id=1, model_file="/share/data/CryoET_Data/lvzy/segVesicle_test/p545/ves_seg/sampled_points.mod")

def save_points_as_mod(points: np.ndarray, object_id: int, model_file: str):
    """
    Saves the sampled points as a .mod file, grouping them by z and creating contours.
    
    :param points: An array of points to save, where each unique z defines a new group.
    :param object_id: The object ID to use in the .mod file.
    :param model_file: The path to save the .mod file.
    """
    # Round z values to the nearest integer
    
    points[:, 2] = np.round(points[:, 2]).astype(int)
    
    # Group points by z-value
    z_groups = {}
    for point in points:
        z = point[2]
        if z not in z_groups:
            z_groups[z] = []
        z_groups[z].append(point)
    
    # Prepare data for DataFrame
    data = []
    contour_count = 0
    for z, group in z_groups.items():
        if len(group) > 1:  # Check group size and limit contour count
            for point in group:
                # object_id, contour_id, x, y, z (1-based for object and contour)
                data.append([object_id, contour_count + 1, point[0], point[1], point[2]])
            contour_count += 1

    # Create DataFrame and write to .mod file
    df = pd.DataFrame(data, columns=["object", "contour", "x", "y", "z"])
    write_model(model_file, df)

def write_model(model_file: str, model_df: pd.DataFrame):
    """
    Converts the point data to a .mod file format using the IMOD tool point2model.
    
    :param model_file: Path where the .mod file will be saved.
    :param model_df: DataFrame containing the point data.
    """
    model = np.asarray(model_df)

    # Ensure the directory exists
    model_dir = os.path.dirname(model_file)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save points to a temporary file and convert to .mod
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=".") as temp_file:
        # Save point data to a temporary .pt file
        point_file = temp_file.name
        np.savetxt(point_file, model, fmt=(['%d']*2 + ['%.2f']*3))


    points[:, 2] = np.round(points[:, 2]).astype(int)
    
    # Group points by z-value
    z_groups = {}
    for point in points:
        z = point[2]
        if z not in z_groups:
            z_groups[z] = []
        z_groups[z].append(point)
    
    # Prepare data for DataFrame
    data = []
    contour_count = 0
    for z, group in z_groups.items():
        if len(group) > 1:  # Check group size and limit contour count
            for point in group:
                # object_id, contour_id, x, y, z (1-based for object and contour)
                data.append([object_id, contour_count + 1, point[0], point[1], point[2]])
            contour_count += 1

    # Create DataFrame and write to .mod file
    df = pd.DataFrame(data, columns=["object", "contour", "x", "y", "z"])
    write_model(model_file, df)

def write_model(model_file: str, model_df: pd.DataFrame):
    """
    Converts the point data to a .mod file format using the IMOD tool point2model.
    
    :param model_file: Path where the .mod file will be saved.
    :param model_df: DataFrame containing the point data.
    """
    model = np.asarray(model_df)

    # Ensure the directory exists
    model_dir = os.path.dirname(model_file)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save points to a temporary file and convert to .mod
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=".") as temp_file:
        # Save point data to a temporary .pt file
        point_file = temp_file.name
        np.savetxt(point_file, model, fmt=(['%d']*2 + ['%.2f']*3))

        # Use point2model to convert the point file to a .mod file
        cmd = f"point2model -op {point_file} {model_file} >/dev/null"
        subprocess.run(cmd, shell=True, check=True)
    print(f".mod file saved successfully to {model_file}")