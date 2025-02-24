import numpy as np
from skimage.morphology import cube, dilation, ball
from scipy.ndimage import binary_dilation

def membrane_point2mask(label_data, points_list):
    """
    Update label data by generating lines between seed points and applying dilation.

    Args:
        label_data (np.ndarray): The 3D label data array to update.
        points (list): A list of seed points where each point is (z, y, x).

    Returns:
        np.ndarray: The updated label data array.
    """
    # Initialize a binary mask
    bimask = np.zeros_like(label_data, dtype=np.int16)

    bimask[points_list[:, 0], points_list[:, 1], points_list[:, 2]] = 1
    # for points in points_list:
    #     for point in points:
    #         z, y, x = point
    #         bimask[z, y, x] = 1

    # Combine dilation operations
    # Step 1: Apply cube dilation
    # selem_cube = cube(3)
    # bimask = dilation(bimask, selem_cube)

    # # Step 2: Apply z-axis dilation
    # selem_z = np.zeros((3, 1, 1), dtype=np.int8)
    # selem_z[:, 0, 0] = 1
    # bimask = dilation(bimask, selem_z)

    return bimask