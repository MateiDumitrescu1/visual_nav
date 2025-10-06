import numpy as np

def estimate_intrinsic_matrix(image_width, image_height, fov_degrees=60):
    """Estimates a basic intrinsic matrix K."""
    #TODO understand what this does
    # Approximate focal length based on FOV. Assumes fx = fy.
    # tan(fov_rad / 2) = (image_width / 2) / fx
    # fx = (image_width / 2) / tan(fov_degrees * pi / 180 / 2)
    fov_rad = np.deg2rad(fov_degrees)
    fx = fy = (image_width / 2.0) / np.tan(fov_rad / 2.0)
    cx = image_width / 2.0
    cy = image_height / 2.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    # print(f"Estimated Intrinsic Matrix K:\n{K}")
    return K
