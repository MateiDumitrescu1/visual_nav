
#! the code for the werko demo, lazer focused on making it look as good as possible
#! be prepared to answer questions about how it works in detail

#TODO estimate rotation with optical flow and match the drone frame at the right rotation to the satellite image
from typing import Dict
import os
import cv2
import numpy as np
from functools import cache
from paths_ import output_dir, images_dir
from dvn.models.xfeat_.xfeat_methods import XFeatModel
from dvn.models.xfeat_.xfeat_utils import save_features_to_folder, load_features_from_folder

rotated_sat_dir = output_dir + '/rotated_sat_img'
drone_frames_dir = images_dir + '/marco_sunny_frames/original'

@cache
def get_sat_img_features(load_device="cuda") -> Dict[float, dict]:
    """
    Read the folder with all the rotations of the sat images.
    For each image, see if the folder already has the features computed. If not, compute and save them.

    ### Params:
        load_device: str
            Device to load features (using `load_features_from_folder`) to ('cpu' or 'cuda'). Default is 'cuda'.
    
    ### Returns:
        Dictionary mapping rotation angles (in degrees) to their computed features.
        Each feature dict contains: 'keypoints', 'descriptors', 'scores', 'image_size'
    """
    # Create XFeat model for feature detection
    xfeat_model = XFeatModel(top_k=4096)

    # Dictionary to store all features, mapping rotation angle -> features
    all_features: Dict[float, dict] = {}

    # Get all PNG images in the rotated satellite directory
    if not os.path.exists(rotated_sat_dir):
        raise FileNotFoundError(f"Rotated satellite image directory not found: {rotated_sat_dir}")

    image_files = [f for f in os.listdir(rotated_sat_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        raise ValueError(f"No image files found in {rotated_sat_dir}")

    print(f"Found {len(image_files)} satellite images to process")

    for image_file in sorted(image_files):
        # Get the filename without extension for use as prefix
        filename_prefix = os.path.splitext(image_file)[0]
        image_path = os.path.join(rotated_sat_dir, image_file)

        # Extract rotation angle from filename (e.g., "sat_rotated_45.png" -> 45.0)
        try:
            # Expecting format: sat_rotated_<angle>.png
            rotation_angle = float(filename_prefix.split('_')[-1])
        except (ValueError, IndexError):
            print(f"Warning: Could not extract rotation angle from {filename_prefix}, skipping...")
            continue

        # Try to load existing features first
        try:
            features = load_features_from_folder(rotated_sat_dir, filename_prefix, device=load_device)
            print(f"✓ Loaded cached features for {filename_prefix} (angle: {rotation_angle}°)")
        except FileNotFoundError:
            # Features don't exist, compute them
            print(f"Computing features for {filename_prefix} (angle: {rotation_angle}°)...")

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to read image {image_path}, skipping...")
                continue

            # Detect and compute features
            features = xfeat_model.xfeat_detect_and_compute(image, top_k=4096)

            # Save features for future use
            save_features_to_folder(features, rotated_sat_dir, filename_prefix)
            print(f"✓ Computed and saved features for {filename_prefix} (angle: {rotation_angle}°)")

        # Store in dictionary using rotation angle as key
        all_features[rotation_angle] = features

    print(f"\nTotal features loaded/computed: {len(all_features)}")
    return all_features

@cache
def get_original_sat_img_features() -> dict:
    """
    Get features for the original (0°, meaning no rotation) satellite image.
    """
    feature_dict = get_sat_img_features()
    if 0.0 not in feature_dict:
        raise ValueError("No features found for the original (0°) satellite image.")
    return feature_dict[0.0]

@cache
def get_drone_frames() -> list[np.ndarray]:
    """
    Read all images from `drone_frames_dir` and return them as a list of numpy arrays.
    """
    drone_frames = []
    for image_file in os.listdir(drone_frames_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(drone_frames_dir, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                drone_frames.append(image)
            else:
                raise ValueError(f"Failed to read image: {image_path}")
    return drone_frames

#! ---------------- TESTING ----------------
def test_get_sat_img_features():
    features_dict = get_sat_img_features()
    for angle, feats in features_dict.items():
        print(f"Angle: {angle}°, Keypoints: {feats['keypoints'].shape[0]}, Descriptors: {feats['descriptors'].shape[0]}")
        
    original_feats = get_original_sat_img_features()
    assert original_feats == features_dict[0.0], "Original features do not match features at 0°"

def test_get_drone_frames() -> None:
    frames = get_drone_frames()
    print(f"Loaded {len(frames)} drone frames")

if __name__ == '__main__':
    # test_get_sat_img_features()
    # test_get_drone_frames()
    pass