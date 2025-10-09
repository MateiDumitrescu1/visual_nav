
#! the code for the werko demo, lazer focused on making it look as good as possible
#! be prepared to answer questions about how it works in detail

#TODO estimate rotation with optical flow and match the drone frame at the right rotation to the satellite image
from typing import Dict
import os
import time
import cv2
import numpy as np
from functools import cache
from paths_ import output_dir, images_dir
from dvn.models.xfeat_.xfeat_methods import XFeatModel
from dvn.models.xfeat_.xfeat_utils import save_features_to_folder, load_features_from_folder
from dvn.utils.cv_utils.warp_corners_and_draw_matches import warp_corners_and_draw_matches

rotated_sat_dir = output_dir + '/rotated_sat_img'
drone_frames_dir = images_dir + '/marco_sunny_frames'
demo_output_dir = output_dir + '/demo_output'

#TODO make the features be saved in folders: sparse_ and dense_ so that we can load them separately
def get_sat_img_features(load_device="cuda", dense:bool = False) -> Dict[float, dict]:
    """
    Read the folder with all the rotations of the sat images.
    For each image, see if the folder already has the features computed. If not, compute and save them.

    ### Params:
        load_device: str
            Device to load features (using `load_features_from_folder`) to ('cpu' or 'cuda'). Default is 'cuda'.
        dense: bool
            If True, use dense feature detection (xfeat_detect_and_compute_DENSE) and save/load with 'dense' prefix.
            Default is False (uses sparse features).

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
    print(f"Using {'dense' if dense else 'sparse'} feature detection")

    for image_file in sorted(image_files):
        # Get the filename without extension for use as prefix
        filename_prefix = os.path.splitext(image_file)[0]

        # Add 'dense' prefix if using dense features
        if dense:
            filename_prefix = f"dense_{filename_prefix}"

        image_path = os.path.join(rotated_sat_dir, image_file)

        # Extract rotation angle from filename (e.g., "sat_rotated_45.png" -> 45.0)
        try:
            # Expecting format: sat_rotated_<angle>.png
            # If dense prefix is used, it will be "dense_sat_rotated_<angle>"
            base_filename = filename_prefix.replace("dense_", "") if dense else filename_prefix
            rotation_angle = float(base_filename.split('_')[-1])
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

            # Detect and compute features (dense or sparse based on parameter)
            if dense:
                features = xfeat_model.xfeat_detect_and_compute_DENSE(image, top_k=8000)
            else:
                features = xfeat_model.xfeat_detect_and_compute(image, top_k=4096)

            # Save features for future use
            save_features_to_folder(features, rotated_sat_dir, filename_prefix)
            print(f"✓ Computed and saved features for {filename_prefix} (angle: {rotation_angle}°)")

        # Store in dictionary using rotation angle as key
        all_features[rotation_angle] = features

    print(f"\nTotal features loaded/computed: {len(all_features)}")
    return all_features


@cache
def get_sat_img_features_sparse(load_device="cuda") -> Dict[float, dict]:
    """
    @cache wrapper to get sparse satellite image features.
    """
    return get_sat_img_features(load_device=load_device, dense=False)

@cache
def get_sat_img_features_dense(load_device="cuda") -> Dict[float, dict]:
    """
    @cache wrapper to get dense satellite image features.
    """
    return get_sat_img_features(load_device=load_device, dense=True)


def get_original_sat_img_features(dense: bool = False) -> dict:
    """
    Get features for the original (0°, meaning no rotation) satellite image.
    """
    if dense:
        feature_dict = get_sat_img_features_dense()
    else:
        feature_dict = get_sat_img_features_sparse()
        
    if 0.0 not in feature_dict:
        raise ValueError("No features found for the original (0°) satellite image.")
    return feature_dict[0.0]

@cache
def get_drone_frames(downsample_factor: float = 1.0) -> list[np.ndarray]:
    """
    Read all images from drone_frames_dir/ the appropriate downsampled folder.
    """
    drone_frames_dir_to_read = None
    if downsample_factor == 1.0: 
        drone_frames_dir_to_read = drone_frames_dir + '/original'
    else:
        folder_name = f"downsampled_{str(downsample_factor).replace('.', '_')}"
        drone_frames_dir_to_read = os.path.join(drone_frames_dir, folder_name)
        if not os.path.exists(drone_frames_dir_to_read):
            raise FileNotFoundError(f"Downsampled frames folder not found: {drone_frames_dir_to_read}")
    
    drone_frames = []
    for image_file in os.listdir(drone_frames_dir_to_read):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(drone_frames_dir_to_read, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                drone_frames.append(image)
            else:
                raise ValueError(f"Failed to read image: {image_path}")
    return drone_frames

#! ---------------- DEMO PIPELINES ----------------
def demo0():
    """
    Use the frames that are downsampled by 0.6
    For each of those frames:
    1. match against all rotations of the satelite (the pre-computed ones)
    2. pick the one with most matches
    3. save the visualization (with draw_matches) to demo_output_dir / demo0 / current_time
    """
    downsample_factor = 0.6

    # Create timestamped output directory for demo0
    # Human-readable, filesystem-safe timestamp (e.g. "2025-10-09_14-32-05")
    current_time: str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    demo0_output_dir = os.path.join(demo_output_dir, 'demo0', current_time)
    os.makedirs(demo0_output_dir, exist_ok=True)
    print(f"Output directory created: {demo0_output_dir}")

    # Load drone frames
    print("Loading drone frames...")
    drone_frames = get_drone_frames(downsample_factor=downsample_factor)
    print(f"Loaded {len(drone_frames)} drone frames")

    # Load satellite image features for all rotations
    print("\nLoading satellite image features for all rotations...")
    sat_features_dict = get_sat_img_features(load_device="cuda")
    print(f"Loaded features for {len(sat_features_dict)} satellite rotations")

    # Load the satellite images (we need them for visualization)
    print("\nLoading satellite images...")
    sat_images_dict = {}
    for image_file in os.listdir(rotated_sat_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename_prefix = os.path.splitext(image_file)[0]
            try:
                rotation_angle = float(filename_prefix.split('_')[-1])
                image_path = os.path.join(rotated_sat_dir, image_file)
                sat_img = cv2.imread(image_path)
                if sat_img is not None:
                    sat_images_dict[rotation_angle] = sat_img
            except (ValueError, IndexError):
                continue
    print(f"Loaded {len(sat_images_dict)} satellite images")

    # Initialize XFeat model for feature detection and matching
    print("\nInitializing XFeat model...")
    xfeat_model = XFeatModel(top_k=4096)

    # Process each drone frame (the frames are already downsampled)
    for frame_idx, drone_frame in enumerate(drone_frames):
        print(f"\n{'='*60}")
        print(f"Processing drone frame {frame_idx + 1}/{len(drone_frames)}")
        print(f"{'='*60}")

        # The drone frame is already downsampled when loaded
        h, w = drone_frame.shape[:2]
        print(f"Drone frame dimensions: {w}x{h}")

        # Compute features for the drone frame
        print("Computing features for drone frame...")
        drone_features = xfeat_model.xfeat_detect_and_compute(drone_frame, top_k=4096)
        print(f"Detected {drone_features['keypoints'].shape[0]} keypoints in drone frame")

        # Match against all satellite rotations
        best_rotation = None
        best_match_count = 0
        best_mkpts_drone = None
        best_mkpts_sat = None

        print("\nMatching against all satellite rotations...")
        for rotation_angle in sorted(sat_features_dict.keys()):
            sat_features = sat_features_dict[rotation_angle]

            # Match features using sparse matching
            mkpts_drone, mkpts_sat = xfeat_model.xfeat_match_sparse_default(
                drone_features,
                sat_features
            )

            match_count = len(mkpts_drone)
            print(f"  Rotation {rotation_angle:6.1f}°: {match_count:4d} matches")

            # Update best match if this rotation has more matches
            if match_count > best_match_count:
                best_match_count = match_count
                best_rotation = rotation_angle
                best_mkpts_drone = mkpts_drone
                best_mkpts_sat = mkpts_sat

        if best_mkpts_sat is None:
            print("No matches found for any rotation, skipping visualization.")
            continue

        print(f"\nBest rotation: {best_rotation}° with {best_match_count} matches")

        # Create visualization with the best match
        if best_rotation is not None and best_mkpts_drone is not None:
            
            print("Creating visualization...")
            sat_img = sat_images_dict[best_rotation]

            sat_h, sat_w = sat_img.shape[:2]
            print(f"Using satellite image (rotation {best_rotation}°): {sat_w}x{sat_h}")

            nr_matches = len(best_mkpts_drone)
            
            # Create the visualization
            viz_canvas = warp_corners_and_draw_matches(
                ref_points=best_mkpts_drone,
                dst_points=best_mkpts_sat,
                img1=drone_frame,
                img2=sat_img,  # Use full-size satellite image
                draw_match_lines=True,
                thickness=1
            )

            if viz_canvas is not None:
                # Save the visualization
                output_path = os.path.join(demo0_output_dir, f'frame_{frame_idx:03d}_rot_{best_rotation:.1f}@{nr_matches}_matches.png')
                cv2.imwrite(output_path, viz_canvas)
                print(f"✓ Saved visualization to: {output_path}")
            else:
                print("⚠️  Visualization failed (homography estimation failed)")
        else:
            print("⚠️  No matches found for this frame")

    print(f"\n{'='*60}")
    print(f"Demo0 completed! Results saved to: {demo0_output_dir}")
    print(f"{'='*60}")

def run_demo():
    demo0()

#! ---------------- TESTING ----------------
def test_get_sat_img_features():
    # Test sparse features
    features_dict = get_sat_img_features_sparse()
    for angle, feats in features_dict.items():
        print(f"Angle: {angle}°, Keypoints: {feats['keypoints'].shape[0]}, Descriptors: {feats['descriptors'].shape[0]}")

    original_feats = get_original_sat_img_features(dense=False)
    assert original_feats is features_dict[0.0], "Original features do not match features at 0°"

    # Test dense features
    dense_features_dict = get_sat_img_features_dense()
    for angle, feats in dense_features_dict.items():
        print(f"[Dense] Angle: {angle}°, Keypoints: {feats['keypoints'].shape}, Descriptors: {feats['descriptors'].shape}")

    dense_original_feats = get_original_sat_img_features(dense=True)
    assert dense_original_feats is dense_features_dict[0.0], "Original dense features do not match features at 0°"

def test_get_drone_frames() -> None:
    frames = get_drone_frames()
    print(f"Loaded {len(frames)} drone frames")

if __name__ == '__main__':
    # test_get_sat_img_features()
    # test_get_drone_frames()
    run_demo()
    pass