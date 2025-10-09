from typing import Dict
import os
import cv2
import numpy as np
from functools import cache
from paths_ import output_dir, images_dir
from dvn.models.xfeat_.xfeat_methods import XFeatModel
from dvn.models.xfeat_.xfeat_utils import save_features_to_folder, load_features_from_folder
from dvn.demos.main import get_drone_frames, rotated_sat_dir, get_original_sat_img_features, get_sat_img_features

def calibrate_min_cossim() -> None:
    """
    Run the sparse matching using the steerer XFeat model to find the value for `min_cossim` that gives the best matches
    against the original satellite image (0° rotation).

    This method tests different min_cossim threshold values (from -1.0 to 1.0) and reports the number of matches
    for each drone frame against the original (0°) satellite image. The goal is to find a good balance between
    having enough matches and filtering out false positives.
    """
    from dvn.utils.cv_utils.warp_corners_and_draw_matches import warp_corners_and_draw_matches
    from dvn.models.xfeat_.xfeat_methods import XFEAT_MODELS

    save_results_dir = output_dir + "/min_cossim_calibration_output"

    # Create output directory if it doesn't exist
    os.makedirs(save_results_dir, exist_ok=True)
    print(f"Results will be saved to: {save_results_dir}\n")

    # Initialize the steerer XFeat model
    xfeat_model_steerer = XFeatModel(top_k=4096, model_name=XFEAT_MODELS.STEERER_PRETRAINED)

    # Get the original satellite image features (0° rotation)
    print("Loading satellite image features...")
    sat_features = get_original_sat_img_features()

    # Load the original satellite image for visualization
    sat_img_path = os.path.join(rotated_sat_dir, "sat_rotated_0.png")
    sat_img = cv2.imread(sat_img_path)
    if sat_img is None:
        raise ValueError(f"Failed to load satellite image at {sat_img_path}")

    # Get the first drone frame only
    print("Loading drone frames...")
    drone_frames = get_drone_frames()
    print(f"Loaded {len(drone_frames)} drone frames")
    drone_frame = drone_frames[0]
    print(f"Using first drone frame for calibration\n")

    # Compute features for the drone frame once (reuse for all tests)
    print("Computing features for drone frame...")
    drone_features = xfeat_model_steerer.xfeat_detect_and_compute(drone_frame, top_k=4096)
    print(f"Detected {drone_features['keypoints'].shape[0]} keypoints\n")

    # Test different min_cossim values
    min_cossim_values = [-1.0, -0.5, 0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    print("=" * 80)
    print("CALIBRATING min_cossim THRESHOLD")
    print("=" * 80)

    # Store results for analysis
    results = {}

    for min_cossim in min_cossim_values:
        print(f"\nTesting min_cossim = {min_cossim}")
        print("-" * 80)

        # Match against satellite image using steerer sparse matching
        mkpts_0, mkpts_1, rot = xfeat_model_steerer.xfeat_match_sparse_steerer(
            drone_features,
            sat_features,
            min_cossim=min_cossim
        )

        num_matches = len(mkpts_0)
        print(f"  Matches: {num_matches} (rotation: {rot})")

        # Visualize and save the matches
        output_canvas = warp_corners_and_draw_matches(
            mkpts_0,
            mkpts_1,
            drone_frame,
            sat_img,
            draw_match_lines=True
        )

        if output_canvas is not None:
            # Save the visualization
            output_filename = f"min_cossim_{min_cossim:.2f}_matches_{num_matches}_rot_{rot}.png"
            output_path = os.path.join(save_results_dir, output_filename)
            cv2.imwrite(output_path, output_canvas)
            print(f"  Saved visualization to: {output_filename}")
        else:
            print(f"  Warning: Failed to create visualization for min_cossim={min_cossim}")

        results[min_cossim] = {
            'matches': num_matches,
            'rotation': rot
        }

    # Print final summary
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)
    print(f"{'min_cossim':<12} {'Matches':<15} {'Rotation':<15}")
    print("-" * 80)

    for min_cossim, stats in results.items():
        print(f"{min_cossim:<12.2f} {stats['matches']:<15} {stats['rotation']:<15}")

    print("\n" + "=" * 80)
    print(f"All visualizations saved to: {save_results_dir}")
    print("=" * 80)

def match_first_drone_frame_against_rotations():
    """
    Use the default XFeat.
    Match the first drone frame against all rotated satellite images that are available in the `get_sat_img_features` method.
    Tests different downsampling factors (1.0, 0.9, 0.8, 0.7, 0.6) to find optimal scale.
    Visualize the matches using draw_matches and save results to output directory.
    """
    from dvn.utils.cv_utils.warp_corners_and_draw_matches import draw_matches
    from dvn.models.xfeat_.xfeat_methods import XFEAT_MODELS

    # Create output directory for results
    save_results_dir = output_dir + "/first_frame_rotation_matches"
    os.makedirs(save_results_dir, exist_ok=True)
    print(f"Results will be saved to: {save_results_dir}\n")

    # Initialize the default XFeat model
    xfeat_model = XFeatModel(top_k=4096)

    # Get the first drone frame
    print("Loading drone frames...")
    drone_frames = get_drone_frames()
    drone_frame_original = drone_frames[0]
    print(f"Using first drone frame (original shape: {drone_frame_original.shape})\n")

    # Define downsampling factors to test
    downsample_factors = [1.0, 0.9, 0.8, 0.7, 0.6]

    # Get all rotated satellite image features
    print("Loading satellite image features for all rotations...")
    all_sat_features = get_sat_img_features()
    print(f"Loaded features for {len(all_sat_features)} rotations: {sorted(all_sat_features.keys())}°\n")

    # Store results for summary: results[downsample_factor][rotation_angle]
    results = {}

    print("=" * 80)
    print("MATCHING FIRST DRONE FRAME AGAINST ALL ROTATIONS WITH DIFFERENT SCALES")
    print("=" * 80)

    # Test each downsampling factor
    for downsample_factor in downsample_factors:
        print(f"\n{'=' * 80}")
        print(f"TESTING DOWNSAMPLE FACTOR: {downsample_factor}")
        print(f"{'=' * 80}\n")

        # Downsample the drone frame if needed
        if downsample_factor == 1.0:
            drone_frame = drone_frame_original
        else:
            new_width = int(drone_frame_original.shape[1] * downsample_factor)
            new_height = int(drone_frame_original.shape[0] * downsample_factor)
            drone_frame = cv2.resize(drone_frame_original, (new_width, new_height), interpolation=cv2.INTER_AREA)

        print(f"Drone frame shape: {drone_frame.shape} (scale: {downsample_factor})")

        # Compute features for the downsampled drone frame
        print("Computing features for drone frame...")
        drone_features = xfeat_model.xfeat_detect_and_compute(drone_frame, top_k=4096)
        print(f"Detected {drone_features['keypoints'].shape[0]} keypoints in drone frame\n")

        # Initialize results for this downsample factor
        results[downsample_factor] = {}

        # Match against each rotation
        for rotation_angle in sorted(all_sat_features.keys()):
            sat_features = all_sat_features[rotation_angle]

            print(f"  Processing rotation: {rotation_angle}°", end=" ")

            # Load the corresponding satellite image for visualization
            sat_img_path = os.path.join(rotated_sat_dir, f"sat_rotated_{int(rotation_angle)}.png")
            sat_img = cv2.imread(sat_img_path)

            if sat_img is None:
                print(f"[Warning: Failed to load satellite image, skipping...]")
                continue

            # Match using XFeat default sparse matching
            mkpts_0, mkpts_1 = xfeat_model.xfeat_match_sparse_default(
                drone_features,
                sat_features
            )

            num_matches = len(mkpts_0)
            print(f"-> {num_matches} matches")

            # Visualize matches using draw_matches
            if num_matches > 0:
                # Create visualization with match lines
                match_visualization = draw_matches(
                    img1=drone_frame,
                    img2=sat_img,
                    ref_points=mkpts_0,
                    dst_points=mkpts_1,
                    inlier_mask=None,  # Draw all matches
                    colors=None,  # Use default green color
                    thickness=1
                )

                # Save the visualization with downsample factor in filename
                output_filename = f"scale_{downsample_factor:.1f}_rotation_{int(rotation_angle):03d}_matches_{num_matches}.png"
                output_path = os.path.join(save_results_dir, output_filename)
                cv2.imwrite(output_path, match_visualization)

                results[downsample_factor][rotation_angle] = {
                    'matches': num_matches,
                    'saved': True
                }
            else:
                results[downsample_factor][rotation_angle] = {
                    'matches': 0,
                    'saved': False
                }

    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MATCHING SUMMARY")
    print("=" * 80)

    # For each downsample factor, show summary
    for downsample_factor in downsample_factors:
        if downsample_factor not in results or not results[downsample_factor]:
            continue

        print(f"\nDownsample Factor: {downsample_factor}")
        print("-" * 80)
        print(f"{'Rotation (°)':<15} {'Matches':<15} {'Visualization Saved':<20}")
        print("-" * 80)

        for rotation_angle in sorted(results[downsample_factor].keys()):
            stats = results[downsample_factor][rotation_angle]
            saved_status = "✓" if stats['saved'] else "✗"
            print(f"{rotation_angle:<15.1f} {stats['matches']:<15} {saved_status:<20}")

        # Find best rotation for this downsample factor
        best_rotation = max(results[downsample_factor].items(), key=lambda x: x[1]['matches'])
        print(f"\nBest rotation for scale {downsample_factor}: {best_rotation[0]}° with {best_rotation[1]['matches']} matches")

    # Find overall best combination
    print("\n" + "=" * 80)
    print("OVERALL BEST MATCH")
    print("=" * 80)

    best_overall = None
    best_matches = 0

    for downsample_factor, rotations in results.items():
        for rotation_angle, stats in rotations.items():
            if stats['matches'] > best_matches:
                best_matches = stats['matches']
                best_overall = (downsample_factor, rotation_angle, stats['matches'])

    if best_overall:
        print(f"Best overall match:")
        print(f"  Scale: {best_overall[0]}")
        print(f"  Rotation: {best_overall[1]}°")
        print(f"  Matches: {best_overall[2]}")
        print("=" * 80)

    print(f"\nAll visualizations saved to: {save_results_dir}")

    
if __name__ == '__main__':
    # calibrate_min_cossim()
    match_first_drone_frame_against_rotations()
    pass