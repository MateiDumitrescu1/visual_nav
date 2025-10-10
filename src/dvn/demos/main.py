
#! the code for the werko demo, lazer focused on making it look as good as possible
#! be prepared to answer questions about how it works in detail

#TODO estimate rotation with optical flow and match the drone frame at the right rotation to the satellite image
from typing import Dict
import os
import time
import copy
import cv2
import json
import numpy as np
from functools import cache
from paths_ import output_dir, images_dir
from dvn.models.xfeat_.xfeat_methods import XFeatModel
from dvn.models.xfeat_.xfeat_utils import save_features_to_folder, load_features_from_folder
from dvn.utils.cv_utils.warp_corners_and_draw_matches import warp_corners_and_draw_matches, warp_and_draw_corners, draw_matches, draw_corners, warp_corners
from dvn.utils.algebra_utils.homo import find_homography
from dvn.utils.cv_utils.shape_degeneration import is_shape_degenerated
from dvn.utils.cv_utils.shape_similarity import compute_shape_similarity

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
    for image_file in sorted(os.listdir(drone_frames_dir_to_read)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(drone_frames_dir_to_read, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                drone_frames.append(image)
            else:
                raise ValueError(f"Failed to read image: {image_path}")
    return drone_frames

def rotate_keypoints(keypoints: np.ndarray, angle_degrees: float, image_center: tuple[float, float]) -> np.ndarray:
    """
    Rotate keypoints around an image center by a given angle.

    This is useful for transforming keypoints from a rotated image back to the original orientation.

    ### Params:
        keypoints: np.ndarray
            nx2 array of (x, y) coordinates
        angle_degrees: float
            Rotation angle in degrees (positive = counterclockwise, negative = clockwise)
        image_center: tuple[float, float]
            (cx, cy) center point of rotation

    ### Returns:
        np.ndarray: Rotated keypoints (nx2 array)
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_degrees)

    # Create 2D rotation matrix
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])

    # Translate keypoints to origin (center of rotation)
    cx, cy = image_center
    kpts_centered = keypoints - np.array([cx, cy])

    # Apply rotation transformation
    kpts_rotated = kpts_centered @ rotation_matrix.T

    # Translate back to original coordinate system
    kpts_final = kpts_rotated + np.array([cx, cy])

    return kpts_final

#! ---------------- DEMO PIPELINES ----------------
cache_path = os.path.join(demo_output_dir, 'demo0', 'cache.json')
def read_frame_cache():
    """This file holds cached information, like what sat image rotation was best for each frame. We save it so we don't compute it again."""
    
    if not os.path.exists(cache_path):
        return {}
    
    with open(cache_path, 'r') as f:
        cache_data = json.load(f)
    return cache_data.get('frames', {})

def write_frame_cache(frame_index: int, best_rotation: float | None = None, trust_fm: int | None = None):
    """
    Write or update cache data for a specific frame.

    ### Params:
        frame_index: int
            The index of the frame to cache
        best_rotation: float | None
            The best satellite image rotation for this frame. If None, don't change this field.
        trust_fm: int | None
            Whether to trust feature matching for this frame. Check README.md for details. If None, don't change this field.
    """
    cache_data = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
    if 'frames' not in cache_data:
        cache_data['frames'] = {}

    # Get existing frame data or create new entry
    frame_key = str(frame_index)
    if frame_key not in cache_data['frames']:
        cache_data['frames'][frame_key] = {}

    # Update only the fields that are not None
    if best_rotation is not None:
        cache_data['frames'][frame_key]['best_rotation'] = best_rotation
    if trust_fm is not None:
        cache_data['frames'][frame_key]['trust_fm'] = trust_fm

    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=4)

def init_cache_from_initial_run(overwrite_existing_rotation: bool = False):
    """
    Read the saved images from the initial run dir and populate the cache.json.

    Parses filenames like 'frame_000_rot_255.0.png' to extract frame index and best rotation,
    then writes each entry to the cache using write_frame_cache.

    ### Params:
        overwrite_existing_rotation: bool
            If False (default), frames that already have a cached rotation will be skipped.
            If True, all frames will be updated with the rotation from the initial run directory.
    """
    initial_run_dir = os.path.join(demo_output_dir, 'demo0', 'initial')

    # Check if the initial run directory exists
    if not os.path.exists(initial_run_dir):
        print(f"Warning: Initial run directory not found: {initial_run_dir}")
        return

    # Get all PNG files in the initial directory
    image_files = [f for f in os.listdir(initial_run_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"Warning: No image files found in {initial_run_dir}")
        return

    print(f"Found {len(image_files)} images in initial run directory")

    # Read existing cache to check which frames already have cached rotations
    existing_cache = read_frame_cache()

    # Track statistics for reporting
    written_count = 0
    skipped_count = 0

    # Parse each filename to extract frame index and rotation angle
    for image_file in sorted(image_files):
        try:
            # Expected format: frame_XXX_rot_YYY.Y.png
            # Example: frame_000_rot_255.0.png
            filename_no_ext = os.path.splitext(image_file)[0]
            parts = filename_no_ext.split('_')

            # Extract frame index (should be after 'frame_')
            frame_idx = int(parts[1])

            # Extract rotation angle (should be after 'rot_')
            # Find the index of 'rot' in parts
            rot_idx = parts.index('rot')
            rotation_str = parts[rot_idx + 1]
            best_rotation = float(rotation_str)

            # Check if frame already has a cached rotation
            frame_key = str(frame_idx)
            if not overwrite_existing_rotation and frame_key in existing_cache:
                existing_rotation = existing_cache[frame_key].get('best_rotation')
                print(f"  ⊘ Skipped frame {frame_idx}: already cached with rotation {existing_rotation}°")
                skipped_count += 1
                continue

            # Write to cache (either new frame or overwrite enabled)
            write_frame_cache(frame_idx, best_rotation)
            print(f"  ✓ Cached frame {frame_idx}: rotation {best_rotation}°")
            written_count += 1

        except (ValueError, IndexError) as e:
            print(f"  ⚠️  Could not parse filename {image_file}: {e}")
            continue

    print(f"\n✓ Cache initialization complete: {written_count} written, {skipped_count} skipped")

def demo0():
    """
    Use the frames that are downsampled by 0.6
    For each of those frames:
    1. match against all rotations of the satelite (the pre-computed ones)
    2. pick the one with most matches
    3. save the visualization (with draw_matches) to demo_output_dir / demo0 / current_time
    """
    # init_cache_from_initial_run()
    # cache_data = None
    cache_data = read_frame_cache()
    
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

    #! rolling parameters and initial configuration
    similarity_threshold = 0.7 # if similarity > threshold, we consider the shapes be similar enough
    initialization_complete = False # when the inter-frame estimation and the feature matching estimation both give similar enough results, we can consider the initialization complete
    #
    prev_frame_features = None
    # initial_rotation_to_try = 
    #* these 2 should always be updated together
    #TODO remove the `current_best_warped_corners` parameter since it can always be inferred from the `current_best_H` parameter
    current_best_warped_corners = None # the current accepted warped corners of the latest fully processed drone frame on the satellite image
    current_best_H = None # the current accepted homography matrix (drone frame -> sat image)
    #! rolling parameters and initial configuration
    
    def compute_new_frame_sat_homography(prev_frame__sat_img_H, inter_frame_H_):
        if prev_frame__sat_img_H is None or inter_frame_H_ is None:
            return None
        
        # prev_frame__sat_img_H: prev_frame -> sat_image
        # inter_frame_H: prev_frame -> current_frame
        # We want: current_frame -> sat_image
        # So: H_new = H_prev @ inv(H_inter) to reverse the inter-frame transformation first
        result = prev_frame__sat_img_H @ np.linalg.inv(inter_frame_H_)
        return result
    
    #! main for loop: Process each drone frame (the frames are already downsampled)
    for frame_idx, drone_frame in enumerate(drone_frames):
        inter_frame_H = None
        
        print(f"\n{'='*60}")
        print(f"Processing drone frame {frame_idx}/{len(drone_frames)}")
        print(f"{'='*60}")

        # The drone frame is already downsampled when loaded
        h, w = drone_frame.shape[:2]
        print(f"Computing features for drone frame with dimensions: {w}x{h}")

        #* Compute features for the drone frame
        drone_features = xfeat_model.xfeat_detect_and_compute(drone_frame, top_k=4096)
        
        #* match against previous frame if available
        if prev_frame_features is not None:
            # Match features between the previous frame and the current one
            mkpts_prev, mkpts_current = xfeat_model.xfeat_match_sparse_default(
                prev_frame_features,
                drone_features
            )

            inter_frame_match_count = len(mkpts_prev)
            print(f"Inter-frame matches: {inter_frame_match_count}")

            #* Compute homography between consecutive frames
            try:
                inter_frame_H, _, _ = find_homography(mkpts_prev, mkpts_current)
                if inter_frame_H is None:
                    print("⚠️  Inter-frame homography estimation failed")
            except Exception as e:
                print(f"⚠️  Exception during inter-frame homography: {e}")

        #! rotation check
        # Check if frame already has a cached best rotation
        frame_cache_key = str(frame_idx)
        cached_rotation = None
        if cache_data is not None and frame_cache_key in cache_data:
            cached_rotation = cache_data[frame_cache_key].get('best_rotation')
            # Convert to float to match sat_features_dict keys (JSON may store as int)
            if cached_rotation is not None:
                cached_rotation = float(cached_rotation)

        best_rotation = None
        best_match_count = 0
        best_mkpts_drone = None
        best_mkpts_sat = None

        if cached_rotation is not None:
            #* Frame has cached rotation - use it directly
            print(f"📋 Using cached rotation: {cached_rotation}°")
            best_rotation = cached_rotation

            # Match features only with the cached rotation
            sat_features = sat_features_dict[cached_rotation]
            mkpts_drone, mkpts_sat = xfeat_model.xfeat_match_sparse_default(
                drone_features,
                sat_features
            )
            best_mkpts_drone = mkpts_drone
            best_mkpts_sat = mkpts_sat
            best_match_count = len(mkpts_drone)
            print(f"  Cached rotation {cached_rotation}°: {best_match_count} matches")
        else:
            #* No cached rotation - match against all satellite rotations
            print("🔍 No cached rotation, searching all rotations...")
            for rotation_angle in sorted(sat_features_dict.keys()):
                sat_features = sat_features_dict[rotation_angle]

                #* Match features using sparse matching
                mkpts_drone, mkpts_sat = xfeat_model.xfeat_match_sparse_default(
                    drone_features,
                    sat_features
                )

                match_count = len(mkpts_drone)
                print(f"  Rotation {rotation_angle:6.1f}°: {match_count:4d} matches")

                #* Update best match if this rotation has more matches
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_rotation = rotation_angle
                    best_mkpts_drone = mkpts_drone
                    best_mkpts_sat = mkpts_sat

            # Cache the newly discovered best rotation for future runs
            if best_rotation is not None:
                write_frame_cache(frame_index=frame_idx, best_rotation=best_rotation)

        if best_mkpts_sat is None:
            print("No matches found for any rotation, skipping visualization.")
            continue

        print(f"\nBest rotation: {best_rotation}° with {best_match_count} matches")

        #! update rolling parameters and create visualizations
        if best_rotation is not None and best_mkpts_drone is not None:
           
           #TODO here do a clustering accept/reject check 
           
            original_sat_img = sat_images_dict[0.0] # Get the original satellite image (0° rotation) for visualization
            sat_h, sat_w = original_sat_img.shape[:2]
            print(f"Using original satellite image (0°): {sat_w}x{sat_h}")

            # Rotate the matched satellite keypoints back to original orientation
            # If best match was at +θ degrees, we need to rotate back by -θ degrees
            sat_center = (sat_w / 2.0, sat_h / 2.0)
            best_mkpts_sat_original = rotate_keypoints(
                best_mkpts_sat,
                -best_rotation,  # Negative angle for reverse rotation
                sat_center
            )
            print(f"Rotated {len(best_mkpts_sat)} keypoints back by {-best_rotation}° to match original satellite image")

            nr_matches = len(best_mkpts_drone)

            #* Compute homography matrix and inlier mask
            try:
                H, inlier_mask, _ = find_homography(best_mkpts_drone, best_mkpts_sat_original)
            except Exception as e:
                print(f"⚠️⚠️⚠️⚠️  Exception during find_homography: {e}")
                H = None
                inlier_mask = None

            if H is None or inlier_mask is None:
                print("⚠️⚠️⚠️⚠️  Homography estimation failed: skipping this pair.")
                continue

            inlier_mask = inlier_mask.flatten()

            #* Warp and draw corners on the satellite image
            result = warp_and_draw_corners(drone_frame, original_sat_img, H)
            if result is None:
                print("⚠️⚠️⚠️⚠️  Warping failed: skipping this pair.")
                continue

            img2_with_corners, warped_corners = result

            #* Check if the warped corners represent a degenerated shape
            is_degenerated, degeneration_diagnostics = is_shape_degenerated(warped_corners)

            #* update the rolling parameters
            #TODO while the demo is going, append to a log file the choices on who to trust for best_corners and why
            # also save the old current_best_H and current_best_corners before updating them, so we can debug later 
            if is_degenerated == False:
                print(f"✓ Non-degenerated shape detected")
                # Initialize best warped corners and homography on first non-degenerated shape
                if current_best_warped_corners is None:
                    current_best_warped_corners = warped_corners.copy()
                    current_best_H = H.copy()
                    print(f"🧙🧙🧙🧙🧙🧙🧙🧙🧙🧙 Initialized current_best_warped_corners and current_best_H")
                else:
                    if inter_frame_H is not None:
                        #* we have inter-frame estimation
                        composition_candidate = compute_new_frame_sat_homography(current_best_H, inter_frame_H)
                        
                        if composition_candidate is not None:
                            #* compute the similarioty between the feature-matching and the inter-frame warped corners
                            try:
                                composition_candidate_warped_corners = warp_corners(drone_frame, composition_candidate)
                                if composition_candidate_warped_corners is None:
                                    print("⚠️  Warping with composition_candidate failed, skipping shape similarity check")
                                    similarity = 0.0
                                else:
                                    similarity = compute_shape_similarity(composition_candidate_warped_corners, warped_corners)
                            except Exception as e:
                                print("🔥" * 80)
                                print(f"🔥🔥🔥  Exception during shape similarity comparison: {e}")
                                similarity = 0.0
                                
                            if similarity >= similarity_threshold: #* inter-frame and feature-matching estimations agree with each other
                                #* they agree -> trust the feature matching
                                print(f"🎉🎉🎉🎉🎉🎉 Inter-frame estimation is similar enough (similarity: {similarity:.3f} >= {similarity_threshold}), updating current_best_H")
                                current_best_H = H.copy()
                                current_best_warped_corners = warped_corners.copy()
                                
                                if initialization_complete == False:
                                    initialization_complete = True
                                    print(f"🎉🎉🎉 Initialization complete! 🎉🎉🎉")
                            else:
                                #* they disagree
                                if initialization_complete == False:
                                    # if we are still initializing, we have no choice but to trust the feature-matching estimation
                                    print(f"🐈 Inter-frame estimation disagrees (similarity: {similarity:.3f} < {similarity_threshold}), but we are still initializing, so trusting feature-matching update")
                                    current_best_H = H.copy() 
                                    current_best_warped_corners = warped_corners.copy()
                                else:
                                    # if we are already initialized, we trust the inter-frame estimation
                                    print(f"🐶 Inter-frame estimation disagrees (similarity: {similarity:.3f} < {similarity_threshold}), trusting inter-frame update")
                                    current_best_H = composition_candidate
                                    current_best_warped_corners = composition_candidate_warped_corners
                        
            else:
                #* degenerated shape: our only option is to compose the inter-frame H with the previous best H and assign the result to current_best_H
                if inter_frame_H is not None and current_best_H is not None:
                    new_H = compute_new_frame_sat_homography(current_best_H, inter_frame_H)
                    if new_H is not None:
                        # Compute new warped corners using the updated homography
                        new_warped_corners = warp_corners(drone_frame, new_H)
                        if new_warped_corners is not None:
                            # Update both H and corners together (as per comment on line 260-261)
                            current_best_H = new_H
                            current_best_warped_corners = new_warped_corners
                            print(f"✓ Updated current_best_H and current_best_warped_corners using inter-frame composition")

            #* Draw the best warped corners if available
            if current_best_warped_corners is not None:
                img2_with_corners = draw_corners(img2_with_corners, current_best_warped_corners, color=(255, 0, 0), thickness=2) 

            # Draw match lines on combined image
            viz_canvas = draw_matches(
                img1=drone_frame,
                img2=img2_with_corners,
                ref_points=best_mkpts_drone,
                dst_points=best_mkpts_sat_original,  # Use rotated-back keypoints
                inlier_mask=inlier_mask,
                thickness=1
            )

            if viz_canvas is not None:
                # Save the visualization
                output_path = os.path.join(demo0_output_dir, f'frame_{frame_idx:03d}_rot_{best_rotation:.1f}@{nr_matches}_matches.png')
                cv2.imwrite(output_path, viz_canvas)
                print(f"✓ Saved visualization to: {output_path}")
            else:
                print("⚠️⚠️⚠️⚠️  Visualization failed (homography estimation failed)")
        else:
            print("⚠️  No matches found for this frame")
            
        prev_frame_features = copy.deepcopy(drone_features) # in the future, only if explicitely desired, ask Claude Code to make this more memory efficient and only copy the numpy arrays
        
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