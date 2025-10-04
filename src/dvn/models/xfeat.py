import numpy as np
import imageio as imio
import os
import torch
import tqdm
from typing import no_type_check
import cv2
import datetime
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple, Dict

from sklearn.cluster import DBSCAN


#! tunable params
ransacReprojThresholdParam = 5.0
# top_k_frames
HOMOGRAPHY_METHOD = cv2.USAC_MAGSAC
top_k_frames = 4096
maxItersParam = 1_000
xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = top_k_frames)

@no_type_check
def warp_corners_and_draw_matches(
    ref_points: np.ndarray,
    dst_points: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    draw_match_lines: bool = True, 
    precomputed_H=None,
    precomputed_mask=None,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """
    Calculates homography, warps the corners of img1 onto img2, draws the warped
    corners, and optionally draws lines for inlier matches.

    Args:
        ref_points: Keypoints from the reference image (img1), shape (N, 2).
        dst_points: Corresponding keypoints from the destination image (img2), shape (N, 2).
        img1: The reference image (BGR format).
        img2: The destination image (BGR format).
        draw_match_lines (bool, optional): If True (default), draws the lines
                                           connecting inlier matches between img1 and img2
                                           in a combined visualization. If False, only
                                           draws the warped polygon outline of img1 onto img2
                                           and returns just img2 with the polygon.

    Returns:
        tuple:
            - np.ndarray | None: The output image.
                - If draw_match_lines is True: A combined image showing img1 and img2
                  side-by-side with match lines and the warped polygon.
                - If draw_match_lines is False: img2 with the warped polygon drawn on it.
                - None if homography estimation or warping fails.
            - float | None: The ratio of inlier matches (inliers / total matches).
                            None if homography estimation fails.
    """
    # Ensure points are float32 for findHomography
    ref_points = np.float32(ref_points).reshape(-1, 1, 2)
    dst_points = np.float32(dst_points).reshape(-1, 1, 2)

    # Calculate the Homography matrix
    H = None
    mask = None
    if precomputed_H is None and precomputed_mask is None:
        H, mask = cv2.findHomography(
            ref_points,
            dst_points,
            HOMOGRAPHY_METHOD,
            ransacReprojThreshold=ransacReprojThresholdParam,
            maxIters=maxItersParam,
            confidence=0.999
        )
    else: 
        H = precomputed_H
        mask = precomputed_mask
        
    if H is None or mask is None:
        print("⚠️  Homography estimation failed: skipping this pair.")
        # pyrefly: ignore  # bad-return
        return None, None, None

    mask = mask.flatten()
    num_inliers = np.sum(mask)
    num_matches = len(mask)
    if num_matches == 0:
        print("⚠️ No matches provided to findHomography.")
        inlier_ratio = 0.0
    else:
        inlier_ratio = num_inliers / num_matches
    print(f'Inlier ratio: {inlier_ratio:.4f} ({int(num_inliers)}/{num_matches})')

    # Get corners of the first image (img1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (img2) space
    try:
        warped_corners = cv2.perspectiveTransform(corners_img1, H)
        if warped_corners is None or not np.all(np.isfinite(warped_corners)):
             print("⚠️ perspectiveTransform resulted in invalid corners.")
             # pyrefly: ignore  # bad-return
             return None, inlier_ratio, None # Homography was found, return ratio but no image
    except cv2.error as e:
        print(f"⚠️ cv2.error during perspectiveTransform: {e}")
        # pyrefly: ignore  # bad-return
        return None, inlier_ratio, None # Homography was found, return ratio but no image


    # --- Draw the warped corners in image2 ---
    img2_with_corners = img2.copy()
    # Check if warped_corners are valid before drawing
    if warped_corners is not None:
        # Convert to integer points for drawing polylines
        pts = np.int32(warped_corners.reshape(-1, 2))
        # pyrefly: ignore  # no-matching-overload
        cv2.polylines(img2_with_corners, [pts], isClosed=True, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)
        # Optional: Draw individual corners if needed
        # for i in range(len(warped_corners)):
        #     pt = tuple(warped_corners[i][0].astype(int))
        #     cv2.circle(img2_with_corners, pt, 5, (0, 0, 255), -1) # Red dots at corners

    # --- Decide what image to return based on the flag ---
    if draw_match_lines:
        # Prepare keypoints and matches for drawMatches function
        # Use original (pre-reshaped) points for KeyPoint creation
        ref_points_orig = ref_points.reshape(-1, 2)
        dst_points_orig = dst_points.reshape(-1, 2)
        keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points_orig]
        keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points_orig]

        # Create DMatch objects only for inliers
        matches = [cv2.DMatch(i, i, 0) for i, m in enumerate(mask) if m]

        # Draw inlier matches onto the combined image
        # Use img2_with_corners which already has the polygon
        # pyrefly: ignore  # no-matching-overload
        img_matches_combined = cv2.drawMatches(
            img1, keypoints1,
            img2_with_corners, keypoints2,
            matches, None, # Draw only inlier matches
            matchColor=(0, 255, 0), # Green lines for matches
            singlePointColor=(255, 0, 0), # Blue single points (if any)
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS # Don't draw unmatched keypoints
        )
        # pyrefly: ignore  # bad-return
        return img_matches_combined, inlier_ratio, warped_corners
    else:
        # Return only the second image with the warped corners drawn
        # pyrefly: ignore  # bad-return
        return img2_with_corners, inlier_ratio, warped_corners
def prepare_np_array_image_for_xfeat(img_src: np.ndarray) -> np.ndarray:
    return np.copy(img_src[..., ::-1])

@no_type_check
def find_homography(points1, points2):
    """Find homography matrix between two sets of points"""
    # Convert to float32
    points1 = np.float32(points1)
    points2 = np.float32(points2)
    
    # Print sample matching points for verification
    print(f"Matching points shape: {points1.shape}, {points2.shape}")
    print("Sample matching points:")
    # pyrefly: ignore  # bad-argument-type
    for i in range(min(5, len(points1))):
        # pyrefly: ignore  # index-error
        print(f"Match {i}: {points1[i]} -> {points2[i]}")
    
    # Find homography
    # pyrefly: ignore  # no-matching-overload
    H, mask = cv2.findHomography(
        points1,
        points2,
        # pyrefly: ignore  # missing-attribute
        method=cv2.HOMOGRAPHY_METHOD,
        ransacReprojThreshold=ransacReprojThresholdParam,
        confidence=0.9999,
        maxIters=10000,
    )
    
    print("Homography matrix:")
    print(H)
    
    # Count inliers
    inliers = mask.ravel().sum()
    # pyrefly: ignore  # bad-argument-type
    print(f"Number of inliers: {inliers} out of {len(points1)} matches")
    
    return H, mask

def create_warped_image(image, target_image, homography):
    """Warp image to align with target_image using the given homography"""
    return cv2.warpPerspective(image, homography, (target_image.shape[1], target_image.shape[0]))

@no_type_check
def xfeat_detect_and_compute(image: np.ndarray, top_k: int = 4096) -> dict:
    """
    Detect and compute features using XFeat.
    """
    # Prepare the image for XFeat
    im = prepare_np_array_image_for_xfeat(image)
    # pyrefly: ignore  # missing-attribute
    output = xfeat.detectAndCompute(im, top_k=top_k)[0]
    output.update({'image_size': (im.shape[1], im.shape[0])})    
    return output

# @no_type_check
def match_xfeat_star(img1: np.ndarray, img2: np.ndarray, top_k: int = 4096) -> tuple:
    """
    Match features between two images using XFeat.
    """
    # Prepare the images for XFeat
    im1 = prepare_np_array_image_for_xfeat(img1)
    im2 = prepare_np_array_image_for_xfeat(img2)

    # Detect and compute features for both images
    
    # pyrefly: ignore  # missing-attribute
    mkpts_0, mkpts_1 = xfeat.match_xfeat_star(img1, img2, top_k = top_k) # pyright: ignore
    # pyrefly: ignore  # bad-unpacking
    canvas,_,_ = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2, draw_match_lines=True)

    # pyrefly: ignore  # bad-return
    return canvas

def match_xfeat(
    image1: np.ndarray,
    feat1: dict,
    image2: np.ndarray,
    feat2: dict,
    top_k: int = 4096,
    draw_match_lines: bool = True # New parameter
) -> np.ndarray:
    """
    Accepts either file paths or pre-loaded BGR arrays.
    """
    # prepare images

    im1 = prepare_np_array_image_for_xfeat(image1)
    output0 = feat1
    if feat1 is None:
        # pyrefly: ignore  # missing-attribute
        output0 = xfeat.detectAndCompute(im1, top_k=top_k)[0]

    
    im2 = prepare_np_array_image_for_xfeat(image2)
    output1 = feat2
    if feat2 is None:
        # pyrefly: ignore  # missing-attribute
        output1 = xfeat.detectAndCompute(im2, top_k=top_k)[0]

    output0.update({'image_size': (im1.shape[1], im1.shape[0])})
    output1.update({'image_size': (im2.shape[1], im2.shape[0])})

    # pyrefly: ignore  # missing-attribute
    mkpts_0, mkpts_1, _ = xfeat.match_lighterglue(output0, output1)
    nr_matches = len(mkpts_0)
    print(f"Number of matches: {nr_matches}")

    if nr_matches < 4:
        # pyrefly: ignore  # bad-return
        return None,None,None,None,None,None

    # pyrefly: ignore  # bad-unpacking
    canvas,inlier_ratio,warped_corners = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2,draw_match_lines)

    # pyrefly: ignore  # bad-return
    return canvas,mkpts_0,mkpts_1,inlier_ratio,nr_matches,warped_corners


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


from dvn.utils.basic_image_utils import generate_90_180_270_rotated_images, downsample_pyramid, generate_rotations
# pyrefly: ignore  # missing-module-attribute
from dvn.utils.basic_image_utils import plot_1_image

def main():
    sat_img = "../../data/marco_video_frames_and_sat/sat.png"
    sat_patch_1 = "../../data/marco_video_frames_and_sat/sat_patch_1.png"
    sat_patch_2 = "../../data/marco_video_frames_and_sat/sat_patch_2.png"
    sat_patch_3 = "../../data/marco_video_frames_and_sat/sat_patch_3.png"
    original_frame_ds =  "../../data/marco_video_frames_and_sat/interesting_frames/original_frame_ds.png"
    original_frame = "../../data/marco_video_frames_and_sat/interesting_frames/original_frame.png"
    
    chosen1 = imio.imread(sat_patch_1)
    chosen2 = imio.imread(sat_patch_2)
    # plot_1_image(chosen1, title="chosen1")
    chosen2_rotated = generate_90_180_270_rotated_images(chosen2)
    # pyrefly: ignore  # missing-attribute
    chosen2_rotated.append(chosen2)
    
    for img in chosen2_rotated:
        # pyrefly: ignore  # missing-argument, bad-argument-type
        canvas,p0,p1,inlier_ratio,_ = match_xfeat(chosen1, img)
        plot_1_image(canvas, title="canvas")

from dvn.utils.basic_image_utils import load_n_images_from_folder, rotate_once,put_images_side_by_side
import shutil
import csv
# pyrefly: ignore  # bad-function-definition
def load_precomputed_xfeat(folder_path: str, n: int = None) -> list:
    """
    Load precomputed XFeat data from a folder.
    """
    files = os.listdir(folder_path)
    files = [f for f in files if f.endswith('.pt')]
    if n is not None:
        files = files[:n]
    
    xfeat_data = []
    # pyrefly: ignore  # not-iterable
    for file in tqdm.tqdm(files):
        file_path = os.path.join(folder_path, file)
        data = torch.load(file_path)
        xfeat_data.append(data)
    
    return xfeat_data


def centre_sat_from_translation(mkpts0, mkpts1, img_shape):
    """Robustly estimate drone-frame centre in satellite coords (translation model)."""
    # 1. translations coming from every correspondence
    t_i = mkpts1 - mkpts0                     # shape (N,2)

    # 2. robust location estimate of t  ➜  median works if >50 % inliers
    t = np.median(t_i, axis=0)                # (dx,dy)

    # 3. add that translation to the known centre of the drone frame
    h, w = img_shape[:2]
    centre_drone = np.array([w/2, h/2], dtype=np.float32)
    centre_sat   = centre_drone + t
    return centre_sat          # (x,y) in satellite pixels


# pyrefly: ignore  # missing-module-attribute
from dvn.utils.basic_image_utils import add_text_to_image

from scipy.spatial.transform import Rotation as R 

def calculate_rotation_angle_from_rotation_matrix(rotation_matrix: np.ndarray,order: str = 'zyx') -> float:
    r = R.from_matrix(rotation_matrix) # Create a Rotation object from the matri
    euler_angles_rad = r.as_euler(order) # Get Euler angles in the specified order (result is in radians)
    euler_angles_deg = np.rad2deg(euler_angles_rad) # Convert radians to degrees
    # Extract angles - the meaning depends on the 'order'
    angle_1 = euler_angles_deg[0]
    angle_2 = euler_angles_deg[1]
    angle_3 = euler_angles_deg[2]
    # pyrefly: ignore  # bad-return
    return angle_1, angle_2, angle_3

def compute_mkpts_for_sat_images(sat_img_list,angles,top_k=4096):
    """
    Compute mkpts for the satellite images.
    """
    sat_img_features = []
    for index,sat_img in enumerate(sat_img_list):
        #* compute xfeat for the sat images, save them to disk
        output = xfeat_detect_and_compute(sat_img, top_k=top_k)
        sat_img_features.append(output)
    
    assert len(sat_img_features) == len(angles)
    # Create a dictionary mapping angles to their corresponding features
    sat_features_by_angle_dict = {angle: feature for angle, feature in zip(angles, sat_img_features)}
    return sat_features_by_angle_dict

#Cluster the translation vectors
#NOTE: correct matches will exhibit more similarity in their translation vectors compared to completely random outliers
def cluster_translation_vectors_dbscan(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    img1_shape: Optional[Tuple[int, int]] = None,
    img2_shape: Optional[Tuple[int, int]] = None,
    eps_scale_factor: float = 0.05,
    dbscan_min_samples: int = 3,
    min_points_to_attempt_clustering: int = 5 # Min # of points to run DBSCAN
) -> Tuple[np.ndarray, float]:
    """
    NOTE: Gemini 2.5 Pro WROTE THIS
    Clusters translation vectors derived from matched keypoints using DBSCAN.

    This function calculates translation vectors (mkpts1 - mkpts0) and then
    applies DBSCAN clustering to these vectors. It returns the raw cluster labels
    assigned by DBSCAN to each input point and the actual epsilon value used.

    The interpretation of these labels (e.g., identifying dominant clusters,
    filtering points, scoring) is intended to be handled by the caller.

    Args:
        mkpts0: Keypoints from the first image, shape (N, 2).
        mkpts1: Corresponding keypoints from the second image, shape (N, 2).
        img1_shape: Optional (height, width) of image 1, for dynamic eps calculation.
        img2_shape: Optional (height, width) of image 2, for dynamic eps calculation.
        eps_scale_factor: Factor to scale an effective image diagonal to get DBSCAN's
                          epsilon, if image shapes are provided.
        dbscan_min_samples: The 'min_samples' parameter for DBSCAN. Minimum number
                            of points in a neighborhood for a point to be a core point.
        min_points_to_attempt_clustering: If the number of initial matches is less
                                          than this (or zero), clustering is skipped,
                                          and an empty labels array and eps_val=0.0
                                          are returned.

    Returns:
        Tuple[np.ndarray, float]:
            - cluster_labels (np.ndarray): An array of shape (N,) where N is the
              number of input keypoints. Each element is the cluster ID assigned
              by DBSCAN (e.g., 0, 1, ...). Noise points are labeled -1.
              If clustering is skipped or fails critically, an empty array is returned.
            - actual_eps_used (float): The epsilon value that was actually used by
              DBSCAN. If clustering is skipped or eps calculation leads to an
              invalid value causing DBSCAN to fail before it can be recorded,
              this might be the intended eps or 0.0.
    """
    num_initial_matches = len(mkpts0)

    if num_initial_matches < min_points_to_attempt_clustering:
        print(f"DBSCAN: Skipping, number of points ({num_initial_matches}) "
              f"is less than min_points_to_attempt_clustering ({min_points_to_attempt_clustering}).")
        return np.array([], dtype=int), 0.0
    
    # compute the translation vector
    translations = mkpts1 - mkpts0

    #* --- Determine eps_val for DBSCAN ---
    eps_val_to_use = 30.0  # Default fixed fallback eps
    effective_diagonal = None
    diag1, diag2 = None, None

    if img1_shape and len(img1_shape) == 2 and img1_shape[0] > 0 and img1_shape[1] > 0:
        diag1 = np.sqrt(img1_shape[0]**2 + img1_shape[1]**2)
    if img2_shape and len(img2_shape) == 2 and img2_shape[0] > 0 and img2_shape[1] > 0:
        diag2 = np.sqrt(img2_shape[0]**2 + img2_shape[1]**2)

    #* By calculating an effective_diagonal and then setting eps_val = effective_diagonal * eps_scale_factor, 
    # the eps_val becomes adaptive to the size of the input images.
    if diag1 and diag2:
        effective_diagonal = (diag1 + diag2) / 2.0
    elif diag1:
        effective_diagonal = diag1
    elif diag2:
        effective_diagonal = diag2

    if effective_diagonal is not None:
        eps_val_to_use = effective_diagonal * eps_scale_factor
        print(f"DBSCAN: Using dynamic eps_val: {eps_val_to_use:.2f}px (factor {eps_scale_factor} of effective diagonal {effective_diagonal:.2f}px).")
    else:
        print(f"DBSCAN: No valid image shapes for dynamic eps, using fixed fallback eps_val: {eps_val_to_use:.2f}px.")
    # --- End of eps_val determination ---

    # Basic validation for DBSCAN parameters
    if dbscan_min_samples < 1:
        # print(f"DBSCAN Warning: dbscan_min_samples ({dbscan_min_samples}) is less than 1. This might not be valid for DBSCAN. Check sklearn docs.")
        # sklearn's DBSCAN requires min_samples >= 1
        pass # Let DBSCAN handle it, or raise an error if preferred
    
    if eps_val_to_use <= 0:
        print(f"DBSCAN Error: Calculated eps_val ({eps_val_to_use:.2f}) is not positive. Clustering cannot proceed.")
        return np.array([], dtype=int), eps_val_to_use # Return the problematic eps for context

    print(f"DBSCAN params: actual eps={eps_val_to_use:.2f}, min_samples={dbscan_min_samples}")

    try:
        db = DBSCAN(eps=eps_val_to_use, min_samples=dbscan_min_samples).fit(translations)
        cluster_labels = db.labels_
        return cluster_labels, eps_val_to_use
    except ValueError as e:
        print(f"⚠️ DBSCAN clustering error: {e}. This can happen with invalid parameters.")
        return np.array([], dtype=int), eps_val_to_use

def get_counts_from_cluster_labels(labels: np.ndarray) -> Tuple[Dict[int, int], int]:
    """
    Processes DBSCAN cluster labels to count points per cluster and noise points.
    Args:
        labels (np.ndarray): An array of cluster labels as returned by
                             DBSCAN. Noise points are typically labeled -1.

    Returns:
        Tuple[Dict[int, int], int]:
            - A dictionary where keys are the cluster IDs (e.g., 0, 1, 2, ...)
              and values are the number of points belonging to that cluster.
            - An integer representing the total count of noise points (labeled -1).
              Returns an empty dictionary and 0 if the input labels array is empty.
    """
    if labels.size == 0:
        return {}, 0

    # Get unique labels and their corresponding counts. # unique_labels will be sorted, e.g., [-1, 0, 1, 2]
    # counts_elements will be the count for each unique_label
    unique_labels, counts_elements = np.unique(labels, return_counts=True)

    cluster_counts: Dict[int, int] = {}
    noise_count: int = 0

    # Iterate through the unique labels and their counts
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        count = counts_elements[i]

        if label == -1:
            # This is a noise point
            noise_count += count
        else:
            # This is a valid cluster
            cluster_counts[label] = count
            
    return cluster_counts, noise_count


def reject_match(cluster_counts, noise_count, nr_matches):
    if nr_matches <= 5:
        return True,f"To few matches, only {nr_matches} matches"
    total_clustered_points = sum(cluster_counts.values())
    if noise_count + 1 >= total_clustered_points: 
        return True, f"We had {noise_count} noise points and {total_clustered_points} clustered points. To many noise points"
    nr_clusters = len(cluster_counts)
    if nr_clusters >=2: 
        small_clusters = [count for count in cluster_counts.values() if count < 0.2 * total_clustered_points]
        nr_small_clusters = len(small_clusters)
        
        if nr_small_clusters == 0:
            return True, f"We had {nr_clusters} clusters. They were all big."
    
    #another rule: if we have under 10 matcches, there can be at most 20% noise points
    # if under 12 points, and we don't have only a single cluster -> reject
    return False, "Accepted"

def save_mkpts_to_file(mkpts, output_folder, filename):
    """
    Save matched keypoints to a text file. Each line contains 'x y' coordinates.
    mkpts: iterable of (x, y) pairs (e.g. numpy array shape (N,2))
    output_folder: directory to write the file into
    filename: name of the file ('.txt' will be appended if missing)
    """
    # ensure .txt extension
    if not filename.lower().endswith('.txt'):
        filename = filename + '.txt'

    # create folder if missing
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, filename)

    try:
        with open(file_path, 'w') as f:
            for pt in mkpts:
                # format with 6 decimal places
                f.write(f"{pt[0]:.6f} {pt[1]:.6f}\n")
        print(f"Saved keypoints to {file_path}")
    except Exception as e:
        print(f"Error saving keypoints to file {file_path}: {e}")
    

def main2(precomputed_angle_step=3,smart_match=False):
    #!
    IGNORE = [12,21,30,54,101]
    last_included_frame = 113
    
    def include_frame(frame_index):
        if frame_index in IGNORE:
            return False
        if frame_index > last_included_frame:
            return False
        return True
    
    #!
    
    frames_folder = "../../../data/marco_video_frames_and_sat/sunny_all_frames"
    video_frames = load_n_images_from_folder(frames_folder, n=None)
    sat_folder = "../../../data/marco_video_frames_and_sat/sat_images"
    sat_images = load_n_images_from_folder(sat_folder, n=None)
    
    rotated_list = []
    angles = [45, 90, 135, 180, 225, 270, 315]
    for sat_img in sat_images:
        # this_rotated_list = generate_90_180_270_rotated_images(sat_img)
        # pyrefly: ignore  # bad-argument-type
        this_rotated_list = generate_rotations(sat_img,angles)
        rotated_list.extend(this_rotated_list)
    rotated_list = zip(rotated_list, angles) #* zip the rotated images with their angles
    #* zip the original images with their angles
    sat_images = [(img, 0.0) for img in sat_images]
    #* unite the original images with the rotated ones
    sat_images.extend(rotated_list)
    
    sat_image_folder_after = "../../../data/marco_video_frames_and_sat/sat_images_after"
    #* Create the folder and wipe its contents if it already exists
    if os.path.exists(sat_image_folder_after):
        shutil.rmtree(sat_image_folder_after)
    os.makedirs(sat_image_folder_after)
    
    sat_save_folder = "../../../data/marco_video_frames_and_sat/sat_images_feat_saved"
    if os.path.exists(sat_save_folder):
        shutil.rmtree(sat_save_folder)
    os.makedirs(sat_save_folder)
    final_angles = []
    for index,(sat_img,angle) in enumerate(sat_images):
        final_angles.append(angle)
        imio.imwrite(os.path.join(sat_image_folder_after, f"sat_img_{index}.png"), sat_img)
        
    
    sat_img_features = None
    #* recompute xfeat for the sat images
    #! compute the mkpoints for the sat images, save them to disk
    all_sat_images = load_n_images_from_folder(sat_image_folder_after, n=None)
    angles_custom = list(range(0, 360, precomputed_angle_step)) # generate all numbers from 
    # pyrefly: ignore  # bad-argument-type
    matei_sat_custom_rotate = generate_rotations(all_sat_images[0], angles_custom) 
    matei_rotated_sat_img_dict = {angle: img for angle, img in zip(angles_custom, matei_sat_custom_rotate)}
    matei_saved_mkpts_dict = compute_mkpts_for_sat_images(matei_sat_custom_rotate, angles_custom, top_k=top_k_frames)
    #! final_angles keeps the angles for these frames
    nr_sat_images = len(all_sat_images)
    print(f"There are {nr_sat_images} sat images to match against")
    
    recompute_sat = True
    if recompute_sat:
        for index,sat_img in enumerate(all_sat_images):
            output = xfeat_detect_and_compute(sat_img, top_k=4096)
            torch.save(output,os.path.join(sat_save_folder, f"sat_img_{index}.pt"))
    
    # pyrefly: ignore  # bad-argument-type
    sat_img_features = load_precomputed_xfeat(sat_save_folder, n=None)
    
    assert len(sat_img_features) == nr_sat_images
    assert len(final_angles) == nr_sat_images
    # for each frame, try to rotate it and match with each rotated
    failed = 0
    success = 0
    total_frames = len(video_frames)
    match_frame_output_path = "../../../data/marco_video_frames_and_sat/match_result"
    match_coord_output_path = "../../../data/marco_video_frames_and_sat/match_result/match_coords"
    match_saved_mkpts_folder = "../../../data/marco_video_frames_and_sat/match_result/match_saved_mkpts"
    
    rejected_matches_file = "../../../data/marco_video_frames_and_sat/match_result/rejected_matches.txt"
    
    #* Ensure the rejected matches file exists (overwrite if it does)
    open(rejected_matches_file, 'w').close()
    print(f"Created and overwrote the file. Rejected matches will be saved to {rejected_matches_file}")
    
    if os.path.exists(match_frame_output_path):
        shutil.rmtree(match_frame_output_path)
    os.makedirs(match_frame_output_path)
    
    print(f"Total frames: {total_frames}")
    match_coord_list = []
    all_sat_images_downsamples = [downsample_pyramid(sat_img, scales=[0.5])[0] for sat_img in all_sat_images]
    mkpts_last_frame = None
    last_frame = None
    total_rotation = 80.0 #TODO find the initial value of this
    
    start_time = datetime.datetime.now()
    #* downscale all frames here, at load time
    downsampled_frames = [downsample_pyramid(frame, scales=[0.66])[0] for frame in video_frames]
    
    considered_frame_count = 0
    accepted_matches = 0
    # pyrefly: ignore  # bad-assignment
    for frame_index,frame in enumerate(downsampled_frames):
        
        if not include_frame(frame_index):
            # print(f"Skipping frame {frame_index} as per inclusion criteria.")
            continue
        considered_frame_count += 1
        
        frame_complete_list = [frame]
        original_frame_index = len(frame_complete_list) - 1 #! because the frame will always be at the end of the list
        # now frame_complete_list has the original frame and the downsampled versions
        matched_this_frame = False
        # print(f"This frame results in a total of {len(frame_complete_list)} images to use for matching")
        
        mkpts_this_frame = xfeat_detect_and_compute(frame, top_k=top_k_frames)
        if mkpts_this_frame is None:
            print("mkpts_this_frame is NONE!!! 🧙‍♀️🧙‍♀️🧙‍♀️🧙‍♀️🧙‍♀️🧙‍♀️🧙‍♀️🧙‍♀️")
            continue
        
        angle_1 = -999 # default uninitialized value
        angle_2 = -999
        angle_3 = -999
        chosen_angle = 0.0 # default uninitialized value
        #* compute feature match with last frame here
        canvas_flow = None
        H_flow = None
        relative_rotation_deg_this = None
        inlier_ratio_flow = -1 # default uninitialized value
        if mkpts_last_frame is None and frame_index > 0:
            print("mkpts_last_frame is None for a frame where it should not be 🧙‍♀️🧙‍♀️🧙‍♀️🧙‍♀️🧙‍♀️🧙‍♀️🧙‍♀️🧙‍♀️")
        if mkpts_last_frame is not None:
            
            # pyrefly: ignore  # missing-attribute
            matched_points = xfeat.match_lighterglue(mkpts_last_frame, mkpts_this_frame) # Assuming keys match image order
            mkpts0_flow, mkpts1_flow, _ = matched_points     
            #* try to compute the homography
            if len(mkpts0_flow) >= 4:
                # Calculate homography H mapping points from last_frame to frame
                H_flow, mask_flow = cv2.findHomography(
                    np.float32(mkpts0_flow).reshape(-1, 1, 2),
                    np.float32(mkpts1_flow).reshape(-1, 1, 2),
                    HOMOGRAPHY_METHOD, # Use the same method as elsewhere
                    ransacReprojThreshold=ransacReprojThresholdParam,
                    maxIters=maxItersParam,
                    confidence=0.999
                )
                if H_flow is None:
                    print("⚠️ Could not calculate frame-to-frame homography.")
            canvas_flow, inlier_ratio_flow = None, None # Placeholder
            
            #* calculate the angle and warp the images according to the homography matrix
            if H_flow is not None:
                # Example: Reuse warp_corners_and_draw_matches (pass correct images!)
                #~ camopute the canvas
                # pyrefly: ignore  # bad-unpacking
                canvas_flow, inlier_ratio_flow, _ = warp_corners_and_draw_matches(
                    mkpts0_flow, mkpts1_flow,
                    # pyrefly: ignore  # bad-argument-type
                    last_frame, frame, # Pass the correct images
                    draw_match_lines=False, # Or False if you just want the outline/H
                    precomputed_H=H_flow,
                    precomputed_mask=mask_flow # Pass the mask if you want to visualize inliers
                )
                K_est = estimate_intrinsic_matrix(frame.shape[1], frame.shape[0],fov_degrees=60)
                num_solutions, Rs, ts, Ns = cv2.decomposeHomographyMat(H_flow, K_est)
                #* we now have some solutions
                plausible_solutions = []
                # If the drone looks down, the ground plane normal points towards the camera (positive Z).
                for i in range(num_solutions):
                    #~ Heuristic: Check if the plane normal's Z component is positive
                    # This assumes the plane normal points towards the second camera
                    if Ns[i][2] > 0: # Check the Z component of the normal vector
                        plausible_solutions.append({
                            'R': Rs[i],
                            't': ts[i],
                            'N': Ns[i],
                            'index': i
                        })
                #TODO now we need some more heuristics to chose the best solution
                if plausible_solutions:
                    best_solution = plausible_solutions[0] # just pick the first one for now
                    R_relative = best_solution['R'] # This is the 3x3 rotation matrix
                    # pyrefly: ignore  # not-iterable
                    angle_1, angle_2, angle_3 = calculate_rotation_angle_from_rotation_matrix(R_relative, order='zyx')
                    chosen_angle = angle_2 # this is the yaw angle
                    print("🩵🩵🩵chosen angle is: ", chosen_angle)
                    total_rotation = total_rotation - chosen_angle
                    # print(f"Frame{frame_index-1} - {frame_index} Angle 1: {angle_1}, Angle 2: {angle_2}, Angle 3: {angle_3}")
                    # print(f"Frame{frame_index-1} - {frame_index} Yaw: {angle_2}")
                    
        
        #! here we try to match according to estimated rotation
        if smart_match == True:
            rounded_total_rotation = round(total_rotation / precomputed_angle_step) * precomputed_angle_step # round to divisible by precomputed_angle_step
            positive_rounded = (rounded_total_rotation+360) % 360 # make sure the angle is positive
            print(f"positive rouner angle is: {positive_rounded}")
            if matei_saved_mkpts_dict[positive_rounded] is None:
                print(f"🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨 No saved mkpts for angle {positive_rounded}")
                continue
            predicted_sat_image_mkpts = matei_saved_mkpts_dict.get(positive_rounded) # get the predicted sat image for this angle
            #mkpts_this_frame -> there are the computed mkpts for the current frame
            #* compute the canvas
            # pyrefly: ignore  # bad-argument-type
            canvas_angle, mkpts0_angle, mkpts1_angle, inlier_ratio_angle, nr_matches_angle,_ = match_xfeat(frame, mkpts_this_frame,matei_rotated_sat_img_dict.get(positive_rounded) ,predicted_sat_image_mkpts, top_k=top_k_frames)
            if canvas_angle is not None:
                #* save the canvas with the inlier ratio
                # inlier_ratio_string = f"{inlier_ratio_angle:.2f}"
                # combined_img_with_text = add_text_to_image(canvas_angle, inlier_ratio_string, color_name="red", position=(200, 200), font_scale=8, thickness=4)
                cv2.imwrite(os.path.join(match_frame_output_path, f"frame_{frame_index}_angle_{positive_rounded}_matched_{inlier_ratio_angle:.2f}_{nr_matches_angle}.png"), canvas_angle)
                matched_this_frame = True
                print("🧠🧠🧠🧠🧠🧠")
            else: 
                # pyrefly: ignore  # bad-argument-type
                combined_fail_match = put_images_side_by_side(frame, matei_rotated_sat_img_dict.get(positive_rounded))  
                plot_1_image(combined_fail_match, title="combined_fail_match")
            
        
        #! now we start matching all frame downsamples with all sat images
        matched_mkpts = []
        nr_matches_for_match = None
        # pyrefly: ignore  # bad-assignment
        for frame_list_index,frame_img in enumerate(frame_complete_list):
            if matched_this_frame==True: break
            
            #* feature match this version of the current frame with all sat images 
            #TODO this all can actually go into a method, to make the code more modular and readable
            for index2,(sat_img_feat,sat_im) in enumerate(zip(sat_img_features,all_sat_images)):
                this_angle = final_angles[index2]
                if frame_list_index == original_frame_index: frame_feat = mkpts_this_frame #~ original frame is already computed
                else: frame_feat = xfeat_detect_and_compute(frame_img, top_k=top_k_frames)
                        
                canvas,mkpts0,mkpts1,inlier_ratio,nr_matches, _ = match_xfeat(frame_img,frame_feat, sat_im,sat_img_feat, top_k=top_k_frames)
                if canvas is not None:
                    matched_this_frame = True
                    nr_matches_for_match = nr_matches
                    matched_mkpts = [mkpts0,mkpts1]
                    # plot_1_image(canvas, title=f"Frame {index} matched with Sat Image")
                    centre_sat = centre_sat_from_translation(mkpts0, mkpts1, frame_img.shape)
                    if centre_sat is not None:

                        x, y = map(int, centre_sat)          # round to integer pixel coords
                        match_coord_list.append((frame_index, x, y, this_angle))
                        sat_im_out = sat_im.copy()

                        # draw a red crosshair
                        cv2.drawMarker(sat_im_out, (x, y),
                                    color=(0, 0, 255),     # BGR red
                                    markerType=cv2.MARKER_CROSS,
                                    markerSize=100,
                                    thickness=3)
                        sat_im_out = rotate_once(sat_im_out, -this_angle) # rotate the image back to the original angle
                        combined_img = put_images_side_by_side(frame, sat_im_out)
                        combined_img = put_images_side_by_side(combined_img, canvas)
                        if canvas_flow is not None:
                            # canvas_flow = add_text_to_image(canvas_flow, "Flow", color_name="white", position=(150,150), font_scale=8, thickness=4)
                            canvas_flow = add_text_to_image(canvas_flow, f"{chosen_angle}", color_name="red", position=(250,250), font_scale=8, thickness=4)
                            # pyrefly: ignore  # bad-argument-type
                            last_frame_rotated = rotate_once(last_frame, chosen_angle)
                            combined_img = put_images_side_by_side(combined_img, canvas_flow)
                            combined_img = put_images_side_by_side(combined_img, last_frame_rotated)
                        inlier_ratio_string = f"{inlier_ratio:.2f}"
                        # combined_img_with_text = add_text_to_image(combined_img,inlier_ratio_string,color_name="red",  position=(200,200), font_scale=8, thickness=4)
                        downsampled_comnbined_img = downsample_pyramid(combined_img, scales=[0.5])[0]
                        # cv2.imwrite(os.path.join(match_frame_output_path, f"frame_{frame_index}_matched_{inlier_ratio:.2f}_{nr_matches}.png"), downsampled_comnbined_img)
                    else:
                        print("⚠️  Centre sat is None")
                                        
                    break
            
        if not matched_this_frame:
            failed += 1
            
            print(f"Frame {frame_index} out of {total_frames}: Failed to match this frame")
        else: 
            success += 1
            # matched_mkpts
            if matched_mkpts == []:
                print("⁉️⁉️⁉️⁉️⁉️matched_mkpts is empty, but we matched the frame⁉️⁉️⁉️⁉️⁉️⁉️⁉️⁉️⁉️⁉️⁉️⁉️⁉️⁉️⁉️⁉️")
            # save the matched_mkpts to this path match_saved_mkpts_folder, in a plain text file 
            save_mkpts_to_file(matched_mkpts[0], match_saved_mkpts_folder, f"frame_{frame_index}_mkpts0.txt")
            save_mkpts_to_file(matched_mkpts[1], match_saved_mkpts_folder, f"frame_{frame_index}_mkpts1.txt")
            cluster_labels, eps_val_to_use = cluster_translation_vectors_dbscan(
                matched_mkpts[0], matched_mkpts[1],
                img1_shape=frame_img.shape[:2],
                img2_shape=sat_im.shape[:2],
                eps_scale_factor=0.05,
                dbscan_min_samples=3,
                min_points_to_attempt_clustering=5
            )
            cluster_counts, noise_count = get_counts_from_cluster_labels(cluster_labels)
            print(f"\033[95mCluster counts: {cluster_counts}\033[0m")
            print(f"\033[95mNoise count: {noise_count}\033[0m")
            # print(f"DBSCAN clustering labels: {cluster_labels}")
            print(f"\033[92m Frame {frame_index} matched. Matches so far {success}. Only {accepted_matches} accepted   \033[0m")
            
            
            rejected,reject_reason = reject_match(cluster_counts, noise_count, nr_matches_for_match)
            if rejected:
                # print(f"Frame {frame_index} out of {total_frames}: Rejected this match")
                with open(rejected_matches_file, 'a') as f:
                    f.write(f"Frame {frame_index} rejected: {reject_reason}\n")
            else:
                # match was accepted
                accepted_matches += 1
                cv2.imwrite(os.path.join(match_frame_output_path, f"frame_{frame_index}_matched.png"), downsampled_comnbined_img)

        # transfer the current frame to the last frame variable
        last_frame = frame.copy()
        mkpts_last_frame = mkpts_this_frame
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    
    print(f"There are {considered_frame_count} frames in the video")
    
    # print(f"Elapsed time: {elapsed_time}")
    # print(f"Success: {success}, Failed: {failed}")
    
    # Ensure the coordinate output directory exists
    if not os.path.exists(match_coord_output_path):
        os.makedirs(match_coord_output_path)

    # Define the output file path
    coord_file_path = os.path.join(match_coord_output_path, "match_coordinates.csv")

    # Save the coordinates to a CSV file
    try:
        with open(coord_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(["frame_index", "x", "y", "angle"])
            # Write the data
            writer.writerows(match_coord_list)
        print(f"Successfully saved match coordinates to {coord_file_path}")
    except Exception as e:
        print(f"Error saving match coordinates: {e}")

    # save a video of the matches: left side is sat image, overlayed with the warped homeography and drone position (if successful), or just a dot with drone position (if failed)


def force_match_single_frame():
    """
    This function is used to force match a single frame with the satellite image.
    """
    root_folder_for_this = "../../../data/marco_video_frames_and_sat/should_match_but_doesnt"
    sat_img = imio.imread("../../../data/marco_video_frames_and_sat/sat_images/sat.png")
    google_19 = imio.imread("../../../data/marco_video_frames_and_sat/should_match_but_doesnt/sat2_zoom19_crop.png")
    google_20 = imio.imread("../../../data/marco_video_frames_and_sat/should_match_but_doesnt/sat2_zoom20.png")
    chosen_sat_img = google_20
    
    frame_filename = "frame_000048.jpg"
    frame_path = os.path.join(root_folder_for_this, frame_filename)
    frame = imio.imread(frame_path)
    ds_frame = downsample_pyramid(frame, scales=[1.0])[0]
    output_subfolder = "./output"
    output_path = os.path.join(root_folder_for_this, output_subfolder)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    #* generate 1 degree rotations of the frame
    angles = [i for i in range(0, 360, 1)]
    # pyrefly: ignore  # bad-argument-type
    rotated_frames_ds = generate_rotations(ds_frame, angles)
    for frame_rt_ds,angle in zip(rotated_frames_ds,angles):
        # pyrefly: ignore  # bad-argument-type
        canvas, mkpts0, mkpts1, inlier_ratio,nr_matches, _ = match_xfeat(image1=chosen_sat_img, feat1=None, image2=frame_rt_ds, feat2=None, top_k=top_k_frames)
        # canvas = match_xfeat_star(sat_img, frame_rt_ds, top_k=8000)[0]
        if canvas is not None:
            #* save the canvas with the inlier ratio
            # inlier_ratio_string = f"{inlier_ratio:.2f}"
            # combined_img_with_text = add_text_to_image(canvas, inlier_ratio_string, color_name="red", position=(200, 200), font_scale=8, thickness=4)
            # print(canvas)
            canvas = canvas[..., ::-1]
            cv2.imwrite(os.path.join(output_path, f"frame_angle_{angle}_matched.png"), canvas)
            
def try_find_initial_angle():
    print("Trying to find the initial angle")
    sat_img = imio.imread("../../../data/marco_video_frames_and_sat/sat_images/sat.png")
    frame = imio.imread("../../../data/marco_video_frames_and_sat/sunny_all_frames/frame_000048.jpg")
    angle_to_try = -80
    frame_rotated = rotate_once(frame, angle_to_try)
    plot_1_image(frame_rotated, title="frame_rotated")

if __name__ == "__main__":
    # main2(3,False)
    main2()
        # plot_1_image
    # match_and_visualize_xfeat(sat_img, original_frame)
