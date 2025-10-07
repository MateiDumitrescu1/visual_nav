import numpy as np
import cv2
from typing import Optional, no_type_check, Tuple

ransacReprojThresholdParam = 5.0

def find_homography(
    points1: np.ndarray,  # shape (N, 2)
    points2: np.ndarray,   # shape (N, 2)
    debug_level: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Uses `cv2.findHomography` to find homography matrix between two sets of points (points1 and points2).
    Also calculates the inlier ratio.

    ### Returns:
    - `H`: Homography matrix of shape (3, 3) that maps points1 to points2.
        If homography cannot be computed, returns None.
    - `inlier_mask`: Binary inlier mask of shape (N, 1) where 1 indicates the point pair
        was classified as an inlier by RANSAC, 0 indicates outlier.
    - `inlier_ratio`: Ratio of inliers to total matches (num_inliers / num_matches).
    """
    
    # Ensure inputs are numpy ndarrays and have dtype float32. Use astype with
    # copy=False so that a numpy ndarray is returned (satisfies type checkers).
    points1 = np.asarray(points1).astype(dtype=np.float32, copy=False)
    points2 = np.asarray(points2).astype(dtype=np.float32, copy=False)
    
    # Print sample matching points for verification
    if debug_level >= 1:
        print(f"Matching points shape: {points1.shape}, {points2.shape}")
        print("Sample matching points:")
        
    for i in range(min(5, len(points1))):
        print(f"  {i}: {points1[i]} -> {points2[i]}")

    # Find homography
    H, inlier_mask = cv2.findHomography(
        points1,
        points2,
        method=cv2.USAC_MAGSAC, #
        ransacReprojThreshold=ransacReprojThresholdParam,
        confidence=0.9999,
        maxIters=1000,
    )
    if debug_level >= 2:
        print(f"Homography matrix: {H}")

    # Calculate inlier ratio
    if inlier_mask is not None:
        inlier_mask_flat = inlier_mask.flatten()
        num_inliers = np.sum(inlier_mask_flat)
        num_matches = len(inlier_mask_flat)
        if num_matches == 0:
            print("⚠️ No matches provided to findHomography.")
            inlier_ratio = 0.0
        else:
            inlier_ratio = float(num_inliers / num_matches)
        print(f'Inlier ratio: {inlier_ratio:.4f} ({int(num_inliers)}/{num_matches})')
    else:
        print("Failed to compute homography - not enough valid point correspondences")
        inlier_ratio = 0.0

    return H, inlier_mask, inlier_ratio


