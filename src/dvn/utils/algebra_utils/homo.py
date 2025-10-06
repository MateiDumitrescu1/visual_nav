import numpy as np
import cv2
from typing import Optional, no_type_check, Tuple

ransacReprojThresholdParam = 5.0

def find_homography(
    points1: np.ndarray,  # shape (N, 2)
    points2: np.ndarray,   # shape (N, 2)
    debug_level: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find homography matrix between two sets of points (points1 and points2).

    ### Returns:
    - `H`: Homography matrix of shape (3, 3) that maps points1 to points2.
        If homography cannot be computed, returns None.
    - `mask`: Binary inlier mask of shape (N, 1) where 1 indicates the point pair
        was classified as an inlier by RANSAC, 0 indicates outlier.
    """
    
    # Ensure inputs are numpy ndarrays and have dtype float32. Use astype with
    # copy=False so that a numpy ndarray is returned (satisfies type checkers).
    points1 = np.asarray(points1).astype(np.float32, copy=False)
    points2 = np.asarray(points2).astype(np.float32, copy=False)
    
    # Print sample matching points for verification
    if debug_level >= 1:
        print(f"Matching points shape: {points1.shape}, {points2.shape}")
        print("Sample matching points:")
        
    for i in range(min(5, len(points1))):
        print(f"  {i}: {points1[i]} -> {points2[i]}")

    # Find homography
    H, mask = cv2.findHomography(
        points1,
        points2,
        method=cv2.USAC_MAGSAC, # 
        ransacReprojThreshold=ransacReprojThresholdParam,
        confidence=0.9999,
        maxIters=10000,
    )
    if debug_level >= 2:
        print(f"Homography matrix: {H}")
 
    # Count inliers
    if mask is not None:
        inliers = mask.ravel().sum()
        print(f"Number of inliers: {inliers} out of {len(points1)} matches")
    else:
        print("Failed to compute homography - not enough valid point correspondences")

    return H, mask


