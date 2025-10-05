import numpy as np
import cv2
from typing import Optional, no_type_check

@no_type_check
def find_homography(points1, points2):
    """Find homography matrix between two sets of points"""
    # Convert to float32
    points1 = np.float32(points1)
    points2 = np.float32(points2)
    
    # Print sample matching points for verification
    print(f"Matching points shape: {points1.shape}, {points2.shape}")
    print("Sample matching points:")
    for i in range(min(5, len(points1))):
        print(f"  {i}: {points1[i]} -> {points2[i]}")

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


