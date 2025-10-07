from typing import Optional, no_type_check
import numpy as np
import cv2
from dvn.utils.algebra_utils.homo import find_homography

# @no_type_check
def warp_corners_and_draw_matches(
    ref_points: np.ndarray,
    dst_points: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    draw_match_lines: bool = True,
    precomputed_H=None,
    precomputed_inlier_mask=None,
) -> np.ndarray | None:
    """
    Calculates homography, warps the corners of img1 onto img2, draws the warped
    corners, and optionally draws lines for inlier matches.

    Args:
        ref_points: Keypoints from the reference image (img1), shape (N, 2).
        dst_points: Corresponding keypoints from the destination image (img2), shape (N, 2).
        img1: The reference image (BGR format).
        img2: The destination image (BGR format).
        draw_match_lines (bool, optional):
            - If True (default), draws the lines
            connecting inlier matches between img1 and img2 in a combined visualization.
            - If False, only draws the warped polygon outline of img1 onto img2
            and returns just img2 with the polygon.

    Returns:
        np.ndarray | None: The output image.
            - If draw_match_lines is True: A combined image showing img1 and img2
              side-by-side with match lines and the warped polygon.
            - If draw_match_lines is False: img2 with the warped polygon drawn on it.
            - None if homography estimation or warping fails.
    """
    
    #* Calculate the Homography matrix
    H = None
    inlier_mask = None
    if precomputed_H is None and precomputed_inlier_mask is None:
        H, inlier_mask, _ = find_homography(ref_points, dst_points)
    else:
        H = precomputed_H
        inlier_mask = precomputed_inlier_mask

    if H is None or inlier_mask is None:
        print("⚠️  Homography estimation failed: skipping this pair.")
        return None

    inlier_mask = inlier_mask.flatten()

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
            return None
    except cv2.error as e:
        print(f"⚠️ cv2.error during perspectiveTransform: {e}")
        return None


    # --- Draw the warped corners in image2 ---
    img2_with_corners = img2.copy()
    # Check if warped_corners are valid before drawing
    if warped_corners is not None:
        # Convert to integer points for drawing polylines
        pts = np.int32(warped_corners.reshape(-1, 2))
        cv2.polylines(img=img2_with_corners, pts=[pts], isClosed=True, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)  # pyright: ignore[reportCallIssue]
        # Optional: Draw individual corners if needed
        # for i in range(len(warped_corners)):
        #     pt = tuple(warped_corners[i][0].astype(int))
        #     cv2.circle(img2_with_corners, pt, 5, (0, 0, 255), -1) # Red dots at corners

    # --- Decide what image to return based on the flag ---
    return_img = None
    if draw_match_lines:
        # Prepare keypoints and matches for drawMatches function

        keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
        keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]

        # Create DMatch objects only for inliers
        matches = [cv2.DMatch(i, i, 0) for i, m in enumerate(inlier_mask) if m]

        # Draw inlier matches onto the combined image
        # Use img2_with_corners which already has the polygon
        img_matches_combined = cv2.drawMatches(                         # pyright: ignore[reportCallIssue]
            img1=img1, keypoints1=keypoints1,
            img2=img2_with_corners, keypoints2=keypoints2,
            matches1to2=matches, outImg=None,
            matchColor=(0, 255, 0), # Green lines for matches
            singlePointColor=(255, 0, 0), # Blue single points (if any)
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS # Don't draw unmatched keypoints
        )
        return_img = img_matches_combined
    else:
        # Return only the second image with the warped corners drawn

        return_img = img2_with_corners

    return return_img