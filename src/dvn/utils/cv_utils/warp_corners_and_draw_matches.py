from typing import Optional, no_type_check
import numpy as np
import cv2
from dvn.utils.algebra_utils.homo import find_homography

def warp_and_draw_corners(
    img1: np.ndarray,
    img2: np.ndarray,
    H: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 4,
) -> np.ndarray | None:
    """
    Warps the corners of img1 onto img2 using the homography matrix H and draws the warped polygon.

    Args:
        img1: The reference image (BGR format).
        img2: The destination image (BGR format).
        H: The homography matrix (3x3) that maps points from img1 to img2.
        color: Color for the polygon in BGR format. Default is green (0, 255, 0).
        thickness: Thickness of the polygon lines. Default is 4.

    Returns:
        np.ndarray | None: img2 with the warped polygon drawn on it, or None if warping fails.
    """
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

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()

    # warped_corners stores the transformed corner coordinates of img1 projected into img2's coordinate space
    if warped_corners is not None:
        pts = warped_corners.reshape(-1, 1, 2).astype(np.int32)
        # pts = np.int32(warped_corners.reshape(-1, 2)) # this was the old way of doing it, but the type error was annoying
        cv2.polylines(
            img=img2_with_corners,
            pts=[pts],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )  # pyright: ignore[reportCallIssue]

    return img2_with_corners


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    ref_points: np.ndarray,
    dst_points: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    colors: Optional[list[tuple[int, int, int]]] = None,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draws match lines between corresponding keypoints in two images placed side by side.
    Allows custom colors for each keypoint pair.

    Args:
        img1: The reference image (BGR format).
        img2: The destination image (BGR format).
        ref_points: Keypoints from the reference image (img1), shape (N, 2).
        dst_points: Corresponding keypoints from the destination image (img2), shape (N, 2).
        inlier_mask: Optional boolean mask indicating which matches are inliers, shape (N,).
                     If None, all matches will be drawn.
        colors: Optional list of BGR colors for each match. If None, all matches are green.
                Length should match the number of matches to draw.
        thickness: Thickness of the match lines. Default is 1.

    Returns:
        np.ndarray: Combined image showing img1 and img2 side-by-side with match lines.
    """
    if thickness is None:
        thickness = 1
    
    #* If no inlier mask provided, create one that selects all matches
    if inlier_mask is None:
        inlier_mask = np.ones(len(ref_points), dtype=bool)
    else:
        inlier_mask = inlier_mask.flatten()

    #* Get inlier indices
    inlier_indices = np.where(inlier_mask)[0]

    #* If no custom colors provided, use green for all matches
    if colors is None:
        colors = [(0, 255, 0)] * len(inlier_indices)
    else:
        if len(colors) != len(inlier_indices):
            raise ValueError("Length of colors list must match number of inliers.")
    
    #* Create combined image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    combined_height = max(h1, h2)
    combined_width = w1 + w2

    combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    combined_img[0:h1, 0:w1] = img1
    combined_img[0:h2, w1:w1+w2] = img2
    
    #* Draw match lines for each inlier (or all if no mask)
    for i, idx in enumerate(inlier_indices):
        pt1 = tuple(ref_points[idx].astype(int))
        pt2 = tuple((dst_points[idx] + np.array([w1, 0])).astype(int))

        color = colors[i] if i < len(colors) else (0, 255, 0)
        cv2.line(combined_img, pt1, pt2, color, thickness, cv2.LINE_AA)

    return combined_img

# @no_type_check
#TODO change the name of this to make it clear it only draw inlier matches
def warp_corners_and_draw_matches(
    ref_points: np.ndarray,
    dst_points: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    # params for draw_matches
    draw_match_lines: bool = True,
    colors: Optional[list[tuple[int, int, int]]] = None,
    thickness: int = 1,
    # precomputed homography and inlier mask (to avoid recomputation)
    precomputed_H=None,
    precomputed_inlier_mask=None,
) -> np.ndarray | None:
    """
    This is a convenience wrapper that combines warp_and_draw_corners and draw_matches.
    """

    # Calculate the Homography matrix
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

    # Warp and draw corners on img2
    img2_with_corners = warp_and_draw_corners(img1, img2, H)
    if img2_with_corners is None:
        return None

    # Decide what image to return based on the `draw_match_lines` flag
    if draw_match_lines:
        # Draw match lines on combined image
        return draw_matches(img1=img1, img2=img2_with_corners, ref_points=ref_points, dst_points=dst_points, inlier_mask=inlier_mask, colors=colors, thickness=thickness)
    else:
        return img2_with_corners