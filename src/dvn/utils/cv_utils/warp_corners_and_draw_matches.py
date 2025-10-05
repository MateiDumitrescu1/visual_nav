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