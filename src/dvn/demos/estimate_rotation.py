import numpy as np
import cv2

def estimate_inplane_rotation(
    mkpts_0: np.ndarray,
    mkpts_1: np.ndarray,
    ransac_reproj_threshold: float = 3.0,
    confidence: float = 0.999,
    max_iters: int = 2000,
    refine_iters: int = 10,
    do_refine: bool = True,
):
    """
    NOTE GPT-5 implemented: https://chatgpt.com/c/68e7df3d-412c-832f-a9d0-f1f2137c8ae7
    Estimate the global 2D rotation between two images from matched points using a
    similarity transform (scale + rotation + translation). The returned angle
    is the in-plane rotation (radians & degrees) that best maps mkpts_0 -> mkpts_1.

    Parameters
    ----------
    mkpts_0 : (N,2) ndarray
        Matched points in image 0 (x, y).
    mkpts_1 : (N,2) ndarray
        Corresponding matched points in image 1 (x, y).
    ransac_reproj_threshold : float
        RANSAC reprojection error (in pixels) for inlier selection.
    confidence : float
        Desired RANSAC confidence (0..1).
    max_iters : int
        Maximum RANSAC iterations.
    refine_iters : int
        Number of local refinement iterations (OpenCV) after RANSAC.
    do_refine : bool
        If True, re-fit a rotation (and scale) with SVD on the inliers for a
        slightly cleaner angle.

    Returns
    -------
    result : dict with keys
        - 'angle_rad': rotation angle (radians) from the similarity matrix.
        - 'angle_deg': same angle in degrees.
        - 'scale': isotropic scale factor.
        - 'translation': (tx, ty) from the similarity.
        - 'inliers_mask': boolean mask of shape (N,) marking RANSAC inliers.
        - 'M': 2x3 similarity matrix mapping mkpts_0 -> mkpts_1.
        - 'angle_refined_rad' (optional): refined angle from Procrustes on inliers.
        - 'angle_refined_deg' (optional): same, in degrees.
    """
    if mkpts_0.ndim != 2 or mkpts_0.shape[1] != 2 or mkpts_1.ndim != 2 or mkpts_1.shape[1] != 2:
        raise ValueError("mkpts_0 and mkpts_1 must be (N,2) arrays.")
    if mkpts_0.shape[0] < 2 or mkpts_1.shape[0] < 2:
        raise ValueError("Need at least 2 matches to estimate a similarity transform.")
    if mkpts_0.shape[0] != mkpts_1.shape[0]:
        raise ValueError("mkpts_0 and mkpts_1 must have the same number of rows (matches).")

    # OpenCV expects float32
    P0 = mkpts_0.astype(np.float32)
    P1 = mkpts_1.astype(np.float32)

    M, inliers = cv2.estimateAffinePartial2D(
        P0, P1,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=confidence,
        maxIters=max_iters,
        refineIters=refine_iters,
    )

    if M is None or inliers is None:
        raise RuntimeError("estimateAffinePartial2D failed; not enough consensus for a similarity model.")

    inliers = inliers.ravel().astype(bool)

    # Decompose similarity: M = [[a, -b, tx],
    #                            [b,  a, ty]]
    a, b = M[0, 0], M[1, 0]
    scale = float(np.hypot(a, b))
    angle_rad = float(np.arctan2(b, a))
    angle_deg = float(np.degrees(angle_rad))
    tx, ty = float(M[0, 2]), float(M[1, 2])

    result = {
        "angle_rad": angle_rad,
        "angle_deg": angle_deg,
        "scale": scale,
        "translation": (tx, ty),
        "inliers_mask": inliers,
        "M": M.copy(),
    }

    if do_refine and inliers.sum() >= 2:
        # Procrustes/Umeyama on inliers for a cleaner angle (still similarity).
        X = mkpts_0[inliers].astype(np.float64)
        Y = mkpts_1[inliers].astype(np.float64)

        Xc = X.mean(axis=0, keepdims=True)
        Yc = Y.mean(axis=0, keepdims=True)
        X0 = X - Xc
        Y0 = Y - Yc

        # Covariance
        H = X0.T @ Y0 / max(1, X0.shape[0])

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Enforce proper rotation (det=+1)
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        # Angle from refined rotation
        angle_refined_rad = float(np.arctan2(R[1, 0], R[0, 0]))
        angle_refined_deg = float(np.degrees(angle_refined_rad))
        result["angle_refined_rad"] = angle_refined_rad
        result["angle_refined_deg"] = angle_refined_deg

    return result

#! ---------------- TESTING ----------------
frame0 = "/devprojects/dronevisnav/data/images/marco_sunny_frames/downsampled_0_6/frame_000266.jpg"
frame1 = "/devprojects/dronevisnav/data/images/marco_sunny_frames/downsampled_0_6/frame_000267.jpg"

def test_estimate_inplane_rotation():
    """
    Test the estimate_inplane_rotation function using XFeat sparse default matching.
    Loads two consecutive frames, matches features using XFeat, and estimates the
    in-plane rotation between them.
    """
    # Import XFeat model
    from dvn.models.xfeat_.xfeat_methods import XFeatModel

    # Load the images
    im0 = cv2.imread(frame0)
    im1 = cv2.imread(frame1)

    if im0 is None or im1 is None:
        raise ValueError(f"Failed to load images: {frame0} or {frame1}")

    print(f"Loaded images: {frame0} and {frame1}")
    print(f"Image 0 shape: {im0.shape}")
    print(f"Image 1 shape: {im1.shape}")

    # Initialize XFeat model with default pretrained weights
    print("\nInitializing XFeat model...")
    xfeat_model = XFeatModel(top_k=4096)

    # Detect and compute features for both images
    print("\nDetecting and computing features...")
    feat0 = xfeat_model.xfeat_detect_and_compute(im0, top_k=4096)
    feat1 = xfeat_model.xfeat_detect_and_compute(im1, top_k=4096)

    print(f"Image 0 features detected: {feat0['keypoints'].shape[0]}")
    print(f"Image 1 features detected: {feat1['keypoints'].shape[0]}")

    # Match features using sparse default matching (LighterGlue)
    print("\nMatching features using XFeat sparse default matching...")
    mkpts_0, mkpts_1 = xfeat_model.xfeat_match_sparse_default(feat0, feat1)

    print(f"Total matches found: {len(mkpts_0)}")

    if len(mkpts_0) < 2:
        raise ValueError("Not enough matches to estimate rotation (need at least 2)")

    # Estimate in-plane rotation
    print("\nEstimating in-plane rotation...")
    result = estimate_inplane_rotation(
        mkpts_0=mkpts_0,
        mkpts_1=mkpts_1,
        ransac_reproj_threshold=3.0,
        confidence=0.999,
        max_iters=2000,
        refine_iters=10,
        do_refine=True,
    )

    # Display results
    print("\n" + "="*60)
    print("ROTATION ESTIMATION RESULTS")
    print("="*60)
    print(f"Angle (radians):           {result['angle_rad']:.6f}")
    print(f"Angle (degrees):           {result['angle_deg']:.3f}°")
    print(f"Scale factor:              {result['scale']:.6f}")
    print(f"Translation (tx, ty):      ({result['translation'][0]:.2f}, {result['translation'][1]:.2f})")
    print(f"Number of inliers:         {result['inliers_mask'].sum()} / {len(mkpts_0)}")
    print(f"Inlier ratio:              {result['inliers_mask'].sum() / len(mkpts_0):.2%}")

    if 'angle_refined_rad' in result:
        print(f"\nRefined angle (radians):   {result['angle_refined_rad']:.6f}")
        print(f"Refined angle (degrees):   {result['angle_refined_deg']:.3f}°")
        angle_diff = abs(result['angle_deg'] - result['angle_refined_deg'])
        print(f"Refinement difference:     {angle_diff:.3f}°")

    print("="*60)

    # Visualize: rotate im0 by the estimated angle and compare
    from dvn.utils.image_utils.plot_images import plot_1_image

    print("\nVisualizing rotation correction...")

    # Get image dimensions
    h, w = im0.shape[:2]
    center = (w / 2, h / 2)

    # Create rotation matrix (negative angle to correct the rotation)
    # Using refined angle if available, otherwise use the RANSAC angle
    angle_to_use = result.get('angle_refined_deg', result['angle_deg'])
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle_to_use, 1.0)

    # Apply rotation
    im0_rotated = cv2.warpAffine(im0, rotation_matrix, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))

    # Create a side-by-side comparison
    # Convert BGR to RGB for plotting
    im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    im0_rotated_rgb = cv2.cvtColor(im0_rotated, cv2.COLOR_BGR2RGB)
    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

    # Concatenate images horizontally
    comparison = np.hstack([im0_rgb, im0_rotated_rgb, im1_rgb])

    # Plot using plot_1_image
    plot_1_image(
        image=comparison,
        title=f'Left: Original Image 0 | Center: Rotated by {-angle_to_use:.3f}° | Right: Image 1 (Target)',
        figsize=(18, 6),
        tight_layout=True
    )

    return result

if __name__ == '__main__':
    test_estimate_inplane_rotation()
    print("Test completed.")