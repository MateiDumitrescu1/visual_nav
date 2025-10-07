import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from hdbscan import HDBSCAN
from dvn.models.xfeat_.xfeat_methods import XFeatModel
from dvn.models.config import FeatureMatchingOutput
from dvn.utils.algebra_utils.homo import find_homography
from dvn.utils.cv_utils.warp_corners_and_draw_matches import warp_corners_and_draw_matches

def cluster_translation_vectors_hdbscan(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    min_points_to_attempt_clustering: int = 5
) -> Tuple[np.ndarray, int]:
    """
    Clusters translation vectors derived from matched keypoints using HDBSCAN.

    This function calculates translation vectors (mkpts1 - mkpts0) and then
    applies HDBSCAN clustering to these vectors. It returns the raw cluster labels
    assigned by HDBSCAN to each input point and the number of clusters found.

    The interpretation of these labels (e.g., identifying dominant clusters,
    filtering points, scoring) is intended to be handled by the caller.

    Args:
        mkpts0: Keypoints from the first image, shape (N, 2).
        mkpts1: Corresponding keypoints from the second image, shape (N, 2).
        img1_shape: Optional (height, width) of image 1 (not used with HDBSCAN).
        img2_shape: Optional (height, width) of image 2 (not used with HDBSCAN).
        min_cluster_size: The minimum size of clusters. HDBSCAN will not produce
                          clusters smaller than this value.
        min_samples: The number of samples in a neighborhood for a point to be
                     considered as a core point.
        min_points_to_attempt_clustering: If the number of initial matches is less
                                          than this (or zero), clustering is skipped,
                                          and an empty labels array and 0 clusters
                                          are returned.

    Returns:
        Tuple[np.ndarray, int]:
            - cluster_labels (np.ndarray): An array of shape (N,) where N is the
              number of input keypoints. Each element is the cluster ID assigned
              by HDBSCAN (e.g., 0, 1, ...). Noise points are labeled -1.
              If clustering is skipped or fails critically, an empty array is returned.
            - num_clusters (int): The number of clusters found (excluding noise).
    """
    num_initial_matches = len(mkpts0)

    if num_initial_matches < min_points_to_attempt_clustering:
        print(f"HDBSCAN: Skipping, number of points ({num_initial_matches}) "
              f"is less than min_points_to_attempt_clustering ({min_points_to_attempt_clustering}).")
        return np.array([], dtype=int), 0

    # compute the translation vector
    translations = mkpts1 - mkpts0

    print(f"HDBSCAN params: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    try:
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(translations)
        cluster_labels = clusterer.labels_
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"HDBSCAN: Found {num_clusters} clusters")
        return cluster_labels, num_clusters
    except ValueError as e:
        print(f"⚠️ HDBSCAN clustering error: {e}. This can happen with invalid parameters.")
        return np.array([], dtype=int), 0

def get_counts_from_cluster_labels(labels: np.ndarray) -> Tuple[Dict[int, int], int]:
    """
    Processes HDBSCAN cluster labels to count points per cluster and noise points.
    Args:
        labels (np.ndarray): An array of cluster labels as returned by
                             HDBSCAN. Noise points are typically labeled -1.

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
        label = int(unique_labels[i])
        count = int(counts_elements[i])

        if label == -1:
            # This is a noise point
            noise_count += count
        else:
            # This is a valid cluster
            cluster_counts[label] = count

    return cluster_counts, noise_count


def reject_match(cluster_counts: Dict[int, int], noise_count: int, nr_matches: int) -> Tuple[bool, str]:
    """
    Determines whether to reject a match based on clustering results.

    Args:
        cluster_counts: Dictionary mapping cluster IDs to their point counts
        noise_count: Number of noise points (labeled -1 by DBSCAN)
        nr_matches: Total number of matches

    Returns:
        Tuple[bool, str]: (should_reject, reason)
    """
    if nr_matches <= 5:
        return True, f"Too few matches, only {nr_matches} matches"

    total_clustered_points = sum(cluster_counts.values())
    if noise_count + 1 >= total_clustered_points:
        return True, f"We had {noise_count} noise points and {total_clustered_points} clustered points. Too many noise points"

    nr_clusters = len(cluster_counts)
    if nr_clusters >= 2:
        small_clusters = [count for count in cluster_counts.values() if count < 0.2 * total_clustered_points]
        nr_small_clusters = len(small_clusters)

        if nr_small_clusters == 0:
            return True, f"We had {nr_clusters} clusters. They were all big."

    return False, "Accepted"

def feature_match_images(
    img1: np.ndarray,
    img2: np.ndarray,
    top_k: int = 4096,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    min_points_to_attempt_clustering: int = 5,
) -> FeatureMatchingOutput | None:
    """
    Use XFeat to feature match two images, then run a clustering algorithm over
    the translation vectors to filter out bad matches.

    ### Params:
    - `img1`: First image as a numpy array
    - `img2`: Second image as a numpy array
    - `top_k`: Number of top features to consider (default 4096)
    - `min_cluster_size`: HDBSCAN minimum cluster size (default 5)
    - `min_samples`: HDBSCAN min_samples parameter (default 3)
    - `min_points_to_attempt_clustering`: Minimum matches required for clustering (default 5)

    ### Returns:
    A FeatureMatchingOutput object with filtered matches and visualization, or None if matching fails.
    """
    # Initialize XFeat model
    xfeat_model = XFeatModel(top_k=top_k)

    # Match features between the two images
    match_result = xfeat_model.match_xfeat(img1, None, img2, None, top_k=top_k)

    if match_result is None:
        print("❌ Feature matching returned None (not enough matches)")
        return None

    mkpts0 = match_result.mkpts_0
    mkpts1 = match_result.mkpts_1
    nr_matches = match_result.nr_matches

    if mkpts0 is None or mkpts1 is None or nr_matches is None:
        print("❌ Missing keypoints or match count")
        return None

    print(f"Initial matches: {nr_matches}")

    # Cluster translation vectors using HDBSCAN
    cluster_labels, num_clusters = cluster_translation_vectors_hdbscan(
        mkpts0,
        mkpts1,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        min_points_to_attempt_clustering=min_points_to_attempt_clustering
    )

    # Get cluster statistics
    cluster_counts, noise_count = get_counts_from_cluster_labels(cluster_labels)
    print(f"Cluster counts: {cluster_counts}")
    print(f"Noise count: {noise_count}")

    # Decide whether to reject the match
    should_reject, reject_reason = reject_match(cluster_counts, noise_count, nr_matches)

    if should_reject:
        print(f"❌ Match rejected: {reject_reason}")
        return None

    print(f"✅ Match accepted: {reject_reason}")

    # Find homography
    H, inlier_mask, inlier_ratio = find_homography(mkpts0, mkpts1)

    if H is None:
        print("❌ Failed to compute homography")
        return None

    # Create visualization
    im1 = xfeat_model.prepare_np_array_image_for_xfeat(img1)
    im2 = xfeat_model.prepare_np_array_image_for_xfeat(img2)

    output_canvas = warp_corners_and_draw_matches(
        mkpts0,
        mkpts1,
        im1,
        im2,
        draw_match_lines=True,
        precomputed_H=H,
        precomputed_inlier_mask=inlier_mask
    )

    # Return complete FeatureMatchingOutput
    return FeatureMatchingOutput(
        output_canvas=output_canvas,
        mkpts_0=mkpts0,
        mkpts_1=mkpts1,
        feat0=match_result.feat0,
        feat1=match_result.feat1,
        inlier_ratio=inlier_ratio,
        nr_matches=nr_matches,
        warped_corners=None  # Could be extracted from warp_corners_and_draw_matches if needed
    )
    
    
#! ------------------- TESTING -------------------

from paths_ import PathLogic, TEST_SET
from dvn.utils.image_utils.plot_images import plot_1_image


def test_feature_match_images():
    """
    Test the feature_match_images function with test image pairs.
    """
    test_img_sets = PathLogic.get_test_image_sets(TEST_SET.FEATURE_MATCHING_TEST)
    test_imgs = test_img_sets.get("xfeat_example", None)

    if test_imgs is None:
        raise ValueError("No test images found in the 'xfeat_example' test set.")

    img1: np.ndarray = cv2.imread(test_imgs[0]) # type: ignore[assignment]
    img2: np.ndarray = cv2.imread(test_imgs[1]) # type: ignore[assignment]

    if img1 is None or img2 is None:
        raise ValueError("Failed to load one or both test images.")

    print(f"Testing feature_match_images with images of shapes: {img1.shape}, {img2.shape}")

    # Run the feature matching with clustering
    result = feature_match_images(img1, img2, top_k=4096)

    if result is None:
        print("❌ Feature matching failed or was rejected by clustering")
        return

    print(f"✅ Feature matching successful!")
    print(f"   - Matches: {result.nr_matches}")
    print(f"   - Inlier ratio: {result.inlier_ratio:.4f}")

    # Visualize the result
    if result.output_canvas is not None:
        # Convert BGR to RGB for display
        output_rgb = cv2.cvtColor(result.output_canvas, cv2.COLOR_BGR2RGB)
        plot_1_image(
            image=output_rgb,
            title=f'Feature Match (Clustered): {result.nr_matches} matches, {result.inlier_ratio:.2f} inlier ratio',
            tight_layout=True
        )
    else:
        print("⚠️ No output canvas generated")

if __name__ == '__main__':
    test_feature_match_images()
    print("Test completed!")
