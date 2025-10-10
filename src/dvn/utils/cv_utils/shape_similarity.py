"""
Shape similarity utilities for comparing warped corner projections from homography matrices.
"""

import numpy as np
import cv2
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def compare_warped_shapes(
    warped_corners1: np.ndarray,
    warped_corners2: np.ndarray,
    normalize: bool = True
) -> Tuple[float, dict]:
    """
    Compare the similarity of two shapes defined by warped corners from homography projections.

    This method computes multiple similarity metrics between two quadrilaterals:
    1. IoU (Intersection over Union) - measures area overlap
    2. Shape Context Distance - measures shape structural similarity
    3. Hausdorff Distance - measures maximum point-to-point distance
    4. Centroid Distance - measures distance between shape centers
    5. Area Ratio - measures relative size difference

    Args:
        warped_corners1: First set of warped corners, shape (4, 1, 2) or (4, 2)
        warped_corners2: Second set of warped corners, shape (4, 1, 2) or (4, 2)
        normalize: If True, normalize distances by the diagonal of the bounding box

    Returns:
        overall_similarity: A combined similarity score in [0, 1] where 1 is identical
        metrics_dict: Dictionary containing individual metrics:
            - 'iou': Intersection over Union
            - 'hausdorff_dist': Hausdorff distance (normalized if normalize=True)
            - 'centroid_dist': Distance between centroids (normalized if normalize=True)
            - 'area_ratio': Ratio of smaller/larger area
            - 'corner_dist': Mean corner-to-corner distance (normalized if normalize=True)
    """
    # Reshape corners to (4, 2) format
    corners1 = warped_corners1.reshape(4, 2).astype(np.float32)
    corners2 = warped_corners2.reshape(4, 2).astype(np.float32)

    # Compute bounding box diagonal for normalization
    all_corners = np.vstack([corners1, corners2])
    bbox_min = all_corners.min(axis=0)
    bbox_max = all_corners.max(axis=0)
    bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)

    # 1. Intersection over Union (IoU)
    iou = _compute_iou(corners1, corners2)

    # 2. Hausdorff Distance (measures max distance between shapes)
    hausdorff_dist = _compute_hausdorff_distance(corners1, corners2)
    if normalize and bbox_diagonal > 0:
        hausdorff_dist /= bbox_diagonal

    # 3. Centroid Distance
    centroid1 = corners1.mean(axis=0)
    centroid2 = corners2.mean(axis=0)
    centroid_dist = np.linalg.norm(centroid1 - centroid2)
    if normalize and bbox_diagonal > 0:
        centroid_dist /= bbox_diagonal

    # 4. Area Ratio (ratio of smaller to larger area)
    area1 = cv2.contourArea(corners1.reshape(-1, 1, 2))
    area2 = cv2.contourArea(corners2.reshape(-1, 1, 2))
    area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0.0

    # 5. Corner-to-corner distance (after optimal alignment)
    # Find best correspondence between corners
    corner_dist = _compute_min_corner_distance(corners1, corners2)
    if normalize and bbox_diagonal > 0:
        corner_dist /= bbox_diagonal

    # Compute overall similarity score
    # Convert distances to similarities (lower distance = higher similarity)
    hausdorff_sim = np.exp(-3 * hausdorff_dist)  # exp decay
    centroid_sim = np.exp(-5 * centroid_dist)    # exp decay
    corner_sim = np.exp(-3 * corner_dist)        # exp decay

    # Weighted combination of all metrics
    overall_similarity = (
        0.4 * iou +           # 40% weight on overlap
        0.2 * hausdorff_sim + # 20% weight on boundary similarity
        0.2 * corner_sim +    # 20% weight on corner alignment
        0.1 * centroid_sim +  # 10% weight on position
        0.1 * area_ratio      # 10% weight on size
    )

    metrics_dict = {
        'iou': float(iou),
        'hausdorff_dist': float(hausdorff_dist),
        'centroid_dist': float(centroid_dist),
        'area_ratio': float(area_ratio),
        'corner_dist': float(corner_dist),
        'hausdorff_sim': float(hausdorff_sim),
        'centroid_sim': float(centroid_sim),
        'corner_sim': float(corner_sim),
    }

    return float(overall_similarity), metrics_dict


def _compute_iou(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Compute Intersection over Union for two quadrilaterals.

    Args:
        corners1: First quadrilateral corners, shape (4, 2)
        corners2: Second quadrilateral corners, shape (4, 2)

    Returns:
        IoU score in [0, 1]
    """
    # Create a canvas large enough to hold both shapes
    all_points = np.vstack([corners1, corners2])
    min_x, min_y = all_points.min(axis=0).astype(int) - 10
    max_x, max_y = all_points.max(axis=0).astype(int) + 10

    width = max_x - min_x
    height = max_y - min_y

    # Shift corners to canvas coordinates
    corners1_shifted = (corners1 - np.array([min_x, min_y])).astype(np.int32)
    corners2_shifted = (corners2 - np.array([min_x, min_y])).astype(np.int32)

    # Create masks for both shapes
    mask1 = np.zeros((height, width), dtype=np.uint8)
    mask2 = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(mask1, [corners1_shifted.reshape(-1, 1, 2)], (255,))
    cv2.fillPoly(mask2, [corners2_shifted.reshape(-1, 1, 2)], (255,))

    # Compute intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def _compute_hausdorff_distance(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Compute Hausdorff distance between two sets of corners.

    Args:
        corners1: First set of corners, shape (4, 2)
        corners2: Second set of corners, shape (4, 2)

    Returns:
        Hausdorff distance
    """
    # Distance from each point in corners1 to closest point in corners2
    dist_1_to_2 = np.array([np.min(np.linalg.norm(corners2 - p, axis=1)) for p in corners1])
    # Distance from each point in corners2 to closest point in corners1
    dist_2_to_1 = np.array([np.min(np.linalg.norm(corners1 - p, axis=1)) for p in corners2])

    # Hausdorff distance is the maximum of these minimum distances
    return max(dist_1_to_2.max(), dist_2_to_1.max())


def _compute_min_corner_distance(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Compute minimum mean corner-to-corner distance by trying all 4 rotational alignments.

    Args:
        corners1: First set of corners, shape (4, 2)
        corners2: Second set of corners, shape (4, 2)

    Returns:
        Minimum mean distance across all alignments
    """
    min_dist = float('inf')

    # Try all 4 rotational alignments
    for shift in range(4):
        corners2_shifted = np.roll(corners2, shift, axis=0)
        dist = np.mean(np.linalg.norm(corners1 - corners2_shifted, axis=1))
        min_dist = min(min_dist, dist)

    return min_dist


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_compare_warped_shapes():
    """
    Test the compare_warped_shapes function with various example shapes and visualize results.

    This test creates several pairs of shapes with known relationships:
    1. Identical shapes
    2. Slightly translated shapes
    3. Scaled shapes
    4. Rotated shapes
    5. Heavily distorted shapes
    """
    print("=" * 80)
    print("Testing Shape Similarity Comparison")
    print("=" * 80)

    # Define base reference shape (a square)
    ref_shape = np.array([
        [100, 100],
        [300, 100],
        [300, 300],
        [100, 300]
    ], dtype=np.float32).reshape(4, 1, 2)

    # Create test cases
    test_cases = []

    # Case 1: Identical shape
    identical_shape = ref_shape.copy()
    test_cases.append(("Identical", identical_shape))

    # Case 2: Small translation
    translated_shape = ref_shape.copy() + np.array([20, 15]).reshape(1, 1, 2)
    test_cases.append(("Small Translation", translated_shape))

    # Case 3: Scaled shape (80% size, same center)
    center = ref_shape.reshape(4, 2).mean(axis=0)
    scaled_shape = (ref_shape.reshape(4, 2) - center) * 0.8 + center
    scaled_shape = scaled_shape.reshape(4, 1, 2)
    test_cases.append(("Scaled (80%)", scaled_shape))

    # Case 4: Rotated shape (30 degrees)
    angle = np.radians(30)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_shape = (ref_shape.reshape(4, 2) - center) @ rotation_matrix.T + center
    rotated_shape = rotated_shape.reshape(4, 1, 2)
    test_cases.append(("Rotated (30�)", rotated_shape))

    # Case 5: Stretched shape (horizontally)
    stretched_shape = ref_shape.copy().reshape(4, 2)
    stretched_shape[:, 0] = (stretched_shape[:, 0] - center[0]) * 1.5 + center[0]
    stretched_shape = stretched_shape.reshape(4, 1, 2)
    test_cases.append(("Stretched (1.5x horizontal)", stretched_shape))

    # Case 6: Perspective distortion
    perspective_shape = np.array([
        [100, 100],
        [320, 110],
        [310, 310],
        [90, 290]
    ], dtype=np.float32).reshape(4, 1, 2)
    test_cases.append(("Perspective Distortion", perspective_shape))

    # Case 7: Very different shape
    different_shape = np.array([
        [400, 400],
        [500, 420],
        [480, 500],
        [420, 490]
    ], dtype=np.float32).reshape(4, 1, 2)
    test_cases.append(("Very Different", different_shape))

    # Run comparisons and collect results
    results = []
    for name, shape in test_cases:
        similarity, metrics = compare_warped_shapes(ref_shape, shape, normalize=True)
        results.append((name, shape, similarity, metrics))

        print(f"\n{name}:")
        print(f"  Overall Similarity: {similarity:.4f}")
        print(f"  IoU:               {metrics['iou']:.4f}")
        print(f"  Hausdorff Dist:    {metrics['hausdorff_dist']:.4f} (sim: {metrics['hausdorff_sim']:.4f})")
        print(f"  Centroid Dist:     {metrics['centroid_dist']:.4f} (sim: {metrics['centroid_sim']:.4f})")
        print(f"  Corner Dist:       {metrics['corner_dist']:.4f} (sim: {metrics['corner_sim']:.4f})")
        print(f"  Area Ratio:        {metrics['area_ratio']:.4f}")

    # Visualize all test cases
    n_cases = len(test_cases)
    n_cols = 3
    n_rows = (n_cases + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_cases > 1 else [axes]

    ref_corners = ref_shape.reshape(4, 2)

    for idx, (name, shape, similarity, metrics) in enumerate(results):
        ax = axes[idx]
        test_corners = shape.reshape(4, 2)

        # Plot reference shape in blue
        ref_polygon = Polygon(ref_corners, fill=False, edgecolor='blue',
                              linewidth=2, label='Reference')
        ax.add_patch(ref_polygon)

        # Plot test shape in red
        test_polygon = Polygon(test_corners, fill=False, edgecolor='red',
                               linewidth=2, label='Test')
        ax.add_patch(test_polygon)

        # Plot centroids
        ref_centroid = ref_corners.mean(axis=0)
        test_centroid = test_corners.mean(axis=0)
        ax.plot(ref_centroid[0], ref_centroid[1], 'bo', markersize=8, label='Ref Center')
        ax.plot(test_centroid[0], test_centroid[1], 'ro', markersize=8, label='Test Center')

        # Set limits to show both shapes
        all_corners = np.vstack([ref_corners, test_corners])
        margin = 50
        ax.set_xlim(all_corners[:, 0].min() - margin, all_corners[:, 0].max() + margin)
        ax.set_ylim(all_corners[:, 1].min() - margin, all_corners[:, 1].max() + margin)

        # Add title with similarity score
        ax.set_title(f"{name}\nSimilarity: {similarity:.3f} | IoU: {metrics['iou']:.3f}",
                     fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # Invert y-axis to match image coordinates
        ax.invert_yaxis()

    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('shape_similarity_test_results.png', dpi=150, bbox_inches='tight')
    print(f"\n{'=' * 80}")
    print("Visualization saved to: shape_similarity_test_results.png")
    print(f"{'=' * 80}\n")
    plt.show()


if __name__ == "__main__":
    test_compare_warped_shapes()
