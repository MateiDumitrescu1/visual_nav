import numpy as np
import cv2
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def is_shape_degenerated(
    warped_corners: np.ndarray,
    area_threshold: float = 10.0,
    aspect_ratio_threshold: float = 10.0,
    angle_threshold_deg: float = 10.0,
    edge_ratio_threshold: float = 10.0,
    collinearity_threshold: float = 0.1
) -> Tuple[bool, dict]:
    """
    Detect if a shape defined by warped corners is degenerated due to bad homography.

    A degenerated shape has poor proportions and typically indicates a bad homography.
    This function checks multiple degeneration criteria:
    1. Very small area (near-zero or collapsed shape)
    2. Extreme aspect ratio (very elongated/thin shape)
    3. Very acute or very obtuse angles
    4. Extreme edge length ratios (one edge much longer/shorter than others)
    5. Near-collinear points (points nearly on a line)

    Args:
        warped_corners: Warped corners array, shape (4, 1, 2) or (4, 2)
        area_threshold: Minimum acceptable area in pixels (default: 10.0)
        aspect_ratio_threshold: Maximum acceptable aspect ratio of bounding box (default: 10.0)
        angle_threshold_deg: Minimum acceptable corner angle in degrees (default: 10.0)
        edge_ratio_threshold: Maximum acceptable ratio between longest/shortest edges (default: 10.0)
        collinearity_threshold: Maximum acceptable collinearity score (0-1, lower is more collinear) (default: 0.1)

    Returns:
        is_degenerated: Boolean indicating if shape is degenerated
        diagnostics: Dictionary with detailed metrics:
            - 'area': Shape area in pixels
            - 'aspect_ratio': Bounding box aspect ratio
            - 'min_angle_deg': Minimum corner angle in degrees
            - 'max_angle_deg': Maximum corner angle in degrees
            - 'edge_ratio': Ratio of longest to shortest edge
            - 'collinearity_score': Measure of how collinear points are (0-1)
            - 'reasons': List of reasons for degeneration (if any)

    Example:
        >>> corners = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
        >>> is_bad, diagnostics = is_shape_degenerated(corners)
        >>> if is_bad:
        >>>     print(f"Bad homography detected: {diagnostics['reasons']}")
    """
    # Reshape corners to (4, 2) format
    corners = warped_corners.reshape(4, 2).astype(np.float32)

    diagnostics = {}
    reasons = []

    # 1. Check area (collapsed or very small shape)
    area = cv2.contourArea(corners.reshape(-1, 1, 2))
    diagnostics['area'] = float(area)

    if area < area_threshold:
        reasons.append(f"Area too small: {area:.2f} < {area_threshold}")

    # 2. Check aspect ratio (very elongated shape)
    # Compute bounding box
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    width = x_coords.max() - x_coords.min()
    height = y_coords.max() - y_coords.min()

    if min(width, height) > 1e-6:  # Avoid division by zero
        aspect_ratio = max(width, height) / min(width, height)
    else:
        aspect_ratio = float('inf')

    diagnostics['aspect_ratio'] = float(aspect_ratio)

    if aspect_ratio > aspect_ratio_threshold:
        reasons.append(f"Aspect ratio too extreme: {aspect_ratio:.2f} > {aspect_ratio_threshold}")

    # 3. Check corner angles (very acute or very obtuse angles)
    angles = []
    for i in range(4):
        # Get three consecutive points (wrapping around)
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        p3 = corners[(i + 2) % 4]

        # Compute vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Compute angle using dot product
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 > 1e-6 and norm2 > 1e-6:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            # Clamp to valid range for arccos
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)
        else:
            angles.append(0.0)  # Degenerate case

    min_angle = min(angles) if angles else 0.0
    max_angle = max(angles) if angles else 0.0

    diagnostics['min_angle_deg'] = float(min_angle)
    diagnostics['max_angle_deg'] = float(max_angle)

    if min_angle < angle_threshold_deg:
        reasons.append(f"Angle too acute: {min_angle:.2f}° < {angle_threshold_deg}°")

    if max_angle > (180.0 - angle_threshold_deg):
        reasons.append(f"Angle too obtuse: {max_angle:.2f}° > {180.0 - angle_threshold_deg}°")

    # 4. Check edge length ratios (one edge much longer/shorter than others)
    edge_lengths = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        length = np.linalg.norm(p2 - p1)
        edge_lengths.append(length)

    min_edge = min(edge_lengths)
    max_edge = max(edge_lengths)

    if min_edge > 1e-6:
        edge_ratio = max_edge / min_edge
    else:
        edge_ratio = float('inf')

    diagnostics['edge_ratio'] = float(edge_ratio)

    if edge_ratio > edge_ratio_threshold:
        reasons.append(f"Edge length ratio too extreme: {edge_ratio:.2f} > {edge_ratio_threshold}")

    # 5. Check collinearity (points nearly on a line)
    # Use cross product area method: fit a line to all 4 points and measure perpendicular distances
    # Compute covariance matrix for PCA
    centroid = corners.mean(axis=0)
    centered = corners - centroid

    # Compute the spread perpendicular to the major axis
    cov_matrix = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)

    # Ratio of minor to major eigenvalue indicates collinearity
    # If points are collinear, minor eigenvalue ≈ 0
    if eigenvalues[1] > 1e-6:
        collinearity_score = eigenvalues[0] / eigenvalues[1]
    else:
        collinearity_score = 0.0

    diagnostics['collinearity_score'] = float(collinearity_score)

    if collinearity_score < collinearity_threshold:
        reasons.append(f"Points nearly collinear: {collinearity_score:.4f} < {collinearity_threshold}")

    # Store reasons in diagnostics
    diagnostics['reasons'] = reasons

    # Shape is degenerated if any criterion is violated
    is_degenerated = len(reasons) > 0

    return is_degenerated, diagnostics


def test_degeneration_detection():
    """
    Test the is_shape_degenerated function with various good and bad shapes.
    Visualizes the results in a grid layout similar to shape_similarity.py.
    """
    print("=" * 80)
    print("Testing Shape Degeneration Detection")
    print("=" * 80)

    test_cases = []

    # Good shape: Normal square
    good_square = np.array([
        [100, 100],
        [300, 100],
        [300, 300],
        [100, 300]
    ], dtype=np.float32)
    test_cases.append(("Good Square", good_square))

    # Bad shape 1: Very small area (collapsed)
    collapsed = np.array([
        [100, 100],
        [101, 100],
        [101, 101],
        [100, 101]
    ], dtype=np.float32)
    test_cases.append(("Collapsed (tiny area)", collapsed))

    # Bad shape 2: Extreme aspect ratio (very thin)
    thin_shape = np.array([
        [100, 200],
        [500, 200],
        [500, 205],
        [100, 205]
    ], dtype=np.float32)
    test_cases.append(("Thin (extreme aspect)", thin_shape))

    # Bad shape 3: Very acute angle
    acute_angle = np.array([
        [100, 100],
        [300, 100],
        [310, 105],
        [100, 300]
    ], dtype=np.float32)
    test_cases.append(("Acute Angle", acute_angle))

    # Bad shape 4: Nearly collinear points
    collinear = np.array([
        [100, 100],
        [200, 105],
        [300, 110],
        [400, 115]
    ], dtype=np.float32)
    test_cases.append(("Nearly Collinear", collinear))

    # Bad shape 5: Extreme edge ratio
    extreme_edges = np.array([
        [100, 100],
        [500, 100],
        [490, 130],
        [110, 130]
    ], dtype=np.float32)
    test_cases.append(("Extreme Edge Ratio", extreme_edges))

    # Good shape 2: Slightly distorted but acceptable
    acceptable = np.array([
        [100, 100],
        [310, 105],
        [305, 305],
        [95, 300]
    ], dtype=np.float32)
    test_cases.append(("Acceptable Distortion", acceptable))

    # Good quadrilateral 1: Trapezoid with reasonable proportions
    good_trapezoid = np.array([
        [100, 100],
        [350, 100],
        [300, 300],
        [150, 300]
    ], dtype=np.float32)
    test_cases.append(("Good Trapezoid", good_trapezoid))

    # Good quadrilateral 2: Perspective-like quad with reasonable proportions
    good_perspective = np.array([
        [100, 100],
        [400, 120],
        [380, 350],
        [120, 330]
    ], dtype=np.float32)
    test_cases.append(("Good Perspective Quad", good_perspective))

    # Bad quadrilateral 1: Very acute angle quadrilateral
    bad_quad_acute = np.array([
        [100, 200],
        [400, 200],
        [405, 205],
        [100, 300]
    ], dtype=np.float32)
    test_cases.append(("Bad Quad (Acute)", bad_quad_acute))

    # Bad quadrilateral 2: Extreme aspect ratio quadrilateral
    bad_quad_extreme = np.array([
        [100, 250],
        [600, 250],
        [595, 255],
        [105, 255]
    ], dtype=np.float32)
    test_cases.append(("Bad Quad (Extreme)", bad_quad_extreme))

    # Run tests and collect results
    results = []
    print()
    for name, corners in test_cases:
        is_bad, diagnostics = is_shape_degenerated(corners)
        results.append((name, corners, is_bad, diagnostics))

        print(f"{name}:")
        print(f"  Degenerated: {is_bad}")
        print(f"  Area: {diagnostics['area']:.2f} px²")
        print(f"  Aspect Ratio: {diagnostics['aspect_ratio']:.2f}")
        print(f"  Min Angle: {diagnostics['min_angle_deg']:.2f}°")
        print(f"  Max Angle: {diagnostics['max_angle_deg']:.2f}°")
        print(f"  Edge Ratio: {diagnostics['edge_ratio']:.2f}")
        print(f"  Collinearity: {diagnostics['collinearity_score']:.4f}")

        if is_bad:
            print(f"  Reasons: {diagnostics['reasons']}")

        print()

    # Visualize all test cases
    n_cases = len(test_cases)
    n_cols = 3
    n_rows = (n_cases + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_cases > 1 else [axes]

    for idx, (name, corners, is_bad, diagnostics) in enumerate(results):
        ax = axes[idx]

        # Determine color based on degeneration status
        color = 'red' if is_bad else 'green'

        # Plot the shape
        polygon = Polygon(corners, fill=True, facecolor=color, alpha=0.2,
                         edgecolor=color, linewidth=2)
        ax.add_patch(polygon)

        # Plot corners
        ax.plot(corners[:, 0], corners[:, 1], 'o', color=color, markersize=8)

        # Plot centroid
        centroid = corners.mean(axis=0)
        ax.plot(centroid[0], centroid[1], 'D', color='black', markersize=10,
               label='Centroid')

        # Set limits with margin
        margin = 50
        ax.set_xlim(corners[:, 0].min() - margin, corners[:, 0].max() + margin)
        ax.set_ylim(corners[:, 1].min() - margin, corners[:, 1].max() + margin)

        # Create title with key metrics
        status = "DEGENERATED" if is_bad else "OK"
        title = f"{name}\nStatus: {status}"
        if is_bad and diagnostics['reasons']:
            # Show first reason if there are multiple
            first_reason = diagnostics['reasons'][0].split(':')[0]
            title += f"\n({first_reason})"

        ax.set_title(title, fontsize=11, fontweight='bold',
                    color=color)

        # Add text with metrics
        metrics_text = (
            f"Area: {diagnostics['area']:.1f}\n"
            f"Aspect: {diagnostics['aspect_ratio']:.1f}\n"
            f"Min∠: {diagnostics['min_angle_deg']:.1f}°\n"
            f"Edge Ratio: {diagnostics['edge_ratio']:.1f}\n"
            f"Collinear: {diagnostics['collinearity_score']:.3f}"
        )
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Invert y-axis to match image coordinates
        ax.invert_yaxis()

    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('shape_degeneration_test_results.png', dpi=150, bbox_inches='tight')

    print("=" * 80)
    print("Visualization saved to: shape_degeneration_test_results.png")
    print("=" * 80 + "\n")
    plt.show()
    
    
if __name__ == "__main__":
    test_degeneration_detection()