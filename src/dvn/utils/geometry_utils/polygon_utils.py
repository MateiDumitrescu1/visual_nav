import numpy as np
import cv2

def expand_polygon(corners: np.ndarray, factor: float) -> np.ndarray:
    """
    Expand a polygon by moving each corner away from the centroid.

    ### Params:
        corners: np.ndarray
            4x2 array of corner points
        factor: float
            Expansion factor (e.g., 0.1 = 10% expansion)

    ### Returns:
        np.ndarray: Expanded polygon corners
    """
    centroid = np.mean(corners, axis=0)
    vectors = corners - centroid
    expanded_corners = centroid + vectors * (1.0 + factor)
    return expanded_corners


def contract_polygon(corners: np.ndarray, factor: float) -> np.ndarray:
    """
    Contract a polygon by moving each corner toward the centroid.

    ### Params:
        corners: np.ndarray
            4x2 array of corner points
        factor: float
            Contraction factor (e.g., 0.1 = 10% contraction)

    ### Returns:
        np.ndarray: Contracted polygon corners
    """
    centroid = np.mean(corners, axis=0)
    vectors = corners - centroid
    contracted_corners = centroid + vectors * (1.0 - factor)
    return contracted_corners


def points_inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Check which points are inside a polygon using OpenCV's pointPolygonTest.

    ### Params:
        points: np.ndarray
            nx2 array of points to test
        polygon: np.ndarray
            mx2 array of polygon vertices

    ### Returns:
        np.ndarray: Boolean array indicating which points are inside the polygon
    """
    # OpenCV's pointPolygonTest requires the polygon as int32
    polygon_int = polygon.astype(np.int32)

    # Test each point
    inside = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        # pointPolygonTest returns: positive (inside), negative (outside), or zero (on edge)
        result = cv2.pointPolygonTest(polygon_int, tuple(point.astype(float)), measureDist=False)
        inside[i] = result >= 0

    return inside