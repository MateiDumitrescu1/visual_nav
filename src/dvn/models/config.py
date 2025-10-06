from __future__ import annotations
from typing import Optional
import numpy as np
from dataclasses import dataclass

class FeatureMatchingOutput:
    """
    Output from feature matching operations.
    This class is used to encapsulate all the useful information returned from feature matching 2 images.

    Attributes:
        output_canvas: The output image with matches visualized.
            - If draw_match_lines is True: A combined image showing img1 and img2
              side-by-side with match lines and the warped polygon.
            - If draw_match_lines is False: img2 with the warped polygon drawn on it.
            - None if less than 4 matches are found.
        mkpts_0: Matched keypoints from the first image (shape (N, 2)). None if less than 4 matches are found.
        mkpts_1: Matched keypoints from the second image (shape (N, 2)). None if less than 4 matches are found.
        inlier_ratio: The ratio of inlier matches (inliers / total matches). None if less than 4 matches are found.
        nr_matches: The number of matches found. None if less than 4 matches are found.
        warped_corners: The warped corner points of img1 in img2's space. None if less than 4 matches are found.
    """
    #TODO understand what the `warped_corners` param does
    output_canvas: Optional[np.ndarray]
    mkpts_0: Optional[np.ndarray]
    mkpts_1: Optional[np.ndarray]
    inlier_ratio: Optional[float]
    nr_matches: Optional[int]
    warped_corners: Optional[np.ndarray]
    

    def __init__(
        self,
        output_canvas: Optional[np.ndarray] = None,
        mkpts_0: Optional[np.ndarray] = None,
        mkpts_1: Optional[np.ndarray] = None,
        inlier_ratio: Optional[float] = None,
        nr_matches: Optional[int] = None,
        warped_corners: Optional[np.ndarray] = None
    ):
        self.output_canvas = output_canvas
        self.mkpts_0 = mkpts_0
        self.mkpts_1 = mkpts_1
        self.inlier_ratio = inlier_ratio
        self.nr_matches = nr_matches
        self.warped_corners = warped_corners