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
        inlier_ratio: The ratio of inlier (aka good) matches (inliers / total matches). None if less than 4 matches are found.
            The bad/incorrect matches are called "outlier" matches. **A more in depth explanation of what "good" and "bad" matches** are
            - Inliers = matches that agree with the estimated geometry → typically correct (good) correspondences.
            - Outliers = matches that don't agree → usually incorrect (bad) correspondences.
        nr_matches: The number of matches found. None if less than 4 matches are found.
        warped_corners: The warped corner points of img1 in img2's space. None if less than 4 matches are found.
        feat0: Features from the first image (dictionary containing 'keypoints', 'descriptors', 'scores'). None if not computed.
        feat1: Features from the second image (dictionary containing 'keypoints', 'descriptors', 'scores'). None if not computed.
    """
    #TODO understand what the `warped_corners` param does
    output_canvas: Optional[np.ndarray]
    mkpts_0: Optional[np.ndarray]
    mkpts_1: Optional[np.ndarray]
    inlier_ratio: Optional[float]
    nr_matches: Optional[int]
    warped_corners: Optional[np.ndarray]
    feat0: Optional[dict]
    feat1: Optional[dict]

    def __init__(
        self,
        output_canvas: Optional[np.ndarray] = None,
        #
        mkpts_0: Optional[np.ndarray] = None,
        mkpts_1: Optional[np.ndarray] = None,
        #
        feat0: Optional[dict] = None,
        feat1: Optional[dict] = None,
        #
        inlier_ratio: Optional[float] = None,
        nr_matches: Optional[int] = None,
        #
        warped_corners: Optional[np.ndarray] = None,

    ):
        self.output_canvas = output_canvas
        self.mkpts_0 = mkpts_0
        self.mkpts_1 = mkpts_1
        self.inlier_ratio = inlier_ratio
        self.nr_matches = self._infer_nr_matches(nr_matches, mkpts_0)
        self.warped_corners = warped_corners
        self.feat0 = feat0
        self.feat1 = feat1

    @staticmethod
    def _infer_nr_matches(nr_matches: Optional[int], mkpts_0: Optional[np.ndarray]) -> Optional[int]:
        """
        Infers the number of matches from the provided value or the length of mkpts_0.

        Args:
            nr_matches: Explicit number of matches, if provided.
            mkpts_0: Matched keypoints array to infer the count from.

        Returns:
            The number of matches, or None if it cannot be determined.
        """
        if nr_matches is not None:
            return nr_matches

        if mkpts_0 is not None:
            return len(mkpts_0)

        return None