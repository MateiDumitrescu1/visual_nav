from __future__ import annotations
#! This file contains methods to use XFeat for feature detection and matching.

from typing import no_type_check, Optional
import os, torch, cv2
import numpy as np
from functools import cache
from dvn.utils.cv_utils.warp_corners_and_draw_matches import warp_corners_and_draw_matches
from dvn.models.config import FeatureMatchingOutput

@cache
def get_xfeat_model(top_k: int = 4096):
    """Returns a singleton XFeat model instance."""
    xfeat = torch.hub.load('verlab/accelerated_features', 
        'XFeat', 
        pretrained = True, 
        top_k = top_k
    )
    return xfeat


class XFeatModel:
    """
    This class holds the logic for using XFeat for feature detection and matching.
    - `top_k`: the cap on how many strongest keypoints/features you keep per image
    """
    def __init__(self, top_k: int = 4096):
        self.top_k = top_k
        self.xfeat = get_xfeat_model(top_k)
    
    
    def prepare_np_array_image_for_xfeat(self, img_src: np.ndarray) -> np.ndarray:
        return np.copy(img_src[..., ::-1])

    def xfeat_detect_and_compute(self, image: np.ndarray, top_k: int = 4096) -> dict:
        """
        Detect and compute features using XFeat.
        Output contains: 'keypoints', 'descriptors', 'scores'.
        ~Look at https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/minimal_example.ipynb 
        for more details about the output dictionary.
        """
        # Prepare the image for XFeat
        im = self.prepare_np_array_image_for_xfeat(image)
        output = self.xfeat.detectAndCompute(im, top_k=top_k)[0] # pyright: ignore[reportAttributeAccessIssue]
        output.update({'image_size': (im.shape[1], im.shape[0])})    
        return output

    def match_xfeat(self,
        image1: np.ndarray,
        feat1: dict | None,
        #
        image2: np.ndarray,
        feat2: dict | None,
        #
        top_k: int = 4096,
        draw_match_lines: bool = True # New parameter
    ) -> FeatureMatchingOutput | None:
        """
        Accepts either file paths or pre-loaded BGR arrays.
        ### Params:
            - image1: First image as a numpy array (BGR format).
            - feat1: Precomputed features for the first image (or None to compute).
            ---
            - image2: Second image as a numpy array (BGR format).
            - feat2: Precomputed features for the second image (or None to compute).
            ---
            - top_k: Number of top features to consider (default 4096).
            - draw_match_lines: If True, draws lines between matched keypoints in the output.
        ### Returns:
        FeatureMatchingOutput object.
        """
        # prepare images

        im1 = self.prepare_np_array_image_for_xfeat(image1)
        output0 = feat1
        if feat1 is None:
            output0 = self.xfeat.detectAndCompute(im1, top_k=top_k)[0] # pyright: ignore[reportAttributeAccessIssue]

        im2 = self.prepare_np_array_image_for_xfeat(image2)
        output1 = feat2
        if feat2 is None:
            output1 = self.xfeat.detectAndCompute(im2, top_k=top_k)[0] # pyright: ignore[reportAttributeAccessIssue]

        if output0 is None or output1 is None:
            raise ValueError("Feature detection failed for one of the images.")

        output0.update({'image_size': (im1.shape[1], im1.shape[0])})
        output1.update({'image_size': (im2.shape[1], im2.shape[0])})

        mkpts_0: np.ndarray  # shape (N, 2)
        mkpts_1: np.ndarray  # shape (N, 2)
        
        mkpts_0, mkpts_1, _ = self.xfeat.match_lighterglue(output0, output1) # pyright: ignore[reportAttributeAccessIssue]
        nr_matches = len(mkpts_0)
        print(f"Number of matches: {nr_matches}")

        if nr_matches < 4:
            return None
        
        result = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2, draw_match_lines)

        return result

def save_mkpts_to_file(mkpts, output_folder, filename):
    """
    Save matched keypoints to a text file. Each line contains 'x y' coordinates.
    mkpts: iterable of (x, y) pairs (e.g. numpy array shape (N,2))
    output_folder: directory to write the file into
    filename: name of the file ('.txt' will be appended if missing)
    """
    # ensure .txt extension
    if not filename.lower().endswith('.txt'):
        filename = filename + '.txt'

    # create folder if missing
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, filename)

    try:
        with open(file_path, 'w') as f:
            for pt in mkpts:
                # format with 6 decimal places
                f.write(f"{pt[0]:.6f} {pt[1]:.6f}\n")
        print(f"Saved keypoints to {file_path}")
    except Exception as e:
        print(f"Error saving keypoints to file {file_path}: {e}")
        
        
#! ------------------- TESTING -------------------
from paths_ import PathLogic, TEST_SET
from dvn.utils.image_utils.plot_images import plot_1_image

def test_():
    xfeat_model = XFeatModel(top_k=4096)
    test_img_sets = PathLogic.get_test_image_sets(TEST_SET.FEATURE_MATCHING_TEST)
    test_imgs = test_img_sets.get("xfeat_example", None)

    if test_imgs is None:
        raise ValueError("No test images found in the 'xfeat_example' test set.")

    img1: np.ndarray = cv2.imread(test_imgs[0]) # type: ignore[assignment]
    img2: np.ndarray = cv2.imread(test_imgs[1]) # type: ignore[assignment]

    if img1 is None or img2 is None:
        raise ValueError("Failed to load one or both test images.")
    
    def test_match_xfeat():
        result = xfeat_model.match_xfeat(img1, None, img2, None, top_k=4096, draw_match_lines=True)
        if result is None:
            raise ValueError("Feature matching returned None (not enough matches).")
        
        if result.output_canvas is None:
            print("Not enough matches found (< 4)")
            return

        print(f"Matches: {result.nr_matches}, Inlier ratio: {result.inlier_ratio:.2%}")

        # Convert BGR to RGB and plot
        output_rgb = cv2.cvtColor(result.output_canvas, cv2.COLOR_BGR2RGB)
        plot_1_image(
            image=output_rgb,
            title=f'XFeat Matches: {result.nr_matches} matches, Inlier ratio: {result.inlier_ratio:.2%}',
            tight_layout=True
        )
    
    def test_xfeat_detect_and_compute():
        # Detect and compute features for both images
        feat1 = xfeat_model.xfeat_detect_and_compute(img1, top_k=4096)
        feat2 = xfeat_model.xfeat_detect_and_compute(img2, top_k=4096)

        print(f"Image 1 features detected: {feat1['keypoints'].shape[0]}")
        print(f"Image 2 features detected: {feat2['keypoints'].shape[0]}")

        # Match using precomputed features
        result = xfeat_model.match_xfeat(img1, feat1, img2, feat2, top_k=4096, draw_match_lines=True)
        
        if result is None:
            raise ValueError("Feature matching returned None (not enough matches).")
        
        if result.output_canvas is None:
            print("Not enough matches found (< 4)")
            return

        print(f"Matches: {result.nr_matches}, Inlier ratio: {result.inlier_ratio:.2%}")

        # Convert BGR to RGB and plot
        output_rgb = cv2.cvtColor(result.output_canvas, cv2.COLOR_BGR2RGB)
        plot_1_image(
            image=output_rgb,
            title=f'XFeat Detect & Compute Test: {result.nr_matches} matches, Inlier ratio: {result.inlier_ratio:.2%}',
            tight_layout=True
        )
    
    #! run the test methods
    # test_match_xfeat()
    test_xfeat_detect_and_compute()
    
if __name__ == '__main__':
    test_()
    print("All tests passed!")