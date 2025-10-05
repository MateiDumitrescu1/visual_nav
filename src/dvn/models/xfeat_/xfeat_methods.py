from __future__ import annotations
#! This file contains methods to use XFeat for feature detection and matching.

from typing import no_type_check, Optional
import os, torch, cv2
import numpy as np
from functools import cache
from dvn.utils.cv_utils.warp_corners_and_draw_matches import warp_corners_and_draw_matches

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
        """
        # Prepare the image for XFeat
        im = self.prepare_np_array_image_for_xfeat(image)
        output = self.xfeat.detectAndCompute(im, top_k=top_k)[0] # pyright: ignore[reportAttributeAccessIssue]
        output.update({'image_size': (im.shape[1], im.shape[0])})    
        return output

    #TODO this needs better types and return types
    def match_xfeat(self,
        image1: np.ndarray,
        feat1: dict | None,
        # 
        image2: np.ndarray,
        feat2: dict | None,
        # 
        top_k: int = 4096,
        draw_match_lines: bool = True # New parameter
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[int], Optional[np.ndarray]]:
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
        a tuple with:
        - **The output image** `(np.ndarray | None)`: 
            - If draw_match_lines is True: A combined image showing img1 and img2
                side-by-side with match lines and the warped polygon.
            - If draw_match_lines is False: img2 with the warped polygon drawn on it.
            - None if less than 4 matches are found.
        - Matched keypoints from the first image `(np.ndarray | None) (shape (N, 2)`. None if less than 4 matches are found.
        - Matched keypoints from the second image `(np.ndarray | None) (shape (N, 2)`. None if less than 4 matches are found.
        - float | None: The ratio of inlier matches (inliers / total matches). None if less than 4 matches are found.
        - int | None: The number of matches found. None if less than 4 matches are found.
        - np.ndarray | None: warped_corners: The warped corner points of img1 in img2's space. None if less than 4 matches are found.
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

        mkpts_0, mkpts_1, _ = self.xfeat.match_lighterglue(output0, output1) # pyright: ignore[reportAttributeAccessIssue]
        nr_matches = len(mkpts_0)
        print(f"Number of matches: {nr_matches}")

        if nr_matches < 4:
            return None,  None, None, None, None, None

        output_canvas,inlier_ratio,warped_corners = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2,draw_match_lines)

        return output_canvas, mkpts_0, mkpts_1, inlier_ratio, nr_matches, warped_corners


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
#TODO implement soem tests for the methods of the XFeatModel class in this file
#TODO from "test_image_pairs", use the "xfeat_example" folder and get the 2 images from there. run tests for all the methods of the XFeatModel class.

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
        from dvn.utils.image_utils.plot_images import plot_1_image

        res = xfeat_model.match_xfeat(img1, None, img2, None, top_k=4096, draw_match_lines=True)
        output_canvas, mkpts_0, mkpts_1, inlier_ratio, nr_matches, warped_corners = res

        if output_canvas is None:
            print("Not enough matches found (< 4)")
            return

        print(f"Matches: {nr_matches}, Inlier ratio: {inlier_ratio:.2%}")

        # Convert BGR to RGB and plot
        output_rgb = cv2.cvtColor(output_canvas, cv2.COLOR_BGR2RGB)
        plot_1_image(
            image=output_rgb,
            title=f'XFeat Matches: {nr_matches} matches, Inlier ratio: {inlier_ratio:.2%}',
            tight_layout=True
        )
    
    def test_xfeat_detect_and_compute():
        raise NotImplementedError()
    
    #! run the test methods
    test_match_xfeat()
    
if __name__ == '__main__':
    test_()
    print("All tests passed!")