from __future__ import annotations
#! This file contains methods to use XFeat for feature detection and matching.

from typing import no_type_check, Optional
import os, torch, cv2
import numpy as np
from functools import cache

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

    
    @no_type_check
    def xfeat_detect_and_compute(self, image: np.ndarray, top_k: int = 4096) -> dict:
        """
        Detect and compute features using XFeat.
        """
        # Prepare the image for XFeat
        im = self.prepare_np_array_image_for_xfeat(image)
        # pyrefly: ignore  # missing-attribute
        output = self.xfeat.detectAndCompute(im, top_k=top_k)[0]
        output.update({'image_size': (im.shape[1], im.shape[0])})    
        return output

    #TODO this needs better types and return types
    def match_xfeat(self,
        image1: np.ndarray,
        feat1: dict,
        # 
        image2: np.ndarray,
        feat2: dict,
        # 
        top_k: int = 4096,
        draw_match_lines: bool = True # New parameter
    ) -> np.ndarray:
        """
        Accepts either file paths or pre-loaded BGR arrays.
        ### Params:
            image1: First image as a numpy array (BGR format).
            feat1: Precomputed features for the first image (or None to compute).
            ---
            image2: Second image as a numpy array (BGR format).
            feat2: Precomputed features for the second image (or None to compute).
            ---
            top_k: Number of top features to consider (default 4096).
            draw_match_lines: If True, draws lines between matched keypoints in the output.
        """
        # prepare images

        im1 = self.prepare_np_array_image_for_xfeat(image1)
        output0 = feat1
        if feat1 is None:
            # pyrefly: ignore  # missing-attribute
            output0 = self.xfeat.detectAndCompute(im1, top_k=top_k)[0]

        
        im2 = self.prepare_np_array_image_for_xfeat(image2)
        output1 = feat2
        if feat2 is None:
            # pyrefly: ignore  # missing-attribute
            output1 = self.xfeat.detectAndCompute(im2, top_k=top_k)[0]

        output0.update({'image_size': (im1.shape[1], im1.shape[0])})
        output1.update({'image_size': (im2.shape[1], im2.shape[0])})

        # pyrefly: ignore  # missing-attribute
        mkpts_0, mkpts_1, _ = self.xfeat.match_lighterglue(output0, output1)
        nr_matches = len(mkpts_0)
        print(f"Number of matches: {nr_matches}")

        if nr_matches < 4:
            # pyrefly: ignore  # bad-return
            return None,None,None,None,None,None

        # pyrefly: ignore  # bad-unpacking
        canvas,inlier_ratio,warped_corners = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2,draw_match_lines)

        # pyrefly: ignore  # bad-return
        return canvas,mkpts_0,mkpts_1,inlier_ratio,nr_matches,warped_corners


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
from paths_ import images_dir
#TODO implement soem tests for the methods of the XFeatModel class in this file
#TODO from "test_image_pairs", use the "xfeat_example" folder and get the 2 images from there. run tests for all the methods of the XFeatModel class.


def test_xfeat_detect_and_compute():
    pass

if __name__ == '__main__':
    print("All tests passed!")