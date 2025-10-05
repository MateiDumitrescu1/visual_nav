from __future__ import annotations
#! This file contains methods to use XFeat for feature detection and matching.

from typing import no_type_check, Optional
import os, torch, cv2
import numpy as np
from functools import cache

@cache
def load_xfeat_model(top_k_frames: int = 4096):
    xfeat = torch.hub.load('verlab/accelerated_features', 
        'XFeat', 
        pretrained = True, 
        top_k = top_k_frames
    )
    return xfeat

@no_type_check
def xfeat_detect_and_compute(image: np.ndarray, top_k: int = 4096) -> dict:
    """
    Detect and compute features using XFeat.
    """
    # Prepare the image for XFeat
    im = prepare_np_array_image_for_xfeat(image)
    # pyrefly: ignore  # missing-attribute
    output = xfeat.detectAndCompute(im, top_k=top_k)[0]
    output.update({'image_size': (im.shape[1], im.shape[0])})    
    return output

def match_xfeat(
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

    im1 = prepare_np_array_image_for_xfeat(image1)
    output0 = feat1
    if feat1 is None:
        # pyrefly: ignore  # missing-attribute
        output0 = xfeat.detectAndCompute(im1, top_k=top_k)[0]

    
    im2 = prepare_np_array_image_for_xfeat(image2)
    output1 = feat2
    if feat2 is None:
        # pyrefly: ignore  # missing-attribute
        output1 = xfeat.detectAndCompute(im2, top_k=top_k)[0]

    output0.update({'image_size': (im1.shape[1], im1.shape[0])})
    output1.update({'image_size': (im2.shape[1], im2.shape[0])})

    # pyrefly: ignore  # missing-attribute
    mkpts_0, mkpts_1, _ = xfeat.match_lighterglue(output0, output1)
    nr_matches = len(mkpts_0)
    print(f"Number of matches: {nr_matches}")

    if nr_matches < 4:
        # pyrefly: ignore  # bad-return
        return None,None,None,None,None,None

    # pyrefly: ignore  # bad-unpacking
    canvas,inlier_ratio,warped_corners = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2,draw_match_lines)

    # pyrefly: ignore  # bad-return
    return canvas,mkpts_0,mkpts_1,inlier_ratio,nr_matches,warped_corners

def prepare_np_array_image_for_xfeat(img_src: np.ndarray) -> np.ndarray:
    return np.copy(img_src[..., ::-1])


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