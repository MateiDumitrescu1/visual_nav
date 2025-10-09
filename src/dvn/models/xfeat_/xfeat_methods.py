from __future__ import annotations
#! This file contains methods to use XFeat for feature detection and matching.

from typing import no_type_check, Optional, Callable
import os, torch, cv2
import numpy as np
from functools import cache
from dvn.utils.cv_utils.warp_corners_and_draw_matches import warp_corners_and_draw_matches
from dvn.models.config import FeatureMatchingOutput
from enum import StrEnum
from paths_ import models_dir

#! config
STEER_PERMUTATIONS = [
    torch.arange(64).reshape(4, 16).roll(k, dims=0).reshape(64)
    for k in range(4)
]
min_cossim_DEFAULT = -1
min_cossim_coarse_DEFAULT = -1
#! config

class XFEAT_MODELS(StrEnum):
    DEFAULT_PRETRAINED = 'default_pretrained'
    STEERER_PRETRAINED = 'steerer_pretrained' #~ https://github.com/verlab/accelerated_features/issues/32?utm_source=chatgpt.com



@cache
def get_default_pretrained_xfeat_model(top_k: int = 4096):
    """Returns a singleton XFeat model instance."""
    xfeat = torch.hub.load('verlab/accelerated_features', 
        'XFeat', 
        pretrained = True, 
        top_k = top_k
    )
    return xfeat

@cache
def get_steerer_pretrained_xfeat_model(top_k: int = 4096):
    xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = False, top_k = top_k)
    model_path = os.path.join(models_dir, 'xfeat_perm_steer.pth')
    sd = torch.load(model_path, map_location='cpu')
    for key in list(sd):
        sd['net.' + key] = sd[key]
        del sd[key]
    xfeat.load_state_dict(sd) # pyright: ignore[reportAttributeAccessIssue]
    return xfeat

def get_xfeat_initialization_method(model_name: XFEAT_MODELS) -> Callable:
    xfeat_init_method: dict[XFEAT_MODELS, Callable] = {
        XFEAT_MODELS.DEFAULT_PRETRAINED: get_default_pretrained_xfeat_model,
        XFEAT_MODELS.STEERER_PRETRAINED: get_steerer_pretrained_xfeat_model,
    }
    if model_name not in xfeat_init_method:
        raise ValueError(f"Invalid XFeat model name: {model_name}. Must be one of {[e.value for e in XFEAT_MODELS]}")
    return xfeat_init_method[model_name]

class XFEAT_MATCHING_FUNCTIONS(StrEnum):
    LIGHTERGLUE = 'lighterglue' 
    STEERER = 'steerer'

class XFeatModel:
    """
    This class holds the logic for using XFeat for feature detection and matching.
    - `top_k`: the cap on how many strongest keypoints/features you keep per image
    """
    def __init__(self, top_k: int = 4096, model_name: XFEAT_MODELS = XFEAT_MODELS.DEFAULT_PRETRAINED):
        self.top_k = top_k
        init_method = get_xfeat_initialization_method(model_name)
        try:
            self.xfeat: object = init_method(top_k)
            print(f"Initialition function run completed for XFeat model '{model_name}'")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize XFeat model '{model_name}': {e}")
        
        if self.xfeat is None:
                raise ValueError(f"Model returned from the init method was None, for model name: '{model_name}'")
            
        self.model_name = model_name
        self.steerer_permutations = STEER_PERMUTATIONS
    
    
    def ensure_we_are_using_steerer_model(self) -> None:
        if self.model_name != XFEAT_MODELS.STEERER_PRETRAINED:
            raise ValueError("XFeat model is not set to STEERER_PRETRAINED")

    def prepare_np_array_image_for_xfeat(self, img_src: np.ndarray) -> np.ndarray:
        return np.copy(img_src[..., ::-1])

    def xfeat_detect_and_compute(self, image: np.ndarray, top_k: int = 4096) -> dict:
        """
        Detect and compute features using XFeat.
        Output contains: 'keypoints', 'descriptors', 'scores'.
        ~Look at https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/minimal_example.ipynb 
        for more details about the output dictionary.
        """
        #* Prepare the image for XFeat (using either `prepare_np_array_image_for_xfeat` or `parse_input`)
        #! the detectAndCompute accepts both, but for detectAndComputeDense only parse_input works
        
        # im = self.prepare_np_array_image_for_xfeat(image) 
        im = self.xfeat.parse_input(image) # pyright: ignore[reportAttributeAccessIssue]
        
        #* Detect and compute features
        features = self.xfeat.detectAndCompute(im, top_k=top_k)[0] # pyright: ignore[reportAttributeAccessIssue]
        
        #* put the required image size metadata
        # features.update({'image_size': (im.shape[1], im.shape[0])})    <----- ❗ this was for when we used prepare_np_array_image_for_xfeat
        h, w = int(im.shape[-2]), int(im.shape[-1])          # BCHW -> H, W
        features['image_size'] = (w, h)                    # (W, H)
        return features

    def xfeat_detect_and_compute_DENSE(self, image: np.ndarray, top_k: int = 8000) -> dict:
        """
        """
        #* Prepare the image for XFeat
        # im = self.prepare_np_array_image_for_xfeat(image)
        im = self.xfeat.parse_input(image) # pyright: ignore[reportAttributeAccessIssue]
        #* Detect and compute dense features
        features = self.xfeat.detectAndComputeDense(im, top_k=top_k) # pyright: ignore[reportAttributeAccessIssue]
        return features
    
    def feature_match_xfeat_full(self,
        image0: np.ndarray,
        precomputed_feat0: dict | None,
        #
        image1: np.ndarray,
        precomputed_feat1: dict | None,
        #
        top_k: int = 4096,
    ) -> FeatureMatchingOutput | None:
        """
        Matches features between two images using XFeat.
        ### Params:
            - image1: First image as a numpy array (BGR format).
            - feat1: Precomputed features for the first image (or None to compute).
            ---
            - image2: Second image as a numpy array (BGR format).
            - feat2: Precomputed features for the second image (or None to compute).
            ---
            - top_k: Number of top features to consider (default 4096).
        ### Returns:
        FeatureMatchingOutput object with matched keypoints and features, or None if less than 4 matches are found.
        """
        if precomputed_feat0 is None:
            feat0 = self.xfeat_detect_and_compute(image0, top_k=top_k)
        else: 
            feat0 = precomputed_feat0

        if precomputed_feat1 is None:
            feat1 = self.xfeat_detect_and_compute(image1, top_k=top_k)
        else:
            feat1 = precomputed_feat1

        if feat0 is None or feat1 is None:
            raise ValueError("Feature detection failed for one of the images.")

        mkpts_0: np.ndarray  # shape (N, 2)
        mkpts_1: np.ndarray  # shape (N, 2)

        mkpts_0, mkpts_1, _ = self.xfeat.match_lighterglue(feat0, feat1) # pyright: ignore[reportAttributeAccessIssue]
        
        nr_matches = len(mkpts_0)
        print(f"Number of matches: {nr_matches}")

        # Return FeatureMatchingOutput with all data
        return FeatureMatchingOutput(
            mkpts_0=mkpts_0,
            mkpts_1=mkpts_1,
            feat0=feat0,
            feat1=feat1,
            nr_matches=nr_matches,
        )

    def xfeat_match_sparse_default(self, feat0: dict, feat1: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Does `sparse` matching.
        Matches features between two images using XFeat.
        ### Params:
        - `feat0`: computed features for the first image.
        - `feat1`: computed features for the second image.
        ### Returns:
        - `mkpts_0`: np.ndarray  # shape (N, 2)
        - `mkpts_1`: np.ndarray  # shape (N, 2)
        """
        mkpts_0: np.ndarray  # shape (N, 2)
        mkpts_1: np.ndarray  # shape (N, 2)

        mkpts_0, mkpts_1, _ = self.xfeat.match_lighterglue(feat0, feat1) # pyright: ignore[reportAttributeAccessIssue]
        
        #TODO remove these prints
        nr_matches = len(mkpts_0)
        print(f"Number of matches: {nr_matches}")

        return mkpts_0, mkpts_1

    def xfeat_match_semi_dense_default(self, image0: np.ndarray, image1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Does `semi-dense` matching.
        
        ### Returns:
        - `mkpts_0`: np.ndarray  # shape (N, 2)
        - `mkpts_1`: np.ndarray  # shape (N, 2)
        """
        im0 = self.prepare_np_array_image_for_xfeat(image0)
        im1 = self.prepare_np_array_image_for_xfeat(image1)
        mkpts_0, mkpts_1 = self.xfeat.match_xfeat_star(im0, im1)  # pyright: ignore[reportAttributeAccessIssue]
        return mkpts_0, mkpts_1

    def xfeat_match_sparse_steerer(self, feat0: dict, feat1: dict, min_cossim: float = min_cossim_DEFAULT) -> tuple[np.ndarray, np.ndarray, int]:
        self.ensure_we_are_using_steerer_model()
        idxs0, idxs1 = self.xfeat.match(feat0['descriptors'], feat1['descriptors'], min_cossim=min_cossim) # pyright: ignore[reportAttributeAccessIssue]
        rot1to2 = 0
        for r in range(1, 4):
            new_idxs0, new_idxs1 = self.xfeat.match(feat0['descriptors'][..., STEER_PERMUTATIONS[r]], feat1['descriptors'], min_cossim=min_cossim)  # pyright: ignore[reportAttributeAccessIssue]
            if len(new_idxs0) > len(idxs0):
                idxs0 = new_idxs0
                idxs1 = new_idxs1
                rot1to2 = r

        return feat0['keypoints'][idxs0].cpu().numpy(), feat1['keypoints'][idxs1].cpu().numpy(), rot1to2

    @torch.inference_mode()
    def xfeat_match_semi_dense_steerer(self, feat0: dict, feat1: dict, min_cossim: float = min_cossim_coarse_DEFAULT) -> tuple[np.ndarray, np.ndarray, int]:
        self.ensure_we_are_using_steerer_model()
        rot1to2 = 0
        idxs_list = self.xfeat.batch_match(feat0['descriptors'], feat1['descriptors'], min_cossim=min_cossim) # pyright: ignore[reportAttributeAccessIssue]
        for r in range(1, 4):
            new_idxs_list = self.xfeat.batch_match(feat0['descriptors'][..., STEER_PERMUTATIONS[r]], feat1['descriptors'], min_cossim=min_cossim) # pyright: ignore[reportAttributeAccessIssue]
            if len(new_idxs_list[0][0]) > len(idxs_list[0][0]):
                idxs_list = new_idxs_list
                rot1to2 = r

        feat1['descriptors'] = feat1['descriptors'][..., STEER_PERMUTATIONS[-rot1to2]]  # align to first image for refinement MLP

        #Refine coarse matches
        #this part is harder to batch, currently iterate
        matches = self.xfeat.refine_matches(feat0, feat1, matches=idxs_list, batch_idx=0) # pyright: ignore[reportAttributeAccessIssue]

        return matches[:, :2].cpu().numpy(), matches[:, 2:].cpu().numpy(), rot1to2
    
#! ------------------- TESTING -------------------
from paths_ import PathLogic, TEST_SET
from dvn.utils.image_utils.plot_images import plot_1_image

def test_():
    xfeat_model = XFeatModel(top_k=4096)
    xfeat_model_steerer = XFeatModel(top_k=4096, model_name=XFEAT_MODELS.STEERER_PRETRAINED)
    
    test_img_sets = PathLogic.get_test_image_sets(TEST_SET.FEATURE_MATCHING_TEST)
    test_imgs = test_img_sets.get("xfeat_example", None)

    if test_imgs is None:
        raise ValueError("No test images found in the 'xfeat_example' test set.")

    img1: np.ndarray = cv2.imread(test_imgs[0]) # type: ignore[assignment]
    img2: np.ndarray = cv2.imread(test_imgs[1]) # type: ignore[assignment]
    rot_im1 = np.rot90(img1, k=1, axes=(0, 1)).copy()
    
    if img1 is None or img2 is None:
        raise ValueError("Failed to load one or both test images.")
    
    def feature_match_xfeat_full():
        result = xfeat_model.feature_match_xfeat_full(img1, None, img2, None, top_k=4096)
        if result is None:
            raise ValueError("Feature matching returned None (not enough matches).")

        print(f"Matches: {result.nr_matches}")

        # Visualize the matches using warp_corners_and_draw_matches
        im1 = xfeat_model.prepare_np_array_image_for_xfeat(img1)
        im2 = xfeat_model.prepare_np_array_image_for_xfeat(img2)

        output_canvas = warp_corners_and_draw_matches(
            result.mkpts_0,  # type: ignore
            result.mkpts_1,  # type: ignore
            im1,
            im2,
            draw_match_lines=True
        )

        if output_canvas is None:
            print("Visualization failed")
            return

        # Convert BGR to RGB and plot
        output_rgb = cv2.cvtColor(output_canvas, cv2.COLOR_BGR2RGB)
        plot_1_image(
            image=output_rgb,
            title=f'XFeat Matches: {result.nr_matches} matches',
            tight_layout=True
        )
    
    def test_xfeat_detect_and_compute():
        # Detect and compute features for both images
        feat1 = xfeat_model.xfeat_detect_and_compute(img1, top_k=4096)
        feat2 = xfeat_model.xfeat_detect_and_compute(img2, top_k=4096)

        print(f"Image 1 features detected: {feat1['keypoints'].shape[0]}")
        print(f"Image 2 features detected: {feat2['keypoints'].shape[0]}")

        # Match using precomputed features
        mkpts_0, mkpts_1 = xfeat_model.xfeat_match_sparse_default(feat1, feat2)

        if mkpts_0 is None or mkpts_1 is None:
            raise ValueError("Feature matching returned None (not enough matches).")

        print(f"Matches: {len(mkpts_0)}")

        # Visualize the matches using warp_corners_and_draw_matches
        im1 = xfeat_model.prepare_np_array_image_for_xfeat(img1)
        im2 = xfeat_model.prepare_np_array_image_for_xfeat(img2)

        output_canvas = warp_corners_and_draw_matches(
            mkpts_0,
            mkpts_1, 
            im1,
            im2,
            draw_match_lines=True
        )

        if output_canvas is None:
            print("Visualization failed")
            return

        # Convert BGR to RGB and plot
        output_rgb = cv2.cvtColor(output_canvas, cv2.COLOR_BGR2RGB)
        plot_1_image(
            image=output_rgb,
            title=f'XFeat Detect & Compute Test',
            tight_layout=True
        )
    
    def test_steerer_sparse():
        xfeat_model_steerer.ensure_we_are_using_steerer_model()
        feat1 = xfeat_model_steerer.xfeat_detect_and_compute(rot_im1, top_k=4096) #! use the rotated image
        feat2 = xfeat_model_steerer.xfeat_detect_and_compute(img2, top_k=4096)
        
        mkpts_0, mkpts_1, rot = xfeat_model_steerer.xfeat_match_sparse_steerer(feat1, feat2, min_cossim=0.9)
        print(f"Matches: {len(mkpts_0)}, rotation: {rot}")

        # Visualize the matches using warp_corners_and_draw_matches
        output_canvas = warp_corners_and_draw_matches(
            mkpts_0,
            mkpts_1, 
            rot_im1,
            img2,
            draw_match_lines=True
        )

        if output_canvas is None:
            print("Visualization failed")
            return

        # Convert BGR to RGB and plot
        output_rgb = cv2.cvtColor(output_canvas, cv2.COLOR_BGR2RGB)
        plot_1_image(
            image=output_rgb,
            title=f'XFeat Steerer Sparse Matches: {len(mkpts_0)} matches',
            tight_layout=True
        )
        
    def test_steerer_semi_dense():
        xfeat_model_steerer.ensure_we_are_using_steerer_model()
        feat1 = xfeat_model_steerer.xfeat_detect_and_compute_DENSE(rot_im1) #! use the rotated image
        feat2 = xfeat_model_steerer.xfeat_detect_and_compute_DENSE(img2)
        
        mkpts_0, mkpts_1, rot = xfeat_model_steerer.xfeat_match_semi_dense_steerer(feat1, feat2, min_cossim=0.9)
        print(f"Matches: {len(mkpts_0)}, rotation: {rot}")

        # Visualize the matches using warp_corners_and_draw_matches
        output_canvas = warp_corners_and_draw_matches(
            mkpts_0,
            mkpts_1, 
            rot_im1,
            img2,
            draw_match_lines=True
        )

        if output_canvas is None:
            print("Visualization failed")
            return

        # Convert BGR to RGB and plot
        output_rgb = cv2.cvtColor(output_canvas, cv2.COLOR_BGR2RGB)
        plot_1_image(
            image=output_rgb,
            title=f'XFeat Steerer Semi-Dense Matches: {len(mkpts_0)} matches',
            tight_layout=True
        )
    
    #! run the test methods
    feature_match_xfeat_full()
    test_xfeat_detect_and_compute()
    test_steerer_sparse()
    test_steerer_semi_dense()
    
if __name__ == '__main__':
    test_()
    print("All tests passed!")