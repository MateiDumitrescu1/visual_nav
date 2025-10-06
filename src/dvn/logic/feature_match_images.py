import numpy as np
from dvn.models.xfeat_.xfeat_methods import XFeatModel
def feature_match_images(img1: np.ndarray, img2: np.ndarray):
    """
    Use XFeat to feature match two images, then run a clustering algorithm over the translation vectors to filter out bad matches.
    
    ### Params:
    - `img1`: First image as a numpy array
    - `img2`: Second image as a numpy array
        
    ### Returns:
        a tuple with:
            - (np.ndarray | None): The output image showing matches and homography. None if homography estimation or warping fails.
            - (float | None): The ratio of inlier matches (inliers / total matches). None if homography estimation fails.
    """
    #TODO add more useful params to the output of this method (maybe the keypoints, etc.)
    
    
#! ------------------- TESTING -------------------

from paths_ import PathLogic, TEST_SET
from dvn.utils.image_utils.plot_images import plot_1_image

def feature_match_images():
    xfeat_model = XFeatModel(top_k=4096)
    
    test_img_sets = PathLogic.get_test_image_sets(TEST_SET.FEATURE_MATCHING_TEST)
    test_imgs = test_img_sets.get("xfeat_example", None)

    if test_imgs is None:
        raise ValueError("No test images found in the 'xfeat_example' test set.")

    img1: np.ndarray = cv2.imread(test_imgs[0]) # type: ignore[assignment]
    img2: np.ndarray = cv2.imread(test_imgs[1]) # type: ignore[assignment]