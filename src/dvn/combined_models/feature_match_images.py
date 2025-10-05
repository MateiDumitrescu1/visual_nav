import numpy as np
def feature_match_images(img1: np.ndarray, img2: np.ndarray):
    """
    Use XFeat to feature match two images, then run a clustering alg 
    over the translation vectors to filter out bad matches.
    
    ### Params:
    - `img1`: First image as a numpy array
    - `img2`: Second image as a numpy array
        
    ### Returns:
        a tuple with:
            - (np.ndarray | None): The output image showing matches and homography. None if homography estimation or warping fails.
            - (float | None): The ratio of inlier matches (inliers / total matches). None if homography estimation fails.
    """
    #TODO add more useful params to the output of this method (maybe the keypoints, etc.)