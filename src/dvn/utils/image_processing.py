#! return cv2.GaussianBlur(img, ksize, sigmaX, sigmaY, borderType=borderType)
import numpy as np
import cv2
# pyrefly: ignore  # bad-function-definition
def blur_image(img: np.ndarray,ksize: tuple = (5, 5),sigmaX: float = 0,sigmaY: float = None,borderType: int = cv2.BORDER_DEFAULT) -> np.ndarray:
    
    """
    Fast Gaussian blur via OpenCV.
    ### Args
    - **img** (np.ndarray): The input image.
    - **ksize** (tuple, optional): The size of the kernel to use for blurring. Defaults to (5, 5).
    - **sigmaX** (float, optional): The standard deviation of the Gaussian kernel in the X direction. Defaults to 0.
        - If sigmaX=0 (default), OpenCV derives σ from ksize, so increasing ksize also increases σ.
    - **sigmaY** (float, optional): The standard deviation of the Gaussian kernel in the Y direction. If None, it is set to sigmaX. Defaults to None.
    - **borderType** (int, optional): The type of border to use when blurring the image. Defaults to cv2.BORDER_DEFAULT.
    Returns:
        np.ndarray: The blurred image.
    """
    if sigmaY is None:
        sigmaY = sigmaX
    # pyrefly: ignore  # no-matching-overload
    return cv2.GaussianBlur(img, ksize, sigmaX, sigmaY, borderType=borderType)

# gaussian noise 
def add_gaussian_noise(img: np.ndarray,mean: float = 0.0,std: float = 10.0) -> np.ndarray:
    """
    Fast Gaussian noise injection using OpenCV RNG.

    ### Args
    - img: HxW or HxWxC numpy array, dtype uint8 or float32/64.
    - mean: noise mean (same units as pixel values).
        - If you set mean > 0, you'll brighten pixels on average; mean < 0 darkens them.
    - std: noise standard deviation.
        - Larger std ⇒ more spread ⇒ heavier noise (stronger grain).
        - Smaller std ⇒ subtler noise.
    
    ### Returns
    - Noisy image, clipped to valid range and same dtype as input.
    """
    # prepare noise container
    noise = np.empty_like(img, dtype=np.float32)
    # fill with Gaussian noise
    # pyrefly: ignore  # no-matching-overload
    cv2.randn(noise, mean, std)

    # add & clip
    if np.issubdtype(img.dtype, np.integer):
        dst = img.astype(np.float32) + noise
        dst = np.clip(dst, 0, 255).astype(img.dtype)
    else:
        # assume floats in [0,1] or beyond
        dst = img.astype(np.float32) + noise
        dst = np.clip(dst, 0.0, 1.0).astype(img.dtype)

    return dst


def add_random_black_rectangles(img: np.ndarray,num_rects: int = 5,min_size: tuple[int,int] = (20, 20), 
                                max_size: tuple[int,int] = (100, 100)
) -> np.ndarray:
    """
    Overlay random black rectangles on the image.

    ### Args
    - img:        Input image, HxW or HxWxC, dtype uint8 or float.
    - num_rects:  How many rectangles to draw.
    - min_size:   Minimum (width, height) of each rect.
    - max_size:   Maximum (width, height) of each rect.

    ### Returns
    - A copy of img with num_rects filled-black rectangles.
    """
    out = img.copy()
    h, w = img.shape[:2]

    for _ in range(num_rects):
        # random rectangle size
        rw = np.random.randint(min_size[0], max_size[0] + 1)
        rh = np.random.randint(min_size[1], max_size[1] + 1)

        # random top-left corner (ensure it fits)
        x1 = np.random.randint(0, w - rw + 1)
        y1 = np.random.randint(0, h - rh + 1)
        x2, y2 = x1 + rw, y1 + rh

        # draw filled rectangle (thickness=-1), works in-place
        # if single-channel, cv2 will interpret (0,) correctly
        cv2.rectangle(out, (x1, y1), (x2, y2), color=0, thickness=-1)

    return out
