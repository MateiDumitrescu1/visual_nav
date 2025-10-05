#! these should be implemented to be as fast and as optimized as possible
import numpy as np
import math
from typing import Union,Tuple, List
import os
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import subprocess
import shutil
import imageio as imio

def pil_to_numpy(image):
    """
    Convert a PIL image to a numpy array.   
    If the input is already a numpy array, return it directly.
    """
    if isinstance(image, np.ndarray):
        return image
    return np.array(image)

def numpy_to_pil(image_array):
    """
    Convert a numpy array to a PIL image.
    If the input is already a PIL Image, return it directly.
    """
    if isinstance(image_array, Image.Image):
        return image_array
    return Image.fromarray(image_array)


#* load n images from a folder path, into an array
def load_n_images_from_folder(folder_path: str | Path, n: int | None = None) -> List[np.ndarray]:
    """
    Load up to *n* images from *folder_path* as NumPy arrays using imio.imread.

    Args:
        folder_path (str | Path): Folder containing images.
        n (int, optional): Maximum number of images to load. If None, load all.

    Returns:
        List[np.ndarray]: Each array has shape (H, W, C) in RGB order.

    Raises:
        ValueError: If *folder_path* does not exist or is not a directory.
    """
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a valid directory")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if n is not None:
        paths = paths[:n]

    images: list[np.ndarray] = []
    for p in paths:
        img = imio.imread(p)  # ndarray, channels last
        # Convert grayscale or RGBA to RGB for consistency
        if img.ndim == 2:                       # grayscale → RGB
            img = np.repeat(img[..., None], 3, axis=2)
        elif img.shape[-1] == 4:                # drop alpha
            img = img[..., :3]
        images.append(img.astype(np.uint8))

    return images

# save an image to a specific folder
# np array save
def save_image_cv2(img_np: np.ndarray, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    # img_np must be H×W or H×W×C (BGR channel order for color)
    cv2.imwrite(path, img_np)

# PIL save
def save_image_pil(img: Image.Image, out_dir: str, filename: str, format=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    # format can be 'PNG', 'JPEG', etc.; if None, inferred from extension
    img.save(path, format=format)

# downsample or upsample an image
#! small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
#! up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
#! up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#! up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

#* calculate the bounds that are possible to do a rotated crop of without exceeding the image boundaries
ArrayOrImage = Union[np.ndarray, Image.Image]
def calc_crop_center_bounds(
    crop_w_rel: float,
    crop_h_rel: float,
    max_angle_deg: float
) -> Tuple[float, float, float, float]:
    """
    Return (min_cx_rel, max_cx_rel, min_cy_rel, max_cy_rel).

    Any centre (cx, cy) chosen inside those bounds is guaranteed to keep the
    rotated crop fully inside the image for every angle in
    ``[-max_angle_deg, +max_angle_deg]``.

    Parameters
    ----------
    crop_w_rel, crop_h_rel : float
        Crop width / height **as fractions of the image** (0–1).
    max_angle_deg : float
        Maximum absolute rotation, in degrees.

    Notes
    -----
    * Works for rectangles or squares.
    * If the function raises, the crop is simply too big for the requested
      rotation and an image‑sized crop centre range does not exist.
    """
    # -------- input guards ---------------------------------------------------
    if not (0 < crop_w_rel <= 1 and 0 < crop_h_rel <= 1):
        raise ValueError("crop sizes must be in (0, 1]")
    if max_angle_deg < 0:
        raise ValueError("max_angle_deg must be non‑negative")

    # effective angle lives in [0, 90°] – anything larger repeats after 90°
    theta_max = math.radians(min(max_angle_deg, 90.0))

    w, h = crop_w_rel, crop_h_rel

    # ------- helper: max bounding box size over angle -----------------------
    def max_bbox_width():
        """max_{θ∈[0,θ_max]} w·cosθ + h·sinθ"""
        if theta_max >= math.atan2(h, w):
            # interior maximum at atan(h / w)
            theta_star = math.atan2(h, w)
            return w * math.cos(theta_star) + h * math.sin(theta_star)
        else:
            # monotonic over the limited range
            return w * math.cos(theta_max) + h * math.sin(theta_max)

    def max_bbox_height():
        """max_{θ∈[0,θ_max]} w·sinθ + h·cosθ"""
        if theta_max >= math.atan2(w, h):
            theta_star = math.atan2(w, h)
            return w * math.sin(theta_star) + h * math.cos(theta_star)
        else:
            return w * math.sin(theta_max) + h * math.cos(theta_max)

    bbox_w_rel = max_bbox_width()
    bbox_h_rel = max_bbox_height()

    if bbox_w_rel > 1 or bbox_h_rel > 1:
        raise ValueError(
            "Crop is too large to stay inside the image for "
            f"±{max_angle_deg} ° rotation."
        )

    half_bw, half_bh = bbox_w_rel / 2, bbox_h_rel / 2

    min_cx = half_bw
    max_cx = 1.0 - half_bw
    min_cy = half_bh
    max_cy = 1.0 - half_bh

    return min_cx, max_cx, min_cy, max_cy

#* # extract a crop or rotated crop an image, at certain center relative coordonates and angle (for the rotated crop)
def rotated_crop(
    image: ArrayOrImage,
    center_x_rel: float,
    center_y_rel: float,
    width_rel: float,
    height_rel: float,
    angle_deg: float,
    *,
    interpolation: str = "linear"   # "nearest" or "linear"
) -> ArrayOrImage:
    """
    Extract a rotated rectangular crop from *either* a Pillow Image
    or an OpenCV/NumPy array.  The returned object has the same type
    as `image`.

    Parameters
    ----------
    image : PIL.Image.Image or np.ndarray (H×W×C)
        Source image in RGB (Pillow) or BGR/RGB (NumPy).  Alpha is ignored.
    center_x_rel, center_y_rel : float
        Centre of the crop, expressed as fractions of width / height.
    width_rel, height_rel : float
        Size of the crop as fractions of width / height.
    angle_deg : float
        Positive values rotate *counter‑clockwise*.
    interpolation : {"nearest", "linear"}, optional
        Resampling filter for the final resize step.
    """
    if isinstance(image, Image.Image):                       # ─── PIL path ───
        img_w, img_h = image.size
        cx = img_w * center_x_rel
        cy = img_h * center_y_rel
        cw = img_w * width_rel
        ch = img_h * height_rel

        rotated = image.rotate(
            -angle_deg,
            resample=Image.Resampling.NEAREST if interpolation == "nearest"
                                             else Image.Resampling.BILINEAR,
            expand=False,
            center=(cx, cy)
        )

        box = (
            int(round(cx - cw / 2)),
            int(round(cy - ch / 2)),
            int(round(cx + cw / 2)),
            int(round(cy + ch / 2)),
        )
        crop = rotated.crop(box)

        tgt_size = (int(round(cw)), int(round(ch)))
        if crop.size != tgt_size:
            crop = crop.resize(tgt_size,
                Image.Resampling.NEAREST if interpolation == "nearest"
                                         else Image.Resampling.BILINEAR)
        return crop

    # ───────────────────────────── NumPy / OpenCV path ──────────────────────
    elif isinstance(image, np.ndarray):
        if image.ndim != 3:
            raise ValueError("expected HxWxC array")

        h, w = image.shape[:2]
        cx = w * center_x_rel
        cy = h * center_y_rel
        cw = w * width_rel
        ch = h * height_rel

        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rot = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_NEAREST if interpolation == "nearest"
                                    else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        x0 = int(round(cx - cw / 2))
        y0 = int(round(cy - ch / 2))
        x1 = int(round(cx + cw / 2))
        y1 = int(round(cy + ch / 2))
        crop = rot[y0:y1, x0:x1]

        tgt_w, tgt_h = int(round(cw)), int(round(ch))
        if crop.shape[0] != tgt_h or crop.shape[1] != tgt_w:
            crop = cv2.resize(
                crop, (tgt_w, tgt_h),
                interpolation=cv2.INTER_NEAREST if interpolation == "nearest"
                                                else cv2.INTER_LINEAR
            )
        return crop

    else:                                                     # unsupported
        raise TypeError("image must be a PIL.Image or a NumPy ndarray")

#* grayscale an image
def to_grayscale(image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
    """
    Convert a PIL Image or an RGB numpy array to grayscale.
    """
    if isinstance(image, Image.Image):
        return image.convert("L")
    elif isinstance(image, np.ndarray):
        # if HxWx3 RGB array, convert properly
        if image.ndim == 3 and image.shape[2] == 3:
            # use RGB→GRAY (not BGR)
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # already single‐channel or unexpected shape
        return image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

#* generate 90, 180 and 270 degree rotated images from a source image (np.array)
def generate_90_180_270_rotated_images(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
    rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # pyrefly: ignore  # bad-return
    return [rotated_90, rotated_180, rotated_270]


def generate_rotations(image: np.ndarray, angles: List[float]) -> List[np.ndarray]:
    """
    Generate rotated images at specified angles.

    Parameters
    ----------
    image : np.ndarray
        Input image as a NumPy array.
    angles : List[float]
        List of angles in degrees for rotation.

    Returns
    -------
    List[np.ndarray]
        List of rotated images as NumPy arrays.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotated_images: List[np.ndarray] = []

    for angle in angles:
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        # Apply affine warp (output size same as input)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        rotated_images.append(rotated)

    return rotated_images

def rotate_once(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by a specified angle.

    ### Parameters
    - `image` : np.ndarray - Input image as a NumPy array.
    - `angle` : float - Angle in degrees for rotation.

    Returns
    - Rotated image as a NumPy array. (np.ndarray)
    """
    
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated

def put_images_side_by_side(image1: np.ndarray, image2: np.ndarray, pad_color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    Places two images (represented as NumPy arrays) side by side horizontally,
    padding the shorter image to match the height of the taller image by creating
    and concatenating padding arrays.

    Args:
        image1: The first image as a NumPy array.
        image2: The second image as a NumPy array.
        pad_color: The color to use for padding (R, G, B). Defaults to black (0, 0, 0).

    Returns:
        A new NumPy array containing the two images placed side by side with matching heights.
    """
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    num_channels = image1.shape[2] if image1.ndim == 3 else 1 # Handle grayscale or color

    max_height = max(height1, height2)

    # Pad image1 if necessary
    if height1 < max_height:
        pad_height = max_height - height1
        # Create a padding array with the specified color
        pad_shape = (pad_height, width1, num_channels) if num_channels > 1 else (pad_height, width1)
        padding = np.full(pad_shape, pad_color[:num_channels], dtype=image1.dtype)
        image1 = np.concatenate((image1, padding), axis=0) # Concatenate vertically

    # Pad image2 if necessary
    if height2 < max_height:
        pad_height = max_height - height2
        # Create a padding array with the specified color
        pad_shape = (pad_height, width2, num_channels) if num_channels > 1 else (pad_height, width2)
        padding = np.full(pad_shape, pad_color[:num_channels], dtype=image2.dtype)
        image2 = np.concatenate((image2, padding), axis=0) # Concatenate vertically

    # Concatenate the images horizontally
    combined_image = np.concatenate((image1, image2), axis=1)

    return combined_image

    
def downsample_pyramid(img,scales):
    pyramid = [cv2.resize(img, None, fx=s, fy=s,
                        interpolation=cv2.INTER_AREA)
            for s in scales]
    return pyramid
    

def split_video_into_frames_random_fps(video_path, output_path,second_lower: int = 1, second_upper: int = 2):
    """
    Split a video into frames at random intervals between second_lower and second_upper.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to the output directory where frames will be saved.
        second_lower (int): Lower bound for random interval in seconds.
        second_upper (int): Upper bound for random interval in seconds.
    """
    os.makedirs(output_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    current_frame = 0
    # pyrefly: ignore  # bad-assignment
    while current_frame < frame_count:
        
        time_interval = np.random.randint(second_lower, second_upper + 1) # randomly roll a time interval
        frame_number = int(current_frame + time_interval * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # read the frame
        ret, frame = cap.read() 
        
        if not ret:
            break
        
        cv2.imwrite(os.path.join(output_path, f"frame_{frame_number}.jpg"), frame) # save the frame
        
        current_frame += time_interval * fps

    cap.release()

#! ---------------- TESTING ----------------
def test_45_rotate():
    sat_img_path = "./test_images/sat.png"
    img = cv2.imread(sat_img_path)
    angles_45 = [i for i in range(0, 360, 45)]
    # pyrefly: ignore  # bad-argument-type
    rotated_images = generate_rotations(img, angles_45)
    for i, img in enumerate(rotated_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
    
def side_by_side_test():
    cat_img_path = "./test_images/cat1.jpg"
    sat_img_path = "./test_images/sat.png"
    cat_img = cv2.imread(cat_img_path)
    sat_img = cv2.imread(sat_img_path)
    # pyrefly: ignore  # bad-argument-type
    combined_image = put_images_side_by_side(cat_img, sat_img, pad_color=(0, 0, 0))
    plt.imshow(combined_image)
    plt.axis('off')
    plt.show()
def test_downsample_pyramid():
    cat_img_path = "./test_images/cat1.jpg"
    img = cv2.imread(cat_img_path)
    pyramid = downsample_pyramid(img, scales=[0.13, 0.20, 0.33])
    for i, img in enumerate(pyramid):
        print(img.shape)
        plt.subplot(1, len(pyramid), i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
    

if __name__ == "__main__":
    test_45_rotate()
    print("All tests passed!")
