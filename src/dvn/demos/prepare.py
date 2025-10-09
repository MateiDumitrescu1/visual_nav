from pathlib import Path
import cv2
from paths_ import output_dir, images_dir

original_frames_folder = images_dir + '/marco_sunny_frames/original'

def downsample_and_save_frames(
    source_folder: str,
    downsample_factor: float
) -> str:
    """
    Downsample all frames from source folder and save to a new folder.

    Args:
        source_folder: Path to folder containing original frames
        downsample_factor: Factor to downsample by (e.g., 0.6 for 60% of original size)

    Returns:
        Path to the downsampled frames folder
    """
    source_path = Path(source_folder)

    # Create output folder name with underscore instead of dot
    folder_name = f"downsampled_{str(downsample_factor).replace('.', '_')}"
    output_folder = source_path.parent / folder_name
    output_folder.mkdir(exist_ok=True)

    # Get all image files from source folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in source_path.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"Downsampling {len(image_files)} frames by factor {downsample_factor}")
    print(f"Output folder: {output_folder}")

    for img_file in sorted(image_files):
        # Read original image
        img = cv2.imread(str(img_file))

        if img is None:
            print(f"Warning: Could not read {img_file.name}, skipping...")
            continue

        # Calculate new dimensions
        height, width = img.shape[:2]
        new_width = int(width * downsample_factor)
        new_height = int(height * downsample_factor)

        # Downsample using INTER_AREA (best for downsampling)
        downsampled = cv2.resize(img, (new_width, new_height),
                                interpolation=cv2.INTER_AREA)

        # Save with same filename in output folder
        output_path = output_folder / img_file.name
        cv2.imwrite(str(output_path), downsampled)

    print(f"Completed downsampling {len(image_files)} frames")

    return str(output_folder)

def use_downsampled_frames() -> None:
    downsample_factor = 0.6
    downsample_and_save_frames(original_frames_folder, downsample_factor)
    
if __name__ == "__main__":
    use_downsampled_frames()
    pass