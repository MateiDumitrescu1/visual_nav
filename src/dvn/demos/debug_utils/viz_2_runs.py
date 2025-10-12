def visualize_2_demo_runs(folder1: str, folder2: str, downsample_factor: float = 0.5):
    """
    Read the frame match results from 2 folders, and visualize them side by side for comparison, in a popup window.
    All images are preloaded and downsampled in memory for instant frame switching with no delay.

    The user can use arrow keys to navigate through the frames:
    - Right arrow: next frame
    - Left arrow: previous frame
    - 'q' or ESC: quit

    ### Params:
        folder1: str
            Path to the first demo run folder (e.g., 'data/output/demo_output/demo0/2025-10-12_02-31-22_v1')
        folder2: str
            Path to the second demo run folder
        downsample_factor: float
            Factor to downsample images for faster display (default 0.5 = 50% of original size)
    """
    import cv2
    import os
    import numpy as np

    # Get sorted list of frame images from both folders
    def get_frame_files(folder: str) -> list[str]:
        """Get sorted list of frame image files from a folder."""
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        files = [f for f in os.listdir(folder)
                if f.startswith('frame_') and f.endswith('.png')]
        return sorted(files)

    files1 = get_frame_files(folder1)
    files2 = get_frame_files(folder2)

    if not files1:
        raise ValueError(f"No frame images found in {folder1}")
    if not files2:
        raise ValueError(f"No frame images found in {folder2}")

    print(f"Folder 1: {len(files1)} frames")
    print(f"Folder 2: {len(files2)} frames")

    # Use the minimum number of frames available in both folders
    max_frames = min(len(files1), len(files2))

    # Preload and downsample all images
    print(f"\nPreloading and downsampling all images (factor: {downsample_factor})...")
    preloaded_canvases = []

    folder1_name = os.path.basename(folder1)
    folder2_name = os.path.basename(folder2)

    for frame_idx in range(max_frames):
        # Load images
        img1_path = os.path.join(folder1, files1[frame_idx])
        img2_path = os.path.join(folder2, files2[frame_idx])

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None:
            print(f"Warning: Could not load {img1_path}")
            preloaded_canvases.append(None)
            continue
        if img2 is None:
            print(f"Warning: Could not load {img2_path}")
            preloaded_canvases.append(None)
            continue

        # Downsample images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        new_w1 = int(w1 * downsample_factor)
        new_h1 = int(h1 * downsample_factor)
        new_w2 = int(w2 * downsample_factor)
        new_h2 = int(h2 * downsample_factor)

        img1 = cv2.resize(img1, (new_w1, new_h1), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (new_w2, new_h2), interpolation=cv2.INTER_AREA)

        # Make both images the same height for side-by-side display
        target_height = max(new_h1, new_h2)

        if new_h1 != target_height:
            aspect_ratio = new_w1 / new_h1
            new_width = int(target_height * aspect_ratio)
            img1 = cv2.resize(img1, (new_width, target_height))
            new_w1 = new_width

        if new_h2 != target_height:
            aspect_ratio = new_w2 / new_h2
            new_width = int(target_height * aspect_ratio)
            img2 = cv2.resize(img2, (new_width, target_height))
            new_w2 = new_width

        # Create side-by-side canvas
        canvas = np.zeros((target_height, new_w1 + new_w2, 3), dtype=np.uint8)
        canvas[:, :new_w1] = img1
        canvas[:, new_w1:new_w1+new_w2] = img2

        # Add labels at the top
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Label for folder 1 (left side)
        text1 = f"{folder1_name} | {files1[frame_idx]}"
        (text_width1, text_height1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
        cv2.rectangle(canvas, (10, 10), (10 + text_width1 + 10, 10 + text_height1 + 10), (0, 0, 0), -1)
        cv2.putText(canvas, text1, (15, 10 + text_height1 + 5), font, font_scale, (255, 255, 255), thickness)

        # Label for folder 2 (right side)
        text2 = f"{folder2_name} | {files2[frame_idx]}"
        (text_width2, text_height2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
        cv2.rectangle(canvas, (new_w1 + 10, 10), (new_w1 + 10 + text_width2 + 10, 10 + text_height2 + 10), (0, 0, 0), -1)
        cv2.putText(canvas, text2, (new_w1 + 15, 10 + text_height2 + 5), font, font_scale, (255, 255, 255), thickness)

        # Add frame counter at the bottom center
        frame_info = f"Frame {frame_idx + 1}/{max_frames}"
        (text_width, text_height), _ = cv2.getTextSize(frame_info, font, font_scale, thickness)
        x_center = (new_w1 + new_w2) // 2 - text_width // 2
        y_bottom = target_height - 20
        cv2.rectangle(canvas, (x_center - 10, y_bottom - text_height - 10),
                     (x_center + text_width + 10, y_bottom + 10), (0, 0, 0), -1)
        cv2.putText(canvas, frame_info, (x_center, y_bottom), font, font_scale, (255, 255, 255), thickness)

        preloaded_canvases.append(canvas)

        # Progress indicator
        if (frame_idx + 1) % 10 == 0 or frame_idx == max_frames - 1:
            print(f"  Loaded {frame_idx + 1}/{max_frames} frames...")

    print(f"✓ All frames preloaded and ready for instant navigation!\n")

    # Create window
    current_frame_idx = 0
    window_name = "Demo Comparison (Left/Right arrows to navigate, Q/ESC to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Main visualization loop
    print(f"Starting visualization...")
    print(f"Controls:")
    print(f"  Right Arrow: Next frame")
    print(f"  Left Arrow: Previous frame")
    print(f"  Q or ESC: Quit\n")

    # Display initial frame
    if preloaded_canvases[current_frame_idx] is not None:
        cv2.imshow(window_name, preloaded_canvases[current_frame_idx])

    while True:
        # Wait for key press (1ms delay for responsive input)
        key = cv2.waitKey(1) & 0xFF

        # Handle key presses
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("Exiting visualization...")
            break
        elif key == 83 or key == 3:  # Right arrow (key codes may vary by system)
            if current_frame_idx < max_frames - 1:
                current_frame_idx += 1
                print(f"Frame {current_frame_idx + 1}/{max_frames}")
                if preloaded_canvases[current_frame_idx] is not None:
                    cv2.imshow(window_name, preloaded_canvases[current_frame_idx])
        elif key == 81 or key == 2:  # Left arrow
            if current_frame_idx > 0:
                current_frame_idx -= 1
                print(f"Frame {current_frame_idx + 1}/{max_frames}")
                if preloaded_canvases[current_frame_idx] is not None:
                    cv2.imshow(window_name, preloaded_canvases[current_frame_idx])

    cv2.destroyAllWindows()
    print("Visualization closed.")
    
from paths_ import demo_output_dir

def run_it():
    folder1 = demo_output_dir + '/demo0/2025-10-10_13-49-10'
    folder2 = demo_output_dir + '/demo0/2025-10-12_02-53-11_v1'
    visualize_2_demo_runs(folder1, folder2)
    
if __name__ == "__main__":
    run_it()