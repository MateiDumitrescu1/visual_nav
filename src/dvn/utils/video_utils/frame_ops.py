import os, shutil, cv2
from pathlib import Path
import subprocess
import numpy as np
from paths_ import data_dir, output_dir
#!
def get_frames_of_video(
    video_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    pattern: str = "frame_%06d.jpg",
    fps: int | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Extract every frame from an .mp4 video—fast—using FFmpeg.

    Parameters
    ----------
    - `video_path`: Path to the input .mp4 file.
    - `output_dir`: Folder where frames are written.
    - `pattern`: FFmpeg‐style filename pattern. Include a numeric directive (e.g. %06d)
        so every frame gets a unique name. Change the extension to .png for lossless output.
    - `fps`: Resample to this many frames per second; None keeps the source FPS.
    - `overwrite`: If the output folder exists, delete it first.
    - `verbose`:  Print the exact FFmpeg command being run.

    Raises
    ------
    FileNotFoundError
        If the input video is missing.
    FileExistsError
        If the output folder already exists and overwrite=False.
    EnvironmentError
        If FFmpeg is not in PATH.
    subprocess.CalledProcessError
        If FFmpeg returns a non-zero exit code.
    """

    video_path = Path(video_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(
                f"Output directory '{out_dir}' already exists – set overwrite=True to replace it."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "FFmpeg executable not found in PATH. Install it from https://ffmpeg.org/download.html"
        )

    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",           # keep output tidy; change to 'info' for full logs
        "-y" if overwrite else "-n",
        "-i", str(video_path),
    ]

    # Optional FPS filter (applied before writing)
    if fps is not None:
        cmd.extend(["-vf", f"fps={fps}"])

    cmd.extend([
        "-vsync", "0",                    # no frame duplication/dropping
        str(out_dir / pattern)            # output filename pattern
    ])

    if verbose:
        print("Executing:", " ".join(cmd))

    subprocess.run(cmd, check=True)
    
    
def stitch_frames_into_video(
    folder_path: str,
    output_folder: str,
    output_file_name: str,
    fps: int = 30,
    codec: str = "mp4v"
) -> None:
    """
    Read all image frames from `folder_path`, pad smaller ones with black to match
    the largest width & height, and write them as a video to
    `output_folder/video_file_name`.

    Args:
        folder_path: directory with per-frame image files.
        output_folder: where to save the resulting video.
        video_file_name: name of the output video file (e.g. "out.mp4").
        fps: frames per second for the output video.
        codec: fourcc codec string (e.g. "mp4v" for .mp4).
    """
    in_dir = Path(folder_path).expanduser().resolve()
    if not in_dir.is_dir():
        raise ValueError(f"{in_dir} is not a valid folder")

    # Gather image paths
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    paths = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in exts)
    print(f"Found {len(paths)} images in {in_dir}")
    if not paths:
        raise ValueError(f"No images found in {in_dir}")

    # Read all images once and determine max dimensions
    images = []
    max_w = max_h = 0
    ch = None
    for p in paths:
        print(f"Reading {p.name}...")
        img = cv2.imread(str(p))
        if img is None:
            continue
        images.append(img)
        h, w = img.shape[:2]
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        ch = img.shape[2] if img.ndim == 3 else 1

    if not images:
        raise ValueError(f"No valid images could be loaded from {in_dir}")

    # Prepare output
    out_dir = Path(output_folder).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_file_name

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (max_w, max_h))

    # Process pre-loaded images
    for idx, img in enumerate(images):
        print(f"Processing frame {idx + 1}/{len(images)}...")
        h, w = img.shape[:2]
        # create black canvas
        if ch == 1:
            canvas = np.zeros((max_h, max_w), dtype=img.dtype)
        else:
            # pyrefly: ignore  # no-matching-overload
            canvas = np.zeros((max_h, max_w, ch), dtype=img.dtype)
        # place image at top-left
        canvas[0:h, 0:w] = img
        # OpenCV VideoWriter expects BGR color & 3‑channel
        if ch == 1:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        writer.write(canvas)

    writer.release()


#! ------------------ EXECUTION ------------------
def compile_demo_video():
    """
    Read all images from the demo output directory, stitch them into a video,
    and save the video in the local directory.
    """
    images_folder: str = output_dir + "/demo_output" + "/demo0" + "/2025-10-10_11-02-23"

    # Define output video name and save location (current working directory)
    output_video_name = "demo_video.mp4"
    output_folder = "."  # Current directory

    # Stitch frames into video using the existing function
    stitch_frames_into_video(
        folder_path=images_folder,
        output_folder=output_folder,
        output_file_name=output_video_name,
        fps=10,  # Adjust fps as needed
        codec="mp4v"
    )

    print(f"Video saved to: {Path(output_folder).resolve() / output_video_name}")


#! ------------------ TESTING ------------------

from paths_ import output_dir
def test_get_frames_of_video():
    video_path = "../../../../data/videos/marco_video_sunny.MP4"
    out_dir = f"{output_dir}/test_video_frames"
    get_frames_of_video(video_path, out_dir, fps=5, overwrite=True, verbose=True)

if __name__ == "__main__":
    # test_get_frames_of_video()
    compile_demo_video()
    print("Completed!")