import os
import re
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


def plot_rotation(demo_run_folder: str, recorded_rotation_offset: float = 90.0) -> None:
    """
    Read the frames from the demo run folder. Also read the rotation.txt file.
    Plot both rotations on a graph.

    Args:
        demo_run_folder: Path to the demo run folder containing frames and rotation.txt
        recorded_rotation_offset: Offset to add to the rotation values from rotation.txt
    """
    demo_path = Path(demo_run_folder)

    # Read rotation.txt file (contains rotation in degrees)
    rotation_file = demo_path / "rotation.txt"
    if not rotation_file.exists():
        raise FileNotFoundError(f"rotation.txt not found in {demo_run_folder}")

    # Parse rotation.txt: frame_number: rotation_degrees
    txt_rotations: Dict[int, float] = {}
    with open(rotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(':')
                if len(parts) == 2:
                    frame_num = int(parts[0].strip())
                    rotation_deg = float(parts[1].strip())
                    # Apply the offset to recorded rotation
                    rotation_deg += recorded_rotation_offset
                    # Normalize to 0-360 range
                    rotation_deg = rotation_deg % 360
                    txt_rotations[frame_num] = rotation_deg

    # Parse frame filenames to extract rotation values
    # Format: frame_XXX_rot_YY.Y@ZZ_matches.png
    frame_pattern = re.compile(r'frame_(\d+)_rot_([\d.]+)@\d+_matches\.png')

    frame_rotations: Dict[int, float] = {}
    for filename in os.listdir(demo_path):
        match = frame_pattern.match(filename)
        if match:
            frame_num = int(match.group(1))
            rotation_deg = float(match.group(2))
            # Normalize to 0-360 range
            rotation_deg = rotation_deg % 360
            frame_rotations[frame_num] = rotation_deg

    # Sort by frame number for plotting
    txt_frames = sorted(txt_rotations.keys())
    txt_values = [txt_rotations[f] for f in txt_frames]

    frame_frames = sorted(frame_rotations.keys())
    frame_values = [frame_rotations[f] for f in frame_frames]

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot both rotation sources
    plt.plot(txt_frames, txt_values, 'b-o', label='rotation.txt', linewidth=2, markersize=6)
    plt.plot(frame_frames, frame_values, 'r-s', label='Frame filenames', linewidth=2, markersize=5, alpha=0.7)

    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Rotation (degrees)', fontsize=12)
    plt.title('Rotation Comparison: rotation.txt vs Frame Filenames', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save the plot
    output_path = demo_path / "rotation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Rotation comparison plot saved to: {output_path}")

    # Also show statistics
    if txt_frames and frame_frames:
        # Find common frames
        common_frames = sorted(set(txt_frames) & set(frame_frames))
        if common_frames:
            differences = [abs(txt_rotations[f] - frame_rotations[f]) for f in common_frames]
            print(f"\nStatistics for {len(common_frames)} common frames:")
            print(f"  Mean difference: {np.mean(differences):.2f} degrees")
            print(f"  Max difference: {np.max(differences):.2f} degrees")
            print(f"  Min difference: {np.min(differences):.2f} degrees")

    plt.show()
    
    
from paths_ import demo_output_dir
def run_plot_rotation():
    folder = demo_output_dir + "/demo0" + "/2025-10-12_02-53-11_v1"
    plot_rotation(folder)
    
if __name__ == "__main__":
    run_plot_rotation()
    print("Done!")