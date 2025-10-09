from pathlib import Path
from time import time
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib.pyplot as plt

drone_img_path = "../../../../../data/images/marco_sunny_frames/frame_000236.jpg"
sat_img_path = "../../../../../data/images/sat.png"

def load_model(model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    """Load the depth estimation model and processor."""
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    return processor, model

processor, model = load_model()

def estimate_depth(image: Image.Image, kwargs: dict | None = None) -> torch.Tensor:
    """
    Estimate depth from an input image.

    Args:
        image: PIL Image in RGB format
        model_id: HuggingFace model identifier for depth estimation

    Returns:
        Depth map as a torch.Tensor of shape [H, W] with normalized values
    """
    if kwargs is None:
        kwargs = {}
        
    inputs = processor(images=image, return_tensors="pt", **kwargs)

    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()})

    # Resize prediction back to original size
    post = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )
    depth_map = post[0]["predicted_depth"]  # torch.Tensor [H, W]

    # Normalize to [0, 1] range
    depth_normalized = depth_map - depth_map.min()
    depth_normalized = depth_normalized / (depth_normalized.max() + 1e-8)

    return depth_normalized

def visualize_depth(depth_map: torch.Tensor, original_image: Image.Image, save_path: Path):
    """
    Visualize depth estimation results.

    Args:
        depth_map: Normalized depth tensor [H, W] with values in [0, 1]
        original_image: Optional original image for overlay visualization
        save_path: Optional path to save visualizations
    """
    # Convert to 8-bit for visualization
    depth_8 = (depth_map * 255.0).cpu().numpy().astype("uint8")

    # Create figure with subplots
    if original_image is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Depth map with colormap
        axes[1].imshow(depth_8, cmap="inferno")
        axes[1].set_title("Depth Map")
        axes[1].axis("off")

        # Overlay
        img_np = np.asarray(original_image).astype(np.float32) / 255.0
        cmap = plt.get_cmap("inferno")(depth_8 / 255.0)[..., :3]  # RGB
        overlay = (0.6 * img_np + 0.4 * cmap).clip(0, 1)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(depth_8, cmap="inferno")
        ax.set_title("Depth Map")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
        print(f"Visualization saved to: {save_path}")

    plt.show()

#! ---------------- TESTING ----------------

def test_dav2():
    """Test depth estimation on the test image and visualize results."""
    # Load test image
    test_img_path_to_use = sat_img_path 
    
    image_path = Path(__file__).parent / test_img_path_to_use
    image = Image.open(image_path).convert("RGB")

    print(f"Running depth estimation on: {image_path}")

    start_time = time()
    # Estimate depth
    depth_map = estimate_depth(image, {"do_resize": False, "do_center_crop": False})

    print(f"depth estimation took {time() - start_time:.2f} seconds")
    
    # Visualize results
    output_path = Path(__file__).parent / "test_output.png"
    visualize_depth(depth_map, original_image=image, save_path=output_path)

    print("Test completed successfully!")

if __name__ == "__main__":
    test_dav2()
    print("All tests passed!")