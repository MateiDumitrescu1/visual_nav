
import numpy as np
import cv2

def add_text_to_image(image: np.ndarray, text: str, color_name: str = "white",
                        position: tuple[int, int] = (10, 30),
                        font_scale: float = 1.0, thickness: int = 2) -> np.ndarray:
    """
    Adds text to an image using OpenCV.

    Note: This function modifies the input image array in-place.

    Args:
        image (np.ndarray): The input image (BGR format expected by OpenCV).
        text (str): The text string to add.
        color_name (str): The desired color name (e.g., "red", "green", "blue",
                            "white", "black", "yellow", "cyan", "magenta").
                            Case-insensitive. Defaults to "white".
        position (tuple[int, int]): The (x, y) coordinates for the bottom-left
                                    corner of the text string in the image.
                                    Defaults to (10, 30) (near top-left).
        font_scale (float): Font size multiplier. Defaults to 1.0.
        thickness (int): Thickness of the text lines. Defaults to 2.

    Returns:
        np.ndarray: The image with the text added (the same array passed as input).

    Raises:
        TypeError: If the input image is not a NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy ndarray.")

    # Define common colors in BGR format (OpenCV default)
    color_map = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
    }

    # Get the BGR color tuple, default to white if color_name is not found
    text_color = color_map.get(color_name.lower(), (255, 255, 255))

    # Choose a standard font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Add the text to the image (modifies the image in-place)
    # Use cv2.LINE_AA for anti-aliased text for better appearance
    cv2.putText(image, text, position, font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Return the modified image array
    return image