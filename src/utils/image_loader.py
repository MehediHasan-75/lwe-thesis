"""Image loading utilities."""

from typing import List, Tuple
from PIL import Image
import numpy as np


def load_image_as_pixels(image_path: str) -> Tuple[List[int], int, int]:
    """Load image and return pixel values."""
    img = Image.open(image_path).convert('L')
    width, height = img.size
    pixels = np.array(img).flatten().tolist()
    return pixels, width, height


def load_image_array(image_path: str) -> np.ndarray:
    """Load image as numpy array."""
    img = Image.open(image_path).convert('L')
    return np.array(img)


def save_image_from_pixels(pixels: List[int], 
                          width: int, 
                          height: int,
                          output_path: str) -> None:
    """Save pixels as image file."""
    img_array = np.array(pixels).reshape(height, width).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(output_path)


def validate_image_dimensions(pixels: List[int], 
                             width: int, 
                             height: int) -> bool:
    """Validate that dimensions match pixel count."""
    return len(pixels) == width * height
