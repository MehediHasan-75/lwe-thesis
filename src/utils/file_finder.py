"""File discovery utilities."""

import glob
import os
from typing import List


def find_all_images(image_dir: str = "image_data") -> List[str]:
    """Recursively find all image files in directory."""
    image_paths = []
    supported_formats = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif')
    
    for fmt in supported_formats:
        image_paths.extend(glob.glob(os.path.join(image_dir, fmt)))
        image_paths.extend(glob.glob(os.path.join(image_dir, '**', fmt), recursive=True))
    
    return sorted(list(set(image_paths)))


def find_images_with_filter(image_dir: str, 
                           min_size: int = 0,
                           max_size: int = float('inf')) -> List[str]:
    """Find images filtered by file size."""
    all_images = find_all_images(image_dir)
    filtered = []
    
    for img_path in all_images:
        size = os.path.getsize(img_path)
        if min_size <= size <= max_size:
            filtered.append(img_path)
    
    return filtered
