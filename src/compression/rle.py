"""Run-Length Encoding (RLE) compression.

Lossless compression for data with repeated values.
Stores (value, count) pairs instead of individual values.
"""

from typing import List, Tuple


def apply_rle(pixels: List[int], max_run: int = 255) -> List[Tuple[int, int]]:
    """Apply Run-Length Encoding."""
    if not pixels:
        return []
    
    rle_data = []
    i = 0
    
    while i < len(pixels):
        current_val = pixels[i]
        run_length = 1
        
        while i + run_length < len(pixels) and \
              pixels[i + run_length] == current_val and \
              run_length < max_run:
            run_length += 1
        
        rle_data.append((current_val, run_length))
        i += run_length
    
    return rle_data


def rle_decode(rle_data: List[Tuple[int, int]]) -> List[int]:
    """Decode RLE back to original values."""
    if not rle_data:
        return []
    
    pixels = []
    for value, count in rle_data:
        pixels.extend([value] * count)
    
    return pixels


def estimate_rle_efficiency(pixels: List[int]) -> float:
    """Estimate RLE compression efficiency."""
    original_size = len(pixels)
    if original_size == 0:
        return 0.0
    
    rle_data = apply_rle(pixels)
    compressed_size = len(rle_data) * 2
    
    return (original_size - compressed_size) / original_size * 100
