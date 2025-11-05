"""Delta DPCM (Differential Pulse Code Modulation) compression.

Lossless compression using first-order differencing.
Reduces entropy by storing pixel differences instead of absolute values.
"""

import numpy as np
from typing import List, Union

def apply_delta_dpcm(pixels: Union[List[int], List]) -> List[int]:
    """
    Apply lossless Delta DPCM encoding using modular arithmetic.
    
    Args:
        pixels: List of pixel values (0-255), can be int or numpy.uint8
    
    Returns:
        List of delta (difference) values
    """
    if not pixels or len(pixels) < 2:
        return pixels
    
    # Convert numpy uint8 to Python int to avoid overflow
    pixels = [int(p) for p in pixels]
    
    dpcm = [pixels[0]]  # Store first pixel as-is
    for i in range(1, len(pixels)):
        # Use modular arithmetic for full range coverage
        diff = (pixels[i] - pixels[i-1]) % 256
        # Convert to signed representation for better compression
        if diff > 127:
            diff = diff - 256
        dpcm.append(diff)
    
    return dpcm



def reverse_delta_dpcm(dpcm_data: List[int]) -> List[int]:
    """Reconstruct original pixels from Delta DPCM encoding."""
    if not dpcm_data:
        return []
    
    pixels = [dpcm_data[0]]  # First pixel is stored as-is
    for i in range(1, len(dpcm_data)):
        # Reconstruct by cumulative sum with modulo
        reconstructed = (pixels[i-1] + dpcm_data[i]) % 256
        pixels.append(reconstructed)
    
    return pixels


def get_delta_dpcm_stats(pixels: List[int]) -> dict:
    """Get statistics about delta values for analysis."""
    if not pixels or len(pixels) < 2:
        return {}
    
    dpcm = apply_delta_dpcm(pixels)
    
    return {
        'min_delta': min(dpcm),
        'max_delta': max(dpcm),
        'mean_delta': float(np.mean(dpcm)),
        'std_delta': float(np.std(dpcm)),
    }
