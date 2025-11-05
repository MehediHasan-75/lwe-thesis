"""Image reconstruction validation and metrics calculation."""

import numpy as np
from typing import List, Tuple, Dict


def validate_reconstruction(original_pixels: List[int], 
                           reconstructed_pixels: List[int]) -> Tuple[bool, str]:
    """Validate that reconstruction is perfect (lossless)."""
    if len(original_pixels) != len(reconstructed_pixels):
        return False, f"Length mismatch: {len(original_pixels)} vs {len(reconstructed_pixels)}"
    
    differences = sum(1 for o, r in zip(original_pixels, reconstructed_pixels) if o != r)
    
    if differences == 0:
        return True, "Perfect reconstruction! ✓"
    else:
        mse = np.mean([(o - r)**2 for o, r in zip(original_pixels, reconstructed_pixels)])
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        return False, f"{differences} pixel errors, MSE={mse:.4f}, PSNR={psnr:.2f}dB"


def calculate_metrics(original: List[int], 
                     reconstructed: List[int]) -> Dict[str, float]:
    """Calculate comprehensive reconstruction metrics."""
    if len(original) != len(reconstructed):
        return None
    
    original = np.array(original)
    reconstructed = np.array(reconstructed)
    
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'mae': float(mae),
        'perfect': mse == 0,
        'max_error': int(np.max(np.abs(original - reconstructed))),
    }


def pixel_wise_comparison(original: List[int], 
                         reconstructed: List[int]) -> Dict:
    """Get detailed pixel-wise comparison."""
    if len(original) != len(reconstructed):
        return None
    
    differences = np.array([abs(o - r) for o, r in zip(original, reconstructed)])
    
    return {
        'perfect_pixels': int((differences == 0).sum()),
        'error_pixels': int((differences != 0).sum()),
        'error_distribution': {
            '0': int((differences == 0).sum()),
            '1-5': int(((differences > 0) & (differences <= 5)).sum()),
            '6-10': int(((differences > 5) & (differences <= 10)).sum()),
            '>10': int((differences > 10).sum()),
        }
    }
