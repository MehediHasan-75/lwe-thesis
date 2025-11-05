"""Main compression orchestrator."""

import time
from typing import List, Tuple
from PIL import Image
import numpy as np

from src.compression.delta_dpcm import apply_delta_dpcm
from src.compression.rle import apply_rle
from src.compression.huffman import huffman_compress_rle
from src.compression.data_structures import CompressedImageData, ImageCompressionResult


class CompressionManager:
    """Manages complete compression pipeline."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def load_image(image_path: str) -> Tuple[List[int], int, int]:
        """Load image and return pixel data."""
        img = Image.open(image_path).convert('L')
        width, height = img.size
        pixels = np.array(img).flatten().tolist()
        return pixels, width, height
    
    @staticmethod
    def compress_image_complete(pixels: List[int], 
                               width: int, 
                               height: int,
                               image_name: str = "") -> CompressedImageData:
        """Execute complete compression pipeline."""
        if not pixels:
            return None
        
        start_time = time.time()
        
        original_bits = len(pixels) * 8
        
        dpcm_data = apply_delta_dpcm(pixels)
        rle_data = apply_rle(dpcm_data)
        encoded_bitstring, code_map, counts, total_bits = huffman_compress_rle(rle_data)
        
        compression_time = time.time() - start_time
        
        compressed_data = CompressedImageData(
            first_pixel=pixels[0],
            width=width,
            height=height,
            huffman_tree=code_map,
            encoded_bitstring=encoded_bitstring,
            rle_counts=counts,
            compressed_bits=total_bits
        )
        
        return compressed_data
