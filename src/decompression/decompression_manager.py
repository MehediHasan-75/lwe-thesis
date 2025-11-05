"""Main decompression orchestrator."""

from typing import List
from PIL import Image
import numpy as np

from src.compression.data_structures import CompressedImageData
from src.compression.delta_dpcm import reverse_delta_dpcm
from src.compression.rle import rle_decode
from src.compression.huffman import huffman_decompress


class DecompressionManager:
    """Manages complete decompression pipeline."""
    
    @staticmethod
    def decompress_image_complete(compressed_data: CompressedImageData) -> List[int]:
        """Execute complete decompression pipeline."""
        values = huffman_decompress(
            compressed_data.encoded_bitstring,
            compressed_data.huffman_tree,
            len(compressed_data.rle_counts)
        )
        
        rle_data = [(val, count) for val, count in 
                   zip(values, compressed_data.rle_counts)]
        
        dpcm_data = rle_decode(rle_data)
        pixels = reverse_delta_dpcm(dpcm_data)
        
        return pixels
    
    @staticmethod
    def save_reconstructed_image(pixels: List[int], 
                                width: int, 
                                height: int,
                                output_path: str) -> None:
        """Save reconstructed pixels as image."""
        reconstructed_img = Image.fromarray(
            np.array(pixels).reshape(height, width).astype(np.uint8)
        )
        reconstructed_img.save(output_path)
