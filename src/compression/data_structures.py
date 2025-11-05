"""Data structures for compression/decompression pipeline."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CompressedImageData:
    """Store all data needed for image decompression."""
    first_pixel: int
    width: int
    height: int
    huffman_tree: Dict[int, str]
    encoded_bitstring: str
    rle_counts: List[int]
    compressed_bits: int


@dataclass
class ImageCompressionResult:
    """Store compression statistics and metrics."""
    original_bits: int
    compressed_bits: int
    compression_ratio: float
    compression_time: float
    image_name: str
