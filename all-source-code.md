# All Source Code Files - Ready to Copy/Paste

## Installation & Quick Start

```bash
# 1. Create project directory
mkdir lwe-thesis && cd lwe-thesis

# 2. Download setup-project.py from previous file
# 3. Run it
python3 setup-project.py

# 4. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# 5. Install dependencies
pip install -r requirements.txt

# 6. Add your images to image_data/
cp /path/to/images/*.png image_data/

# 7. Now copy the following code files to their respective locations
```

---

## COMPLETE SOURCE CODE FOR ALL MODULES

### 1. src/compression/delta_dpcm.py

```python
"""Delta DPCM (Differential Pulse Code Modulation) compression.

Lossless compression using first-order differencing.
Reduces entropy by storing pixel differences instead of absolute values.
"""

import numpy as np
from typing import List


def apply_delta_dpcm(pixels: List[int]) -> List[int]:
    """
    Apply lossless Delta DPCM encoding using modular arithmetic.
    
    Args:
        pixels: List of pixel values (0-255)
    
    Returns:
        List of delta (difference) values
    """
    if not pixels or len(pixels) < 2:
        return pixels
    
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
    """
    Reconstruct original pixels from Delta DPCM encoding.
    
    Args:
        dpcm_data: List of delta values from Delta DPCM
    
    Returns:
        Reconstructed original pixel values
    """
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
```

### 2. src/compression/rle.py

```python
"""Run-Length Encoding (RLE) compression.

Lossless compression for data with repeated values.
Stores (value, count) pairs instead of individual values.
"""

from typing import List, Tuple


def apply_rle(pixels: List[int], max_run: int = 255) -> List[Tuple[int, int]]:
    """
    Apply Run-Length Encoding.
    
    Args:
        pixels: List of pixel or delta values
        max_run: Maximum run length (limited to fit in byte)
    
    Returns:
        List of (value, count) tuples
    """
    if not pixels:
        return []
    
    rle_data = []
    i = 0
    
    while i < len(pixels):
        current_val = pixels[i]
        run_length = 1
        
        # Count consecutive identical values
        while i + run_length < len(pixels) and \
              pixels[i + run_length] == current_val and \
              run_length < max_run:
            run_length += 1
        
        rle_data.append((current_val, run_length))
        i += run_length
    
    return rle_data


def rle_decode(rle_data: List[Tuple[int, int]]) -> List[int]:
    """
    Decode RLE back to original values.
    
    Args:
        rle_data: List of (value, count) tuples
    
    Returns:
        Decoded pixel/delta values
    """
    if not rle_data:
        return []
    
    pixels = []
    for value, count in rle_data:
        pixels.extend([value] * count)
    
    return pixels


def estimate_rle_efficiency(pixels: List[int]) -> float:
    """
    Estimate RLE compression efficiency.
    
    Args:
        pixels: Input values
    
    Returns:
        Compression ratio percentage
    """
    original_size = len(pixels)
    if original_size == 0:
        return 0.0
    
    rle_data = apply_rle(pixels)
    compressed_size = len(rle_data) * 2  # (value, count) pairs
    
    return (original_size - compressed_size) / original_size * 100
```

### 3. src/compression/huffman.py

```python
"""Huffman Coding compression.

Variable-length prefix coding based on symbol frequency.
Optimal for compression when combined with other techniques.
"""

import heapq
from typing import Dict, List, Tuple, Optional


class HuffmanNode:
    """Huffman tree node."""
    
    def __init__(self, char: Optional[int], freq: int):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq


def get_frequency_map(values: List[int]) -> Dict[int, int]:
    """Build frequency map from values."""
    freq_map = {}
    for val in values:
        freq_map[val] = freq_map.get(val, 0) + 1
    return freq_map


def build_huffman_tree(freq_map: Dict[int, int]) -> Optional[HuffmanNode]:
    """Build Huffman tree from frequency map."""
    if not freq_map:
        return None
    
    # Create leaf nodes for each symbol
    heap = [HuffmanNode(char, freq) for char, freq in freq_map.items()]
    heapq.heapify(heap)
    
    # Handle single symbol case
    if len(heap) == 1:
        node = heapq.heappop(heap)
        root = HuffmanNode(None, node.freq)
        root.left = node
        return root
    
    # Build tree bottom-up
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        heapq.heappush(heap, parent)
    
    return heap[0]


def generate_huffman_codes(root: Optional[HuffmanNode], 
                          code: str = '', 
                          codes: Optional[Dict[int, str]] = None) -> Dict[int, str]:
    """Generate Huffman codes from tree."""
    if codes is None:
        codes = {}
    
    if root is None:
        return codes
    
    # Leaf node
    if root.char is not None:
        codes[root.char] = code if code else '0'
        return codes
    
    # Internal node
    if root.left:
        generate_huffman_codes(root.left, code + '0', codes)
    if root.right:
        generate_huffman_codes(root.right, code + '1', codes)
    
    return codes


def huffman_compress_rle(rle_data: List[Tuple[int, int]]) -> Tuple[str, Dict[int, str], List[int], int]:
    """
    Compress RLE data using Huffman coding.
    
    Args:
        rle_data: List of (value, count) tuples from RLE
    
    Returns:
        Tuple of (encoded_bitstring, code_map, counts, total_bits)
    """
    if not rle_data:
        return '', {}, [], 0
    
    # Extract values and counts
    values = [val for val, count in rle_data]
    counts = [count for val, count in rle_data]
    
    # Build Huffman tree
    freq_map = get_frequency_map(values)
    root = build_huffman_tree(freq_map)
    code_map = generate_huffman_codes(root)
    
    # Handle single symbol case
    if len(code_map) == 1:
        single_val = next(iter(code_map))
        code_map[single_val] = '0'
    
    # Encode values
    canonical_overhead = len(code_map) * 8  # Codebook storage
    encoded_values = ''.join(code_map[val] for val in values)
    total_bits = len(encoded_values) + (len(counts) * 8) + canonical_overhead
    
    return encoded_values, code_map, counts, total_bits


def huffman_decompress(encoded_bitstring: str, 
                      code_map: Dict[int, str], 
                      num_values: int) -> List[int]:
    """
    Decompress Huffman encoded bitstring back to values.
    
    Args:
        encoded_bitstring: Binary string from Huffman encoding
        code_map: Huffman code mapping
        num_values: Expected number of values to decode
    
    Returns:
        Decoded values
    """
    if not encoded_bitstring or not code_map:
        return []
    
    # Create reverse mapping: code → value
    reverse_map = {code: value for value, code in code_map.items()}
    
    decoded_values = []
    current_code = ''
    
    for bit in encoded_bitstring:
        current_code += bit
        if current_code in reverse_map:
            decoded_values.append(reverse_map[current_code])
            current_code = ''
            if len(decoded_values) >= num_values:
                break
    
    return decoded_values


def huffman_decompress_rle(encoded_bitstring: str, 
                          code_map: Dict[int, str], 
                          rle_counts: List[int]) -> List[Tuple[int, int]]:
    """
    Decompress Huffman encoded RLE values.
    
    Args:
        encoded_bitstring: Binary string from Huffman
        code_map: Huffman code mapping
        rle_counts: RLE run lengths
    
    Returns:
        Reconstructed RLE data as (value, count) tuples
    """
    values = huffman_decompress(encoded_bitstring, code_map, len(rle_counts))
    rle_data = [(val, count) for val, count in zip(values, rle_counts)]
    return rle_data
```

### 4. src/compression/data_structures.py

```python
"""Data structures for compression/decompression pipeline."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CompressedImageData:
    """Store all data needed for image decompression."""
    first_pixel: int           # First pixel value
    width: int                 # Image width
    height: int                # Image height
    huffman_tree: Dict[int, str]  # Huffman code mapping
    encoded_bitstring: str     # Huffman encoded data
    rle_counts: List[int]      # RLE run lengths
    compressed_bits: int       # Total bits after compression


@dataclass
class ImageCompressionResult:
    """Store compression statistics and metrics."""
    original_bits: int
    compressed_bits: int
    compression_ratio: float
    compression_time: float
    image_name: str


@dataclass
class CompressionMetrics:
    """Detailed compression metrics."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    delta_dpcm_efficiency: float
    rle_efficiency: float
    huffman_efficiency: float
    compression_time: float
    decompression_time: float
    is_perfect: bool
```

### 5. src/compression/compression_manager.py

```python
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
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        width, height = img.size
        pixels = np.array(img).flatten().tolist()
        return pixels, width, height
    
    @staticmethod
    def compress_image_complete(pixels: List[int], 
                               width: int, 
                               height: int,
                               image_name: str = "") -> CompressedImageData:
        """
        Execute complete compression pipeline.
        
        Steps:
        1. Delta DPCM
        2. RLE
        3. Huffman
        """
        if not pixels:
            return None
        
        start_time = time.time()
        
        original_bits = len(pixels) * 8
        
        # Step 1: Delta DPCM
        dpcm_data = apply_delta_dpcm(pixels)
        
        # Step 2: RLE
        rle_data = apply_rle(dpcm_data)
        
        # Step 3: Huffman
        encoded_bitstring, code_map, counts, total_bits = huffman_compress_rle(rle_data)
        
        compression_time = time.time() - start_time
        
        # Create compressed data structure
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
```

### 6. src/decompression/decompression_manager.py

```python
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
        """
        Execute complete decompression pipeline.
        
        Steps:
        1. Huffman decode
        2. RLE decode
        3. Reverse Delta DPCM
        """
        # Step 1: Huffman decode
        values = huffman_decompress(
            compressed_data.encoded_bitstring,
            compressed_data.huffman_tree,
            len(compressed_data.rle_counts)
        )
        
        # Step 2: Reconstruct RLE pairs
        rle_data = [(val, count) for val, count in 
                   zip(values, compressed_data.rle_counts)]
        
        # Step 3: RLE decode
        dpcm_data = rle_decode(rle_data)
        
        # Step 4: Reverse Delta DPCM
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
```

### 7. src/validation/image_validator.py

```python
"""Image reconstruction validation and metrics calculation."""

import numpy as np
from typing import List, Tuple, Dict


def validate_reconstruction(original_pixels: List[int], 
                           reconstructed_pixels: List[int]) -> Tuple[bool, str]:
    """
    Validate that reconstruction is perfect (lossless).
    
    Args:
        original_pixels: Original pixel values
        reconstructed_pixels: Reconstructed pixel values
    
    Returns:
        Tuple of (is_perfect, message)
    """
    # Check length
    if len(original_pixels) != len(reconstructed_pixels):
        return False, f"Length mismatch: {len(original_pixels)} vs {len(reconstructed_pixels)}"
    
    # Check pixel-by-pixel
    differences = sum(1 for o, r in zip(original_pixels, reconstructed_pixels) if o != r)
    
    if differences == 0:
        return True, "Perfect reconstruction! ✓"
    else:
        mse = np.mean([(o - r)**2 for o, r in zip(original_pixels, reconstructed_pixels)])
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        return False, f"{differences} pixel errors, MSE={mse:.4f}, PSNR={psnr:.2f}dB"


def calculate_metrics(original: List[int], 
                     reconstructed: List[int]) -> Dict[str, float]:
    """
    Calculate comprehensive reconstruction metrics.
    
    Args:
        original: Original pixel values
        reconstructed: Reconstructed pixel values
    
    Returns:
        Dictionary of metrics
    """
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
```

### 8. src/utils/image_loader.py

```python
"""Image loading utilities."""

from typing import List, Tuple
from PIL import Image
import numpy as np


def load_image_as_pixels(image_path: str) -> Tuple[List[int], int, int]:
    """
    Load image and return pixel values.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (pixels, width, height)
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
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
```

### 9. src/utils/file_finder.py

```python
"""File discovery utilities."""

import glob
import os
from typing import List


def find_all_images(image_dir: str = "image_data") -> List[str]:
    """
    Recursively find all image files in directory.
    
    Args:
        image_dir: Directory to search
    
    Returns:
        Sorted list of image file paths
    """
    image_paths = []
    supported_formats = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif')
    
    for fmt in supported_formats:
        # Non-recursive
        image_paths.extend(glob.glob(os.path.join(image_dir, fmt)))
        # Recursive in subdirectories
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
```

---

## USAGE EXAMPLE

Create a test script `test_compression.py`:

```python
"""Test compression and decompression."""

import os
from src.utils.image_loader import load_image_as_pixels, save_image_from_pixels
from src.compression.compression_manager import CompressionManager
from src.decompression.decompression_manager import DecompressionManager
from src.validation.image_validator import validate_reconstruction, calculate_metrics


def test_single_image(image_path):
    """Test compression/decompression on single image."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Load image
    pixels, width, height = load_image_as_pixels(image_path)
    print(f"✓ Loaded: {width}×{height} = {len(pixels)} pixels")
    
    # Compress
    compressed_data = CompressionManager.compress_image_complete(
        pixels, width, height, image_path
    )
    print(f"✓ Compressed: {len(pixels)*8} → {compressed_data.compressed_bits} bits")
    print(f"  Ratio: {(compressed_data.compressed_bits / (len(pixels)*8)) * 100:.2f}%")
    
    # Decompress
    reconstructed_pixels = DecompressionManager.decompress_image_complete(compressed_data)
    print(f"✓ Decompressed: {len(reconstructed_pixels)} pixels")
    
    # Validate
    is_perfect, message = validate_reconstruction(pixels, reconstructed_pixels)
    print(f"✓ Validation: {message}")
    
    if is_perfect:
        metrics = calculate_metrics(pixels, reconstructed_pixels)
        print(f"  Perfect reconstruction! MSE=0, PSNR=∞")
    
    # Save reconstructed
    output_name = f"reconstructed_{os.path.basename(image_path)}"
    save_image_from_pixels(reconstructed_pixels, width, height, output_name)
    print(f"✓ Saved: {output_name}")
    
    return is_perfect


if __name__ == "__main__":
    from src.utils.file_finder import find_all_images
    
    images = find_all_images("image_data")
    
    if not images:
        print("❌ No images found in image_data/")
    else:
        print(f"Found {len(images)} images")
        
        perfect_count = 0
        for img_path in images[:3]:  # Test first 3
            if test_single_image(img_path):
                perfect_count += 1
        
        print(f"\n{'='*60}")
        print(f"Results: {perfect_count}/{min(3, len(images))} perfect reconstructions")
        print(f"{'='*60}")
```

Run with:
```bash
python test_compression.py
```

---

## Summary

All files provided above. Each module is:
- ✅ Fully documented
- ✅ Type-hinted
- ✅ Ready to use
- ✅ Tested structure

Copy each to its respective location in your project directory structure.
