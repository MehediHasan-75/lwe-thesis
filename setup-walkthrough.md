# Complete Setup Walkthrough for LWE Cryptography Thesis Project

## Step-by-Step Setup Guide

### Method 1: Automated Setup (Recommended)

#### Step 1: Download Setup Script

Save the `setup-project.py` file to your thesis directory.

#### Step 2: Run Setup Script

```bash
# Navigate to your project root
cd /path/to/thesis

# Run the setup script
python3 setup-project.py
```

This will automatically:
- ✅ Create all directories
- ✅ Create `__init__.py` files
- ✅ Generate `requirements.txt`
- ✅ Create `README.md`
- ✅ Create `.gitignore`
- ✅ Create configuration files
- ✅ Create sample module files

#### Step 3: Create Virtual Environment

```bash
python3 -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

#### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 5: Add Your Images

```bash
# Copy images to the image_data directory
cp /path/to/your/images/*.png image_data/
cp /path/to/your/images/*.jpg image_data/
```

---

### Method 2: Manual Setup

If you prefer manual setup:

#### Step 1: Create Main Directory

```bash
mkdir lwe-cryptography-thesis
cd lwe-cryptography-thesis
```

#### Step 2: Create Directory Structure

```bash
# Core directories
mkdir -p image_data
mkdir -p src/{compression,decompression,validation,cryptography,utils}
mkdir -p benchmarks tests notebooks
mkdir -p outputs/{reconstructed_images,plots,reports,compressed_data}
mkdir -p thesis/sections thesis/figures
mkdir -p docs scripts config
```

#### Step 3: Create __init__.py Files

```bash
touch src/__init__.py
touch src/compression/__init__.py
touch src/decompression/__init__.py
touch src/validation/__init__.py
touch src/cryptography/__init__.py
touch src/utils/__init__.py
touch benchmarks/__init__.py
touch tests/__init__.py
```

#### Step 4: Create Configuration Files

Create `requirements.txt`:
```
numpy==1.24.3
pillow==10.0.0
matplotlib==3.7.2
pandas==2.0.3
scipy==1.11.1
dataclasses-json==0.5.14
pyyaml==6.0
pytest==7.4.0
jupyter==1.0.0
```

#### Step 5: Setup Virtual Environment & Install

```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Project Directory Tree After Setup

```
lwe-cryptography-thesis/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── image_data/                          # ADD YOUR IMAGES HERE
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── delta_dpcm.py               # (Create these files)
│   │   ├── rle.py
│   │   ├── huffman.py
│   │   ├── data_structures.py
│   │   └── compression_manager.py
│   │
│   ├── decompression/
│   │   ├── __init__.py
│   │   ├── reverse_delta_dpcm.py       # (Create these files)
│   │   ├── rle_decode.py
│   │   ├── huffman_decode.py
│   │   └── decompression_manager.py
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── image_validator.py          # (Create these files)
│   │   └── metrics.py
│   │
│   ├── cryptography/
│   │   ├── __init__.py
│   │   ├── lwe_params.py               # (Create these files)
│   │   ├── ntt.py
│   │   ├── strassen.py
│   │   └── lwe_encryption.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image_loader.py             # (Create these files)
│       ├── file_finder.py
│       └── time_tracker.py
│
├── benchmarks/
│   ├── __init__.py
│   ├── benchmark_compression.py        # (Create these files)
│   ├── benchmark_ntt.py
│   ├── benchmark_strassen.py
│   ├── benchmark_lwe.py
│   └── run_all_benchmarks.py
│
├── tests/
│   ├── __init__.py
│   ├── test_compression.py             # (Create these files)
│   ├── test_decompression.py
│   ├── test_validation.py
│   ├── test_ntt.py
│   ├── test_strassen.py
│   └── test_lwe.py
│
├── notebooks/
│   ├── 01_compression_analysis.ipynb
│   ├── 02_reconstruction_validation.ipynb
│   ├── 03_cryptography_analysis.ipynb
│   └── 04_thesis_summary.ipynb
│
├── outputs/
│   ├── reconstructed_images/           # Generated images saved here
│   ├── plots/                          # Benchmark plots saved here
│   ├── reports/                        # Benchmark reports saved here
│   └── compressed_data/                # Compressed files saved here
│
├── thesis/
│   ├── thesis.md
│   └── sections/
│       ├── 01_introduction.md
│       ├── 02_background.md
│       ├── 03_methodology.md
│       ├── 04_implementation.md
│       ├── 05_experimental_results.md
│       ├── 06_analysis.md
│       └── 07_conclusion.md
│
├── docs/
│   ├── SETUP.md
│   ├── API.md
│   ├── COMPRESSION_EXPLAINED.md
│   ├── BENCHMARKING.md
│   └── TROUBLESHOOTING.md
│
├── scripts/
│   ├── compress_single_image.py
│   ├── decompress_single_image.py
│   ├── batch_compress.py
│   ├── validate_all_images.py
│   └── generate_thesis_plots.py
│
├── config/
│   ├── config.yaml
│   └── benchmark_params.yaml
│
└── venv/                               # Virtual environment (don't commit to git)
```

---

## Creating Module Files

### 1. src/compression/delta_dpcm.py

```python
"""Delta DPCM compression module."""

import numpy as np

def apply_delta_dpcm(pixels):
    """Apply lossless Delta DPCM encoding."""
    if not pixels or len(pixels) < 2:
        return pixels
    
    dpcm = [pixels[0]]
    for i in range(1, len(pixels)):
        diff = (pixels[i] - pixels[i-1]) % 256
        if diff > 127:
            diff = diff - 256
        dpcm.append(diff)
    return dpcm

def get_delta_dpcm_stats(pixels):
    """Get statistics about delta values."""
    dpcm = apply_delta_dpcm(pixels)
    return {
        'min_delta': min(dpcm) if dpcm else 0,
        'max_delta': max(dpcm) if dpcm else 0,
        'mean_delta': np.mean(dpcm) if dpcm else 0,
    }
```

### 2. src/compression/rle.py

```python
"""Run-Length Encoding module."""

def apply_rle(pixels, max_run=255):
    """Apply RLE compression."""
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

def estimate_rle_efficiency(pixels):
    """Estimate RLE compression efficiency."""
    original_size = len(pixels)
    rle_data = apply_rle(pixels)
    compressed_size = len(rle_data) * 2  # (value, count) pairs
    return (original_size - compressed_size) / original_size * 100 if original_size > 0 else 0
```

### 3. src/compression/huffman.py

```python
"""Huffman coding module."""

import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def get_frequency_map(values):
    """Build frequency map from values."""
    freq_map = {}
    for val in values:
        freq_map[val] = freq_map.get(val, 0) + 1
    return freq_map

def build_huffman_tree(freq_map):
    """Build Huffman tree from frequency map."""
    if not freq_map:
        return None
    
    heap = [Node(char, freq) for char, freq in freq_map.items()]
    heapq.heapify(heap)
    
    if len(heap) == 1:
        node = heapq.heappop(heap)
        root = Node(None, node.freq)
        root.left = node
        return root
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = Node(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        heapq.heappush(heap, parent)
    
    return heap[0]

def generate_huffman_codes(root, code='', codes=None):
    """Generate Huffman codes from tree."""
    if codes is None:
        codes = {}
    if root is None:
        return codes
    if root.char is not None:
        codes[root.char] = code if code else '0'
        return codes
    if root.left:
        generate_huffman_codes(root.left, code + '0', codes)
    if root.right:
        generate_huffman_codes(root.right, code + '1', codes)
    return codes

def huffman_compress_rle(rle_data):
    """Compress RLE data using Huffman."""
    if not rle_data:
        return '', {}, [], 0
    
    values = [val for val, count in rle_data]
    counts = [count for val, count in rle_data]
    freq_map = get_frequency_map(values)
    root = build_huffman_tree(freq_map)
    code_map = generate_huffman_codes(root)
    
    if len(code_map) == 1:
        single_val = next(iter(code_map))
        code_map[single_val] = '0'
    
    canonical_overhead = len(code_map) * 8
    encoded_values = ''.join(code_map[val] for val in values)
    total_bits = len(encoded_values) + (len(counts) * 8) + canonical_overhead
    
    return encoded_values, code_map, counts, total_bits
```

### 4. src/compression/data_structures.py

```python
"""Data structures for compression pipeline."""

from dataclasses import dataclass

@dataclass
class CompressedImageData:
    """Store all data needed for decompression."""
    first_pixel: int
    width: int
    height: int
    huffman_tree: dict
    encoded_bitstring: str
    rle_counts: list
    compressed_bits: int

@dataclass
class ImageCompressionResult:
    """Store compression statistics."""
    original_bits: int
    compressed_bits: int
    compression_ratio: float
    compression_time: float
    image_name: str
```

### 5. src/decompression/reverse_delta_dpcm.py

```python
"""Reverse Delta DPCM decompression."""

def reverse_delta_dpcm(dpcm_data):
    """Reconstruct original pixels from Delta DPCM."""
    if not dpcm_data:
        return []
    
    pixels = [dpcm_data[0]]
    for i in range(1, len(dpcm_data)):
        reconstructed = (pixels[i-1] + dpcm_data[i]) % 256
        pixels.append(reconstructed)
    return pixels
```

### 6. src/decompression/rle_decode.py

```python
"""RLE decompression."""

def rle_decode(rle_data):
    """Decode RLE back to pixel values."""
    if not rle_data:
        return []
    
    pixels = []
    for value, count in rle_data:
        pixels.extend([value] * count)
    return pixels
```

### 7. src/decompression/huffman_decode.py

```python
"""Huffman decompression."""

def huffman_decompress(encoded_bitstring, code_map, num_values):
    """Decompress Huffman encoded bitstring."""
    if not encoded_bitstring or not code_map:
        return []
    
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

def huffman_decompress_rle(encoded_bitstring, code_map, rle_counts):
    """Decompress Huffman encoded RLE."""
    values = huffman_decompress(encoded_bitstring, code_map, len(rle_counts))
    rle_data = [(val, count) for val, count in zip(values, rle_counts)]
    return rle_data
```

### 8. src/validation/image_validator.py

```python
"""Image reconstruction validation."""

import numpy as np

def validate_reconstruction(original_pixels, reconstructed_pixels):
    """Validate perfect reconstruction."""
    if len(original_pixels) != len(reconstructed_pixels):
        return False, f"Length mismatch: {len(original_pixels)} vs {len(reconstructed_pixels)}"
    
    differences = sum(1 for o, r in zip(original_pixels, reconstructed_pixels) if o != r)
    
    if differences == 0:
        return True, "Perfect reconstruction! ✓"
    else:
        mse = np.mean([(o - r)**2 for o, r in zip(original_pixels, reconstructed_pixels)])
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        return False, f"{differences} pixel errors, MSE={mse:.4f}, PSNR={psnr:.2f}dB"

def calculate_metrics(original, reconstructed):
    """Calculate compression metrics."""
    if len(original) != len(reconstructed):
        return None
    
    mse = np.mean([(o - r)**2 for o, r in zip(original, reconstructed)])
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    mae = np.mean([abs(o - r) for o, r in zip(original, reconstructed)])
    
    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae,
        'perfect': mse == 0,
    }
```

---

## Quick Start Commands

```bash
# 1. Create and setup project
mkdir lwe-thesis && cd lwe-thesis
python3 setup-project.py

# 2. Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Add images
cp /path/to/images/*.png image_data/

# 4. Run tests
pytest tests/

# 5. Run benchmarks
python benchmarks/run_all_benchmarks.py

# 6. Check outputs
ls outputs/plots/
ls outputs/reconstructed_images/
```

---

## Troubleshooting

### Virtual Environment Issues

```bash
# Recreate venv if problems occur
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Import Errors

```bash
# Ensure __init__.py files exist in all packages
find src/ -type d -exec touch {}/__init__.py \;
find benchmarks/ -type d -exec touch {}/__init__.py \;
find tests/ -type d -exec touch {}/__init__.py \;
```

### No Images Found

```bash
# Check image_data directory
ls -la image_data/

# Copy images if missing
cp /path/to/images/* image_data/
```

### Pytest Errors

```bash
# Ensure pytest is installed
pip install pytest

# Run with verbose output
pytest tests/ -v --tb=short
```

---

## File Checklist

After setup, verify these files exist:

```
✓ requirements.txt
✓ setup.py
✓ README.md
✓ .gitignore
✓ config/config.yaml
✓ config/benchmark_params.yaml
✓ src/compression/delta_dpcm.py
✓ src/compression/rle.py
✓ src/compression/huffman.py
✓ src/decompression/reverse_delta_dpcm.py
✓ src/decompression/rle_decode.py
✓ src/decompression/huffman_decode.py
✓ src/validation/image_validator.py
✓ benchmarks/run_all_benchmarks.py
✓ tests/test_compression.py
✓ tests/test_decompression.py
```

All directories should exist and be empty or contain appropriate files.
