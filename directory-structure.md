# Complete Directory Structure for LWE Cryptography Thesis Project

## Full Project Structure

```
lwe-cryptography-thesis/
│
├── README.md                          # Project overview & setup instructions
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup (optional)
│
├── image_data/                        # Input images directory
│   ├── sample_1.png
│   ├── sample_2.jpg
│   ├── sample_3.bmp
│   └── ...more images...
│
├── src/                               # Source code directory
│   ├── __init__.py
│   │
│   ├── compression/                   # Compression module
│   │   ├── __init__.py
│   │   ├── delta_dpcm.py              # Delta DPCM compression
│   │   ├── rle.py                     # Run-Length Encoding
│   │   ├── huffman.py                 # Huffman coding
│   │   ├── data_structures.py         # CompressedImageData, etc.
│   │   └── compression_manager.py     # Main compression orchestrator
│   │
│   ├── decompression/                 # Decompression module
│   │   ├── __init__.py
│   │   ├── reverse_delta_dpcm.py      # Delta DPCM decompression
│   │   ├── rle_decode.py              # RLE decompression
│   │   ├── huffman_decode.py          # Huffman decompression
│   │   └── decompression_manager.py   # Main decompression orchestrator
│   │
│   ├── validation/                    # Validation & testing module
│   │   ├── __init__.py
│   │   ├── image_validator.py         # Image reconstruction validation
│   │   └── metrics.py                 # MSE, PSNR, compression ratio
│   │
│   ├── cryptography/                  # LWE cryptography module
│   │   ├── __init__.py
│   │   ├── lwe_params.py              # LWE parameters
│   │   ├── ntt.py                     # Number Theoretic Transform
│   │   ├── strassen.py                # Strassen matrix multiplication
│   │   └── lwe_encryption.py          # LWE encryption/decryption
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── image_loader.py            # Image loading utilities
│       ├── file_finder.py             # Find images in directories
│       └── time_tracker.py            # Benchmark timing
│
├── benchmarks/                        # Benchmarking scripts
│   ├── __init__.py
│   ├── benchmark_compression.py       # Image compression benchmarks
│   ├── benchmark_ntt.py               # NTT polynomial benchmarks
│   ├── benchmark_strassen.py          # Strassen matrix benchmarks
│   ├── benchmark_lwe.py               # LWE encryption benchmarks
│   └── run_all_benchmarks.py          # Master benchmark runner
│
├── tests/                             # Unit & integration tests
│   ├── __init__.py
│   ├── test_compression.py            # Compression tests
│   ├── test_decompression.py          # Decompression tests
│   ├── test_validation.py             # Validation tests
│   ├── test_ntt.py                    # NTT tests
│   ├── test_strassen.py               # Strassen tests
│   └── test_lwe.py                    # LWE tests
│
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── 01_compression_analysis.ipynb
│   ├── 02_reconstruction_validation.ipynb
│   ├── 03_cryptography_analysis.ipynb
│   └── 04_thesis_summary.ipynb
│
├── outputs/                           # Output directory (generated)
│   ├── reconstructed_images/          # Decompressed images
│   │   ├── reconstructed_sample_1.png
│   │   └── ...
│   │
│   ├── plots/                         # Generated benchmark plots
│   │   ├── 1_image_compression_results.png
│   │   ├── 2_ntt_polynomial_results.png
│   │   ├── 3_strassen_matrix_results.png
│   │   ├── 4_thesis_summary_comparison.png
│   │   └── ...
│   │
│   ├── reports/                       # Benchmark reports
│   │   ├── compression_report.txt
│   │   ├── cryptography_report.txt
│   │   └── thesis_summary_report.txt
│   │
│   └── compressed_data/               # Compressed image files (optional)
│       ├── compressed_sample_1.cmp
│       └── ...
│
├── thesis/                            # Thesis document files
│   ├── thesis.md                      # Main thesis document
│   ├── sections/
│   │   ├── 01_introduction.md
│   │   ├── 02_background.md
│   │   ├── 03_methodology.md
│   │   ├── 04_implementation.md
│   │   ├── 05_experimental_results.md
│   │   ├── 06_analysis.md
│   │   └── 07_conclusion.md
│   ├── figures/                       # Thesis figures/plots
│   └── tables/                        # Thesis tables (CSV/data)
│
├── docs/                              # Documentation
│   ├── SETUP.md                       # Setup instructions
│   ├── API.md                         # API documentation
│   ├── COMPRESSION_EXPLAINED.md       # Technical explanation
│   ├── BENCHMARKING.md                # Benchmarking guide
│   └── TROUBLESHOOTING.md             # Common issues & fixes
│
├── scripts/                           # Standalone scripts
│   ├── compress_single_image.py       # Compress one image
│   ├── decompress_single_image.py     # Decompress one image
│   ├── batch_compress.py              # Batch compression
│   ├── validate_all_images.py         # Validate reconstruction
│   └── generate_thesis_plots.py       # Generate all thesis plots
│
├── config/                            # Configuration files
│   ├── config.yaml                    # Main configuration
│   ├── lwe_params.yaml                # LWE parameters
│   └── benchmark_params.yaml          # Benchmark parameters
│
└── venv/                              # Python virtual environment (local)
    └── (virtual environment files)
```

---

## Detailed File Descriptions

### Core Source Files

#### `src/compression/delta_dpcm.py`
```python
"""Delta DPCM compression module."""

def apply_delta_dpcm(pixels):
    """Apply lossless Delta DPCM encoding."""
    pass

def get_delta_dpcm_stats(pixels):
    """Get statistics about delta values."""
    pass
```

#### `src/compression/rle.py`
```python
"""Run-Length Encoding compression module."""

def apply_rle(pixels, max_run=255):
    """Apply RLE compression."""
    pass

def estimate_rle_efficiency(pixels):
    """Estimate RLE compression efficiency."""
    pass
```

#### `src/compression/huffman.py`
```python
"""Huffman coding compression module."""

class HuffmanTree:
    """Huffman tree implementation."""
    pass

def build_huffman_tree(freq_map):
    """Build Huffman tree from frequency map."""
    pass

def huffman_compress_rle(rle_data):
    """Compress RLE data using Huffman coding."""
    pass
```

#### `src/compression/data_structures.py`
```python
"""Data structures for compression pipeline."""

@dataclass
class CompressedImageData:
    """Store compressed image data."""
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

#### `src/compression/compression_manager.py`
```python
"""Main compression orchestrator."""

class CompressionManager:
    """Manages complete compression pipeline."""
    
    def compress_image(self, image_path):
        """Compress a single image."""
        pass
    
    def compress_batch(self, image_dir):
        """Compress multiple images."""
        pass
```

#### `src/decompression/decompression_manager.py`
```python
"""Main decompression orchestrator."""

class DecompressionManager:
    """Manages complete decompression pipeline."""
    
    def decompress_image(self, compressed_data):
        """Decompress a single image."""
        pass
```

#### `src/validation/image_validator.py`
```python
"""Image reconstruction validation."""

def validate_reconstruction(original_pixels, reconstructed_pixels):
    """Validate perfect reconstruction."""
    pass

def calculate_metrics(original, reconstructed):
    """Calculate MSE, PSNR, etc."""
    pass
```

#### `src/cryptography/ntt.py`
```python
"""Number Theoretic Transform implementation."""

class SimplifiedNTT:
    """NTT for polynomial multiplication."""
    
    def multiply(self, p1, p2):
        """Multiply polynomials using NTT."""
        pass
```

#### `src/cryptography/strassen.py`
```python
"""Strassen matrix multiplication."""

class StrassenMultiplier:
    """Strassen algorithm implementation."""
    
    @staticmethod
    def multiply(A, B, threshold=32):
        """Multiply matrices using Strassen."""
        pass
```

#### `src/cryptography/lwe_encryption.py`
```python
"""LWE encryption implementation."""

@dataclass
class LWEParams:
    """LWE parameters."""
    n: int
    q: int
    sigma: float

def generate_lwe_key(params):
    """Generate LWE secret key."""
    pass

def lwe_encrypt_byte(byte_value, key, params):
    """Encrypt single byte."""
    pass
```

### Benchmark Scripts

#### `benchmarks/benchmark_compression.py`
```python
"""Compression benchmarking."""

def benchmark_image_compression_lwe(image_paths):
    """Benchmark Delta DPCM + RLE + Huffman + LWE."""
    pass

def plot_image_compression_results(compression_results):
    """Plot compression results."""
    pass
```

#### `benchmarks/benchmark_ntt.py`
```python
"""NTT benchmarking."""

def benchmark_ntt_polynomial(sizes=[128, 256, 512]):
    """Benchmark NTT vs standard polynomial multiplication."""
    pass
```

#### `benchmarks/benchmark_strassen.py`
```python
"""Strassen benchmarking."""

def benchmark_strassen_matrix(sizes=[64, 128, 256]):
    """Benchmark Strassen vs standard matrix multiplication."""
    pass
```

#### `benchmarks/run_all_benchmarks.py`
```python
"""Master benchmark runner."""

def run_all_benchmarks(image_dir="image_data"):
    """Run all benchmarks and generate plots."""
    pass
```

### Test Files

#### `tests/test_compression.py`
```python
"""Unit tests for compression."""

def test_delta_dpcm():
    """Test Delta DPCM."""
    pass

def test_rle():
    """Test RLE."""
    pass

def test_huffman():
    """Test Huffman."""
    pass
```

#### `tests/test_decompression.py`
```python
"""Unit tests for decompression."""

def test_perfect_reconstruction():
    """Test that reconstruction is perfect."""
    pass

def test_lossless_compression():
    """Test losslessness."""
    pass
```

### Utility Scripts

#### `scripts/compress_single_image.py`
```python
"""Compress a single image."""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()
    # Compress image
```

#### `scripts/batch_compress.py`
```python
"""Batch compress all images in directory."""

if __name__ == "__main__":
    # Find all images
    # Compress each
    # Generate report
```

---

## Setup Instructions

### 1. Create Project Structure

```bash
cd /path/to/thesis/directory

# Create directories
mkdir -p lwe-cryptography-thesis/{src/{compression,decompression,validation,cryptography,utils},benchmarks,tests,notebooks,outputs/{reconstructed_images,plots,reports,compressed_data},thesis/sections,docs,scripts,config,image_data}

cd lwe-cryptography-thesis
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Create `requirements.txt`

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

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Create `__init__.py` Files

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

### 6. Add Images

```bash
# Copy your grayscale images to:
cp /path/to/images/* image_data/
```

### 7. Run Tests

```bash
pytest tests/ -v
```

### 8. Run Benchmarks

```bash
python benchmarks/run_all_benchmarks.py --image-dir image_data
```

---

## File Organization Best Practices

### For Thesis Writing
- Store all thesis sections in `thesis/sections/`
- Keep figures in `thesis/figures/`
- Reference generated plots: `outputs/plots/`

### For Development
- Keep source code modular in `src/`
- Each module has single responsibility
- Tests mirror source structure: `tests/test_*.py`

### For Benchmarking
- Scripts in `benchmarks/` generate plots to `outputs/plots/`
- Reports saved to `outputs/reports/`
- Reconstructed images saved to `outputs/reconstructed_images/`

### For Configuration
- YAML files in `config/` for easy parameter adjustment
- No hardcoded values in main code

---

## Quick Start Command Sequence

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Add images to image_data/

# 3. Run tests
pytest tests/

# 4. Run benchmarks
python benchmarks/run_all_benchmarks.py

# 5. Check outputs
ls -la outputs/plots/
ls -la outputs/reconstructed_images/
ls -la outputs/reports/
```

---

## Thesis Integration Points

When writing your thesis, reference:
- **Compression Results:** `outputs/reports/compression_report.txt`
- **Cryptography Results:** `outputs/reports/cryptography_report.txt`
- **Plots:** `outputs/plots/`
- **Validation Data:** `tests/test_*.py` results
- **Source Code:** `src/` for implementation details

All generated files are automatically timestamped and organized by test/benchmark run.
