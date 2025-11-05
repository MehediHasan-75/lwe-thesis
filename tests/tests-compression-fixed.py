"""
Comprehensive tests for LWE compression project.
Run with: python -m pytest tests/test_compression.py -v
"""

import pytest
import os
import tempfile
from PIL import Image
import numpy as np

from src.compression.delta_dpcm import apply_delta_dpcm, reverse_delta_dpcm
from src.compression.rle import apply_rle, rle_decode
from src.compression.huffman import huffman_compress_rle, huffman_decompress
from src.compression.compression_manager import CompressionManager
from src.decompression.decompression_manager import DecompressionManager
from src.validation.image_validator import validate_reconstruction, calculate_metrics
from src.utils.image_loader import load_image_as_pixels, save_image_from_pixels
from src.utils.file_finder import find_all_images


class TestDeltaDPCM:
    """Test Delta DPCM compression/decompression."""
    
    def test_simple_pixels(self):
        """Test with simple pixel values."""
        pixels = [100, 105, 110, 108, 120, 115, 125, 130]
        compressed = apply_delta_dpcm(pixels)
        reconstructed = reverse_delta_dpcm(compressed)
        assert pixels == reconstructed
    
    def test_constant_pixels(self):
        """Test with constant pixel values."""
        pixels = [128] * 10
        compressed = apply_delta_dpcm(pixels)
        reconstructed = reverse_delta_dpcm(compressed)
        assert pixels == reconstructed
    
    def test_extreme_values(self):
        """Test with extreme pixel values (0 and 255)."""
        pixels = [0, 255, 0, 255, 128, 64]
        compressed = apply_delta_dpcm(pixels)
        reconstructed = reverse_delta_dpcm(compressed)
        assert pixels == reconstructed
    
    def test_random_pixels(self):
        """Test with random pixel values."""
        np.random.seed(42)
        pixels = list(np.random.randint(0, 256, 100))
        compressed = apply_delta_dpcm(pixels)
        reconstructed = reverse_delta_dpcm(compressed)
        assert pixels == reconstructed


class TestRLE:
    """Test Run-Length Encoding compression/decompression."""
    
    def test_simple_rle(self):
        """Test basic RLE encoding."""
        data = [5, 5, 5, 7, 7, 9, 9, 9, 9]
        rle_data = apply_rle(data)
        decoded = rle_decode(rle_data)
        assert data == decoded
    
    def test_no_repetition(self):
        """Test RLE with no repeated values."""
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        rle_data = apply_rle(data)
        decoded = rle_decode(rle_data)
        assert data == decoded
    
    def test_all_same(self):
        """Test RLE with all same values."""
        data = [42] * 100
        rle_data = apply_rle(data)
        decoded = rle_decode(rle_data)
        assert data == decoded
    
    def test_max_run_length(self):
        """Test RLE with max run length."""
        data = [7] * 300  # Exceeds max_run=255
        rle_data = apply_rle(data, max_run=255)
        decoded = rle_decode(rle_data)
        assert data == decoded


class TestHuffman:
    """Test Huffman coding compression/decompression."""
    
    def test_simple_huffman(self):
        """Test simple Huffman encoding."""
        rle_data = [(5, 3), (7, 2), (9, 4)]
        encoded, code_map, counts, total_bits = huffman_compress_rle(rle_data)
        decoded = huffman_decompress(encoded, code_map, len(counts))
        original_values = [5, 7, 9]
        assert decoded == original_values
    
    def test_single_symbol(self):
        """Test Huffman with single unique symbol."""
        rle_data = [(42, 10)]
        encoded, code_map, counts, total_bits = huffman_compress_rle(rle_data)
        decoded = huffman_decompress(encoded, code_map, len(counts))
        assert decoded == [42]


class TestCompressionPipeline:
    """Test complete compression pipeline."""
    
    def test_pipeline_simple(self):
        """Test complete compression pipeline with simple data."""
        pixels = [100, 105, 110, 108, 120, 115, 125, 130]
        
        # Compress
        dpcm = apply_delta_dpcm(pixels)
        rle = apply_rle(dpcm)
        encoded, code_map, counts, total_bits = huffman_compress_rle(rle)
        
        # Decompress
        huffman_vals = huffman_decompress(encoded, code_map, len(counts))
        rle_pairs = [(v, c) for v, c in zip(huffman_vals, counts)]
        rle_decoded = rle_decode(rle_pairs)
        reconstructed = reverse_delta_dpcm(rle_decoded)
        
        # Verify
        assert pixels == reconstructed
    
    def test_pipeline_random(self):
        """Test pipeline with random pixels."""
        np.random.seed(42)
        pixels = list(np.random.randint(0, 256, 100))
        
        # Compress
        dpcm = apply_delta_dpcm(pixels)
        rle = apply_rle(dpcm)
        encoded, code_map, counts, total_bits = huffman_compress_rle(rle)
        
        # Decompress
        huffman_vals = huffman_decompress(encoded, code_map, len(counts))
        rle_pairs = [(v, c) for v, c in zip(huffman_vals, counts)]
        rle_decoded = rle_decode(rle_pairs)
        reconstructed = reverse_delta_dpcm(rle_decoded)
        
        # Verify
        assert pixels == reconstructed


class TestImageCompression:
    """Test image compression and decompression."""
    
    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        # Create simple gradient image
        img_data = np.linspace(0, 255, 100*100).reshape(100, 100).astype(np.uint8)
        return Image.fromarray(img_data), img_data.flatten().tolist(), 100, 100
    
    def test_compress_decompress_image(self, test_image):
        """Test image compression and decompression."""
        img, pixels, width, height = test_image
        
        # Compress
        compressed = CompressionManager.compress_image_complete(
            pixels, width, height, "test"
        )
        assert compressed is not None
        assert compressed.width == width
        assert compressed.height == height
        
        # Decompress
        reconstructed = DecompressionManager.decompress_image_complete(compressed)
        
        # Validate
        is_perfect, msg = validate_reconstruction(pixels, reconstructed)
        assert is_perfect, msg
    
    def test_reconstruction_perfect(self, test_image):
        """Test that reconstruction is perfect."""
        img, pixels, width, height = test_image
        
        compressed = CompressionManager.compress_image_complete(
            pixels, width, height, "test"
        )
        reconstructed = DecompressionManager.decompress_image_complete(compressed)
        
        # Check metrics
        metrics = calculate_metrics(pixels, reconstructed)
        assert metrics['mse'] == 0.0
        assert metrics['perfect'] == True
        assert metrics['max_error'] == 0
    
    def test_compression_ratio(self, test_image):
        """Test that compression ratio is calculated."""
        img, pixels, width, height = test_image
        
        compressed = CompressionManager.compress_image_complete(
            pixels, width, height, "test"
        )
        
        original_bits = len(pixels) * 8
        compression_ratio = (compressed.compressed_bits / original_bits) * 100
        
        # Should compress to less than 100%
        assert compression_ratio >= 0
        assert compression_ratio <= 100


class TestValidation:
    """Test validation functions."""
    
    def test_perfect_reconstruction(self):
        """Test validation of perfect reconstruction."""
        pixels = [100, 105, 110, 108, 120]
        is_perfect, msg = validate_reconstruction(pixels, pixels)
        assert is_perfect
        assert "Perfect" in msg
    
    def test_imperfect_reconstruction(self):
        """Test validation of imperfect reconstruction."""
        pixels = [100, 105, 110, 108, 120]
        reconstructed = [100, 105, 110, 108, 121]  # Last pixel different
        is_perfect, msg = validate_reconstruction(pixels, reconstructed)
        assert not is_perfect
        assert "error" in msg.lower()
    
    def test_metrics_perfect(self):
        """Test metrics for perfect reconstruction."""
        pixels = [100, 105, 110, 108, 120]
        metrics = calculate_metrics(pixels, pixels)
        assert metrics['mse'] == 0.0
        assert metrics['perfect'] == True
        assert metrics['mae'] == 0.0
    
    def test_metrics_imperfect(self):
        """Test metrics for imperfect reconstruction."""
        pixels = [100, 105, 110, 108, 120]
        reconstructed = [100, 105, 110, 108, 125]
        metrics = calculate_metrics(pixels, reconstructed)
        assert metrics['mse'] > 0
        assert metrics['perfect'] == False
        assert metrics['max_error'] == 5


class TestUtilities:
    """Test utility functions."""
    
    def test_find_images_empty_dir(self):
        """Test finding images in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = find_all_images(tmpdir)
            assert len(images) == 0
    
    def test_find_images_with_images(self):
        """Test finding images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(3):
                img_data = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
                img = Image.fromarray(img_data)
                img.save(os.path.join(tmpdir, f"test_{i}.png"))
            
            images = find_all_images(tmpdir)
            assert len(images) == 3
    
    def test_save_and_load_image(self):
        """Test saving and loading images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            pixels = list(np.linspace(0, 255, 100*100).astype(np.uint8))
            width, height = 100, 100
            
            # Save
            filepath = os.path.join(tmpdir, "test.png")
            save_image_from_pixels(pixels, width, height, filepath)
            assert os.path.exists(filepath)
            
            # Load
            loaded_pixels, w, h = load_image_as_pixels(filepath)
            assert w == width
            assert h == height
            assert len(loaded_pixels) == len(pixels)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create test image
        img_data = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        pixels = img_data.flatten().tolist()
        width, height = 50, 50
        
        # Compress
        compressed = CompressionManager.compress_image_complete(
            pixels, width, height, "test"
        )
        
        # Calculate compression ratio
        ratio = (compressed.compressed_bits / (len(pixels) * 8)) * 100
        assert 0 <= ratio <= 100
        
        # Decompress
        reconstructed = DecompressionManager.decompress_image_complete(compressed)
        
        # Validate
        is_perfect, msg = validate_reconstruction(pixels, reconstructed)
        assert is_perfect
        
        # Calculate metrics
        metrics = calculate_metrics(pixels, reconstructed)
        assert metrics['perfect']
        assert metrics['mse'] == 0.0


# Parametrized test for multiple scenarios
@pytest.mark.parametrize("size,values", [
    (10, [100] * 10),
    (50, list(range(50))),
    (100, list(np.random.randint(0, 256, 100))),
])
def test_parametrized_compression(size, values):
    """Test compression with different inputs."""
    compressed = apply_delta_dpcm(values)
    reconstructed = reverse_delta_dpcm(compressed)
    assert values == reconstructed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
