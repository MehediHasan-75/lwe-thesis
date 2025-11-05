#!/usr/bin/env python3
"""
Complete working examples for LWE Compression Project.
Run this script to see all compression/decompression examples in action.

Usage:
    python examples.py
"""

import os
import sys
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, '.')

from src.utils.image_loader import load_image_as_pixels, save_image_from_pixels
from src.compression.compression_manager import CompressionManager
from src.decompression.decompression_manager import DecompressionManager
from src.validation.image_validator import validate_reconstruction, calculate_metrics
from src.utils.file_finder import find_all_images


def create_test_images():
    """Create test images if they don't exist."""
    print("\n" + "="*70)
    print("STEP 1: Creating Test Images")
    print("="*70)
    
    os.makedirs("image_data", exist_ok=True)
    
    # Create gradient image
    img_data = np.linspace(0, 255, 100*100).reshape(100, 100).astype(np.uint8)
    img = Image.fromarray(img_data)
    img.save("image_data/test_gradient.png")
    print("✓ Created: image_data/test_gradient.png")
    
    # Create checkerboard pattern
    checkerboard = np.zeros((100, 100), dtype=np.uint8)
    checkerboard[::10, ::10] = 255
    checkerboard[5::10, 5::10] = 255
    img2 = Image.fromarray(checkerboard)
    img2.save("image_data/test_checkerboard.png")
    print("✓ Created: image_data/test_checkerboard.png")
    
    # Create random noise
    noise = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    img3 = Image.fromarray(noise)
    img3.save("image_data/test_noise.png")
    print("✓ Created: image_data/test_noise.png")


def example_1_compress_single_image():
    """Example 1: Compress a single image."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Compress a Single Image")
    print("="*70)
    
    # Load image
    pixels, width, height = load_image_as_pixels("image_data/test_gradient.png")
    print(f"\n✓ Loaded image: {width}×{height} = {len(pixels)} pixels")
    
    # Compress
    compressed = CompressionManager.compress_image_complete(
        pixels, width, height, "test_gradient.png"
    )
    
    # Calculate compression ratio
    original_bits = len(pixels) * 8
    compression_ratio = (compressed.compressed_bits / original_bits) * 100
    
    print(f"\nCompression Results:")
    print(f"  Original size:     {original_bits} bits ({original_bits//8} bytes)")
    print(f"  Compressed size:   {compressed.compressed_bits} bits ({compressed.compressed_bits//8} bytes)")
    print(f"  Compression ratio: {compression_ratio:.1f}%")
    print(f"  Space saved:       {original_bits - compressed.compressed_bits} bits")
    
    return compressed, pixels, width, height


def example_2_decompress_and_validate(compressed, original_pixels, width, height):
    """Example 2: Decompress and validate reconstruction."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Decompress and Validate")
    print("="*70)
    
    # Decompress
    reconstructed = DecompressionManager.decompress_image_complete(compressed)
    print(f"\n✓ Decompressed: {len(reconstructed)} pixels")
    
    # Validate
    is_perfect, msg = validate_reconstruction(original_pixels, reconstructed)
    print(f"✓ Validation: {msg}")
    
    # Calculate detailed metrics
    metrics = calculate_metrics(original_pixels, reconstructed)
    print(f"\nReconstruction Metrics:")
    print(f"  MSE:          {metrics['mse']:.6f}")
    print(f"  PSNR:         {metrics['psnr']:.2f} dB")
    print(f"  MAE:          {metrics['mae']:.6f}")
    print(f"  Perfect:      {metrics['perfect']}")
    print(f"  Max Error:    {metrics['max_error']} pixels")
    
    # Save reconstructed image
    output_path = "outputs/reconstructed_gradient.png"
    os.makedirs("outputs", exist_ok=True)
    DecompressionManager.save_reconstructed_image(
        reconstructed, width, height, output_path
    )
    print(f"\n✓ Saved reconstructed image: {output_path}")
    
    return metrics


def example_3_batch_process_images():
    """Example 3: Batch process all images."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Process All Images")
    print("="*70)
    
    images = find_all_images("image_data")
    print(f"\n✓ Found {len(images)} images to process\n")
    
    results = []
    
    for i, img_path in enumerate(images, 1):
        try:
            filename = os.path.basename(img_path)
            print(f"{i}. Processing: {filename}")
            
            # Load image
            pixels, width, height = load_image_as_pixels(img_path)
            
            # Compress
            compressed = CompressionManager.compress_image_complete(
                pixels, width, height, filename
            )
            ratio = (compressed.compressed_bits / (len(pixels)*8)) * 100
            
            # Decompress
            reconstructed = DecompressionManager.decompress_image_complete(compressed)
            
            # Validate
            is_perfect, msg = validate_reconstruction(pixels, reconstructed)
            
            print(f"   Size: {len(pixels)} pixels ({width}×{height})")
            print(f"   Compression: {ratio:.1f}%")
            print(f"   Result: {msg}")
            print()
            
            results.append({
                'filename': filename,
                'pixels': len(pixels),
                'ratio': ratio,
                'perfect': is_perfect,
            })
            
        except Exception as e:
            print(f"   Error: {e}\n")
    
    # Summary
    print("="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"\nProcessed: {len(results)} images\n")
    
    for result in results:
        status = "✓" if result['perfect'] else "✗"
        print(f"{status} {result['filename']:<30} {result['pixels']:>6} px  {result['ratio']:>6.1f}%")
    
    if results:
        avg_ratio = sum(r['ratio'] for r in results) / len(results)
        print(f"\nAverage compression: {avg_ratio:.1f}%")


def example_4_test_different_image_types():
    """Example 4: Test with different types of images."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Test Different Image Types")
    print("="*70)
    
    test_images = [
        ("image_data/test_gradient.png", "Smooth Gradient"),
        ("image_data/test_checkerboard.png", "Checkerboard Pattern"),
        ("image_data/test_noise.png", "Random Noise"),
    ]
    
    for img_path, description in test_images:
        if not os.path.exists(img_path):
            continue
            
        print(f"\n{description}: {os.path.basename(img_path)}")
        print("-" * 50)
        
        # Load
        pixels, width, height = load_image_as_pixels(img_path)
        
        # Compress
        compressed = CompressionManager.compress_image_complete(
            pixels, width, height, img_path
        )
        ratio = (compressed.compressed_bits / (len(pixels)*8)) * 100
        
        # Decompress
        reconstructed = DecompressionManager.decompress_image_complete(compressed)
        
        # Validate
        is_perfect, msg = validate_reconstruction(pixels, reconstructed)
        
        print(f"Size: {width}×{height} = {len(pixels)} pixels")
        print(f"Compression: {ratio:.1f}%")
        print(f"Validation: {msg}")


def main():
    """Run all examples."""
    print("\n" + "█"*70)
    print("█  LWE COMPRESSION PROJECT - COMPLETE EXAMPLES")
    print("█"*70)
    
    try:
        # Step 1: Create test images
        create_test_images()
        
        # Example 1: Compress single image
        compressed, pixels, width, height = example_1_compress_single_image()
        
        # Example 2: Decompress and validate
        metrics = example_2_decompress_and_validate(compressed, pixels, width, height)
        
        # Example 3: Batch process
        example_3_batch_process_images()
        
        # Example 4: Different image types
        example_4_test_different_image_types()
        
        # Final summary
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("""
Next steps:
  1. Try with your own images:
     - Copy your images to image_data/
     - Run: python examples.py

  2. Run tests:
     - python -m pytest tests/test_compression.py -v

  3. For your thesis:
     - Add NTT polynomial multiplication (src/cryptography/ntt.py)
     - Add Strassen matrix multiplication (src/cryptography/strassen.py)
     - Integrate with LWE encryption (src/cryptography/lwe_encryption.py)
     - Run end-to-end benchmarks
        """)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
