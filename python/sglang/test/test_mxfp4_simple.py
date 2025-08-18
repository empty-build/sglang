#!/usr/bin/env python3
"""
Simple test script for MXFP4QuantizeUtil
Tests: bf16 -> mxfp4 -> bf16 pipeline
"""

import torch
from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

use_packed = True

def test_mxfp4_simple():
    """Simple test of MXFP4QuantizeUtil"""

    print("Testing MXFP4QuantizeUtil")
    print("=" * 50)

    # Set random seed
    torch.manual_seed(42)

    print("1. Generating random bf16 tensor...")
    input_tensor = torch.randn(42, 8, 128, dtype=torch.bfloat16)
    # input_tensor = torch.randn(2048, 1024, dtype=torch.bfloat16)
    input_tensor = input_tensor * 0.1

    print(f"   Input shape: {input_tensor.shape}")
    print(f"   Input dtype: {input_tensor.dtype}")
    print(f"   Input range: [{input_tensor.min().item():.4f}, {input_tensor.max().item():.4f}]")

    print("\n2. Quantizing to mxfp4...")
    if use_packed:
        quantized_tensor = MXFP4QuantizeUtil.quantize_packed(input_tensor, 32)
        print(f"   Quantized shape: {quantized_tensor.shape}")
        print(f"   Quantized dtype: {quantized_tensor.dtype}")
    else:
        quantized_tensor, scale = MXFP4QuantizeUtil.quantize(input_tensor, 32)
        print(f"   Quantized shape: {quantized_tensor.shape}")
        print(f"   Quantized dtype: {quantized_tensor.dtype}")
        print(f"   Scale shape: {scale.shape}")
        print(f"   Scale dtype: {scale.dtype}")

    print("\n3. Dequantizing back to bf16...")
    if use_packed:
        dequantized_tensor = MXFP4QuantizeUtil.dequantize_packed(
            quantized_data=quantized_tensor,
            dtype=torch.bfloat16,
            block_sizes=[32]
        )
    else:
        dequantized_tensor = MXFP4QuantizeUtil.dequantize(
            quantized_data=quantized_tensor,
            scale=scale,
            dtype=torch.bfloat16,
            block_sizes=[32]
        )
    print(f"   Dequantized shape: {dequantized_tensor.shape}")
    print(f"   Dequantized dtype: {dequantized_tensor.dtype}")

    print("\n4. Error Analysis...")
    # Calculate errors
    abs_error = torch.abs(input_tensor - dequantized_tensor)
    rel_error = torch.abs((input_tensor - dequantized_tensor) / (input_tensor.abs() + 1e-8))

    print(f"   Absolute Error - Mean: {abs_error.mean().item():.8f}")
    print(f"   Absolute Error - Max: {abs_error.max().item():.8f}")
    print(f"   Relative Error - Mean: {rel_error.mean().item():.8f}")
    print(f"   Relative Error - Max: {rel_error.max().item():.8f}")

    # RMSE
    rmse = torch.sqrt(torch.mean((input_tensor - dequantized_tensor) ** 2))
    print(f"   RMSE: {rmse.item():.8f}")

    # Memory comparison
    input_memory = input_tensor.numel() * 2  # bf16 = 2 bytes
    quantized_memory = quantized_tensor.numel() * 1  # uint8 = 1 byte
    scale_memory = 0 if use_packed else scale.numel() * 1  # uint8 = 1 byte
    total_quantized = quantized_memory + scale_memory

    print(f"\n5. Memory Usage:")
    print(f"   Original: {input_memory / 1024 / 1024:.2f} MB")
    print(f"   Quantized: {total_quantized / 1024 / 1024:.2f} MB")
    print(f"   Compression: {input_memory / total_quantized:.2f}x")
    print(f"   Saved: {(1 - total_quantized / input_memory) * 100:.1f}%")

    print("\nTest completed!")


if __name__ == "__main__":
    test_mxfp4_simple()

