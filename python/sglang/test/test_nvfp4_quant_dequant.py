#!/usr/bin/env python3

import torch
import time
from sglang.srt.layers.quantization.nvfp4_tensor import NVFP4QuantizeUtil
import numpy as np

def calculate_accuracy_metrics(original, reconstructed):
    """Calculate accuracy metrics between original and reconstructed tensors."""
    mse = torch.mean((original - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(original - reconstructed)).item()
    
    # PSNR calculation
    max_val = torch.max(torch.abs(original)).item()
    psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # Relative error
    rel_error = torch.mean(torch.abs(original - reconstructed) / (torch.abs(original) + 1e-8)).item()
    
    return {
        'MSE': mse,
        'MAE': mae, 
        'PSNR': psnr,
        'Relative Error': rel_error
    }

def benchmark_quantization_performance():
    num_runs = 100
    E, M, N = 256, 8192, 4096
    print(f"Tensor size: [{E}, {M}, {N}]")
    tensor_bf16 = torch.randn(E, M, N, dtype=torch.bfloat16, device='cuda')

    # warm_up
    for _ in range(10):
        _ = tensor_bf16 * 2
    torch.cuda.synchronize()

    print("\n=== FP8 Quant/Dequant ===")
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        tensor_fp8 = tensor_bf16.to(torch.float8_e4m3fn)
    torch.cuda.synchronize()
    fp8_quant_time = (time.time() - start_time) / num_runs

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        tensor_fp8_dequant = tensor_fp8.to(torch.bfloat16)
    torch.cuda.synchronize()
    fp8_dequant_time = (time.time() - start_time) / num_runs

    print(f"FP8 Quant: {fp8_quant_time*1000:.2f} ms")
    print(f"FP8 Dequant: {fp8_dequant_time*1000:.2f} ms")
    # FP8 Accuracy
    fp8_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp8_dequant)
    print(f"FP8 Accuracy - MSE: {fp8_metrics['MSE']:.6f}, MAE: {fp8_metrics['MAE']:.6f}, PSNR: {fp8_metrics['PSNR']:.2f}dB")
    del tensor_fp8
    del tensor_fp8_dequant

    print("\n=== NVFP4 Quant/Dequant ===")
    global_sf = (448 * 6) / tensor_bf16.abs().max().float()
    sf_vec_size = 16
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        tensor_fp4, scale_factors = NVFP4QuantizeUtil.batched_quantize(
            tensor_bf16, global_sf, sf_vec_size
        )
    torch.cuda.synchronize()
    fp4_quant_time = (time.time() - start_time) / num_runs

    # torch.compile() warm up
    _ = NVFP4QuantizeUtil.batched_dequantize(
        tensor_fp4, scale_factors, global_sf, torch.bfloat16
    )
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        tensor_fp4_dequant = NVFP4QuantizeUtil.batched_dequantize(
            tensor_fp4, scale_factors, global_sf, torch.bfloat16
        )
    torch.cuda.synchronize()
    fp4_dequant_time = (time.time() - start_time) / num_runs

    print(f"NVFP4 Quant: {fp4_quant_time*1000:.2f} ms")
    print(f"NVFP4 Dequant: {fp4_dequant_time*1000:.2f} ms")

    # FP4 Accuracy
    fp4_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp4_dequant)
    print(f"FP4 Accuracy - MSE: {fp4_metrics['MSE']:.6f}, MAE: {fp4_metrics['MAE']:.6f}, PSNR: {fp4_metrics['PSNR']:.2f}dB")
    
    print("\n=== Accuracy Comparison ===")
    print(f"FP8 vs FP4 - MSE Ratio: {fp8_metrics['MSE']/fp4_metrics['MSE']:.2f}x")
    print(f"FP8 vs FP4 - PSNR Diff: {fp8_metrics['PSNR']-fp4_metrics['PSNR']:.2f}dB")


if __name__ == "__main__":
    print(f"Use GPU: {torch.cuda.get_device_name()}")
    benchmark_quantization_performance()
