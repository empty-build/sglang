import torch

#python/sglang/srt/layers/quantization/mxfp4_tensor.py
from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

def test_quant_dequant():
    torch.manual_seed(42)
    T, H, D = 2, 4, 128  # 小尺寸方便測試
    block_size = 32

    # 模擬 FP32 原始 tensor
    x = torch.randn(T,H,D)

    # 量化
    q_bytes, q_scales = MXFP4QuantizeUtil.quantize_tokenwise(x, block_size=block_size)
    print("q_bytes shape:", q_bytes.shape)       # [T,H,64]
    print("q_scales shape:", q_scales.shape)     # [T,H,4]

    # 反量化
    x_recon = MXFP4QuantizeUtil.dequantize_tokenwise(q_bytes, q_scales, block_size=block_size)
    print("x_recon shape:", x_recon.shape)       # [T,H,128]

    # 檢查誤差
    abs_err = (x - x_recon).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()

    print(f"Max error: {max_err}")
    print(f"Mean error: {mean_err}")

    # 小誤差判斷 (FP4 本身不可能完全無損，通常 <= 0.5 ~ 1.0)
    assert max_err < 2.0, "Quantization error too large!"
    print("Test passed.")

if __name__ == "__main__":
    test_quant_dequant()
