import torch

class MHATokenToKVPool:
    def __init__(self):
        self.FP4_VALUES = torch.tensor([
            -6.0, -3.0, -2.0, -1.0,
            -0.5, -0.25, -0.125, -0.0,
             0.0,  0.125, 0.25, 0.5,
             1.0, 2.0, 3.0, 6.0
        ], dtype=torch.float32)

    def quantize_bf16_to_fp4_e2m1(self, tensor_bf16: torch.Tensor) -> torch.Tensor:
        tensor_f32 = tensor_bf16.float().clamp(-6.0, 6.0).view(-1)
        distances = torch.abs(tensor_f32.unsqueeze(1) - self.FP4_VALUES.to(tensor_f32.device))  # [N,16]
        indices = distances.argmin(dim=1).to(torch.uint8)  # [N]

        # 補偶數長度
        if indices.numel() % 2 != 0:
            indices = torch.cat([indices, torch.zeros(1, dtype=torch.uint8, device=indices.device)])

        packed = indices.view(-1, 2)
        packed_uint8 = (packed[:, 0] & 0x0F) | ((packed[:, 1] & 0x0F) << 4)  # pack兩個4bit成1byte
        new_shape = list(tensor_bf16.shape)
        new_shape[-1] = new_shape[-1] // 2  # 尾維度/2
        return packed_uint8.view(*new_shape)

    def dequantize_fp4_e2m1_to_bf16(self, tensor_uint8: torch.Tensor) -> torch.Tensor:
        fp4_values = self.FP4_VALUES.to(tensor_uint8.device)
        flat = tensor_uint8.flatten()
        low_nibble = flat & 0x0F
        high_nibble = (flat >> 4) & 0x0F
        # indices = torch.stack([low_nibble, high_nibble], dim=1).flatten()
        indices = torch.stack([low_nibble, high_nibble], dim=1).flatten().to(torch.long)
        values = fp4_values[indices].view(-1)
        # reshape回原始tensor shape尾維*2
        orig_shape = list(tensor_uint8.shape)
        orig_shape[-1] *= 2
        return values.view(*orig_shape).to(torch.bfloat16)


def test_quant_dequant():
    pool = MHATokenToKVPool()

    # 測試資料，尾維是2的倍數 (batch=2, seq=4, features=8, 8是偶數)
    data = torch.linspace(-6.0, 6.0, steps=2*4*8, dtype=torch.bfloat16).view(2, 4, 8)

    quantized = pool.quantize_bf16_to_fp4_e2m1(data)
    dequantized = pool.dequantize_fp4_e2m1_to_bf16(quantized)

    print("原始資料:")
    print(data)
    print("量化後 (uint8):")
    print(quantized)
    print("反量化後 (bfloat16):")
    print(dequantized)

    max_error = torch.abs(data.float() - dequantized.float()).max().item()
    print(f"最大誤差: {max_error:.6f}")

if __name__ == "__main__":
    test_quant_dequant()