# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
# import tensorrt_llm

@torch.compile
def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]

@torch.compile
def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    *batch_dims, m, n = a.shape

    # Flatten batch dimensions
    a_flat = a.view(-1, m, n)
    batch_size = a_flat.size(0)

    results = []
    for b in range(batch_size):
        a_batch = a_flat[b].flatten()
        high = (a_batch & 0xF0) >> 4
        low = a_batch & 0x0F

        combined = torch.stack((low, high), dim=1).flatten()
        signs = (combined & 0x08).to(torch.bool)
        abs_vals = (combined & 0x07).to(torch.long)

        kE2M1ToFloat = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float32, device=a.device
        )
        values = kE2M1ToFloat[abs_vals] * torch.where(signs, -1.0, 1.0)
        results.append(values.reshape(m, n * 2).to(dtype=dtype))

    result = torch.stack(results).view(*batch_dims, m, n * 2)
    return result

class NVFP4Dequantizer:
    # @classmethod
    # def batched_quantize(cls, tensor_bf16, global_sf, sf_vec_size):
    #     tensor_fp4, scale_factors = torch.ops.trtllm.fp4_batched_quantize(
    #         tensor_bf16, global_sf, sf_vec_size, False
    #     )
    #     return tensor_fp4, scale_factors

    @classmethod
    @torch.compile
    def batched_dequantize(
        cls, tensor_fp4, tensor_sf, global_scale, dtype, block_size=16
    ):
        """Dequantize fp4 tensor back to high precision."""
        assert tensor_fp4.dtype == torch.uint8
        *batch_dims, m, packed_k = tensor_fp4.shape
        k = packed_k * 2
        
        # Process each batch
        batch_size = tensor_fp4.view(-1, m, packed_k).size(0)
        tensor_fp4_flat = tensor_fp4.view(batch_size, m, packed_k)
        tensor_sf_flat = tensor_sf.view(batch_size, -1)
        
        results = []
        for b in range(batch_size):
            # Dequantize FP4 values
            tensor_f32 = break_fp4_bytes(tensor_fp4_flat[b:b+1], dtype).squeeze(0)
            tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
            
            # Process scale factors
            sf_view = tensor_sf_flat[b].view(torch.float8_e4m3fn)
            sf_linear = convert_swizzled_to_linear(sf_view.unsqueeze(0), m, k, block_size)
            sf_dtype = sf_linear.to(torch.float32) / global_scale
            
            # Apply scaling
            out = (tensor_f32 * sf_dtype.unsqueeze(-1)).reshape(m, k)
            results.append(out.to(dtype=dtype))
        
        return torch.stack(results).view(*batch_dims, m, k)
