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


# https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/mxfp4_tensor.py
class MXFP4QuantizeUtil:
    E2M1_max = 6.0

    E2M1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])

    # E2M1_bounds = torch.tensor([0.0, 0.33333334, 0.6666667], dtype=torch.float32)
    # E2M1_max = 0.6666667  # max representable magnitude in E2M1

    # @classmethod
    # def quantize(cls, input: torch.Tensor, block_size: int | None) -> tuple:
    #     """Converting a tensor to a quantized format based on MXFP4 quantization. Only E4M3 is supported.
    #     Args:
    #         input (torch.Tensor): The input tensor to be quantized.
    #         block_sizes (dict | None): The block sizes for quantization.
    #     """

    #     def cast_fp4(x):
    #         sign = torch.sign(x)
    #         sign_bit = (2 - sign) // 2
    #         ord_ = torch.sum(
    #             (x.abs().unsqueeze(-1) - cls.E2M1_bounds.to(x.device)) > 0, dim=-1
    #         )
    #         fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)
    #         return fp4_val

    #     def fuse_uint4_to_uint8(x):
    #         # If the last dimension is odd, pad with zeros
    #         # If this behavior is not desired, please modify the code accordingly
    #         left_side = x[..., 0::2]  # Even indices (0, 2, 4...)
    #         right_side = x[..., 1::2]  # Odd indices (1, 3, 5...)
    #         new_data = (
    #             right_side.clone() << 4
    #         )  # Put odd indices (higher addresses) in high bits
    #         new_data[
    #             ..., : left_side.shape[-1]
    #         ] += left_side  # Put even indices in low bits
    #         return new_data

    #     if block_size is None:
    #         block_size = 32

    #     original_shape = input.shape
    #     original_dtype = input.dtype
    #     input = input.view(-1, block_size)
    #     # get scales
    #     input_amax = input.abs().max(dim=-1, keepdim=True).values
    #     descale = input_amax / cls.E2M1_max
    #     min_value = torch.tensor(-127.0, device=descale.device)
    #     e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))

    #     input = (input / torch.exp2(e8m0_scale)).view(original_shape)
    #     input_q = cast_fp4(input)
    #     input_q = fuse_uint4_to_uint8(input_q)
    #     e8m0_scale = (e8m0_scale + 127).to(torch.uint8)
    #     # return cls(original_shape, original_dtype, input_q), e8m0_scale
    #     return input_q, e8m0_scale

    @classmethod
    def quantize(cls, input: torch.Tensor, block_size: int | None) -> tuple:
        """Converting a tensor to a quantized format based on MXFP4 quantization. Only E4M3 is supported.
        Args:
            input (torch.Tensor): The input tensor to be quantized.
            block_sizes (dict | None): The block sizes for quantization.
        """

        def cast_fp4(x):
            sign = torch.sign(x)
            sign_bit = (2 - sign) // 2
            ord_ = torch.sum(
                (x.abs().unsqueeze(-1) - cls.E2M1_bounds.to(x.device)) > 0, dim=-1
            )
            fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)
            return fp4_val

        def fuse_uint4_to_uint8(x):
            # If the last dimension is odd, pad with zeros
            # If this behavior is not desired, please modify the code accordingly
            left_side = x[..., 0::2]  # Even indices (0, 2, 4...)
            right_side = x[..., 1::2]  # Odd indices (1, 3, 5...)
            new_data = (
                right_side.clone() << 4
            )  # Put odd indices (higher addresses) in high bits
            new_data[
                ..., : left_side.shape[-1]
            ] += left_side  # Put even indices in low bits
            return new_data

        if block_size is None:
            block_size = 32

        # Flatten safely (handles non-contiguous tensors)
        flat = input.reshape(-1)
        total_elems = flat.numel()
        # Pad to multiple of block_size so each block is full
        pad_len = (block_size - (total_elems % block_size)) % block_size
        if pad_len:
            pad = torch.zeros(pad_len, dtype=flat.dtype, device=flat.device)
            flat = torch.cat([flat, pad], dim=0)
        # Split into blocks
        blocks = flat.view(-1, block_size)
        # Per-block scale
        input_amax = blocks.abs().max(dim=-1, keepdim=True).values
        descale = input_amax / cls.E2M1_max
        min_value = torch.tensor(-127.0, device=descale.device)
        e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))
        # Scale and quantize to FP4
        scaled_blocks = blocks / torch.exp2(e8m0_scale)
        fp4_blocks = cast_fp4(scaled_blocks)
        # Pack each block's 32 nibbles into 16 bytes
        q_bytes = fuse_uint4_to_uint8(fp4_blocks)
        # Store E8M0 scale as uint8 per block
        e8m0_scale = (e8m0_scale + 127).to(torch.uint8).view(-1)
        return q_bytes, e8m0_scale

    @classmethod
    def dequantize(cls, quantized_data, dtype: torch.dtype, scale, block_sizes):
        """Dequantze MXFP4 packed tensor to a target dtype."""

        def unfuse_uint8_to_uint4(x):
            """Unfuse uint8 values back to uint4 values.
            This is the inverse operation of fuse_uint4_to_uint8.
            """
            # Extract the lower 4 bits (even indices)
            left_side = x & 0x0F

            # Extract the upper 4 bits (odd indices)
            right_side = (x >> 4) & 0x0F

            # Create a new tensor with alternating values
            shape = list(x.shape)
            shape[-1] = shape[-1] * 2
            result = torch.zeros(shape, dtype=torch.uint8, device=x.device)

            # Fill in the values - even indices get low bits, odd indices get high bits
            result[..., 0::2] = left_side  # Even indices from low bits
            result[..., 1::2] = right_side  # Odd indices from high bits

            return result

        e8m0_scale = scale
        block_size = block_sizes[-1]

        # Unfuse the uint8 values back to uint4
        x_unfused = unfuse_uint8_to_uint4(quantized_data)
        # Extract sign and magnitude
        sign = 1 - 2 * ((x_unfused & 0b1000) >> 3).to(
            torch.float32
        )  # Extract sign bit and convert to +1/-1
        magnitude = x_unfused & 0b0111  # Extract magnitude bits
        magnitude = magnitude.to(torch.long)

        # Create a tensor with the E2M1 values
        values = torch.tensor(cls.E2M1_values, device=quantized_data.device)

        # Use gather to index the values tensor properly
        # We need to reshape magnitude to match the dimensions we want to gather along
        original_shape = magnitude.shape
        x_float = values[magnitude.reshape(-1)].reshape(original_shape)

        # Apply sign and scale
        x_float = sign.float() * x_float

        # Reshape to apply block-wise scaling
        x_float = x_float.reshape(-1, block_size)

        # Apply the E8M0 scale
        scale_factor = torch.exp2(e8m0_scale.float() - 127)
        scale_factor = scale_factor.reshape(-1, 1)  # Reshape for proper broadcasting

        # Apply scaling and reshape back to original shape
        x_float = x_float * scale_factor

        # Reshape back to the original shape
        return x_float.reshape(original_shape).to(dtype)

    @classmethod
    def quantize_packed(cls, input: torch.Tensor, block_size: int | None) -> torch.Tensor:
       
        def pack_quantized_data(quantized_tensor: torch.Tensor, 
                                scale: torch.Tensor, 
                                original_shape: tuple) -> torch.Tensor:            
            shape_size = len(original_shape)
            # Always use uint64 (8 bytes per dimension);
            metadata_size = 1 + shape_size * 8
            scale_size = scale.numel()
            quantized_size = quantized_tensor.numel()
            total_size = metadata_size + scale_size + quantized_size
            
            # Pre-allocate result tensor
            packed = torch.empty(total_size, dtype=torch.uint8, device=quantized_tensor.device)
            packed[0] = shape_size  # Number of dimensions (uint8)

            # Convert shape to uint64 and pack directly
            shape_tensor = torch.tensor(original_shape, dtype=torch.uint64, device=quantized_tensor.device)
            shape_bytes = shape_tensor.view(torch.uint8)
            packed[1:metadata_size] = shape_bytes
            
            # Fill scale data
            scale_start = metadata_size
            scale_end = scale_start + scale_size
            packed[scale_start:scale_end] = scale.flatten()
            
            # Fill quantized data
            quantized_start = scale_end
            packed[quantized_start:] = quantized_tensor.flatten()
            
            return packed
        
        input_q, e8m0_scale = cls.quantize(input, block_size)
        return pack_quantized_data(
            input_q, e8m0_scale, input.shape
        )
    
    @classmethod
    def dequantize_packed(cls, quantized_data, dtype: torch.dtype, block_sizes): 
        
        def unpack_quantized_data(packed_tensor: torch.Tensor, block_size) -> tuple:
            # Extract metadata efficiently
            shape_size = int(packed_tensor[0].item())
            metadata_size = 1 + shape_size * 8

            if (shape_size == 0):
                return None, None, None
            
            # Extract shape values (always uint64)
            shape_start = 1
            shape_end = metadata_size
            # Clone to ensure storage_offset == 0 for safe view to uint64
            shape_bytes = packed_tensor[shape_start:shape_end].clone()
            
            # Convert bytes back to shape tensor (uint64)
            shape_tensor = shape_bytes.view(torch.uint64)
            original_shape = tuple(shape_tensor.tolist())
            
            # Calculate sizes
            total_elements = torch.prod(torch.tensor(original_shape)).item()
            num_blocks = (total_elements + block_size - 1) // block_size

            # Extract data using slicing
            scale_start = metadata_size
            scale_end = scale_start + num_blocks

            # Jack convert
            scale_start = int(scale_start)
            scale_end = int(scale_end)
            num_blocks = int(num_blocks)

            quantized_start = scale_end
            
            scale = packed_tensor[scale_start:scale_end]
            quantized_flat = packed_tensor[quantized_start:]
            
            # Reshape
            quantized_tensor = quantized_flat.view(num_blocks, 16)
            
            return quantized_tensor, scale, original_shape

        block_size = block_sizes[-1]
        quantized_data, scale, original_shape = unpack_quantized_data(quantized_data, block_size)
        if quantized_data != None:
            dequantized_tensor = cls.dequantize(quantized_data, dtype, scale, block_sizes)
            dequantized_tensor = dequantized_tensor.view(original_shape)
        else:
            dequantized_tensor = None

        return dequantized_tensor

    @classmethod
    def quantize_tokenwis_v1(cls, input: torch.Tensor, block_size: int = 32):
        """
        Quantize [T, 8, D] bfloat16 tensor into FP4-packed format per token, fully vectorized.
        Output:
        q_bytes: [T, 8, D/2] (uint8, packed FP4)
        e8m0_scale: [T, 8, D/block_size] (uint8, per-block scale)
        """
        assert input.shape[-1] % block_size == 0, "Last dim must be divisible by block_size"
        T, H, D = input.shape
        num_blocks = D // block_size

        def cast_fp4(x):
            sign = torch.sign(x)
            sign_bit = (2 - sign) // 2
            ord_ = torch.sum(
                (x.abs().unsqueeze(-1) - cls.E2M1_bounds.to(x.device)) > 0, dim=-1
            )
            return (sign_bit * 0b1000 + ord_).to(torch.uint8)

        def fuse_uint4_to_uint8(x):
            left = x[..., 0::2]   # even index
            right = x[..., 1::2]  # odd index
            return (right << 4) | left

        # [T, H, num_blocks, block_size]
        blocks = input.view(T, H, num_blocks, block_size)

        # per-block scale
        input_amax = blocks.abs().amax(dim=-1, keepdim=True)  # [T, H, num_blocks, 1]
        descale = input_amax / cls.E2M1_max
        min_value = torch.tensor(-127.0, device=input.device)
        e8m0 = torch.ceil(torch.maximum(torch.log2(descale), min_value))  # [T, H, num_blocks, 1]

        # scale and quantize
        scaled_blocks = blocks / torch.exp2(e8m0)
        fp4_blocks = cast_fp4(scaled_blocks)  # [T, H, num_blocks, block_size]

        # pack FP4 into uint8
        packed = fuse_uint4_to_uint8(fp4_blocks)  # [T, H, num_blocks, block_size/2]
        q_bytes = packed.reshape(T, H, D // 2)

        # store scale
        e8m0_scale = (e8m0.squeeze(-1) + 127).to(torch.uint8)  # [T, H, num_blocks]

        return q_bytes, e8m0_scale

    @classmethod
    def dequantize_tokenwise_v1(cls, q_bytes: torch.Tensor, e8m0_scale: torch.Tensor, block_size: int = 32):
        """
        Dequantize FP4-packed q_bytes per token using token-wise scale.
        Args:
            q_bytes: [T, H, D/2] uint8
            e8m0_scale: [T, H, num_blocks] uint8
        Returns:
            dequantized: [T, H, D] bfloat16
        """
        T, H, half_D = q_bytes.shape
        D = half_D * 2
        num_blocks = e8m0_scale.shape[-1]

        # unpack uint8 -> FP4 nibbles
        left = q_bytes & 0x0F
        right = (q_bytes >> 4) & 0x0F
        fp4 = torch.empty((T, H, D), dtype=torch.uint8, device=q_bytes.device)
        fp4[..., 0::2] = left
        fp4[..., 1::2] = right

        # decode sign and magnitude
        sign = torch.where((fp4 & 0b1000) == 0, 1.0, -1.0)
        ord_ = (fp4 & 0b0111).to(torch.float32)

        # reconstruct FP4 values: multiply by cls.E2M1_bounds
        # Note: cls.E2M1_bounds.shape = [3] -> broadcast over ord_
        # clamp ord_ to max 2
        ord_ = torch.clamp(ord_, 0, 2)
        bounds = cls.E2M1_bounds.to(fp4.device)
        val = sign * bounds[ord_.long()]

        # reshape to blocks
        val = val.view(T, H, num_blocks, block_size)

        # recover scale
        scale = e8m0_scale.to(torch.float32) - 127  # [T, H, num_blocks]
        scale = scale.unsqueeze(-1)  # broadcast over block_size

        dequantized = val * torch.exp2(scale)  # [T, H, num_blocks, block_size]

        return dequantized.reshape(T, H, D).to(torch.bfloat16)
    
    @classmethod
    def quantize_tokenwise_v2(cls, input: torch.Tensor, block_size: int = 32):
        """
        Token-wise FP4 quantization
        Args:
            input: [T, H, D] bfloat16
        Returns:
            q_bytes: [T, H, D/2] uint8
            e8m0_scale: [T, H, D//block_size] uint8
        """
        T, H, D = input.shape
        num_blocks = D // block_size
        assert D % block_size == 0, "D must be divisible by block_size"

        # reshape into blocks
        blocks = input.view(T, H, num_blocks, block_size)

        # per-block max
        amax = blocks.abs().amax(dim=-1, keepdim=True)  # [T,H,num_blocks,1]
        descale = amax / cls.E2M1_max
        min_value = torch.tensor(-127.0, device=input.device)
        e8m0 = torch.ceil(torch.maximum(torch.log2(descale), min_value))  # [T,H,num_blocks,1]

        # scale blocks
        scaled = blocks / torch.exp2(e8m0)

        # cast to FP4
        sign_bit = (scaled < 0).to(torch.uint8)
        abs_scaled = scaled.abs()
        # ord_ = sum(abs_scaled > bound, dim=-1) broadcasted
        ord_ = ((abs_scaled.unsqueeze(-1) - cls.E2M1_bounds.to(input.device)) > 0).sum(dim=-1).to(torch.uint8)
        fp4 = (sign_bit << 3) | ord_  # [T,H,num_blocks,block_size]

        # pack FP4
        left = fp4[..., 0::2]
        right = fp4[..., 1::2]
        q_bytes = (right << 4) | left
        q_bytes = q_bytes.reshape(T, H, D//2)

        # store scale
        e8m0_scale = (e8m0.squeeze(-1) + 127).to(torch.uint8)

        return q_bytes, e8m0_scale

    @classmethod
    def dequantize_tokenwise_v2(cls, q_bytes: torch.Tensor, e8m0_scale: torch.Tensor, block_size: int = 32):
        """
        Token-wise FP4 dequantization
        Args:
            q_bytes: [T,H,D/2] uint8
            e8m0_scale: [T,H,D//block_size] uint8
        Returns:
            dequantized: [T,H,D] bfloat16
        """
        T, H, half_D = q_bytes.shape
        D = half_D * 2
        num_blocks = e8m0_scale.shape[-1]

        # unpack FP4
        left = q_bytes & 0x0F
        right = (q_bytes >> 4) & 0x0F
        fp4 = torch.empty((T, H, D), dtype=torch.uint8, device=q_bytes.device)
        fp4[..., 0::2] = left
        fp4[..., 1::2] = right

        # decode FP4
        sign = torch.where((fp4 >> 3) == 0, 1.0, -1.0)
        ord_ = fp4 & 0b011  # E2M1 ord 0/1/2

        bounds = cls.E2M1_bounds.to(fp4.device)  # <- 保證同 device
        val = sign * bounds[ord_.long()]

        # reshape into blocks
        val = val.view(T, H, num_blocks, block_size)

        # recover scale
        scale = e8m0_scale.to(fp4.device).float() - 127.0
        scale = scale.unsqueeze(-1)  # broadcast
        dequantized = val * torch.exp2(scale)

        return dequantized.reshape(T, H, D).to(torch.bfloat16)


    @classmethod
    @torch.compile
    def quantize_tokenwise(cls, input: torch.Tensor, block_size: int = 32):
        T, H, D = input.shape
        device = input.device
        num_blocks = D // block_size
        assert D % block_size == 0, "D must be divisible by block_size"

        blocks = input.view(T, H, num_blocks, block_size).float()
        amax = blocks.abs().amax(dim=-1, keepdim=True)
        descale = amax / cls.E2M1_max
        scale = torch.ceil(torch.log2(torch.clamp(descale, min=1e-8)))
        scale_uint8 = (scale + 127).to(torch.uint8).squeeze(-1)

        scaled = blocks / torch.pow(2.0, scale)
        sign = torch.sign(scaled)
        sign_bit = ((sign < 0).to(torch.uint8) << 3)
        abs_scaled = scaled.abs().unsqueeze(-1)
        bounds = cls.E2M1_bounds.to(device)
        ord_ = (abs_scaled > bounds).sum(dim=-1).to(torch.uint8)
        fp4 = sign_bit + ord_

        left = fp4[..., 0::2]
        right = fp4[..., 1::2]
        q_bytes = (right << 4) + left
        q_bytes = q_bytes.reshape(T, H, D // 2)

        return q_bytes, scale_uint8

    @classmethod
    @torch.compile
    def dequantize_tokenwise(cls, q_bytes: torch.Tensor, e8m0_scale: torch.Tensor = None, *, scale_uint8: torch.Tensor = None, block_size: int = 32):
        """
        支援舊呼叫 e8m0_scale 或新呼叫 scale_uint8
        """
        if scale_uint8 is None:
            if e8m0_scale is None:
                raise ValueError("Must provide either scale_uint8 or e8m0_scale")
            scale_uint8 = e8m0_scale

        T, H, half_D = q_bytes.shape
        D = half_D * 2
        num_blocks = D // block_size
        device = q_bytes.device

        q_bytes_blocks = q_bytes.view(T, H, num_blocks, block_size // 2)
        left = q_bytes_blocks & 0x0F
        right = (q_bytes_blocks >> 4) & 0x0F
        fp4 = torch.empty((T, H, num_blocks, block_size), dtype=torch.uint8, device=device)
        fp4[..., 0::2] = left
        fp4[..., 1::2] = right

        sign = torch.where((fp4 >> 3) == 0, 1.0, -1.0)
        ord_ = fp4 & 0b111
        vals = torch.tensor(cls.E2M1_values, dtype=torch.float32, device=device)
        val = torch.take(vals, ord_.long())
        val = val * sign

        scale = scale_uint8.to(device).float().unsqueeze(-1) - 127.0
        dequantized = val * torch.pow(2.0, scale)

        return dequantized.reshape(T, H, D).to(torch.bfloat16)