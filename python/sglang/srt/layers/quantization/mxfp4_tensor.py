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
    # E2M1_max = 6.0

    # E2M1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    # E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])

    # E2M1_bounds = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float32)  # example
    # E2M1_max = 1.0  # example

    # 使用 E2M1: 1 sign | 2 exp | 1 mantissa
    # payload (exp<<1 | mant) in [0..7] maps to magnitude:
    # mag = (1 + mant*0.5) * 2**(exp - 1)
    # payload 0..7 => mags: [0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0]
    FP4_PAYLOAD_MAGS = torch.tensor([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    FP4_MAX_MAG = float(FP4_PAYLOAD_MAGS.max())  # 6.0

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
    def quantize_tokenwise(cls, x: torch.Tensor, block_size: int = 32):
        """
        Quantize batch tokens.
        Args:
            x: [T, H, D] float32, D must be divisible by block_size (here D=128)
            block_size: usually 32
        Returns:
            q_bytes: uint8 tensor [T, H, D//2]  (packed 2 FP4 per byte)
            scales: uint8 tensor [T, H, D//block_size]  (scale per block stored as 0..255 -> factor=val/255)
        """
        assert x.dtype == torch.float32 or x.dtype == torch.float16 or x.dtype == torch.bfloat16
        T, H, D = x.shape
        assert D % block_size == 0
        device = x.device
        n_blocks = D // block_size

        # reshape to [T,H,n_blocks,block_size]
        x_blocks = x.view(T, H, n_blocks, block_size).to(torch.float32)

        # compute amax per block -> scale_factor = amax / FP4_MAX_MAG
        amax = x_blocks.abs().amax(dim=-1, keepdim=True)  # [T,H,n_blocks,1]
        # avoid zero scale
        eps = 1e-12
        scale_factor = (amax / cls.FP4_MAX_MAG).clamp(min=eps)  # >= eps

        # normalized values
        x_norm = x_blocks / scale_factor  # target magnitude in approx [-FP4_MAX_MAG, FP4_MAX_MAG]

        # sign and abs
        sign_mask = (x_norm < 0)  # bool
        abs_norm = x_norm.abs()  # float

        # find nearest payload magnitude (vectorized)
        # FP4_PAYLOAD_MAGS shape [8] -> expand to [T,H,n_blocks,block_size,8] for distance calc
        mags = cls.FP4_PAYLOAD_MAGS.to(device)  # [8]
        # we compute distances: |abs_norm[...,None] - mags[None,...]|
        # abs_norm: [T,H,n_blocks,block_size]
        diff = torch.abs(abs_norm.unsqueeze(-1) - mags.view(1,1,1,1,-1))  # [...,8]
        idx = diff.argmin(dim=-1).to(torch.uint8)  # chosen payload 0..7; shape [T,H,n_blocks,block_size]

        # Compose 4-bit fp4: sign<<3 | payload
        sign_bit = sign_mask.to(torch.uint8) << 3  # 1 in high bit if negative
        fp4 = (sign_bit | idx)  # uint8, values 0..15

        # pack 2 fp4 into 1 byte: left = fp4[...,0::2], right = fp4[...,1::2]
        left = fp4[..., 0::2] & 0xF
        right = fp4[..., 1::2] & 0xF
        packed = (left | (right << 4)).to(torch.uint8)  # shape [T,H,n_blocks, block_size//2]

        # reshape packed to [T,H,D//2]
        q_bytes = packed.reshape(T, H, D // 2)

        # store scale_factor as uint8: scale_uint8 = round(scale_factor * 255)
        # scale_factor in [eps, +inf) but typically <=~1 (since amax <= scale*FP4_MAX)
        # clamp to [1,255] (reserve 0 if want but avoid zero)
        scale_uint8 = torch.clamp((scale_factor.squeeze(-1) * 255.0).round(), min=1, max=255).to(torch.uint8)  # [T,H,n_blocks]

        return q_bytes, scale_uint8

    @classmethod
    def dequantize_tokenwise(cls, q_bytes: torch.Tensor, scales: torch.Tensor, block_size: int = 32):
        """
        Batch dequantize.
        Args:
            q_bytes: uint8 [T,H,D//2]
            scales: uint8 [T,H,D//block_size]
        Returns:
            x_recon: float32 [T,H,D]
        """
        T, H, D2 = q_bytes.shape
        D = D2 * 2
        assert D % block_size == 0
        n_blocks = D // block_size
        device = q_bytes.device

        # reshape to [T,H,n_blocks, block_size//2]
        packed = q_bytes.view(T, H, n_blocks, block_size // 2)

        # unpack: left = packed & 0xF, right = packed >> 4
        left = packed & 0xF
        right = (packed >> 4) & 0xF
        # interleave: get shape [T,H,n_blocks, block_size]
        # we want order [v0_left, v0_right, v1_left, v1_right, ...] to match packing
        # left/right are both [..., block_size//2]
        # construct fp4 array
        fp4 = torch.empty((T, H, n_blocks, block_size), dtype=torch.uint8, device=device)
        fp4[..., 0::2] = left
        fp4[..., 1::2] = right

        # sign and payload
        sign = torch.where((fp4 >> 3) & 0x1 == 1, -1.0, 1.0).to(torch.float32)  # [T,H,n_blocks,block_size]
        payload = (fp4 & 0x7).long()  # 0..7

        # lookup payload magnitude
        mags = cls.FP4_PAYLOAD_MAGS.to(device)  # [8]
        # use gather-like: expand mags to match payload shape
        # easiest: take with flatten and reshape back
        mags_taken = torch.take(mags, payload)  # returns shape same as payload
        mags_taken = mags_taken.view(T, H, n_blocks, block_size).to(torch.float32)

        # reconstruct normalized values
        x_norm = mags_taken * sign  # still needs scale_factor multiply

        # reconstruct scale_factor
        scale_factor = (scales.float() / 255.0).unsqueeze(-1)  # [T,H,n_blocks,1]

        # apply scale
        x_blocks = x_norm * scale_factor  # [T,H,n_blocks,block_size]

        # reshape back to [T,H,D]
        x_recon = x_blocks.reshape(T, H, D)

        return x_recon