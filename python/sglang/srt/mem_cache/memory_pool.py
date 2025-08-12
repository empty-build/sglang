"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

import abc
import logging
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import get_bool_env_var, is_cuda, next_power_of_2

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()

from sglang.jack_utils import hcdprint
# from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
USE_KV4_CUDA = 0

class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (size, max_context_len), dtype=torch.int32, device=device
            )

        self.free_slots = list(range(size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))


class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError()


# Quant: bf16 → packed uint8 (fp4)
@triton.jit
def _quant_fp4_kernel(x_ptr, out_ptr, n_elements, table_ptr):
    pid = tl.program_id(0)
    idx = pid * 2
    if idx + 1 < n_elements:
        v0 = tl.cast(tl.load(x_ptr + idx), tl.float32)
        v1 = tl.cast(tl.load(x_ptr + idx + 1), tl.float32)

        v0 = tl.maximum(tl.minimum(v0, 6.0), -6.0)
        v1 = tl.maximum(tl.minimum(v1, 6.0), -6.0)

        table = tl.load(table_ptr + tl.arange(0, 16))
        diff0 = tl.abs(v0 - table)
        diff1 = tl.abs(v1 - table)
        idx0 = tl.argmin(diff0, axis=0).to(tl.uint8)
        idx1 = tl.argmin(diff1, axis=0).to(tl.uint8)

        packed = (idx0 & 0x0F) | ((idx1 & 0x0F) << 4)
        tl.store(out_ptr + pid, packed)

# Dequant: packed uint8 (fp4) → bf16
@triton.jit
def _dequant_fp4_kernel(x_ptr, out_ptr, n_elements, table_ptr):
    pid = tl.program_id(0)
    offsets = pid + tl.arange(0, 1)  # 生成长度为 1 的 vector
    mask = offsets < n_elements
    byte_val = tl.load(x_ptr + offsets, mask=mask, other=0)[0]

    idx0 = byte_val & 0x0F
    idx1 = (byte_val >> 4) & 0x0F

    v0 = tl.cast(tl.load(table_ptr + idx0), tl.bfloat16)
    v1 = tl.cast(tl.load(table_ptr + idx1), tl.bfloat16)

    out_idx = pid * 2
    tl.store(out_ptr + out_idx, v0)
    tl.store(out_ptr + out_idx + 1, v1)



class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim

        # for disagg with nvlink
        self.enable_custom_mem_pool = get_bool_env_var(
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "false"
        )
        if self.enable_custom_mem_pool:
            # TODO(shangming): abstract custom allocator class for more backends
            from mooncake.allocator import NVLinkAllocator

            allocator = NVLinkAllocator.get_allocator(self.device)
            self.custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
        else:
            self.custom_mem_pool = None

        self._create_buffers()

        self.layer_transfer_counter = None
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
        )
        self.mem_usage = (k_size + v_size) / GB


        if not USE_KV4_CUDA:
            # [horenc]
            # 預建 FP4 lookup table # FP4 值（已排序）與 thresholds（相鄰中點）
            hcdprint(f"[horenc] init FP4_VALUES table in class MHATokenToKVPool:__init__()")

            self._max_elems = 10_000_000_00  # 這裡你可以改成你模型的最大 tensor 大小

            self.FP4_VALUES = torch.tensor([
                    -6.0, -3.0, -2.0, -1.0,
                    -0.5, -0.25, -0.125, -0.0,
                    0.0,  0.125, 0.25, 0.5,
                    1.0, 2.0, 3.0, 6.0
            ], dtype=torch.float32, device=self.device) # set device for CUDA graph
            # # Perf pre-do: thresholds 長度為 15：介於相鄰兩個 quant level 的中點
            # fp = self.FP4_VALUES
            # self.FP4_THRESHOLDS = ((fp[:-1] + fp[1:]) / 2.0).to(self.device)  # shape [15]
            
            # self.FP4_THRESHOLDS = ((self.FP4_VALUES[:-1] + self.FP4_VALUES[1:]) / 2.0)
            self.FP4_VALUES_expand = self.FP4_VALUES.unsqueeze(0)  # [1,16]

            
            # # 預分配 buffer
            # self._indices_buf = torch.empty(self._max_elems, dtype=torch.uint8, device=device)
            # self._packed_buf = torch.empty((self._max_elems + 1) // 2, dtype=torch.uint8, device=device)
            # self._dequant_buf = torch.empty(self._max_elems, dtype=torch.float32, device=device)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                hcdprint(f"[horenc] class MHATokenToKVPool:_create_buffers(): TODO WIP")
                hcdprint(f"[horenc] class MHATokenToKVPool:_create_buffers(): self.store_dtype = {self.store_dtype}")
                if self.store_dtype == torch.float4_e2m1fn_x2:
                    if USE_KV4_CUDA:
                        # hcdprint(f"[horenc] class MHATokenToKVPool:_create_buffers(): Jack hack - "
                        #     f"Milestone2 layout")
                        # special layout: 1D kv buffer
                        m = self.size + self.page_size
                        n = self.head_num
                        k = self.head_dim
                        # TODO1: assert % 32 = 0
                        # TODO2: change 32 to var
                        buffer_size = ((m * n * k) // 2) + ((m * n * k) // 32) + (1 + (8 * 3))
                        print(f"[horenc] class MHATokenToKVPool:_create_buffers(): Jack hack - "
                            f"Milestone2 layout: m={m}, n={n}, k={k}, per-layer buffer_size = {buffer_size}")
                        self.k_buffer = [
                            torch.zeros(
                                buffer_size, # 1D
                                dtype=torch.uint8, 
                                device=self.device,
                            )
                            for _ in range(self.layer_num)
                        ]
                        self.v_buffer = [
                            torch.zeros(
                                buffer_size, # 1D
                                dtype=torch.uint8,
                                device=self.device,
                            )
                            for _ in range(self.layer_num)
                        ]
                    else:
                        hcdprint(f"[horenc] class MHATokenToKVPool:_create_buffers(): Jack hack - "
                                f"force torch.uint8 + dividiveby2 for matching fp4 shape[42, 8, 64] (ori:[42, 8, 128])")
                        self.k_buffer = [
                            torch.zeros( # TODO change this 3D to 1D
                                (self.size + self.page_size, self.head_num, self.head_dim // 2), # 3D
                                dtype=torch.uint8,
                                device=self.device,
                            )
                            # TODO: Change to something like this torch.zeros( MNK),  dtype=torch.uint8,device=self.device,)
                            for _ in range(self.layer_num)
                        ]
                            # loc now = 3D, TODO: change it to 1D
                            # [layer_id][loc] => [loc]==[ self.size + self.page_size, 2D(head), 3D(dim)]
                            # self.size + self.page_size = M
                            # self.head_num = N
                            # self.head_dim = K
                            # Use yichen's formula to calculate 1D's k/v buffer size
                        self.v_buffer = [
                            torch.zeros(
                                (self.size + self.page_size, self.head_num, self.head_dim // 2),
                                dtype=torch.uint8,
                                device=self.device,
                            )
                            for _ in range(self.layer_num)
                        ]

                else: # origin
                    # [size, head_num, head_dim] for each layer
                    # The padded slot 0 is used for writing dummy outputs from padded tokens.
                    self.k_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        torch.cuda.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # hcdprint(f"[horenc] going to hack MHATokenToKVPool get_key_buffer()")
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != torch.float4_e2m1fn_x2:
            # ori
            return self._get_key_buffer(layer_id)
        else:
            if USE_KV4_CUDA:
                from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
                # cache_k = MXFP4QuantizeUtil.quantize_packed(self._get_key_buffer(layer_id), 32)
                print(f"\t [horenc]   layer_id = {layer_id}")
                print(f"\t [horenc]   bf-Dequantized k shape: {self._get_key_buffer(layer_id).shape}")
                print(f"\t [horenc]   bf-Dequantized k dtype: {self._get_key_buffer(layer_id).dtype}")
                dequantized_cache_k = MXFP4QuantizeUtil.dequantize_packed(
                                            quantized_data=self._get_key_buffer(layer_id),
                                            dtype=torch.bfloat16,
                                            block_sizes=[32]
                                        )
                print(f"\t [horenc]   af-Dequantized k shape {dequantized_cache_k.shape}")
                print(f"\t [horenc]   af-Dequantized k dtype: {dequantized_cache_k.dtype}")
                return dequantized_cache_k
            else:
                # # new - horenc 250806
                return self.dequantize_fp4_e2m1_to_bf16(self._get_key_buffer(layer_id))

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        # hcdprint(f"[horenc] going to hack MHATokenToKVPool get_value_buffer()")
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        if self.store_dtype != torch.float4_e2m1fn_x2:
            # ori
            return self._get_value_buffer(layer_id)
        else:
            if USE_KV4_CUDA:
                from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
                # cache_k = MXFP4QuantizeUtil.quantize_packed(self._get_value_buffer(layer_id), 32)
                print(f"\t [horenc]   layer_id = {layer_id}")
                print(f"\t [horenc]   bf-Dequantized v shape: {self._get_value_buffer(layer_id).shape}")
                print(f"\t [horenc]   bf-Dequantized v dtype: {self._get_value_buffer(layer_id).dtype}")
                dequantized_cache_v = MXFP4QuantizeUtil.dequantize_packed(
                                            quantized_data=self._get_value_buffer(layer_id),
                                            dtype=torch.bfloat16,
                                            block_sizes=[32]
                                        )
                print(f"\t [horenc]   af-Dequantized v shape: {dequantized_cache_v.shape}")
                print(f"\t [horenc]   af-Dequantized v dtype: {dequantized_cache_v.dtype}")
                return dequantized_cache_v
            else:
                # # new - horenc 250806
                return self.dequantize_fp4_e2m1_to_bf16(self._get_value_buffer(layer_id))

    def get_kv_buffer(self, layer_id: int):
        # hcdprint(f"[horenc] MHATokenToKVPool:get_kv_buffer() some attention do not enter here like torch_native_backend")
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)


    # # ----------------------
    # # Optimize - pytorch
    # # ----------------------
    @torch.inference_mode()
    def quantize_bf16_to_fp4_e2m1(self, tensor_bf16: torch.Tensor) -> torch.Tensor:
        shape = tensor_bf16.shape
        last_dim = shape[-1]
        assert last_dim % 2 == 0, "最後維度必須是偶數"

        tensor_f32 = tensor_bf16.float().clamp(-6.0, 6.0)
        tensor_2d = tensor_f32.view(-1, last_dim)  # [N, last_dim]

        # 距離計算
        distances = torch.abs(tensor_2d.unsqueeze(-1) - self.FP4_VALUES_expand)  # [N, last_dim, 16]
        indices = distances.argmin(dim=-1).to(torch.uint8)  # [N, last_dim]

        # pack 兩個4bit成1byte, 避免 concat, 直接用切片操作
        # [N, last_dim//2], 直接取奇偶位組合
        packed_uint8 = (indices[:, 0::2] & 0x0F) | ((indices[:, 1::2] & 0x0F) << 4)

        new_shape = list(shape[:-1]) + [last_dim // 2]
        return packed_uint8.contiguous().view(*new_shape)

    @torch.inference_mode()
    def dequantize_fp4_e2m1_to_bf16(self, tensor_uint8: torch.Tensor) -> torch.Tensor:
        shape = tensor_uint8.shape
        last_dim = shape[-1]

        tensor_2d = tensor_uint8.view(-1, last_dim)  # [N, last_dim]

        low_nibble = tensor_2d & 0x0F  # [N, last_dim]
        high_nibble = (tensor_2d >> 4) & 0x0F  # [N, last_dim]

        # 利用拼接的新維度，交錯合併 (interleave) 兩個 nibble
        # 預先建立空 tensor, 並利用索引賦值避免 concat 和 stack 開銷
        N = tensor_2d.shape[0]
        out_len = last_dim * 2
        out_indices = torch.empty((N, out_len), dtype=torch.long, device=tensor_uint8.device)

        out_indices[:, 0::2] = low_nibble
        out_indices[:, 1::2] = high_nibble

        values = self.FP4_VALUES[out_indices]  # [N, last_dim*2]

        new_shape = list(shape[:-1]) + [last_dim * 2]
        return values.contiguous().view(*new_shape).to(torch.bfloat16)


    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        hcdprint(f"[horenc] MHATokenToKVPool cache_k.dtype = {cache_k.dtype}")
        hcdprint(f"[horenc] MHATokenToKVPool self.dtype = {self.dtype}")
        hcdprint(f"[horenc] MHATokenToKVPool k_scale = {k_scale}")
        # print(f"\t [horenc]  Quantized k/v check")
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            if self.dtype != torch.float4_e2m1fn_x2:
                # origin
                cache_k = cache_k.to(self.dtype)
                cache_v = cache_v.to(self.dtype)
            else:
                # new - yichen's quant
                # hcdprint(f"[horenc] hack .to -> scaled_fp4_quant()")
                # hcdprint(f"[horenc] cache_k.shape: {list(cache_k.shape)}")
                if USE_KV4_CUDA:
                    from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
                    print(f"\t [horenc]   layer_id ({layer_id}) - self.start_layer ({self.start_layer})  = {layer_id - self.start_layer}, loc = {loc}")
                    print(f"\t [horenc]   bf-Quantize k/v shape: {cache_k.shape}")
                    print(f"\t [horenc]   bf-Quantize k/v dtype: {cache_k.dtype}")
                    cache_k = MXFP4QuantizeUtil.quantize_packed(cache_k, 32)
                    cache_v = MXFP4QuantizeUtil.quantize_packed(cache_v, 32)
                    print(f"\t [horenc]   af-Quantized k/v shape: {cache_k.shape}")
                    print(f"\t [horenc]   af-Quantized k/v dtype: {cache_k.dtype}")
                    print(f"\t [horenc]   self.k_buffer[layer_id - self.start_layer].shape = {self.k_buffer[layer_id - self.start_layer].shape}")
                    print(f"\t [horenc]   self.k_buffer[layer_id - self.start_layer][loc].shape = {self.k_buffer[layer_id - self.start_layer][loc].shape}")
                else:
                    # new 250809 - working
                    cache_k = self.quantize_bf16_to_fp4_e2m1(cache_k)
                    cache_v = self.quantize_bf16_to_fp4_e2m1(cache_v)
         
        # print("[debug] cache_k.shape =", cache_k.shape)
        # print("[debug] target shape =", self.k_buffer[layer_id - self.start_layer][loc].shape)
            
        hcdprint(f"[horenc] self.store_dtype = {self.store_dtype}")
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v


    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        copy_all_layer_kv_cache[(len(self.data_ptrs),)](
            self.data_ptrs,
            self.data_strides,
            tgt_loc,
            src_loc,
            len(tgt_loc),
            next_power_of_2(len(tgt_loc)),
        )


class SWAKVPool(KVCache):
    """KV cache with separate pools for full and SWA attention layers."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
    ):
        self.size = size
        self.size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = 1
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose
        TokenToKVPoolClass = MHATokenToKVPool
        self.swa_kv_pool = TokenToKVPoolClass(
            size=size_swa,
            page_size=self.page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=self.swa_layer_nums,
            device=device,
            enable_memory_saver=False,
        )
        self.full_kv_pool = TokenToKVPoolClass(
            size=size,
            page_size=self.page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=self.full_layer_nums,
            device=device,
            enable_memory_saver=False,
        )
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        k_size, v_size = self.full_kv_pool.get_kv_size_bytes()
        k_size_swa, v_size_swa = self.swa_kv_pool.get_kv_size_bytes()
        return k_size + k_size_swa, v_size + v_size_swa

    def get_contiguous_buf_infos(self):
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = (
            self.full_kv_pool.get_contiguous_buf_infos()
        )
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = (
            self.swa_kv_pool.get_contiguous_buf_infos()
        )

        kv_data_ptrs = full_kv_data_ptrs + swa_kv_data_ptrs
        kv_data_lens = full_kv_data_lens + swa_kv_data_lens
        kv_item_lens = full_kv_item_lens + swa_kv_item_lens

        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_key_buffer(layer_id_pool)

    def get_value_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_value_buffer(layer_id_pool)

    def get_kv_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):

        layer_id = layer.layer_id
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            if self.full_to_swa_index_mapping is not None:
                loc = self.translate_loc_from_full_to_swa(loc)
            self.swa_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )
        else:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )


class AscendTokenToKVPool(MHATokenToKVPool):

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # Continuous memory improves the efficiency of Ascend`s transmission backend,
            # while other backends remain unchanged.
            self.kv_buffer = torch.zeros(
                (
                    2,
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.k_buffer = self.kv_buffer[0]
            self.v_buffer = self.kv_buffer[1]

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self.get_key_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        import torch_npu

        torch_npu._npu_reshape_and_cache(
            key=cache_k,
            value=cache_v,
            key_cache=self.k_buffer[layer_id].view(
                -1, self.page_size, self.head_num, self.head_dim
            ),
            value_cache=self.v_buffer[layer_id].view(
                -1, self.page_size, self.head_num, self.head_dim
            ),
            slot_indices=loc,
        )


@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    else:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )

    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        # for disagg with nvlink
        self.enable_custom_mem_pool = get_bool_env_var(
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "false"
        )
        if self.enable_custom_mem_pool:
            # TODO(shangming): abstract custom allocator class for more backends
            from mooncake.allocator import NVLinkAllocator

            allocator = NVLinkAllocator.get_allocator(self.device)
            self.custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
        else:
            self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.kv_buffer = [
                    torch.zeros(
                        (size + page_size, 1, kv_lora_rank + qk_rope_head_dim),
                        dtype=self.store_dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.layer_transfer_counter = None

        kv_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, KV size: {kv_size / GB:.2f} GB"
        )
        self.mem_usage = kv_size / GB

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += np.prod(kv_cache.shape) * kv_cache.dtype.itemsize
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k_nope.dtype != self.dtype:
            cache_k_nope = cache_k_nope.to(self.dtype)
            cache_k_rope = cache_k_rope.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k_nope = cache_k_nope.view(self.store_dtype)
            cache_k_rope = cache_k_rope.view(self.store_dtype)

        set_mla_kv_buffer_triton(
            self.kv_buffer[layer_id], loc, cache_k_nope, cache_k_rope
        )

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        torch.cuda.synchronize()


class AscendMLAPagedTokenToKVPool(MLATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super(MLATokenToKVPool, self).__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.kv_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )

        self.layer_transfer_counter = None

        kv_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, KV size: {kv_size / GB:.2f} GB"
        )
        self.mem_usage = kv_size / GB

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [self.kv_buffer[i][0].nbytes for i in range(self.layer_num)]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(store_dtype)

        import torch_npu

        torch_npu._npu_reshape_and_cache_siso(
            key=cache_k.view(-1, 1, self.kv_lora_rank + self.qk_rope_head_dim),
            key_cache=self.kv_buffer[layer_id - self.start_layer].view(
                -1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim
            ),
            slot_indices=loc,
        )


class DoubleSparseTokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        heavy_channel_num: int,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            self.k_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (size + page_size, head_num, head_dim), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

            # [size, head_num, heavy_channel_num] for each layer
            self.label_buffer = [
                torch.zeros(
                    (size + 1, head_num, heavy_channel_num), dtype=dtype, device=device
                )
                for _ in range(layer_num)
            ]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_label_buffer(self, layer_id: int):
        return self.label_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_label: torch.Tensor,
    ):
        # NOTE(Andy): ignore the dtype check
        layer_id = layer.layer_id
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v
        self.label_buffer[layer_id - self.start_layer][loc] = cache_label


@triton.jit
def copy_all_layer_kv_cache(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128

    bid = tl.program_id(0)
    stride = tl.load(strides + bid)

    data_ptr = tl.load(data_ptrs + bid)
    data_ptr = tl.cast(data_ptr, tl.pointer_type(tl.uint8))

    num_locs_offset = tl.arange(0, num_locs_upper)
    tgt_locs = tl.load(tgt_loc_ptr + num_locs_offset, mask=num_locs_offset < num_locs)
    src_locs = tl.load(src_loc_ptr + num_locs_offset, mask=num_locs_offset < num_locs)

    # NOTE: we cannot parallelize over the tgt_loc_ptr dim with cuda blocks
    # because this copy is an inplace operation.

    num_loop = tl.cdiv(stride, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = (num_locs_offset < num_locs)[:, None] and (copy_offset < stride)[None, :]
        value = tl.load(
            data_ptr + src_locs[:, None] * stride + copy_offset[None, :], mask=mask
        )
        tl.store(
            data_ptr + tgt_locs[:, None] * stride + copy_offset[None, :],
            value,
            mask=mask,
        )
