import functools
import hashlib
import logging
import pickle
import threading
import time
from queue import Empty, PriorityQueue, Queue
from typing import TYPE_CHECKING, Iterable, List, Optional

import torch

from sglang.srt.layers.moe.token_dispatcher.deepep import use_deepep
from sglang.srt.managers.cache_controller import (
    CacheOperation,
    HiCacheController,
    LayerDoneCounter,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.eic_memory_pool import EICBaseTokenToKVPoolHost
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def get_content_hash(
    content: Iterable, page_size: int, prev_hash: Optional[int] = None
) -> List[int]:
    """
    Get the hash of the content.
    """

    def hash_func(input):
        input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
        return int.from_bytes(hashlib.sha256(input_bytes).digest(), byteorder="big")

    if prev_hash is None:
        prev_hash = 0
    result = []
    for i in range(len(content) // page_size):
        page = content[i * page_size : (i + 1) * page_size]
        page_hash = hash_func((prev_hash, page))
        prev_hash = page_hash
        result.append(page_hash)
    return result


class EICCacheOperation(CacheOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        content_hash: Optional[List[int]] = None,
        priority: Optional[int] = None,
    ):
        self.content_hash = content_hash
        super().__init__(
            host_indices=host_indices,
            device_indices=device_indices,
            node_id=node_id,
            priority=priority,
        )


class EICCacheController(HiCacheController):
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: EICBaseTokenToKVPoolHost,
        page_size: int,
        load_cache_event: threading.Event = None,
        write_policy: str = "write_through",
        server_args: Optional[ServerArgs] = None,
    ):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.load_cache_event = load_cache_event
        self.layer_done_counter = LayerDoneCounter(self.mem_pool_device.layer_num)
        self.server_args = server_args
        self.disable_shared = server_args.disable_eic_shared

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        self.write_queue = Queue()
        self.load_queue = Queue()

        self.ack_write_queue = Queue()
        self.ack_load_queue = Queue()

        self.stop_event = threading.Event()
        self.write_wait_event = threading.Event()
        self.load_wait_event = threading.Event()

        self.write_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()
        self.device = token_to_kv_pool_allocator.device

        # synchronize for write or load operation
        self.scheduler_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.scheduler_stream.synchronize = lambda: None
        self.sync_before_write = self.need_sync_before_write()
        if server_args.enable_deepep_moe:
            self.hook_model_forward()
            self.hook_deepep_dispatch()

        self.write_parallel = 1
        self.load_parallel = 1

        self.write_thread_pool = [
            threading.Thread(target=self.write_thread_func_direct, daemon=True)
            for _ in range(self.write_parallel)
        ]
        self.load_thread_pool = [
            threading.Thread(target=self.load_thread_func_direct, daemon=True)
            for _ in range(self.load_parallel)
        ]

        for th in self.write_thread_pool:
            th.start()
        for th in self.load_thread_pool:
            th.start()

    def reset(self):
        self.stop_event.set()

        for th in self.write_thread_pool:
            th.join()
        for th in self.load_thread_pool:
            th.join()

        self.write_queue.queue.clear()
        self.load_queue.queue.clear()
        self.ack_write_queue.queue.clear()
        self.ack_load_queue.queue.clear()

        self.write_thread_pool = [
            threading.Thread(target=self.write_thread_func_direct, daemon=True)
            for _ in range(self.write_parallel)
        ]
        self.load_thread_pool = [
            threading.Thread(target=self.load_thread_func_direct, daemon=True)
            for _ in range(self.load_parallel)
        ]

        self.stop_event.clear()
        self.write_wait_event.clear()
        self.load_wait_event.clear()

        for th in self.write_thread_pool:
            th.start()
        for th in self.load_thread_pool:
            th.start()

    def hook_empty_cache(self):
        """
        Safely empty the CUDA stream.
        """
        original_empty_cache = torch.cuda.empty_cache

        def safe_empty_cache():
            self.write_wait_event.set()
            write_stream = self.write_stream
            write_event = torch.cuda.Event()
            write_event.record(write_stream)
            write_event.synchronize()
            original_empty_cache()
            self.write_wait_event.clear()

        torch.cuda.empty_cache = safe_empty_cache

    def hook_model_forward(self):
        """
        Hook the model forward to synchronize the write stream.
        """
        from sglang.srt.model_executor.model_runner import ModelRunner

        original_forward = ModelRunner.forward

        def synced_forward(*args, **kwargs):
            self.write_wait_event.set()
            self.write_stream.synchronize()
            result = original_forward(*args, **kwargs)
            self.write_wait_event.clear()
            return result

        ModelRunner.forward = synced_forward
        logger.info("Hooked model forward with synchronized write stream.")

    def hook_deepep_dispatch(self):
        if not use_deepep:
            return
        from deep_ep import Buffer

        original_dispatch = Buffer.internode_dispatch

        def syned_dispatch(*args, **kwargs):
            while self.load_wait_event.is_set():
                time.sleep(0.1)
            result = original_dispatch(*args, **kwargs)
            return result

        Buffer.internode_dispatch = syned_dispatch
        logger.info("Hooked deepep dispatch with synchronized load.")

    def need_sync_before_write(self):
        return (
            self.write_policy == "write_through"
            and not self.server_args.disable_overlap_schedule
        )

    def write_operation_exclusive(self, operation: EICCacheOperation):
        """
        Write the KV cache to host memory.
        """
        if operation.data is None:
            operation.data = self.mem_pool_device.get_flat_data(
                operation.device_indices
            )
        ret = self.mem_pool_host.transfer(operation.host_indices, operation.data)
        if not ret:
            logger.error(f"Failed to write to host memory {operation.node_ids}")
            self.mem_pool_host.free(operation.host_indices)
            for node_id in operation.node_ids:
                if node_id != 0:
                    self.ack_write_queue.put((node_id, False))
            return
        self.mem_pool_host.complete_io(operation.host_indices)
        for node_id in operation.node_ids:
            if node_id != 0:
                self.ack_write_queue.put((node_id, True))

    def load_operation_exclusive(self, operation: EICCacheOperation):
        """
        Load the KV cache from host memory to device memory.
        """
        operation.data, mask = self.mem_pool_host.get_flat_data(operation.host_indices)
        if operation.data is None:
            logger.error(f"Failed to load from host memory {operation.node_ids}")
            for node_id in operation.node_ids:
                if node_id != 0:
                    self.ack_load_queue.put((node_id, False))
            return
        self.mem_pool_device.transfer(operation.device_indices, operation.data)
        self.mem_pool_host.complete_io(operation.host_indices)
        for node_id in operation.node_ids:
            if node_id != 0:
                self.ack_load_queue.put((node_id, True))

    def write_operation_shared(self, operation: EICCacheOperation):
        """
        Write the KV cache to host memory.
        """
        assert len(operation.host_indices) == self.page_size * len(
            operation.content_hash
        )
        while self.write_wait_event.is_set():
            time.sleep(0.01)
        logger.debug(f"write device indices: {operation.device_indices}")
        if operation.data is None:
            operation.data = self.mem_pool_device.get_flat_data(
                operation.device_indices
            )
        ret = self.mem_pool_host.assign_page_data(
            operation.content_hash, operation.data
        )
        if not ret:
            logger.error(f"Failed to write to host memory {operation.node_ids}")
            for node_id in operation.node_ids:
                if node_id != 0:
                    self.ack_write_queue.put((node_id, False))
            return
        for node_id in operation.node_ids:
            if node_id != 0:
                self.ack_write_queue.put((node_id, True))

    def load_operation_shared(self, operation: EICCacheOperation):
        """
        Load the KV cache from host memory to device memory.
        """
        assert len(operation.host_indices) == self.page_size * len(
            operation.content_hash
        )
        operation.data, mask = self.mem_pool_host.get_page_data(operation.content_hash)
        if operation.data is None:
            logger.error(f"Failed to load from host memory {operation.node_ids}")
            for node_id in operation.node_ids:
                if node_id != 0:
                    self.ack_load_queue.put((node_id, False))
            return
        self.mem_pool_device.transfer(operation.device_indices, operation.data)
        for node_id in operation.node_ids:
            if node_id != 0:
                self.ack_load_queue.put((node_id, True))

    def write_to_eic(self, operation: EICCacheOperation):
        """
        Write the KV cache to host memory.
        """
        if self.disable_shared:
            self.write_operation_exclusive(operation)
        else:
            self.write_operation_shared(operation)

    def load_from_eic(self, operation: EICCacheOperation):
        """
        Load the KV cache from host memory to device memory.
        """
        if self.disable_shared:
            self.load_operation_exclusive(operation)
        else:
            self.load_operation_shared(operation)

    def write_thread_func_direct(self):
        """
        Directly write through KV caches to host memory without buffering.
        """
        with torch.cuda.stream(self.write_stream):
            while not self.stop_event.is_set():
                logger.debug("write thread eventloop running")
                try:
                    operation = self.write_queue.get(block=True, timeout=1)
                    if self.sync_before_write:
                        self.scheduler_stream.synchronize()
                    self.write_to_eic(operation)
                except Empty:
                    continue
                except Exception as e:
                    logger.exception("Exception in write thread: %s", e)
                    from sglang.srt.utils import pyspy_dump_schedulers

                    pyspy_dump_schedulers()

    def load_thread_func_direct(self):
        """
        Directly load KV caches from host memory to device memory without buffering.
        """
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(self.load_stream):
            while not self.stop_event.is_set():
                logger.debug("load thread eventloop running")
                # self.load_cache_event.wait(timeout=1)
                # if not self.load_cache_event.is_set():
                #     continue
                # self.load_cache_event.clear()
                try:
                    operation = self.load_queue.get(block=True, timeout=1)
                    self.load_wait_event.set()
                    self.load_from_eic(operation)
                    self.load_wait_event.clear()
                except Empty:
                    continue
                except Exception as e:
                    logger.exception("Exception in load thread: %s", e)
                    from sglang.srt.utils import pyspy_dump_schedulers

                    pyspy_dump_schedulers()

    def host_allocate(self, size):
        """
        Allocate memory on the host.
        """
        return self.mem_pool_host.alloc(size)

    def find_longest_prefix_in_eic(self, prompt, prev_hash=None):
        """
        Find the longest prefix in the EIC cache.
        """
        if len(prompt) == 0:
            return [], []
        content_hash = get_content_hash(prompt, self.page_size, prev_hash)
        exist_result = self.mem_pool_host.exist_page(content_hash)
        return exist_result, prompt[: len(exist_result) * self.page_size]

    def batch_find_longest_prefix_in_eic(self, prompts, prev_hashes):
        assert len(prompts) == len(
            prev_hashes
        ), "prompts and prev_hashes must have the same length"
        prompt_key_offset = [
            0,
        ]
        eic_prefix_len = []
        content_hashes = []
        for i in range(len(prompts)):
            content_hash = get_content_hash(prompts[i], self.page_size, prev_hashes[i])
            prompt_key_offset.append(len(content_hash) + prompt_key_offset[-1])
            content_hashes.extend(content_hash)
        if len(content_hashes) == 0:
            return []
        exist_result = self.mem_pool_host.batch_exist_page(content_hashes)
        for i in range(len(prompts)):
            eic_prefix_len.append(
                exist_result[prompt_key_offset[i] : prompt_key_offset[i + 1]].count(
                    True
                )
                * self.page_size
            )

        return eic_prefix_len

    def write_page(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = 0,
        content_hash: List[int] = None,
        data_copy: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = device_indices.clone().cpu()
        cache_operation = EICCacheOperation(
            host_indices, device_indices, node_id, content_hash, priority
        )
        if data_copy:
            cache_operation.data = self.mem_pool_device.get_flat_data(device_indices)
        self.write_queue.put(cache_operation)
        return host_indices

    def load_page(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = 0,
        content_hash: List[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        # to ensure the device indices are ready before accessed by another CUDA stream
        torch.cuda.current_stream().synchronize()
        self.load_queue.put(
            EICCacheOperation(
                host_indices, device_indices, node_id, content_hash, priority
            )
        )
        return device_indices

    def evict_device(
        self, device_indices: torch.Tensor, host_indices: torch.Tensor
    ) -> int:
        if self.disable_shared and not self.mem_pool_host.is_synced(host_indices):
            raise ValueError(
                f"Inconsistent states: {self.mem_pool_host.get_state(host_indices)}"
            )
        self.mem_pool_device_allocator.free(device_indices)
        self.mem_pool_host.update_backup(host_indices)
        return len(device_indices)
