# SPDX-License-Identifier: Apache-2.0

import torch
import torch.utils.benchmark as benchmark
# from benchmark_shapes import WEIGHT_SHAPES_MOE

from vllm.utils import FlexibleArgumentParser
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.moe.ep_moe.kernels import (
    pre_reorder_triton_kernel,
    pre_reorder_triton_kernel_for_cutlass_moe,
    run_moe_ep_preproess,
    run_cutlass_moe_ep_preproess,
)
from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe

import time
import os
from typing import Callable, List, Optional, Dict

# DEFAULT_MODELS = [
#   "ibm-granite/granite-3.0-3b-a800m"
# ]
# DEFAULT_BATCH_SIZES = [1,2,4,8,16,32,64,128,256,512,1024,2048]
# DEFAULT_BATCH_SIZES = [24,32,64]
# DEFAULT_BATCH_SIZES = [8,16,32,64,128,512]
# DEFAULT_BATCH_SIZES = [2,4,8,16,32,64,128,256,512,1024,2048]
DEFAULT_BATCH_SIZES = [4, 8, 16, 32, 64, 128,  256, 512,1024, 2048, 4096, 8192, 16384, 32768]
# DEFAULT_TP_SIZES = [1]

# PER_ACT_TOKEN_OPTS = [False]
# PER_OUT_CH_OPTS = [False]


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)


def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError(
            "int4_values_interleaved 的最后一个维度的大小必须是偶数。"
        )

    input_tensor_int8 = int4_values_interleaved.to(torch.int8)

    low_nibbles = input_tensor_int8[..., 0::2]
    high_nibbles = input_tensor_int8[..., 1::2]

    packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)
    
    return packed_tensor.to(torch.int8)


def pack_interleave(num_experts, ref_weight, ref_scale):
    n, k = ref_weight.shape[1], ref_weight.shape[2]
    # packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4

    # weight = packer(ref_weight.cpu()).cuda()
    weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8)
    # w_q = w_q.contiguous().transpose(1, 2)
    w_q = w_q.contiguous()

    ###############################################################
    # scale interleave, [E, K, N]
    scale = ref_scale.permute(0, 2, 1)  # [E, N, K]
    # scale = ref_scale
    scale_interleaved = scale.reshape(
        scale.shape[0], scale.shape[1], (scale.shape[2] // 4), 4
    )  # [E, N, K/4, 4]
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)  # [E, K/4, N, 4]
    scale_interleaved = scale_interleaved.reshape(
        scale.shape[0], scale.shape[2] // 4, scale.shape[1] * 4
    )  # [E, K/4, N*4]
    w_scale = scale_interleaved.contiguous()

    return w_q, w_scale


def bench_run(results: list[benchmark.Measurement], model: str,
              num_experts: int, topk: int, mkn: tuple[int, int, int]):
    label = "Quant Matmul"

    sub_label = (
        "{}, num_experts={}, topk={}, MKN=({})".format(model, num_experts, topk, mkn))

    print(f"Testing: {sub_label}")
    # 64 7168
    (m, k, n) = mkn

    dtype = torch.bfloat16
    device = "cuda"

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randint(-8, 8, (num_experts, n * 2, k), dtype=torch.int8, device="cuda")
    w2 = torch.randint(-8, 8, (num_experts, k, n), dtype=torch.int8, device="cuda")
    a1_scale = torch.randn(1, dtype=torch.float32, device="cuda")
    a2_scale = torch.randn(1, dtype=torch.float32, device="cuda")
    affine_coeff = 0.005
    scale_1 = torch.randn(
            num_experts, k // 128, n * 2, dtype=dtype,
            device="cuda") * affine_coeff
    scale_2 = torch.randn(
        num_experts, n // 128, k, dtype=dtype,
        device="cuda") * affine_coeff
    w1_q, w1_scale = pack_interleave(num_experts, w1, scale_1)
    w2_q, w2_scale = pack_interleave(num_experts, w2, scale_2)

    a_strides1 = torch.full((num_experts, 3),
                                    k,
                                    device=device,
                                    dtype=torch.int64)
    c_strides1 = torch.full((num_experts, 3),
                                    2 * n,
                                    device=device,
                                    dtype=torch.int64)
    a_strides2 = torch.full((num_experts, 3),
                                    n,
                                    device=device,
                                    dtype=torch.int64)
    c_strides2 = torch.full((num_experts, 3),
                                    k,
                                    device=device,
                                    dtype=torch.int64)
    b_strides1 = a_strides1
    s_strides13 = c_strides1
    b_strides2 = a_strides2
    s_strides2 = c_strides2

    E = 256
    expert_map = torch.arange(E, dtype=torch.int32, device="cuda")
    expert_map[num_experts:] = E

    score = torch.randn((m, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = select_experts(
        hidden_states=a,
        router_logits=score,
        top_k=topk,
        use_grouped_topk=False,
        renormalize=False,
    )
    # print(".......tok_ids {}.......".format(topk_ids))

    def run_cutlass_moe(a: torch.Tensor,
                        w1_q: torch.Tensor,
                        w2_q: torch.Tensor,
                        w1_scale: torch.Tensor,
                        w2_scale: torch.Tensor,
                        topk_weights: torch.Tensor,
                        topk_ids_: torch.Tensor,
                        a_strides1: torch.Tensor,
                        b_strides1: torch.Tensor,
                        c_strides1: torch.Tensor,
                        a_strides2: torch.Tensor,
                        b_strides2: torch.Tensor,
                        c_strides2: torch.Tensor,
                        s_strides13: torch.Tensor,
                        s_strides2: torch.Tensor,
                        start_expert_id: int,
                        end_expert_id: int,
                        E: int,
                        a1_scale: Optional[torch.Tensor] = None,
                        a2_scale: Optional[torch.Tensor] = None,
                        expert_map: Optional[torch.Tensor] = None,
                        apply_router_weight_on_input: bool = False,
                        num_repeats: int = 0):

        torch.cuda.synchronize()
        device = a.device
        expert_offsets = torch.empty((num_experts + 1),
                                 dtype=torch.int32,
                                 device=device)
        problem_sizes1 = torch.empty((num_experts, 3),
                                    dtype=torch.int32,
                                    device=device)
        problem_sizes2 = torch.empty((num_experts, 3),
                                    dtype=torch.int32,
                                    device=device)
        start_time = time.time()
        for _ in range(num_repeats):
            local_topk_ids = topk_ids_
            local_topk_ids = torch.where(expert_map[topk_ids_] != E,
                                            expert_map[topk_ids_], E)
            gateup_input = torch.empty(
                ((a.shape[0] * 8), a.shape[1]),
                device=device,
                dtype=torch.float8_e4m3fn,
            )
            reorder_topk_ids, src2dst, seg_indptr = run_cutlass_moe_ep_preproess(
                local_topk_ids, end_expert_id - start_expert_id + 1,
            )
            # reorder_topk_ids, src2dst = run_cutlass_moe_ep_preproess(
            #     local_topk_ids, E
            # )
            pre_reorder_triton_kernel_for_cutlass_moe[(a.shape[0],)](
                a,
                gateup_input,
                src2dst,
                local_topk_ids,
                a1_scale,
                E,
                8,
                a.shape[1],
                BLOCK_SIZE=512,
            )
            cutlass_w4a8_moe(
                a,
                w1_q,
                w2_q,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_ids_,
                local_topk_ids,
                a_strides1,
                b_strides1,
                c_strides1,
                a_strides2,
                b_strides2,
                c_strides2,
                s_strides13,
                s_strides2,
                gateup_input,
                src2dst,
                # seg_indptr,
                start_expert_id,
                end_expert_id,
                expert_offsets,
                problem_sizes1,
                problem_sizes2,
                a1_scale,
                a2_scale,
                expert_map,
            )
        
        torch.cuda.synchronize()   
        end_time = time.time()
        if num_repeats > 3:
            label = os.getenv("label")
            batch_size = a.shape[0]
            label_value = -1
            if label:
                label_value = int(label)
            test_time = (end_time - start_time) * 1000
            print("batch_size: {} label: {} test_time: {} ms".format(batch_size, label_value, test_time))
            # print("k_rate: {} cutlass time: {} ms".format(k_rate, (end_time-start_time) * 1000 ))

            # print("cutlass dtype:",  cutlass_ret_dtype)

    min_run_time = 1
    num_warmup = 3
    num_run = 5

    # Warmup
    # print("\n>>>>>> cutlass warmup time <<<<<<<\n")
    run_cutlass_moe(a, w1_q, w2_q, w1_scale, w2_scale, topk_weights, topk_ids,
                    a_strides1, b_strides1, c_strides1, a_strides2, b_strides2,
                    c_strides2, s_strides13, s_strides2, 0, num_experts - 1, 256,
                    a1_scale=a1_scale, a2_scale=a2_scale, expert_map=expert_map, apply_router_weight_on_input=False, num_repeats=num_warmup)

    print("\n>>>>>> cutlass execute time <<<<<<\n")
    run_cutlass_moe(a, w1_q, w2_q, w1_scale, w2_scale, topk_weights, topk_ids,
                    a_strides1, b_strides1, c_strides1, a_strides2, b_strides2,
                    c_strides2, s_strides13, s_strides2, 0, num_experts - 1, 256,
                    a1_scale=a1_scale, a2_scale=a2_scale, expert_map=expert_map, apply_router_weight_on_input=False, num_repeats=num_run)


def main(args): 
    print("Benchmarking models:")
    # for i, model in enumerate(args.models):
    #     print(f"[{i}]  {model}")
    model = "DeepSeek-W4AFP8"

    results: list[benchmark.Measurement] = []

    # for model in args.models:
        # for tp in args.tp_sizes:
            # for layer in WEIGHT_SHAPES_MOE[model]:
    num_experts = 32
    topk = 8
    size_k = 7168
    size_n = 2048 * 2
            
            # if len(args.limit_k) > 0 and size_k not in args.limit_k:
            #     continue

            # if len(args.limit_n) > 0 and size_n not in args.limit_n:
            #     continue

                # for per_act_token in PER_ACT_TOKEN_OPTS:
                    # for per_out_ch in PER_OUT_CH_OPTS:
    for size_m in DEFAULT_BATCH_SIZES:
        mkn = (size_m, size_k, size_n)
        print("\n+++++++ test mkn :", mkn, " start++++++++")
        print("num_experts and topk:", num_experts, topk)
        bench_run(results, model, num_experts, topk,mkn)
        print("+++++++ test mkn :", mkn, " end++++++++\n")

    # compare = benchmark.Compare(results)
    # compare.print()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark Marlin across specified models/shapes/batches")
    # parser.add_argument(
    #     "--models",
    #     nargs="+",
    #     type=str,
    #     default=DEFAULT_MODELS,
    #     choices=WEIGHT_SHAPES_MOE.keys(),
    # )
    # parser.add_argument("--tp-sizes",
    #                     nargs="+",
    #                     type=int,
    #                     default=DEFAULT_TP_SIZES)
    parser.add_argument("--batch-sizes",
                        nargs="+",
                        type=int,
                        default=DEFAULT_BATCH_SIZES)
    # parser.add_argument("--limit-k", nargs="+", type=int, default=[])
    # parser.add_argument("--limit-n", nargs="+", type=int, default=[])
    # parser.add_argument("--limit-num-groups", nargs="+", type=int, default=[])
    # parser.add_argument("--limit-per-act-token",
    #                     nargs="+",
    #                     type=int,
    #                     default=[])
    # parser.add_argument("--limit-per-out-ch", nargs="+", type=int, default=[])

    args = parser.parse_args()
    main(args)
