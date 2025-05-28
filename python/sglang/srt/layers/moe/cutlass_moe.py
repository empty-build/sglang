# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""
from typing import Optional # TODO 

import torch

# import _custom_ops as ops # TODO
from sgl_kernel import silu_and_mul
from vllm import _custom_ops as vllm_ops

# Functions that require adjustment for integrating vLLM’s code into sglang.
# ops.scaled_fp8_quant() -> scaled_fp8_quant # ref: fused_moe()
# torch.ops._C.silu_and_mul(intermediate, c1) -> silu_and_mul() # ref: fused_moe() 
# ops.get_cutlass_moe_mm_data(local_topk_ids, expert_offsets, problem_sizes1) -> CUDA (no replacement) 先暫時寫在這邊 到時候可以搬, vllm是在_custom_ops.py下
# ops.cutlass_moe_mm(c1, rep_a_q, w1_q, rep_a1_scales, w1_scale) -> CUDA (no replacement)

from sgl_kernel import get_cutlass_moe_mm_data # func name
from sgl_kernel import cutlass_moe_mm # func name


# def get_cutlass_moe_mm_data(
#         topk_ids: torch.Tensor, expert_offsets: torch.Tensor,
#         problem_sizes1: torch.Tensor, problem_sizes2: torch.Tensor,
#         input_permutation: torch.Tensor, output_permutation: torch.Tensor,
#         num_experts: int, n: int, k: int):
#     """
#     Prepare data necessary to perform CUTLASS grouped matrix multiplications
#     used in CUTLASS-based fused MoE.

#     The function takes in topk_ids (token-expert mapping) and uses it to
#     compute:
#     - expert_offsets: Indices that mark at which token index each expert begins
#                       its computation after the input is sorted with
#                       input_permutation. The number of tokens computed with
#                       expert E is expert_offsets[E + 1] - expert_offsets[E]
#     - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
#                                       multiplication in two grouped MMs used in
#                                       the fused MoE operation.
#     - input_permutation: Permutation that must be used to shuffle the input
#                          before executing the MMs.
#     - output_permutation: Permutation that must be used to shuffle the output
#                           after executing the MMs.
#     """
#     # torch.ops._C.get_cutlass_moe_mm_data(topk_ids, expert_offsets,
#     #                                      problem_sizes1, problem_sizes2,
#     #                                      input_permutation, output_permutation,
#     #                                      num_experts, n, k)
#     torch.ops._C.get_cutlass_moe_mm_data.default(topk_ids, expert_offsets,
#                                          problem_sizes1, problem_sizes2,
#                                          input_permutation, output_permutation,
#                                          num_experts, n, k)

#TODO make the grouped gemm kernel consistent with scaled gemm kernel
def cutlass_moe_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids_: torch.Tensor,
    ab_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.half,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of fp8-quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2_q (torch.Tensor): The second set of fp8-quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - ab_strides1 (torch.Tensor): The input and weights strides of the first
        grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - ab_strides2 (torch.Tensor): The input and weights strides of the second
        grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]
    - out_dtype (torch.Tensor): The output tensor type.
    - expert_map (Optional[torch.Tensor]): In the case of Expert parallel,
        every Rank is responsible for a subset of experts. expert_map is a
        mapping from global expert-id to local expert-id. When expert_map[i]
        is -1, it means that this Rank is not responsible for global
        expert-id i.
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.

    Returns:
    - torch.Tensor: The fp16 output tensor after applying the MoE layer.
    """

    assert topk_weights.shape == topk_ids_.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert a1_scale is None or a1_scale.dim(
    ) == 0 or a1_scale.shape[0] == 1 or a1_scale.shape[0] == a.shape[
        0], "Input scale shape mismatch"
    assert w1_scale.dim() == 1 or w1_scale.shape[1] == 1 or w1_scale.shape[
        1] == w1_q.shape[2], "W1 scale shape mismatch"
    assert w2_scale.dim() == 1 or w2_scale.shape[1] == 1 or w2_scale.shape[
        1] == w2_q.shape[2], "W2 scale shape mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[
        0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[
        0], "w2 scales expert number mismatch"
    assert a2_scale is None or a1_scale is None or a2_scale.shape == a1_scale.shape, "Intermediate scale shape mismatch"  # noqa: E501
    assert ab_strides1.shape[0] == w1_q.shape[
        0], "AB Strides 1 expert number mismatch"
    assert c_strides1.shape[0] == w1_q.shape[
        0], "C Strides 1 expert number mismatch"
    assert ab_strides2.shape[0] == w2_q.shape[
        0], "AB Strides 2 expert number  mismatch"
    assert c_strides2.shape[0] == w2_q.shape[
        0], "C Strides 2 expert number mismatch"
    assert out_dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(1)
    n = w2_q.size(1)

    local_topk_ids = topk_ids_
    if expert_map is not None:
        "Translate info from expert_map to topk_ids"
        local_topk_ids = torch.where(expert_map[topk_ids_] != -1,
                                     expert_map[topk_ids_], -1)

    topk = local_topk_ids.size(1)

    per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
        a2_scale.numel() != 1 if a2_scale is not None else False)
    if apply_router_weight_on_input:
        assert topk == 1, \
            "apply_router_weight_on_input is only implemented for topk=1"
        # TODO: this only works for topK=1, will need to update for topK>1
        a = a * topk_weights.to(out_dtype)

    # a_q, a1_scale = ops.scaled_fp8_quant(
    #     a, a1_scale, use_per_token_if_dynamic=per_act_token)
    # from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
    # a_q, a1_scale = scaled_fp8_quant(
    #     a, a1_scale, use_per_token_if_dynamic=per_act_token)

    a_q, a1_scale  = vllm_ops.scaled_fp8_quant(
            a, a1_scale, use_per_token_if_dynamic=per_act_token)

    device = a_q.device
    expert_offsets = torch.empty((num_experts + 1),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes1 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes2 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)

    a_map_initializer = torch.empty
    c2_initializer = torch.empty
    if expert_map is not None:
        # With expert_map each Rank processes only a subset of experts. As
        # a result not all of a_map and c2 tensors are filled. We fill it
        # zeros for correctness.
        a_map_initializer = torch.zeros
        c2_initializer = torch.zeros

    a_map = a_map_initializer((local_topk_ids.numel()),
                              dtype=torch.int32,
                              device=device)
    c_map = torch.empty((local_topk_ids.numel()),
                        dtype=torch.int32,
                        device=device)

    # ops.get_cutlass_moe_mm_data(local_topk_ids, expert_offsets, problem_sizes1,
    #                             problem_sizes2, a_map, c_map, num_experts, n,
    #                             k)
    get_cutlass_moe_mm_data(local_topk_ids, expert_offsets, problem_sizes1,
                            problem_sizes2, a_map, c_map, num_experts,
                            n, k)

    rep_a_q = a_q.view(dtype=torch.uint8)[a_map].view(dtype=a_q.dtype)
    rep_a1_scales = a1_scale[a_map] if per_act_token else a1_scale

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
    c2 = c2_initializer((m * topk, k), device=device, dtype=out_dtype)

    if 0:
        print("===---------------===")
        print(f"({a.device.index}) 1st: cutlass_moe_mm() ({a.device.index})")
        print("===---------------===")
    if 0:
        print(f"({a.device.index}) a.shape: {a.shape}")
        print(f"({a.device.index}) w1_q.shape: {w1_q.shape}")
        print(f"({a.device.index}) w1_scale.shape: {w1_scale.shape}")

        print(f"({a.device.index}) c1.shape: {c1.shape} tok(M) * topk")
        print(f"({a.device.index}) topk_weights.shape: {topk_weights.shape}")
        print(f"({a.device.index}) rep_a1_scales.shape: {rep_a1_scales.shape}")
        print(f"({a.device.index}) ab_strides1.shape: {ab_strides1.shape}")
        print(f"({a.device.index}) c_strides1.shape: {c_strides1.shape}")
        print(f"({a.device.index}) expert_offsets[:-1].shape: {expert_offsets[:-1].shape}")
        print(f"({a.device.index}) problem_sizes1.shape: {problem_sizes1.shape}")
    # ops.cutlass_moe_mm(c1, rep_a_q, w1_q, rep_a1_scales, w1_scale,
    #                    expert_offsets[:-1], problem_sizes1, ab_strides1,
    #                    ab_strides1, c_strides1)
    cutlass_moe_mm(c1, rep_a_q, w1_q, rep_a1_scales, w1_scale,
                       expert_offsets[:-1], problem_sizes1, ab_strides1,
                       ab_strides1, c_strides1)


    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    # print(f"intermediate.shape: {intermediate.shape} d")
    # print(f"c1.shape: {c1.shape} 2d")
    # print(f"intermediate.shape[-1]: {intermediate.shape[-1]} [-1] d")
    # print(f"c1.shape[-1]: {c1.shape[-1]} [-1] 2d")
    # torch.ops._C.silu_and_mul(intermediate, c1)
    # silu_and_mul(intermediate, c1) # 怪
    # from vllm import _custom_ops as vllm_ops
    """
    if _is_cuda:
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
    else:
        vllm_ops.silu_and_mul(
            intermediate_cache2, intermediate_cache1.view(-1, N)
        )
    """
    # print("c1.shape: ", c1.shape)
    # print("itermdeidate shape: ", intermediate.shape )
    silu_and_mul(c1, intermediate)

    # intemediate_q, a2_scale = ops.scaled_fp8_quant(
    #     intermediate, a2_scale, use_per_token_if_dynamic=per_act_token)
    # intemediate_q, a2_scale = scaled_fp8_quant(
    #     intermediate, a2_scale, use_per_token_if_dynamic=per_act_token)
    intemediate_q, a2_scale = vllm_ops.scaled_fp8_quant(
        intermediate, a2_scale, use_per_token_if_dynamic=per_act_token)

    if 0:
        print("===---------------===")
        print(f"({a.device.index}) 2nd: cutlass_moe_mm() ({a.device.index})")
        print("===---------------===")
    if 0:
        print(f"({a.device.index}) a.shape: {a.shape}")
        print(f"({a.device.index}) w2_q.shape: {w2_q.shape}")
        print(f"({a.device.index}) w2_scale.shape: {w2_scale.shape}")

        print(f"({a.device.index}) c2.shape: {c2.shape}")
        print(f"({a.device.index}) a2_scale.shape: {a2_scale.shape}")
        print(f"({a.device.index}) ab_strides2.shape: {ab_strides2.shape}")
        print(f"({a.device.index}) c_strides2.shape: {c_strides2.shape}")
        print(f"({a.device.index}) problem_sizes2.shape: {problem_sizes2.shape}")

    # ops.cutlass_moe_mm(c2, intemediate_q, w2_q, a2_scale, w2_scale,
    #                    expert_offsets[:-1], problem_sizes2, ab_strides2,
    #                    ab_strides2, c_strides2)
    cutlass_moe_mm(c2, intemediate_q, w2_q, a2_scale, w2_scale,
                       expert_offsets[:-1], problem_sizes2, ab_strides2,
                       ab_strides2, c_strides2)
    # Gather tokens
    c2 = c2[c_map].view(m, topk, k)
    if not apply_router_weight_on_input:
        c2 = c2 * topk_weights.view(m, topk, 1).to(out_dtype)
    return c2.sum(dim=1)
