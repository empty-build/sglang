import torch


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    token_cnts_buffer,
    cumsum_buffer,
):
    torch.ops.sgl_kernel.moe_align_block_size.default(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: float,
) -> None:
    torch.ops.sgl_kernel.topk_softmax.default(
        topk_weights, topk_ids, token_expert_indices, gating_output
    )


def moe_fused_gate(
    input_tensor,
    bias,
    num_expert_group,
    topk_group,
    topk,
    n_share_experts_fusion=0,
    routed_scaling_factor=0,
):
    # This fused kernel function is used to select topk expert in a hierarchical 2-layer fashion
    # it split group of expert into num_expert_group, and use top2 expert weight sum in each group
    # as the group weight to select exerpt groups and then select topk experts within the selected groups
    # the #experts is decided by the input tensor shape and we currently only support power of 2 #experts
    # and #experts should be divisible by num_expert_group. #expert/num_expert_group <= 32 is limitted for now.
    # for non-supported case, we suggestion to use the biased_grouped_topk func in sglang.srt.layers.moe.topk
    # n_share_experts_fusion: if > 0, the last expert will be replaced with a round-robin shared expert
    # routed_scaling_factor: if > 0, the last expert will be scaled by this factor
    return torch.ops.sgl_kernel.moe_fused_gate.default(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        n_share_experts_fusion,
        routed_scaling_factor,
    )


def fp8_blockwise_scaled_grouped_mm(
    output,
    a,
    b,
    scales_a,
    scales_b,
    stride_a,
    stride_b,
    stride_c,
    layout_sfa,
    layout_sfb,
    problem_sizes,
    expert_offsets,
):
    torch.ops.sgl_kernel.fp8_blockwise_scaled_grouped_mm.default(
        output,
        a,
        b,
        scales_a,
        scales_b,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
    )

def get_cutlass_moe_mm_data(
        topk_ids: torch.Tensor, expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor, problem_sizes2: torch.Tensor,
        input_permutation: torch.Tensor, output_permutation: torch.Tensor,
        num_experts: int, n: int, k: int):
    """
    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in topk_ids (token-expert mapping) and uses it to
    compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation after the input is sorted with
                      input_permutation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    - input_permutation: Permutation that must be used to shuffle the input
                         before executing the MMs.
    - output_permutation: Permutation that must be used to shuffle the output
                          after executing the MMs.
    """
    torch.ops.sgl_kernel.get_cutlass_moe_mm_data.default(topk_ids, expert_offsets,
                                         problem_sizes1, problem_sizes2,
                                         input_permutation, output_permutation,
                                         num_experts, n, k)

def cutlass_w4a8_moe_mm(
    d: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    experts_offsets: torch.tensor,
    problem_sizes: torch.tensor,
    a_strides: torch.tensor,
    b_strides: torch.tensor,
    d_strides: torch.tensor,
    s_strides: torch.tensor,
    chunk_size: int = 0,
    topk: int = 0,
):
    """
    Perform grouped matrix multiplication between int4 weights and fp8 activations.
    
    This function executes multiple GEMM operations in parallel, which is useful for
    scenarios like Mixture of Experts (MoE) where different inputs go through different
    experts. The implementation leverages NVIDIA Hopper architecture features for
    optimal performance with quantized weights.
    
    Args:
        d: Output matrices of shape [total_m, total_n]
        a: Activation matrices in FP8 (float_e4m3_t) format
           Each tensor should be of shape [total_m, K] in row-major layout
        b: Weight matrices in packed int4 format
           Each tensor should be of shape [E, N, K//2] in column-major layout
           where each byte contains two 4-bit integers
        a_scales: Scale factors for the inputs
        b_scales: Scale factors for the quantized weights
           Each tensor should be of shape [E, K//512, N*8]
        experts_offsets: Tensor containing expert offsets for determining group boundaries
        problem_sizes: with shape [num_experts, 3] (M, N, K for each group) (int32)
        a_strides: Strides information for A matrices
        b_strides: Strides information for B matrices
        d_strides: Strides information for D matrices
        s_strides: Strides information for b_scales matrices
        chunk_size: Number of elements each scale value applies to (K//512), default to 128
        
    Requirements:
        - All tensors must be on a CUDA device
        - Requires an NVIDIA Hopper GPU (H100)
        - A tensors must be in float8_e4m3fn format
        - B tensors must contain packed int4 values (stored as int8)
        
    Note:
        The function computes: D = (A * (B * scales))
        for each group of tensors in parallel
    """

    torch.ops.sgl_kernel.cutlass_w4a8_moe_mm.default(
        d, a, b, a_scales, b_scales, experts_offsets, problem_sizes, 
        a_strides, b_strides, d_strides, s_strides, chunk_size, topk)