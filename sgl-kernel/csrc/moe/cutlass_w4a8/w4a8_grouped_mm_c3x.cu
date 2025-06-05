#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "w4a8_grouped_mm_c3x.cuh"

using namespace cute;

namespace {

void dispatch_w4a8_moe_mm_sm90(
    torch::Tensor& d_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& d_strides,
    torch::Tensor const& s_strides, int64_t chunk_size, int64_t M) {
  constexpr int TileShapeK = 512;
  using KernelSchedule =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  
  uint32_t const m = M / 32;
  uint32_t const n = d_tensors.size(1);
  uint32_t const k = a_tensors.size(1);

  if (m <= 16) {
    using TileShape = Shape<_128, _16, cute::Int<TileShapeK>>;
    using ClusterShape = Shape<_2, _1, _1>;
    cutlass_w4a8_group_gemm_caller<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>(
        d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size
    );
  } else if (m <= 32) {
    using TileShape = Shape<_128, _32, cute::Int<TileShapeK>>;
    using ClusterShape = Shape<_1, _1, _1>;
    cutlass_w4a8_group_gemm_caller<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>(
        d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size
    );
  } else if (m <= 64) {
    using TileShape = Shape<_128, _64, cute::Int<TileShapeK>>;
    using ClusterShape = Shape<_2, _1, _1>;
    cutlass_w4a8_group_gemm_caller<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>(
        d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size
    );
  } else {
    using TileShape = Shape<_128, _64, cute::Int<TileShapeK>>;
    using ClusterShape = Shape<_1, _2, _1>;
    cutlass_w4a8_group_gemm_caller<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>(
        d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size
    );
  }
}

}  // namespace

void cutlass_w4a8_moe_mm_sm90(
    torch::Tensor& d_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& d_strides,
    torch::Tensor const& s_strides, int64_t chunk_size, int64_t M) {
  dispatch_w4a8_moe_mm_sm90(d_tensors, a_tensors, b_tensors, a_scales,
                            b_scales, expert_offsets, problem_sizes,
                            a_strides, b_strides, d_strides, s_strides,
                            chunk_size, M);
}