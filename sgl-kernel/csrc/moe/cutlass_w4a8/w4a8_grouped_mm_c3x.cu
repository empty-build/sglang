#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "w4a8_grouped_mm_c3x.cuh"

using namespace cute;

namespace {

#define JOIN_STRUCT_NAME(m, n, k, a, b, c) \
    sm90_fp8_config##_##m##_##n##_##k##_##a##_##b##_##c

#define JOIN_STRUCT_NAME_CO(m, n, k, a, b, c) \
    sm90_fp8_co_config##_##m##_##n##_##k##_##a##_##b##_##c

<<<<<<< HEAD
#define GENERATE_SM90_W4A8_PP_CONFIG(M, N, K, A, B, C) \
=======
#define GENERATE_SM90_FP8_CONFIG(M, N, K, A, B, C) \
>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)
struct JOIN_STRUCT_NAME(M, N, K, A, B, C) { \
  using KernelSchedule = \
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong; \
  using EpilogueSchedule = \
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong; \
  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>; \
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>; \
  \
  using Cutlass3xW4A8Gemm = \
  cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, \
                            KernelSchedule, EpilogueSchedule>; \
};

<<<<<<< HEAD
#define GENERATE_SM90_W4A8_CO_CONFIG(M, N, K, A, B, C) \
=======
#define GENERATE_SM90_FP8_CO_CONFIG(M, N, K, A, B, C) \
>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)
struct JOIN_STRUCT_NAME_CO(M, N, K, A, B, C) { \
  using KernelSchedule = \
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative; \
  using EpilogueSchedule = \
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative; \
  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>; \
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>; \
  \
  using Cutlass3xW4A8Gemm = \
  cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, \
                            KernelSchedule, EpilogueSchedule>; \
};

<<<<<<< HEAD
GENERATE_SM90_W4A8_PP_CONFIG(64,16,512,1,1,1)
GENERATE_SM90_W4A8_PP_CONFIG(64,32,512,2,1,1)

GENERATE_SM90_W4A8_CO_CONFIG(128,16,512,1,1,1)
GENERATE_SM90_W4A8_CO_CONFIG(128,16,512,2,1,1)
GENERATE_SM90_W4A8_CO_CONFIG(128,32,512,1,1,1)
GENERATE_SM90_W4A8_CO_CONFIG(128,32,512,2,1,1)
GENERATE_SM90_W4A8_CO_CONFIG(128,64,512,1,1,1)
=======
GENERATE_SM90_FP8_CONFIG(64,16,512,1,1,1)
GENERATE_SM90_FP8_CONFIG(64,16,512,1,2,1)
GENERATE_SM90_FP8_CONFIG(64,16,512,2,1,1)
GENERATE_SM90_FP8_CONFIG(64,16,512,2,2,1)
GENERATE_SM90_FP8_CONFIG(64,32,512,1,1,1)
GENERATE_SM90_FP8_CONFIG(64,32,512,1,2,1)
GENERATE_SM90_FP8_CONFIG(64,32,512,2,1,1)
GENERATE_SM90_FP8_CONFIG(64,32,512,2,2,1)
GENERATE_SM90_FP8_CONFIG(64,64,512,1,1,1)
GENERATE_SM90_FP8_CONFIG(64,64,512,1,2,1)
GENERATE_SM90_FP8_CONFIG(64,64,512,2,1,1)
GENERATE_SM90_FP8_CONFIG(64,64,512,2,2,1)
GENERATE_SM90_FP8_CONFIG(64,128,512,1,1,1)
GENERATE_SM90_FP8_CONFIG(64,128,512,1,2,1)
GENERATE_SM90_FP8_CONFIG(64,128,512,2,1,1)
GENERATE_SM90_FP8_CONFIG(64,128,512,2,2,1)
GENERATE_SM90_FP8_CONFIG(128,16,512,1,1,1)
GENERATE_SM90_FP8_CONFIG(128,16,512,1,2,1)
GENERATE_SM90_FP8_CONFIG(128,16,512,2,1,1)
GENERATE_SM90_FP8_CONFIG(128,16,512,2,2,1)
GENERATE_SM90_FP8_CONFIG(128,32,512,1,1,1)
GENERATE_SM90_FP8_CONFIG(128,32,512,1,2,1)
GENERATE_SM90_FP8_CONFIG(128,32,512,2,1,1)
GENERATE_SM90_FP8_CONFIG(128,32,512,2,2,1)
GENERATE_SM90_FP8_CONFIG(128,64,512,1,1,1)
GENERATE_SM90_FP8_CONFIG(128,64,512,1,2,1)
GENERATE_SM90_FP8_CONFIG(128,64,512,2,1,1)
GENERATE_SM90_FP8_CONFIG(128,64,512,2,2,1)
GENERATE_SM90_FP8_CONFIG(128,128,512,1,1,1)
GENERATE_SM90_FP8_CONFIG(128,128,512,1,2,1)
GENERATE_SM90_FP8_CONFIG(128,128,512,2,1,1)
GENERATE_SM90_FP8_CONFIG(128,128,512,2,2,1)
GENERATE_SM90_FP8_CONFIG(256,16,512,1,1,1)
GENERATE_SM90_FP8_CONFIG(256,16,512,1,2,1)
GENERATE_SM90_FP8_CONFIG(256,16,512,2,1,1)
GENERATE_SM90_FP8_CONFIG(256,16,512,2,2,1)
GENERATE_SM90_FP8_CONFIG(256,32,512,1,1,1)
GENERATE_SM90_FP8_CONFIG(256,32,512,1,2,1)
GENERATE_SM90_FP8_CONFIG(256,32,512,2,1,1)
GENERATE_SM90_FP8_CONFIG(256,32,512,2,2,1)

GENERATE_SM90_FP8_CO_CONFIG(128,16,512,1,1,1)
GENERATE_SM90_FP8_CO_CONFIG(128,16,512,1,2,1)
GENERATE_SM90_FP8_CO_CONFIG(128,16,512,2,1,1)
GENERATE_SM90_FP8_CO_CONFIG(128,16,512,2,2,1)
GENERATE_SM90_FP8_CO_CONFIG(128,32,512,1,1,1)
GENERATE_SM90_FP8_CO_CONFIG(128,32,512,1,2,1)
GENERATE_SM90_FP8_CO_CONFIG(128,32,512,2,1,1)
GENERATE_SM90_FP8_CO_CONFIG(128,32,512,2,2,1)
GENERATE_SM90_FP8_CO_CONFIG(128,64,512,1,1,1)
GENERATE_SM90_FP8_CO_CONFIG(128,64,512,1,2,1)
GENERATE_SM90_FP8_CO_CONFIG(128,64,512,2,1,1)
GENERATE_SM90_FP8_CO_CONFIG(128,64,512,2,2,1)
GENERATE_SM90_FP8_CO_CONFIG(256,16,512,1,1,1)
GENERATE_SM90_FP8_CO_CONFIG(256,16,512,1,2,1)
GENERATE_SM90_FP8_CO_CONFIG(256,16,512,2,1,1)
GENERATE_SM90_FP8_CO_CONFIG(256,16,512,2,2,1)
GENERATE_SM90_FP8_CO_CONFIG(256,32,512,1,1,1)
GENERATE_SM90_FP8_CO_CONFIG(256,32,512,1,2,1)
GENERATE_SM90_FP8_CO_CONFIG(256,32,512,2,1,1)
GENERATE_SM90_FP8_CO_CONFIG(256,32,512,2,2,1)
>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)

void dispatch_w4a8_moe_mm_sm90(
    torch::Tensor& d_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& d_strides,
<<<<<<< HEAD
    torch::Tensor const& s_strides, int64_t chunk_size, int64_t topk) {
//   constexpr int TileShapeK = 512;
=======
    torch::Tensor const& s_strides, int64_t chunk_size, int64_t M) {
  constexpr int TileShapeK = 512;
>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)
  using KernelSchedule =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  
<<<<<<< HEAD
  uint32_t const m = a_tensors.size(0) / topk;
=======
  // uint32_t const m = M / 32;
  uint32_t const m = a_tensors.size(0) / 8;
>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)
  uint32_t const n = d_tensors.size(1);
  uint32_t const k = a_tensors.size(1);
  
  if (n == 4096 && k == 7168) {
    // group gemm 1
    if (m <= 4) {
        using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME(64,32,512,2,1,1)::Cutlass3xW4A8Gemm;
        cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
            d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
    } else if (m <= 16) {
        using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,16,512,2,1,1)::Cutlass3xW4A8Gemm;
        cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
            d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
    } else if (m <= 256) {
        using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,16,512,1,1,1)::Cutlass3xW4A8Gemm;
        cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
            d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
    } else if (m <= 1024) {
        using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,32,512,2,1,1)::Cutlass3xW4A8Gemm;
        cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
            d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
    } else {
        using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,512,1,1,1)::Cutlass3xW4A8Gemm;
        cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
            d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
    }
  } else if (n == 7168 && k == 2048) {
    // group gemm 2
    if (m <= 8) {
        using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME(64,16,512,1,1,1)::Cutlass3xW4A8Gemm;
        cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
            d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
    } else if (m <= 512) {
        using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,32,512,1,1,1)::Cutlass3xW4A8Gemm;
        cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
            d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
    } else {
        using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,512,1,1,1)::Cutlass3xW4A8Gemm;
        cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
            d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
    }
  } else {
        using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,32,512,1,1,1)::Cutlass3xW4A8Gemm;
        cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
            d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
  }
<<<<<<< HEAD
=======

//   if (M > 0) {
//     switch (M) {
//       // pingpong
//     //   case 110611: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,16,512,1,1,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 110621: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,16,512,1,2,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 110711: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,16,512,2,1,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 110721: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,16,512,2,2,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 113111: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,32,512,1,1,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 113121: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,32,512,1,2,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 113211: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,32,512,2,1,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 113221: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,32,512,2,2,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 118111: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,64,512,1,1,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 118121: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,64,512,1,2,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 118211: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,64,512,2,1,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 118221: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,64,512,2,2,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 128111: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,128,512,1,1,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 128121: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,128,512,1,2,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 128211: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,128,512,2,1,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 128221: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(64,128,512,2,2,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     case 210611: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,16,512,1,1,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 210621: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,16,512,1,2,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 210711: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,16,512,2,1,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 210721: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,16,512,2,2,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 213111: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,32,512,1,1,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 213121: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,32,512,1,2,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 213211: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,32,512,2,1,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 213221: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,32,512,2,2,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 218111: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,512,1,1,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 218121: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,512,1,2,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 218211: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,512,2,1,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 218221: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,512,2,2,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     // case 228111: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,128,512,1,1,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 228121: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,128,512,1,2,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 228211: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,128,512,2,1,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     // case 228221: {
//     //     using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(128,128,512,2,2,1)::Cutlass3xW4A8Gemm;
//     //     cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//     //         d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//     //         problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//     //     break;
//     // }
//     case 410611: {
//       using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(256,16,512,1,1,1)::Cutlass3xW4A8Gemm;
//       cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//           d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//           problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//       break;
//     }
//     case 410621: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(256,16,512,1,2,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 410711: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(256,16,512,2,1,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 410721: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(256,16,512,2,2,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 413111: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(256,32,512,1,1,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 413121: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(256,32,512,1,2,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 413211: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(256,32,512,2,1,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     case 413221: {
//         using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO(256,32,512,2,2,1)::Cutlass3xW4A8Gemm;
//         cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
//             d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
//             problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
//         break;
//     }
//     }
//   }

  // if (m <= 16) {
  //   using TileShape = Shape<_128, _16, cute::Int<TileShapeK>>;
  //   using ClusterShape = Shape<_2, _1, _1>;
  //   cutlass_w4a8_group_gemm_caller<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>(
  //       d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
  //       problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size
  //   );
  // } else if (m <= 32) {
  //   using TileShape = Shape<_128, _32, cute::Int<TileShapeK>>;
  //   using ClusterShape = Shape<_1, _1, _1>;
  //   cutlass_w4a8_group_gemm_caller<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>(
  //       d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
  //       problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size
  //   );
  // } else if (m <= 64) {
  //   using TileShape = Shape<_128, _64, cute::Int<TileShapeK>>;
  //   using ClusterShape = Shape<_2, _1, _1>;
  //   cutlass_w4a8_group_gemm_caller<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>(
  //       d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
  //       problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size
  //   );
  // } else {
  //   using TileShape = Shape<_128, _64, cute::Int<TileShapeK>>;
  //   using ClusterShape = Shape<_1, _2, _1>;
  //   cutlass_w4a8_group_gemm_caller<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>(
  //       d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
  //       problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size
  //   );
  // }
>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)
}

}  // namespace

void cutlass_w4a8_moe_mm_sm90(
    torch::Tensor& d_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& d_strides,
<<<<<<< HEAD
    torch::Tensor const& s_strides, int64_t chunk_size, int64_t topk) {
  dispatch_w4a8_moe_mm_sm90(d_tensors, a_tensors, b_tensors, a_scales,
                            b_scales, expert_offsets, problem_sizes,
                            a_strides, b_strides, d_strides, s_strides,
                            chunk_size, topk);
=======
    torch::Tensor const& s_strides, int64_t chunk_size, int64_t M) {
  dispatch_w4a8_moe_mm_sm90(d_tensors, a_tensors, b_tensors, a_scales,
                            b_scales, expert_offsets, problem_sizes,
                            a_strides, b_strides, d_strides, s_strides,
                            chunk_size, M);
>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)
}