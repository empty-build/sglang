#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "grouped_mm_c3x.cuh"

using namespace cute;

namespace {

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_default {
  // M in (16, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_256, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M16 {
  // M in [1, 16]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_64, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_4, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_K8192 {
  // K in [8192, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_8, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_N8192 {
  // N in [8192, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_128, cute::_256>;
  using ClusterShape = cute::Shape<cute::_1, cute::_8, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};


template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_K7168 {
  // N in [8192, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_64, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};


template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_K256 {
  // N in [8192, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};


#define JOIN_STRUCT_NAME(m, n, k, a, b, c) \
    sm90_fp8_config##_##m##_##n##_##k##_##a##_##b##_##c

#define JOIN_STRUCT_NAME_CO(m, n, k, a, b, c) \
    sm90_fp8_co_config##_##m##_##n##_##k##_##a##_##b##_##c

#define GENERATE_SM90_FP8_CONFIG(M, N, K, A, B, C) \
template <typename InType, typename OutType, \
          template <typename, typename, typename> typename Epilogue> \
struct JOIN_STRUCT_NAME(M, N, K, A, B, C) { \
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>()); \
  using KernelSchedule = \
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum; \
  using EpilogueSchedule = \
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong; \
  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>; \
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>; \
  \
  using Cutlass3xGemm = \
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape, \
                            KernelSchedule, EpilogueSchedule>; \
};

#define GENERATE_SM90_FP8_CO_CONFIG(M, N, K, A, B, C) \
template <typename InType, typename OutType, \
          template <typename, typename, typename> typename Epilogue> \
struct JOIN_STRUCT_NAME_CO(M, N, K, A, B, C) { \
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>()); \
  using KernelSchedule = \
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum; \
  using EpilogueSchedule = \
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative; \
  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>; \
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>; \
  \
  using Cutlass3xGemm = \
      cutlass_3x_group_gemm<InType, OutType, Epilogue, TileShape, ClusterShape, \
                            KernelSchedule, EpilogueSchedule>; \
};

GENERATE_SM90_FP8_CONFIG(64,64,128,1,1,1)
GENERATE_SM90_FP8_CONFIG(64,64,128,1,1,2)
GENERATE_SM90_FP8_CONFIG(64,64,128,1,2,1)
GENERATE_SM90_FP8_CONFIG(64,64,128,1,2,2)
GENERATE_SM90_FP8_CONFIG(64,64,128,2,1,1)
GENERATE_SM90_FP8_CONFIG(64,64,128,2,1,2)
GENERATE_SM90_FP8_CONFIG(64,64,128,2,2,1)
GENERATE_SM90_FP8_CONFIG(64,64,128,2,2,2)
GENERATE_SM90_FP8_CONFIG(64,256,128,1,1,1)
GENERATE_SM90_FP8_CONFIG(64,256,128,1,1,2)
GENERATE_SM90_FP8_CONFIG(64,256,128,1,2,1)
GENERATE_SM90_FP8_CONFIG(64,256,128,1,2,2)
GENERATE_SM90_FP8_CONFIG(64,256,128,2,1,1)
GENERATE_SM90_FP8_CONFIG(64,256,128,2,1,2)
GENERATE_SM90_FP8_CONFIG(64,256,128,2,2,1)
GENERATE_SM90_FP8_CONFIG(64,256,128,2,2,2)

GENERATE_SM90_FP8_CO_CONFIG(128,64,128,1,1,1)
GENERATE_SM90_FP8_CO_CONFIG(128,64,128,1,1,2)
GENERATE_SM90_FP8_CO_CONFIG(128,64,128,1,2,1)
GENERATE_SM90_FP8_CO_CONFIG(128,64,128,1,2,2)
GENERATE_SM90_FP8_CO_CONFIG(128,64,128,2,1,1)
GENERATE_SM90_FP8_CO_CONFIG(128,64,128,2,1,2)
GENERATE_SM90_FP8_CO_CONFIG(128,64,128,2,2,1)
GENERATE_SM90_FP8_CO_CONFIG(128,64,128,2,2,2)
GENERATE_SM90_FP8_CO_CONFIG(128,256,128,1,1,1)
GENERATE_SM90_FP8_CO_CONFIG(128,256,128,1,1,2)
GENERATE_SM90_FP8_CO_CONFIG(128,256,128,1,2,1)
GENERATE_SM90_FP8_CO_CONFIG(128,256,128,1,2,2)
GENERATE_SM90_FP8_CO_CONFIG(128,256,128,2,1,1)
GENERATE_SM90_FP8_CO_CONFIG(128,256,128,2,1,2)
GENERATE_SM90_FP8_CO_CONFIG(128,256,128,2,2,1)
GENERATE_SM90_FP8_CO_CONFIG(128,256,128,2,2,2)


template <typename InType, typename OutType>
void run_cutlass_moe_mm_sm90(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides, int64_t force_kernel_id) {
  TORCH_CHECK(a_tensors.size(0) > 0, "No input A tensors provided.");
  TORCH_CHECK(b_tensors.size(0) > 0, "No input B tensors provided.");
  TORCH_CHECK(out_tensors.size(0) > 0, "No output tensors provided.");

  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn,
              "A tensors must be of type float8_e4m3fn.");
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn,
              "B tensors must be of type float8_e4m3fn.");

  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn);

  using Cutlass3xGemmN8192 = typename sm90_fp8_config_N8192<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmK8192 = typename sm90_fp8_config_K8192<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmM16 = typename sm90_fp8_config_M16<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmDefault = typename sm90_fp8_config_default<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

  using Cutlass3xGemmK7168 = typename sm90_fp8_config_K7168<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmK256 = typename sm90_fp8_config_K256<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

  uint32_t const m = a_tensors.size(0);
  uint32_t const n = out_tensors.size(1);
  uint32_t const k = a_tensors.size(1);
  if (force_kernel_id >=0 ) {
      switch (force_kernel_id) {
        case 112111: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,64,128,1,1,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 112112: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,64,128,1,1,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 112121: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,64,128,1,2,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 112122: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,64,128,1,2,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 112211: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,64,128,2,1,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 112212: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,64,128,2,1,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 112221: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,64,128,2,2,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 112222: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,64,128,2,2,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 142111: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,256,128,1,1,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 142112: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,256,128,1,1,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 142121: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,256,128,1,2,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 142122: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,256,128,1,2,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 142211: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,256,128,2,1,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 142212: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,256,128,2,1,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 142221: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,256,128,2,2,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 142222: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME(64,256,128,2,2,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1212111: {
        using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,128,1,1,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
        cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
            out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, c_strides);
        break;
        }
        case 1212112: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,128,1,1,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1212121: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,128,1,2,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1212122: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,128,1,2,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1212211: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,128,2,1,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1212212: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,128,2,1,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1212221: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,128,2,2,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1212222: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,64,128,2,2,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1242111: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,256,128,1,1,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1242112: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,256,128,1,1,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1242121: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,256,128,1,2,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1242122: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,256,128,1,2,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1242211: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,256,128,2,1,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1242212: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,256,128,2,1,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1242221: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,256,128,2,2,1)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        case 1242222: {
            using Cutlass3xGemmKSelected = typename JOIN_STRUCT_NAME_CO(128,256,128,2,2,2)<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
            break;
        }
        default : {
            using Cutlass3xGemmKSelected = typename sm90_fp8_config_default<InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
            cutlass_group_gemm_caller<Cutlass3xGemmKSelected>(
                out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
                problem_sizes, a_strides, b_strides, c_strides);
        }
      }
  } else {
    if (n >= 8192) {
        cutlass_group_gemm_caller<Cutlass3xGemmN8192>(
            out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, c_strides);
    } else if (k >= 8192) {
        cutlass_group_gemm_caller<Cutlass3xGemmK8192>(
            out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, c_strides);
    } else if (m <= 16) {
        cutlass_group_gemm_caller<Cutlass3xGemmM16>(
            out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, c_strides);
    } else {
        cutlass_group_gemm_caller<Cutlass3xGemmDefault>(
            out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, c_strides);
    }
  } 
}

// bool generate_cutlass_template_args(int force_kernel_id, std::vector<int>& cluster_args, std::vector<int>& tile_args) {
//     std::vector<std::vector<int>> c_options = {{1,2},{1,2},{1}};

//     std::vector<std::vector<int>> t_options = {{64, 128, 256}, {32, 64, 128, 256}, {128, 256}};

//     int max_label_value = c_options[0].size() *  1e5 +
//                           c_options[1].size() *  1e4 +
//                           c_options[2].size() *  1e3 +
//                           t_options[0].size() *  1e2 +
//                           t_options[1].size() *  1e1 +
//                           t_options[2].size() *  1e0; 

//     if ( force_kernel_id > max_label_value) {
//         std::cerr << "invaild kernel id: " << force_kernel_id << std::endl;
//         return false;
//     }

//     int c1,c2,c3;
//     int t1,t2,t3;
//     int remain = force_kernel_id;

//     c1 = c_options[0][(int)(remain / (int)1e5 -1)];
//     remain = remain % ((int)1e5);
    
//     c2 = c_options[1][(int)(remain / (int)1e4 -1)];
//     remain = remain % ((int)1e4);

//     c3 = c_options[2][(int)(remain / (int)1e3 -1)];
//     remain = remain % ((int)1e3);

//     t1 = t_options[0][(int)(remain / (int)1e2 -1)];
//     remain = remain % ((int)1e2);
    
//     t2 = t_options[1][(int)(remain / (int)1e1 -1)];
//     remain = remain % ((int)1e1);

//     t3 = t_options[2][(int)(remain / (int)1e0 -1)];

//     cluster_args = {c1, c2, c3};
//     tile_args = {t1, t2, t3};
//     return true;
// }

void dispatch_moe_mm_sm90(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    int64_t force_kernel_id ) {    
    if (out_tensors.dtype() == torch::kBFloat16) {
        run_cutlass_moe_mm_sm90<cutlass::float_e4m3_t, cutlass::bfloat16_t>(
            out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, c_strides, force_kernel_id);
    } else if (out_tensors.dtype() == torch::kHalf) {
        run_cutlass_moe_mm_sm90<cutlass::float_e4m3_t, cutlass::half_t>(
            out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, c_strides, force_kernel_id);
    } else if (out_tensors.dtype() == torch::kFloat8_e4m3fn) {
        run_cutlass_moe_mm_sm90<cutlass::float_e4m3_t, cutlass::float_e4m3_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, force_kernel_id);
    }
}

}  // namespace

void cutlass_moe_mm_sm90(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides, int64_t force_kernel_id) {
    dispatch_moe_mm_sm90(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                       expert_offsets, problem_sizes, a_strides, b_strides,
                       c_strides, force_kernel_id);
}
