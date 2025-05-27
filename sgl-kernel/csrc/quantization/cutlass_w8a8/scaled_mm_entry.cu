#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

// TESTING
#include "pytorch_extension_utils.h"

#define ENABLE_CUTLASS_MOE_SM90 1 // -> in CMakeLists.txt but doesn't work. TODO: use SGL_KERNEL_ENABLE_SM90A in CMakeLists.txt e.g., make build -DSGL_KERNEL_ENABLE_SM90A. And check SGL_KERNEL_ENABLE_SM90A below.
// #include "cutlass_extensions/common.hpp"
// get_sm_version_num

/* TODO */
// /nvme0n1/jack/vllm-w8a8-cutlass/csrc/quantization/cutlass_w8a8/moe/grouped_mm_c3x.cu
// cutlass_moe_mm_sm90()

/*
HACK
hope this doesn't need common.hpp.....
*/
// #include "cutlass_extensions/common.hpp"

int32_t get_sm_version_num() {
  int32_t major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}


/*
 HACK
 vllm/csrc/quantization/cutlass_w8a8/moe/moe_data.cu
*/
/*
// #include <cudaTypedefs.h>

// #include <c10/cuda/CUDAGuard.h>
// #include <torch/all.h>

#include <iostream>

constexpr uint64_t THREADS_PER_EXPERT = 512;

__global__ void compute_problem_sizes(const int* __restrict__ topk_ids,
                                      int32_t* problem_sizes1,
                                      int32_t* problem_sizes2,
                                      int32_t* atomic_buffer,
                                      const int topk_length, const int n,
                                      const int k) {
  int expert_id = blockIdx.x;

  int occurrences = 0;
  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    occurrences += (topk_ids[i] == expert_id);
  }
  atomicAdd(&atomic_buffer[expert_id], occurrences);
  __syncthreads();

  if (threadIdx.x == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    problem_sizes1[expert_id * 3] = final_occurrences;
    problem_sizes1[expert_id * 3 + 1] = 2 * n;
    problem_sizes1[expert_id * 3 + 2] = k;
    problem_sizes2[expert_id * 3] = final_occurrences;
    problem_sizes2[expert_id * 3 + 1] = k;
    problem_sizes2[expert_id * 3 + 2] = n;
  }
}

__global__ void compute_expert_offsets(
    const int32_t* __restrict__ problem_sizes1, int32_t* expert_offsets,
    int32_t* atomic_buffer, const int num_experts) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += problem_sizes1[i * 3];
    expert_offsets[i + 1] = tot_offset;
  }
}

__global__ void compute_arg_sorts(const int* __restrict__ topk_ids,
                                  const int32_t* __restrict__ expert_offsets,
                                  int32_t* input_permutation,
                                  int32_t* output_permutation,
                                  int32_t* atomic_buffer, const int topk_length,
                                  const int topk) {
  int const blk_expert_id = blockIdx.x;
  int const num_experts = gridDim.x;
  int32_t const num_tokens = expert_offsets[num_experts];

  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    int const expert_id = topk_ids[i];
    if (expert_id == -1 && blockIdx.x == 0) {
      // output_permutation is used to re-order the moe outputs. It is
      // used as c2 = c2[c_map], where c2 is a torch.tensor that is the
      // output of the cutlass kernels and c_map is the output_permutation.
      // c2 is initialized to zeros, therefore by setting the output_permutation
      // to num_tokens, we are guaranteed to fill the moe outputs to zero
      // for "invalid" topk_ids.
      output_permutation[i] = num_tokens;
    } else if (expert_id == blk_expert_id) {
      int start = atomicAdd(&atomic_buffer[expert_id], 1);
      input_permutation[start] = i / topk;
      output_permutation[i] = start;
    }
  }
}


void get_cutlass_moe_mm_data_caller(
  const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
  torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
  torch::Tensor& input_permutation, torch::Tensor& output_permutation,
  const int64_t num_experts, const int64_t n, const int64_t k) {
auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
auto options_int32 =
    torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
torch::Tensor atomic_buffer = torch::zeros(num_experts, options_int32);

int num_threads = min(THREADS_PER_EXPERT, topk_ids.numel());
compute_problem_sizes<<<num_experts, num_threads, 0, stream>>>(
    static_cast<const int32_t*>(topk_ids.data_ptr()),
    static_cast<int32_t*>(problem_sizes1.data_ptr()),
    static_cast<int32_t*>(problem_sizes2.data_ptr()),
    static_cast<int32_t*>(atomic_buffer.data_ptr()), topk_ids.numel(), n, k);
compute_expert_offsets<<<1, 1, 0, stream>>>(
    static_cast<const int32_t*>(problem_sizes1.data_ptr()),
    static_cast<int32_t*>(expert_offsets.data_ptr()),
    static_cast<int32_t*>(atomic_buffer.data_ptr()), num_experts);
compute_arg_sorts<<<num_experts, num_threads, 0, stream>>>(
    static_cast<const int32_t*>(topk_ids.data_ptr()),
    static_cast<const int32_t*>(expert_offsets.data_ptr()),
    static_cast<int32_t*>(input_permutation.data_ptr()),
    static_cast<int32_t*>(output_permutation.data_ptr()),
    static_cast<int32_t*>(atomic_buffer.data_ptr()), topk_ids.numel(),
    topk_ids.size(1));
}
*/




/*
void cutlass_scaled_mm_sm75(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_sm80(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_sm89(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias);
*/

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
/*
void cutlass_scaled_mm_sm90(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias);
*/
void cutlass_moe_mm_sm90(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides);

void get_cutlass_moe_mm_data_caller(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k);

#endif

/*
#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
void cutlass_scaled_mm_sm100(torch::Tensor& c, torch::Tensor const& a,
                             torch::Tensor const& b,
                             torch::Tensor const& a_scales,
                             torch::Tensor const& b_scales,
                             std::optional<torch::Tensor> const& bias);
#endif
*/

/*
void cutlass_scaled_mm_azp_sm75(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm80(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm89(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);
*/

/*
#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
void cutlass_scaled_mm_azp_sm90(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);
#endif
*/

/*
bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability) {
  // CUTLASS FP8 kernels need at least
  //   CUDA 12.0 on SM90 systems (Hopper)
  //   CUDA 12.4 on SM89 systems (Lovelace)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12000;
  } else if (cuda_device_capability >= 89) {
    return CUDA_VERSION >= 12040;
  }
#endif

  return false;
}


bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability) {
  // CUTLASS block-quantized FP8 kernels need at least CUDA 12.0
  // and at least SM90 (Hopper)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 90 && cuda_device_capability < 100) {
    return CUDA_VERSION >= 12000;
  }
#endif

  return false;
}
*/

/*
bool cutlass_group_gemm_supported(int64_t cuda_device_capability) {
  // CUTLASS groped FP8 kernels need at least CUDA 12.3
  // and SM90 (Hopper)

#if defined CUDA_VERSION
  if (cuda_device_capability == 90) {
    return CUDA_VERSION >= 12030;
  }
#endif

  return false;
}
*/

/*
void cutlass_scaled_mm(torch::Tensor& c, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias) {
  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment

  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous() &&
                bias->dim() == 1);
  }

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
  if (version_num >= 100) {
    cutlass_scaled_mm_sm100(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

  // Guard against compilation issues for sm90 kernels
#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  if (version_num >= 90 && version_num < 100) {
    // Hopper
    cutlass_scaled_mm_sm90(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_C2X && ENABLE_SCALED_MM_C2X
  if (version_num == 89) {
    // Ada Lovelace
    cutlass_scaled_mm_sm89(c, a, b, a_scales, b_scales, bias);
    return;
  }

  if (version_num >= 80) {
    // Ampere
    cutlass_scaled_mm_sm80(c, a, b, a_scales, b_scales, bias);
    return;
  }

  if (version_num >= 75) {
    // Turing
    cutlass_scaled_mm_sm75(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}
*/

void cutlass_moe_mm(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  int32_t version_num = get_sm_version_num();
#if defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90
  cutlass_moe_mm_sm90(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                      expert_offsets, problem_sizes, a_strides, b_strides,
                      c_strides);
  return;
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm for CUDA device capability: ", version_num,
      ". Required capability: 90");
}

void get_cutlass_moe_mm_data(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k) {
  // This function currently gets compiled only if we have a valid cutlass moe
  // mm to run it for.
#if defined ENABLE_CUTLASS_MOE_SM90
  #if 0
    printf("Jack get_cutlass_moe_mm_data(): torch_binding working defined ENABLE_CUTLASS_MOE_SM90 %d\n", ENABLE_CUTLASS_MOE_SM90);
  #endif
#else
  #if 0:
    printf("Jack get_cutlass_moe_mm_data(): torch_binding working not defined ENABLE_CUTLASS_MOE_SM90\n");
  #endif
#endif
  int32_t version_num = get_sm_version_num();
#if defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90
  get_cutlass_moe_mm_data_caller(topk_ids, expert_offsets, problem_sizes1,
                                 problem_sizes2, input_permutation,
                                 output_permutation, num_experts, n, k);
  return;
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled get_cutlass_moe_mm_data: no cutlass_scaled_mm kernel for "
      "CUDA device capability: ",
      version_num, ". Required capability: 90");
}

/*
void cutlass_scaled_mm_azp(torch::Tensor& c, torch::Tensor const& a,
                           torch::Tensor const& b,
                           torch::Tensor const& a_scales,
                           torch::Tensor const& b_scales,
                           torch::Tensor const& azp_adj,
                           std::optional<torch::Tensor> const& azp,
                           std::optional<torch::Tensor> const& bias) {
  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));
  TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  // bias, azp, azp_adj are all 1d
  // bias and azp_adj have n elements, azp has m elements
  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous());
  }
  if (azp) {
    TORCH_CHECK(azp->numel() == a.size(0) && azp->is_contiguous());
  }
  TORCH_CHECK(azp_adj.numel() == b.size(1) && azp_adj.is_contiguous());

  // azp & bias types
  TORCH_CHECK(azp_adj.dtype() == torch::kInt32);
  TORCH_CHECK(!azp || azp->dtype() == torch::kInt32);
  TORCH_CHECK(!bias || bias->dtype() == c.dtype(),
              "currently bias dtype must match output dtype ", c.dtype());

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));

  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  if (version_num >= 90) {
    cutlass_scaled_mm_azp_sm90(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_C2X && ENABLE_SCALED_MM_C2X
  if (version_num == 89) {
    // Ada Lovelace
    cutlass_scaled_mm_azp_sm89(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }

  if (version_num >= 80) {
    // Ampere
    cutlass_scaled_mm_azp_sm80(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }

  // Turing
  TORCH_CHECK(version_num >= 75);
  cutlass_scaled_mm_azp_sm75(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
  return;
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm_azp for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}

*/