#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/arch/arch.h>
#include <torch/all.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass_moe_helper.cu"
#include "utils.h"

#include "cutlass/util/reference/device/tensor_fill.h"

using namespace cute;

/* Jack debugging */

void print_tensor_info(const torch::Tensor& t, const std::string& name, int max_elements = 20) {
    std::cout << name << " shape: " << t.sizes() << std::endl;
    auto flattened = t.flatten();
    auto numel = std::min((int)flattened.numel(), max_elements);
    std::cout << name << " values (first " << numel << "): [";
    for (int i = 0; i < numel; ++i) {
        std::cout << flattened[i].item<float>(); // 你可以根據實際 dtype 換成 int, half 等
        if (i < numel - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void print_tensor_meta(const torch::Tensor& t, const std::string& name) {
  std::cout << "Tensor: " << name << "\n";
  std::cout << "  Shape: " << t.sizes() << "\n";
  std::cout << "  Dtype: " << t.dtype().name() << "\n";
  std::cout << "  Device: " << t.device() << "\n\n";
}

template <typename T>
void print_tensor_cpu(torch::Tensor t, const std::string& name, int count = 20) {
    t = t.contiguous().cpu();
    std::cout << name << ": ";
    auto ptr = t.data_ptr<T>();
    for (int i = 0; i < std::min((int)t.numel(), count); ++i) {
        std::cout << ptr[i] << " ";
    }
    std::cout << std::endl;
}

// If the DeviceAllocation<> was wrapped as Tensor, convert it back to host array (optional helper)
template <typename T>
void print_device_allocation(const T* device_ptr, int count, const std::string& name) {
    std::vector<T> host_data(count);
    cudaMemcpy(host_data.data(), device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << name << ": ";
    for (int i = 0; i < count; ++i) {
        std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;
}

/* 250609 */

// // Print shape + first 20 values of a regular tensor
// void print_tensor_debug(const std::string& name, const torch::Tensor& t) {
//   std::cout << "[Tensor] " << name << " shape: " << t.sizes() << std::endl;
//   auto flat = t.flatten();
//   int64_t n = std::min<int64_t>(20, flat.numel());
//   std::cout << "  values: [ ";
//   for (int64_t i = 0; i < n; ++i) {
//     std::cout << flat[i].item<float>() << " ";
//   }
//   std::cout << "]" << std::endl;
// }
// void print_tensor_debug(const std::string& name, const at::Tensor& t) {
//   std::cout << "[Tensor] " << name << " shape: " << t.sizes()
//             << ", dtype: " << t.scalar_type() << std::endl;

//   int64_t n = std::min<int64_t>(20, t.numel());
//   auto flat = t.view({-1});

//   std::cout << "  values: [ ";
//   if (t.scalar_type() == torch::kFloat32) {
//     for (int64_t i = 0; i < n; ++i)
//       std::cout << flat[i].item<float>() << " ";
//   } else if (t.scalar_type() == torch::kFloat16) {
//     for (int64_t i = 0; i < n; ++i)
//       std::cout << static_cast<float>(flat[i].item<at::Half>()) << " ";
//   } else if (t.scalar_type() == torch::kBFloat16) {
//     for (int64_t i = 0; i < n; ++i)
//       std::cout << static_cast<float>(flat[i].item<at::BFloat16>()) << " ";
//   } else if (t.scalar_type() == torch::kInt32) {
//     for (int64_t i = 0; i < n; ++i)
//       std::cout << flat[i].item<int32_t>() << " ";
//   } else if (t.scalar_type() == torch::kInt64) {
//     for (int64_t i = 0; i < n; ++i)
//       std::cout << flat[i].item<int64_t>() << " ";
//   } else if (t.scalar_type() == torch::ScalarType::Float8_e4m3fn ||
//              t.scalar_type() == torch::ScalarType::Float8_e5m2 ||
//              t.scalar_type() == torch::kUInt8) {
//     auto tmp = torch::empty({n}, torch::kUInt8);
//     cudaMemcpy(tmp.data_ptr(), t.data_ptr(), n, cudaMemcpyDeviceToHost);
//     auto acc = tmp.accessor<uint8_t, 1>();
//     for (int64_t i = 0; i < n; ++i)
//       std::cout << static_cast<int>(acc[i]) << " ";
//   } else {
//     std::cout << "(unsupported dtype)";
//   }
//   std::cout << "]" << std::endl;
// }

// template <typename T>
// void print_tensor_debug(const std::string& name, const at::Tensor& t, int num_values = 20) {
//     std::cout << "============start==============" << std::endl;
//     std::cout << "[Tensor] " << name << " shape: " << t.sizes() << ", dtype: " << t.dtype() << std::endl;

//     try {
//         // 降維
//         auto flat = t.flatten();

//         // 嘗試轉換 dtype 並複製到 CPU（不直接用 .data()）
//         at::Tensor cpu_tensor;
//         if (flat.device().is_cuda()) {
//             cpu_tensor = flat.to(at::kFloat, /*non_blocking=*/false).cpu();
//         } else {
//             cpu_tensor = flat.to(at::kFloat);
//         }

//         // 印出前 num_values 個值
//         auto accessor = cpu_tensor.accessor<float, 1>();
//         std::cout << "  values: [ ";
//         for (int i = 0; i < std::min(num_values, (int)cpu_tensor.numel()); ++i) {
//             std::cout << accessor[i] << " ";
//         }
//         std::cout << "]" << std::endl;
//     } catch (const std::exception& e) {
//         std::cerr << "  ERROR converting or printing tensor " << name << ": " << e.what() << std::endl;
//     }
// }
// .view報錯
// void print_tensor_debug(const std::string& name, const at::Tensor& t, int max_elements = 20) {
//     std::cout << "[Tensor] " << name << " shape: " << t.sizes() 
//               << ", dtype: " << t.dtype() << std::endl;

//     at::Tensor cpu_tensor;
//     try {
//         // 轉換為 float32 或 int64 以供打印（根據 dtype）
//         switch (t.scalar_type()) {
//             case at::kFloat:
//                 cpu_tensor = t.flatten().cpu();
//                 break;
//             case at::kHalf:
//             case at::kBFloat16:
//             case at::kFloat8_e4m3fn:
//             case at::kFloat8_e5m2:
//                 cpu_tensor = t.flatten().to(at::kFloat).cpu();
//                 break;
//             case at::kInt:
//             case at::kShort:
//             case at::kByte:
//             case at::kLong:
//                 cpu_tensor = t.flatten().to(at::kLong).cpu();
//                 break;
//             case at::kBool:
//                 cpu_tensor = t.flatten().to(at::kByte).cpu();
//                 break;
//             default:
//                 std::cerr << "Unsupported dtype: " << t.dtype() << std::endl;
//                 return;
//         }

//         // 根據實際 dtype 印出內容
//         std::cout << "  values: [ ";
//         int64_t total = cpu_tensor.numel();
//         int64_t count = std::min<int64_t>(total, max_elements);

//         if (cpu_tensor.scalar_type() == at::kFloat) {
//             auto accessor = cpu_tensor.accessor<float, 1>();
//             for (int64_t i = 0; i < count; ++i) {
//                 std::cout << accessor[i] << " ";
//             }
//         } else if (cpu_tensor.scalar_type() == at::kLong) {
//             auto accessor = cpu_tensor.accessor<int64_t, 1>();
//             for (int64_t i = 0; i < count; ++i) {
//                 std::cout << accessor[i] << " ";
//             }
//         } else if (cpu_tensor.scalar_type() == at::kByte) {
//             auto accessor = cpu_tensor.accessor<uint8_t, 1>();
//             for (int64_t i = 0; i < count; ++i) {
//                 std::cout << static_cast<int>(accessor[i]) << " ";
//             }
//         } else {
//             std::cerr << "Unsupported print format for dtype: " << cpu_tensor.scalar_type() << std::endl;
//             return;
//         }

//         if (total > count) std::cout << "...";
//         std::cout << "]" << std::endl;
//     } catch (const std::exception& e) {
//         std::cerr << "Error printing tensor: " << e.what() << std::endl;
//     }
// }

void print_tensor_debug(const std::string& name, const at::Tensor& t, int max_elements = 20) {
    std::cout << "[Tensor] " << name << " shape: " << t.sizes() 
              << ", dtype: " << t.dtype() << std::endl;

    at::Tensor cpu_tensor;
    try {
        switch (t.scalar_type()) {
            case at::kFloat:
                cpu_tensor = t.contiguous().reshape({-1}).cpu();
                break;
            case at::kHalf:
            case at::kBFloat16:
            case at::kFloat8_e4m3fn:
            case at::kFloat8_e5m2:
                cpu_tensor = t.contiguous().to(at::kFloat).reshape({-1}).cpu();
                break;
            case at::kInt:
            case at::kShort:
            case at::kByte:
            case at::kLong:
                cpu_tensor = t.contiguous().to(at::kLong).reshape({-1}).cpu();
                break;
            case at::kBool:
                cpu_tensor = t.contiguous().to(at::kByte).reshape({-1}).cpu();
                break;
            default:
                std::cerr << "Unsupported dtype: " << t.dtype() << std::endl;
                return;
        }

        std::cout << "  values: " << name << " [ ";
        int64_t total = cpu_tensor.numel();
        int64_t count = std::min<int64_t>(total, max_elements);

        if (cpu_tensor.scalar_type() == at::kFloat) {
            auto accessor = cpu_tensor.accessor<float, 1>();
            for (int64_t i = 0; i < count; ++i) {
                std::cout << accessor[i] << " ";
            }
        } else if (cpu_tensor.scalar_type() == at::kLong) {
            auto accessor = cpu_tensor.accessor<int64_t, 1>();
            for (int64_t i = 0; i < count; ++i) {
                std::cout << accessor[i] << " ";
            }
        } else if (cpu_tensor.scalar_type() == at::kByte) {
            auto accessor = cpu_tensor.accessor<uint8_t, 1>();
            for (int64_t i = 0; i < count; ++i) {
                std::cout << static_cast<int>(accessor[i]) << " ";
            }
        } else {
            std::cerr << "Unsupported print format for dtype: " << cpu_tensor.scalar_type() << std::endl;
            return;
        }

        if (total > count) std::cout << "...";
        std::cout << "]" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error printing tensor: " << e.what() << std::endl;
    }
}

void print_tensor_from_ptr_debug(const std::string& name, const void* ptr, at::ScalarType dtype, std::vector<int64_t> shape) {
  std::cout << "[From Ptr] " << name << " shape: " << shape
            << ", dtype: " << dtype << std::endl;

  int64_t numel = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  int64_t n = std::min<int64_t>(20, numel);

  std::cout << "[From Ptr] " << name << "  values: [ ";

  if (dtype == torch::kFloat32) {
    std::vector<float> tmp(n);
    cudaMemcpy(tmp.data(), ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (auto v : tmp) std::cout << v << " ";
  } else if (dtype == torch::kFloat16) {
    std::vector<at::Half> tmp(n);
    cudaMemcpy(tmp.data(), ptr, n * sizeof(at::Half), cudaMemcpyDeviceToHost);
    for (auto v : tmp) std::cout << static_cast<float>(v) << " ";
  } else if (dtype == torch::kBFloat16) {
    std::vector<at::BFloat16> tmp(n);
    cudaMemcpy(tmp.data(), ptr, n * sizeof(at::BFloat16), cudaMemcpyDeviceToHost);
    for (auto v : tmp) std::cout << static_cast<float>(v) << " ";
  } else if (dtype == torch::kUInt8 || dtype == torch::ScalarType::Float8_e4m3fn || dtype == torch::ScalarType::Float8_e5m2) {
    std::vector<uint8_t> tmp(n);
    cudaMemcpy(tmp.data(), ptr, n, cudaMemcpyDeviceToHost);
    for (auto v : tmp) std::cout << static_cast<int>(v) << " ";
  } else if (dtype == torch::kInt32) {
    std::vector<int32_t> tmp(n);
    cudaMemcpy(tmp.data(), ptr, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (auto v : tmp) std::cout << v << " ";
  } else if (dtype == torch::kInt64) {
    std::vector<int64_t> tmp(n);
    cudaMemcpy(tmp.data(), ptr, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
    for (auto v : tmp) std::cout << v << " ";
  } else {
    std::cout << "(unsupported dtype)";
  }
  std::cout << "]" << std::endl;
}

void print_tensor_full_debug(const std::string& name, const at::Tensor& t, int max_print = 20) {
    try {
        at::Tensor t_cpu = t.to(torch::kCPU, /*non_blocking=*/true).contiguous();
        at::Tensor flat = t_cpu.reshape({-1});

        std::cout << name << " shape: " << t.sizes()
                  << ", stride: " << t.strides()
                  << ", dtype: " << t.dtype() << std::endl;

        std::cout << "  values: " << name << " [ ";
        for (int i = 0; i < std::min<int64_t>(max_print, flat.numel()); ++i) {
            std::cout << flat[i].item<float>() << " ";
        }
        std::cout << "]" << std::endl;

    } catch (const std::exception& e) {
        std::cout << name << " shape: " << t.sizes()
                  << ", stride: " << t.strides()
                  << ", dtype: " << t.dtype() << std::endl;
        std::cout << "  values: " << name << " [Error] " << e.what() << std::endl;
    }
}


// // Print shape + first 20 values of a pointer tensor (int64)
// void print_pointer_tensor_debug(const std::string& name, const torch::Tensor& ptr_tensor) {
//   std::cout << "[Pointer] " << name << " shape: " << ptr_tensor.sizes() << std::endl;
//   int64_t n = std::min<int64_t>(20, ptr_tensor.numel());
//   auto ptr_data = ptr_tensor.cpu().data_ptr<int64_t>();
//   std::cout << "  addresses: [ ";
//   for (int64_t i = 0; i < n; ++i) {
//     std::cout << ptr_data[i] << " ";
//   }
//   std::cout << "]" << std::endl;
// }

// // Print actual tensor from pointer tensor's first address (e.g., a_ptrs[0])
// void print_tensor_from_ptr(const std::string& name, const torch::Tensor& full_tensor, int64_t index) {
//   torch::Tensor slice;
//   if (full_tensor.dim() >= 2) {
//     slice = full_tensor[index];  // e.g., a[0]
//   } else {
//     std::cout << "[TensorFromPtr] " << name << " has dim < 2, skip." << std::endl;
//     return;
//   }

//   std::cout << "[From Ptr] " << name << "[" << index << "] shape: " << slice.sizes() << std::endl;
//   auto flat = slice.flatten();
//   int64_t n = std::min<int64_t>(20, flat.numel());
//   std::cout << "  values: [ ";
//   for (int64_t i = 0; i < n; ++i) {
//     std::cout << flat[i].item<float>() << " ";
//   }
//   std::cout << "]" << std::endl;
// }

/* Jack porting */
#if 1 // all
#if 0 // wrong way
/// Fills a tensor with random values with a uniform random distribution.
template <typename Element>
void BlockFillRandomUniform(
  Element *ptr,
  size_t capacity,
  uint64_t seed,                          ///< seed for RNG
  typename RealType<Element>::Type max,   ///< upper bound of distribution
  typename RealType<Element>::Type min,   ///< lower bound for distribution
  int bits = -1,                          ///< If non-negative, specifies number of fractional bits that
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.
  double pnan = 0,                        ///< Percentage of NaN elements.
  cudaStream_t stream = nullptr) {

  using RandomFunc = detail::RandomUniformFunc<Element>;

  typename RandomFunc::Params params(seed, max, min, bits, pnan);

  BlockForEach<Element, RandomFunc>(ptr, capacity, params, /*grid_size*/0, /*block_size*/0, stream);
}
#endif

template <class Element, class ScopeMin = std::nullopt_t, class ScopeMax = std::nullopt_t>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023,
  ScopeMin scope_min = std::nullopt, ScopeMax scope_max = std::nullopt) {

  double _scope_max, _scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  if (bits_input == 1) {
    _scope_max = 2;
    _scope_min = 0;
  } else if (bits_input <= 8) {
    _scope_max = 2;
    _scope_min = -2;
  } else if (bits_input == 16) {
    _scope_max = 5;
    _scope_min = -5;
  } else {
    _scope_max = 8;
    _scope_min = -8;
  }
  if constexpr (!std::is_same_v<ScopeMax, std::nullopt_t>) {
    _scope_max = scope_max;
  }
  if constexpr (!std::is_same_v<ScopeMin, std::nullopt_t>) {
    _scope_min = scope_min;
  }

  // BlockFillRandomUniform( // wrong way
  /* Solv: include #include "cutlass/util/reference/device/tensor_fill.h" */
  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, (Element) _scope_max, (Element) _scope_min, 0);

  return true;
}

#if 0 // wrong way
namespace detail {

/// Computes a random uniform distribution
template <typename Element>                ///< Element type 
struct RandomUniformFunc {

  using FloatType = typename std::conditional<
    (sizeof(Element) > 4),
    double,
    float>::type;

  using IntType = typename std::conditional<
    (sizeof(Element) > 4),
    int64_t,
    int>::type;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    uint64_t seed;
    FloatType range;
    FloatType max;
    int int_scale;
    double pnan;
    FloatType float_scale_up;
    FloatType float_scale_down;
    int exclude_zero;           ///< If non-negative, excludes zeros

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      uint64_t seed_ = 0, 
      Element max_ = 1,
      Element min = 0,
      int int_scale_ = -1,
      double pnan_ = 0,
      int exclude_zero_ = -1
    ):
      seed(seed_), 
      range(static_cast<FloatType>(max_) - static_cast<FloatType>(min)), 
      max(static_cast<FloatType>(max_)),
      int_scale(int_scale_),
      pnan(pnan_),
      exclude_zero(exclude_zero_) {
      
      float_scale_up = FloatType(IntType(1) << int_scale); // scale up to clamp low order bits
      float_scale_down = FloatType(1) / FloatType(IntType(1) << int_scale);

      // Handle cases where min = 0 or max = 0 for excluding zeros
      if (exclude_zero >= 0) {
        range = (min == Element(0)) ? range - FloatType(1): range;
        max = (max_ == Element(0)) ? max - FloatType(1): max; 
      }
    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// RNG state object
  curandState_t rng_state;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  RandomUniformFunc(Params const &params): params(params) {

    uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(params.seed, gtid, 0, &rng_state);
  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  Element operator()() {

    // Draw random float in [0.0, 1.0] to determine if element should be NaN.
    if constexpr (std::numeric_limits<Element>::has_quiet_NaN) {
      if (params.pnan > 0 && (curand_uniform(&rng_state) < (params.pnan))) {
        return Element(NAN);
      }
    }

    FloatType rnd = random_uniform_float<FloatType>(&rng_state);
    rnd = params.max - params.range * rnd;

    // Random values are cast to integer after scaling by a power of two to facilitate error
    // testing
    Element result;

    if (params.int_scale >= 0) {
      rnd = FloatType(std::llround(rnd * params.float_scale_up));
      result = Element(rnd * params.float_scale_down);
    }
    else {
      result = Element(rnd);
    }

    if (params.exclude_zero >=0 && result == Element(0.0)) {
      if (rnd > FloatType(0)) {
        rnd = std::min(params.max, rnd + FloatType(1));
      } else {
        rnd = std::max((params.max - params.range), rnd - FloatType(1));
      }
      result = Element(rnd);
    }

    return result;
  }
};

} // namespace detail
#endif

#endif

/* exp68 (yichen recommened) */
using ElementBlockScale   = float;

cutlass::DeviceAllocation<ElementBlockScale> blockscale_block_A;
cutlass::DeviceAllocation<ElementBlockScale> blockscale_block_B;

std::vector<int64_t> offset_blockscale_A;
std::vector<int64_t> offset_blockscale_B;

// 全局不能用 typename, 函數內有模板才行
// std::vector<typename ScheduleConfig::LayoutSFA> layout_SFA_host;
// std::vector<typename ScheduleConfig::LayoutSFB> layout_SFB_host;

// cutlass::DeviceAllocation<typename ScheduleConfig::LayoutSFA> layout_SFA;
// cutlass::DeviceAllocation<typename ScheduleConfig::LayoutSFB> layout_SFB;

constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;
using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;
using LayoutSFA     = decltype(ScaleConfig::deduce_layoutSFA());    // Layout type for SFA matrix operand
using LayoutSFB     = decltype(ScaleConfig::deduce_layoutSFB());    // Layout type for SFB matrix operand

std::vector<LayoutSFA> layout_SFA_host;
std::vector<LayoutSFB> layout_SFB_host;

cutlass::DeviceAllocation<LayoutSFA> layout_SFA;
cutlass::DeviceAllocation<LayoutSFB> layout_SFB;

using ElementAccumulator = float;
cutlass::DeviceAllocation<const ElementAccumulator *> ptr_blockscale_A;
cutlass::DeviceAllocation<const ElementAccumulator *> ptr_blockscale_B;


// using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
template <typename OutType, typename ScheduleConfig, typename LayoutD>
void launch_sm90_fp8_blockwise_scaled_group_mm(
    torch::Tensor& out_ptrs,
    const torch::Tensor& a_ptrs,
    const torch::Tensor& b_ptrs,
    const torch::Tensor& a_scales_ptrs,
    const torch::Tensor& b_scales_ptrs,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = OutType; // Jack 250602: origin以下解釋
  // using ElementC = cutlass::float_e4m3_t; // Jack 250602: 強制改成exp68type 單純改exp68format (保留後面那邊nullptr) 結果一樣.
  // epx68 uses ElementC = cutlass::float_e4m3_t; 
  // 但這邊是傳入的 <OutType>
  // sm90_fp8_blockwise_group_mm_dispatch_shape<cutlass::bfloat16_t>(
  // or
  // sm90_fp8_blockwise_group_mm_dispatch_shape<cutlass::half_t>(
  using ElementD = ElementC;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = LayoutD;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using FusionOperation   = cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>; // 250602 new
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      void, // 250602 origin
      // ElementC, // 250602 new
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutC*, // 250602 D=C
      AlignmentC, // 250602 D=C
      // typename ScheduleConfig::EpilogueSchedule, // 250602 new
      // FusionOperation // 250602 new new
      // >::CollectiveOp; // 250602 new
      typename ScheduleConfig::EpilogueSchedule>::CollectiveOp; // 250602 origin
      // typename ScheduleConfig::EpilogueSchedule, // 250612 new from non-group
      // typename ScheduleConfig::StoreEpilogueCompute>::CollectiveOp; // 250612 new from non-group


  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutA*, typename ScheduleConfig::LayoutSFA*>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB*, typename ScheduleConfig::LayoutSFB*>,
      AlignmentB,
      ElementAccumulator,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename ScheduleConfig::KernelSchedule>::CollectiveOp;

  // using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>; // 250602 origin
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>; // 250602 new
  // using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, cutlass::gemm::StreamKScheduler>; // 250612 new - 1 - fail cannot compile
  // using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, cutlass::gemm::PersistentScheduler>; // 250612 new - 2 - fail cannot compile

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  int num_experts = (int)expert_offsets.size(0);
  // Create an instance of the GEMM
  Gemm gemm_op;

  /* Jack: TODO example68, test it with no cuda graph */
  /* Jack: TODO example68, test it with no cuda graph */
  /* Jack: TODO example68, test it with no cuda graph */
  /* Jack: TODO example68, test it with no cuda graph */
  /* Jack: TODO example68, test it with no cuda graph */
  /* Jack: TODO example68, test it with no cuda graph */
  // int groups = num_experts; int options.groups = num_experts;

  auto problem_sizes_cpu = problem_sizes.to(torch::kCPU);
  auto* problem_sizes_cpu_ptr = problem_sizes_cpu.data_ptr<int32_t>();

  


  /* exp68 (yichen recommened) */
  ////////////////////////////////////////////
  // /* Jack: 0602 alloc()-1 */
  ////////////////////////////////////////////
  int64_t total_elements_blockscale_A = 0;
  int64_t total_elements_blockscale_B = 0;
  offset_blockscale_A.clear();
  offset_blockscale_B.clear();

  for (int32_t i = 0; i < num_experts; ++i) {
      // auto problem = problem_sizes_cpu_ptr[i];
      // int32_t M = get<0>(problem);
      // int32_t N = get<1>(problem);
      // int32_t K = get<2>(problem);
      int32_t M = problem_sizes_cpu_ptr[(i * 3) + 0];
      int32_t N = problem_sizes_cpu_ptr[(i * 3) + 1];
      int32_t K = problem_sizes_cpu_ptr[(i * 3) + 2];

      auto group_layout_SFA = ScheduleConfig::ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
      auto group_layout_SFB = ScheduleConfig::ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

      offset_blockscale_A.push_back(total_elements_blockscale_A);
      offset_blockscale_B.push_back(total_elements_blockscale_B);


      int64_t elements_blockscale_A = size(filter_zeros(group_layout_SFA)); // 
      int64_t elements_blockscale_B = size(filter_zeros(group_layout_SFB)); //
 
      total_elements_blockscale_A += elements_blockscale_A;
      total_elements_blockscale_B += elements_blockscale_B;

      layout_SFA_host.push_back(group_layout_SFA);
      layout_SFB_host.push_back(group_layout_SFB);
  }
  // /* Jack: 0602 alloc-3 */
  blockscale_block_A.reset(total_elements_blockscale_A);
  blockscale_block_B.reset(total_elements_blockscale_B);


  ////////////////////////////////////////////
  // /* Jack: 0602 initialize()-1 */
  ////////////////////////////////////////////
  // save to gpu
  layout_SFA.reset(num_experts);
  layout_SFA.copy_from_host(layout_SFA_host.data()); // final one, use it
  layout_SFB.reset(num_experts);
  layout_SFB.copy_from_host(layout_SFB_host.data()); // final one, use it

  // further process it
  std::vector<ElementBlockScale *> ptr_blockscale_A_host(num_experts);
  std::vector<ElementBlockScale *> ptr_blockscale_B_host(num_experts);
  for (int i = 0; i < num_experts; i++) {
    ptr_blockscale_A_host.at(i) = blockscale_block_A.get() + offset_blockscale_A.at(i);
    ptr_blockscale_B_host.at(i) = blockscale_block_B.get() + offset_blockscale_B.at(i);
  }
  // save to gpu
  ptr_blockscale_A.reset(num_experts);
  ptr_blockscale_A.copy_from_host(ptr_blockscale_A_host.data()); // final one, use it

  ptr_blockscale_B.reset(num_experts);
  ptr_blockscale_B.copy_from_host(ptr_blockscale_B_host.data()); // final one, use it

  uint64_t seed = 2023;
  initialize_block(blockscale_block_A, seed + 2025, -1, 1);
  initialize_block(blockscale_block_B, seed + 2026, -1, 1);
  /* exp68 (yichen recommened) end after here use it */


#if 0 // origin
  typename GemmKernel::MainloopArguments mainloop_args{
    static_cast<const ElementA**>(a_ptrs.data_ptr()),
    static_cast<StrideA*>(stride_a.data_ptr()),
    static_cast<const ElementB**>(b_ptrs.data_ptr()),
    static_cast<StrideB*>(stride_b.data_ptr()),
    static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
    reinterpret_cast<typename ScheduleConfig::LayoutSFA*>(layout_sfa.data_ptr()), // original
    // layout_SFA.get(), // exp68 (yichen recommened)
    static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
    reinterpret_cast<typename ScheduleConfig::LayoutSFB*>(layout_sfb.data_ptr())}; // original
    // layout_SFB.get() // exp68 (yichen recommened)
    // }; // exp68 (yichen recommened)
#else  // exp68
  typename GemmKernel::MainloopArguments mainloop_args{
    static_cast<const ElementA**>(a_ptrs.data_ptr()),
    static_cast<StrideA*>(stride_a.data_ptr()),
    static_cast<const ElementB**>(b_ptrs.data_ptr()),
    static_cast<StrideB*>(stride_b.data_ptr()),
    // static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()), // origin ->
    ptr_blockscale_A.get(), // exp68 (yichen recommened)
    // reinterpret_cast<typename ScheduleConfig::LayoutSFA*>(layout_sfa.data_ptr()), // original ->
    layout_SFA.get(), // exp68 (yichen recommened)
    // static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()), // origin ->
    ptr_blockscale_B.get(), // exp68 (yichen recommened)
    // reinterpret_cast<typename ScheduleConfig::LayoutSFB*>(layout_sfb.data_ptr())}; // original ->
    layout_SFB.get() // exp68 (yichen recommened)
    }; // exp68 (yichen recommened)
#endif
  /* Jack debug */
  // print_device_allocation(layout_SFA.get(), 20, "layout_SFA");
  // print_device_allocation(layout_SFB.get(), 20, "layout_SFB");



  // sm100 origin
  // typename GemmKernel::MainloopArguments mainloop_args{
  //     static_cast<const ElementA**>(a_ptrs.data_ptr()),
  //     static_cast<StrideA*>(stride_a.data_ptr()),
  //     static_cast<const ElementB**>(b_ptrs.data_ptr()),
  //     static_cast<StrideB*>(stride_b.data_ptr()),
  //     static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
  //     reinterpret_cast<typename ScheduleConfig::LayoutSFA*>(layout_sfa.data_ptr()),
  //     static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
  //     reinterpret_cast<typename ScheduleConfig::LayoutSFB*>(layout_sfb.data_ptr())};

  // sm_count is the number of SMs on the current device, since we here support H20 (SM90), so we set it to 78
  // hw_info.sm_count = 78;
  int device_id = 0;
  cutlass::KernelHardwareInfo hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(device_id);
  // cutlass::KernelHardwareInfo hw_info;
  // hw_info.device_id = a_tensors.device().index();
  // hw_info.sm_count =
  //     cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
  //         hw_info.device_id);

  // 250602 origin
  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      static_cast<StrideC*>(stride_c.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(stride_c.data_ptr())};

  // 250602 new - fail
  // typename GemmKernel::EpilogueArguments epilogue_args{
  //     {},
  //     // nullptr, // 這個不改成exp68format的話會報錯 單純改exp68format保留這邊nullptr結果一樣
  //     static_cast<ElementC**>(out_ptrs.data_ptr()), // 改成exp68format 依然錯 所以這條路不通
  //     static_cast<StrideC*>(stride_c.data_ptr()),
  //     static_cast<ElementD**>(out_ptrs.data_ptr()),
  //     static_cast<StrideC*>(stride_c.data_ptr())};

  UnderlyingProblemShape* problem_sizes_as_shapes = static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  at::cuda::CUDAGuard device_guard{(char)a_ptrs.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a_ptrs.get_device());

  auto can_implement_status = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess, "Failed to implement GEMM");

  auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}


template <typename OutType>
void sm90_fp8_blockwise_group_mm_dispatch_shape(
    torch::Tensor& output,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  // Check the first matrix size to decide on the configuration
  // Assuming all matrices in the group have similar size characteristics
  // bool use_small_config = a[0].size(0) <= 128;
  // Ref: origin sm100
  struct MmaConfig1_origin {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _32, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
    using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    // using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    // using KernelSchedule =
    //           cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
    // using EpilogueSchedule =
    //           cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
    // using ScaleConfig = // TODO cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
    //     cutlass::detail::Sm100BlockwiseScaleConfig<128, 1, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<128, 1, 128>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  // (WIP) SM90-friendly configs
  struct MmaConfig1_old {
    /* Jack: note
     * MmaTileShape & ClusterShape 不影響精度
     * ScaleConfig 影響精度
     */
    
    // origin
    // using ElementA = cutlass::float_e4m3_t;
    // using MmaTileShape = Shape<_128, _32, _128>;
    // using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    // // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    // // using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    // using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
    // using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    // using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<128, 1, 128>;
    //     // cutlass::detail::Sm100BlockwiseScaleConfig<128, 1, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    // using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    // using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    // gpt建議改origin精度的解法, 滿近說真話的
    // using ElementA = cutlass::float_e4m3_t;
    // using MmaTileShape = Shape<_128, _128, _128>;
    // using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    // // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    // // using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    // using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
    // using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    // using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<32, 1, 128>;
    // // using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<1, 32, 128>; // [ret=cannot compile]
    //     // cutlass::detail::Sm100BlockwiseScaleConfig<128, 1, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    // using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    // using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    // 250527
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _128, _128>;
    // using MmaTileShape = Shape<_128, _32, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // gpt - Layout type for SFB matrix operand
    // using ClusterShape = Shape<_1, _2, _1>;  // exp68 - Layout type for SFB matrix operand
    // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    // using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
    using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    // using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<32, 1, 128>; // try [ret=胡話]
    using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<128, 1, 128>; // 這跟sm100 config2一樣了, example68=1,128,128  [ret=胡話]
    // using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>; // example68=1,128,128. sm100也有這種shpae 但, mmatile也跟著比較小=32 [ret=illegal memory]
        // cutlass::detail::Sm100BlockwiseScaleConfig<128, 1, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    // // (gpt)
    // using ElementA = cutlass::float_e4m3_t;
    // using MmaTileShape = Shape<_64, _128, _64>;  // Hopper-friendly
    // using ClusterShape = Shape<_1, _2, _1>;      // TMA 需要協作兩個 warp
    // // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative; // gpt
    // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum; // mine
    // using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    // // using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<64, 2, 128, cute::UMMA::Major::M, cute::UMMA::Major::K>;
    // using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<64, 2, 128>;
    // using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());  // = ColumnMajor
    // using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());  // = RowMajor

    // 全用sm90 68example (works but still accuracy wrong)
    // using ElementA = cutlass::float_e4m3_t;
    // using MmaTileShape = Shape<_128, _128, _128>;
    // using ClusterShape = Shape<_1, _2, _1>;  // Layout type for SFB matrix operand
    // using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
    // using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    // using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
    // using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    // using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };

  // struct MmaConfig1 {
  //   using ElementA = cutlass::float_e4m3_t;
  //   using MmaTileShape = Shape<_128, _128, _128>;
  //   using ClusterShape = Shape<_1, _2, _1>;  // Layout type for SFB matrix operand
  //   using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
  //   using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  //   using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
  //   using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  //   using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  // };

  // 0529 with yichen
  // struct MmaConfig2 {
  //   using ElementA = cutlass::float_e4m3_t;
  //   using MmaTileShape = Shape<_128, _128, _128>;
  //   using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
  //   using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
  //   using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  //   using ScaleConfig =
  //       cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
  //   using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  //   using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  // };


  // try origin config1 again cuz colum major
  struct MmaConfig1 {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _32, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
    using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<128, 1, 128>; // origin compile fail
    // using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>; // new cuz compile fail
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  // 0612 trying
  struct MmaConfig2 {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum; // origin
    //  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum; // 250612 new from non-groupq (fail)
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    using ScaleTileShape = Shape<_1, _128, _128>;
    using ScaleConfig =
          // decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(ScaleTileShape{})); // 250612 new from non-group
        cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>; // origin
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
    // using StoreEpilogueCompute = typename cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>; // 250612 new from non-group
  };
  // struct MmaConfig2 {
  //   using ElementA = cutlass::float_e4m3_t;
  //   using MmaTileShape = Shape<_128, _128, _128>;
  //   // using ClusterShape = Shape<_1, _2, _1>;  // or this?
  //   using ClusterShape = Shape<_1, _2, _1>;  // Layout type for SFB matrix operand
  //   using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
  //   using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  //   using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>; // illigal when serving
  //   // using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<128, 1, 128>; // illigal killed when starup
  //   using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  //   using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  // };


#if 0 // testing: only use MmaConfig1 for testing,  turn  2,3 off 這兩個都還沒用對. 基本上config2=config3.....
  struct MmaConfig2 {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
    using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    // using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    // using KernelSchedule =
    //           cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
    // using EpilogueSchedule =
    //           cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
    // using ScaleConfig = // TODO cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
    //     cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    // using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
    using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<128, 128, 128>; // O
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  struct MmaConfig3 {
    using ElementA = cutlass::float_e4m3_t;
    // using MmaTileShape = Shape<_64, _128, _128>;
    using MmaTileShape = Shape<_128, _128, _128>; // 90用KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccu 沒辦法
    using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
    using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    // using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    // using KernelSchedule =
    //           cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
    // using EpilogueSchedule =
    //           cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
    // using ScaleConfig = // TODO cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
    //     cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    // using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
    using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<128, 128, 128>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
#endif
  int num_experts = (int)expert_offsets.size(0);
  torch::TensorOptions options_int = torch::TensorOptions().dtype(torch::kInt64).device(a.device());
  torch::Tensor problem_sizes_transpose = torch::empty(num_experts * 3, options_int);
  torch::Tensor output_t = output.t();
  torch::Tensor a_t = a.t();
  torch::Tensor b_t = b.transpose(1, 2);
  torch::Tensor scales_a_t = scales_a.t();
  torch::Tensor scales_b_t = scales_b.transpose(1, 2);

  /* [基本上rets=胡話] */
  // torch::Tensor output_t = output; // [remove ret=胡話] [remove "output = output_t.t()"" ret=胡話]
  // torch::Tensor a_t = a; // [remove ret=胡話]
  // torch::Tensor b_t = b;  // [remove ret=胡話]
  // torch::Tensor scales_a_t = scales_a; // [remove ret=胡話]
  // torch::Tensor scales_b_t = scales_b; // [remove ret=胡話]


#if 1 // Jack for testing
  // Jack for testing
  // // if (a.size(0) <= 512 && a.size(1) >= 2048) {
  //   run_get_group_gemm_starts<MmaConfig1::LayoutSFA, MmaConfig1::LayoutSFB, MmaConfig1::ScaleConfig>(
  //         expert_offsets,
  //         a_ptrs,
  //         b_ptrs,
  //         out_ptrs,
  //         a_scales_ptrs,
  //         b_scales_ptrs,
  //         b_t,
  //         a_t,
  //         output_t,
  //         scales_b_t,
  //         scales_a_t,
  //         layout_sfa,
  //         layout_sfb,
  //         problem_sizes,
  //         problem_sizes_transpose,
  //         true);
  //   // launch_sm90_fp8_blockwise_scaled_group_mm<OutType, MmaConfig1, cutlass::layout::RowMajor>( // [ret=illegal memory]
  //   launch_sm90_fp8_blockwise_scaled_group_mm<OutType, MmaConfig1, cutlass::layout::ColumnMajor>(
  //       out_ptrs,
  //       a_ptrs,
  //       b_ptrs,
  //       a_scales_ptrs,
  //       b_scales_ptrs,
  //       stride_a,
  //       stride_b,
  //       stride_c,
  //       layout_sfa,
  //       layout_sfb,
  //       problem_sizes_transpose,
  //       expert_offsets,
  //       workspace);
  //   output = output_t.t(); // [remove doens't help]

  // 250605 - Debug info print 跟exp68比較
  // std::cout << "=== Tensor Info Debug ===" << std::endl;
  // print_tensor_meta(a_t, "meta a_t");
  // print_tensor_meta(b_t, "meta b_t");
  // print_tensor_meta(a_ptrs, "meta a_ptrs");
  // print_tensor_meta(b_ptrs, "meta b_ptrs");
  // print_tensor_meta(a_scales_ptrs, "meta a_scales_ptrs");
  // print_tensor_meta(b_scales_ptrs, "meta b_scales_ptrs");
  // print_tensor_meta(layout_sfa, "meta layout_sfa");
  // print_tensor_meta(layout_sfb, "meta layout_sfb");
  // print_tensor_meta(problem_sizes, "meta problem_sizes");

  // print_tensor_info(a, "a_t");
  // // print_tensor_info(b, "b_t");
  // print_tensor_info(a_ptrs, "a_ptrs");
  // print_tensor_info(b_ptrs, "b_ptrs");
  // print_tensor_info(a_scales_ptrs, "a_scales_ptrs");
  // print_tensor_info(b_scales_ptrs, "b_scales_ptrs");
  // print_tensor_info(layout_sfa, "layout_sfa");
  // print_tensor_info(layout_sfb, "layout_sfb");
  // print_tensor_info(problem_sizes, "problem_sizes");


  // std::cout << "============start==============" << std::endl;

  // print_tensor_cpu<uint8_t>(a_t, "a_t");  // float8 用 uint8_t 看原始位元
  // // print_tensor_cpu<uint8_t>(b_t, "b_t");

  // // print_tensor_cpu<int64_t>(a_ptrs, "a_ptrs");
  // // print_tensor_cpu<int64_t>(b_ptrs, "b_ptrs");

  // print_tensor_cpu<int64_t>(a_scales_ptrs, "a_scales_ptrs"); // pointer 所以 int64_t
  // print_tensor_cpu<int64_t>(b_scales_ptrs, "b_scales_ptrs");

  // print_tensor_cpu<int64_t>(layout_sfa, "layout_sfa");
  // print_tensor_cpu<int64_t>(layout_sfb, "layout_sfb");

  // print_tensor_cpu<int64_t>(problem_sizes, "problem_sizes");

  // // print_device_allocation(layout_SFA.get(), 20, "layout_SFA");
  // // print_device_allocation(layout_SFB.get(), 20, "layout_SFB");


  // 250609
  // // 分成兩類印：指標 + 資料 tensor
  // print_pointer_tensor_debug("a_ptrs", a_ptrs);
  // print_pointer_tensor_debug("b_ptrs", b_ptrs);
  // print_pointer_tensor_debug("a_scales_ptrs", a_scales_ptrs);
  // print_pointer_tensor_debug("b_scales_ptrs", b_scales_ptrs);

  // print_tensor_debug("a", a);
  // print_tensor_debug("b", b);
  // print_tensor_debug("scales_a", scales_a);
  // print_tensor_debug("scales_b", scales_b);
  // print_tensor_debug("layout_sfa", layout_sfa);
  // print_tensor_debug("layout_sfb", layout_sfb);
  // print_tensor_debug("problem_sizes", problem_sizes);

  // // 額外印第一個 expert 的真實 tensor 資料內容
  // print_tensor_from_ptr("a", a, 0);
  // print_tensor_from_ptr("b", b, 0);
  // print_tensor_from_ptr("scales_a", scales_a, 0);
  // print_tensor_from_ptr("scales_b", scales_b, 0);


  // // 普通 tensor
  // print_tensor_debug("a", a);
  // print_tensor_debug("b", b);
  // print_tensor_debug("scales_a", scales_a);
  // print_tensor_debug("scales_b", scales_b);
  // print_tensor_debug("layout_sfa", layout_sfa);
  // print_tensor_debug("layout_sfb", layout_sfb); //
  // print_tensor_debug("problem_sizes", problem_sizes);

  // // // pointer tensor address
  // auto a_ptrs_cpu = a_ptrs.to(torch::kCPU);
  // auto b_ptrs_cpu = b_ptrs.to(torch::kCPU); //
  // auto a_scales_ptrs_cpu = a_scales_ptrs.to(torch::kCPU); //
  // auto b_scales_ptrs_cpu = b_scales_ptrs.to(torch::kCPU); //

  // std::cout << "[Pointer] a_ptrs: ";
  // for (int i = 0; i < std::min<int64_t>(20, a_ptrs.numel()); ++i)
  //   std::cout << a_ptrs_cpu[i].item<int64_t>() << " ";
  // std::cout << std::endl;

  // //
  // std::cout << "[Pointer] b_ptrs: ";
  // for (int i = 0; i < std::min<int64_t>(20, b_ptrs.numel()); ++i)
  //   std::cout << b_ptrs_cpu[i].item<int64_t>() << " ";
  // std::cout << std::endl;

  // //
  // std::cout << "[Pointer] a_scales_ptrs: ";
  // for (int i = 0; i < std::min<int64_t>(20, a_scales_ptrs.numel()); ++i)
  //   std::cout << a_scales_ptrs_cpu[i].item<int64_t>() << " ";
  // std::cout << std::endl;

  // //
  // std::cout << "[Pointer] b_scales_ptrs: ";
  // for (int i = 0; i < std::min<int64_t>(20, b_scales_ptrs.numel()); ++i)
  //   std::cout << b_scales_ptrs_cpu[i].item<int64_t>() << " ";
  // std::cout << std::endl;

  // // // deref 第一個實際內容
  // void* a0 = reinterpret_cast<void*>(a_ptrs_cpu[0].item<int64_t>());
  // void* b0 = reinterpret_cast<void*>(b_ptrs_cpu[0].item<int64_t>()); //
  // void* sa0 = reinterpret_cast<void*>(a_scales_ptrs_cpu[0].item<int64_t>()); //
  // void* sb0 = reinterpret_cast<void*>(b_scales_ptrs_cpu[0].item<int64_t>()); //

  // print_tensor_from_ptr_debug("a_ptrs[0]", a0, a.scalar_type(), {a.size(1)});
  // print_tensor_from_ptr_debug("b_ptrs[0]", b0, b.scalar_type(), {b.size(1)}); //
  // print_tensor_from_ptr_debug("a_scales_ptrs[0]", sa0, scales_a.scalar_type(), {scales_a.size(1)}); //
  // print_tensor_from_ptr_debug("b_scales_ptrs[0]", sb0, scales_b.scalar_type(), {scales_b.size(1)}); //

  // print_tensor_full_debug("stride_a", stride_a);
  // print_tensor_full_debug("stride_b", stride_b);
  // print_tensor_full_debug("stride_c", stride_c);


  // std::cout << "==========================" << std::endl;


  // 250602 因為exp68是elementC/D 是 column-major
  // exp68 => using LayoutC = cutlass::layout::ColumnMajor; // Layout type for C and D matrix operands
  // run_get_group_gemm_starts<MmaConfig1::LayoutSFA, MmaConfig1::LayoutSFB, MmaConfig1::ScaleConfig>(
  //     expert_offsets,
  //     a_ptrs,
  //     b_ptrs,
  //     out_ptrs,
  //     a_scales_ptrs,
  //     b_scales_ptrs,
  //     b_t,
  //     a_t,
  //     output_t,
  //     scales_b_t,
  //     scales_a_t,
  //     layout_sfa,
  //     layout_sfb,
  //     problem_sizes,
  //     problem_sizes_transpose,
  //     true);
  // launch_sm90_fp8_blockwise_scaled_group_mm<OutType, MmaConfig1, cutlass::layout::ColumnMajor>(
  //     out_ptrs,
  //     a_ptrs,
  //     b_ptrs,
  //     a_scales_ptrs,
  //     b_scales_ptrs,
  //     stride_a,
  //     stride_b,
  //     stride_c,
  //     layout_sfa,
  //     layout_sfb,
  //     problem_sizes_transpose,
  //     expert_offsets,
  //     workspace);
  // output = output_t.t();

  // 0529 with yichen origin config2
  run_get_group_gemm_starts<MmaConfig2::LayoutSFA, MmaConfig2::LayoutSFB, MmaConfig2::ScaleConfig>(
      expert_offsets,
      a_ptrs,
      b_ptrs,
      out_ptrs,
      a_scales_ptrs,
      b_scales_ptrs,
      a,
      b,
      output,
      scales_a,
      scales_b,
      layout_sfa,
      layout_sfb,
      problem_sizes,
      problem_sizes_transpose);

  launch_sm90_fp8_blockwise_scaled_group_mm<OutType, MmaConfig2, cutlass::layout::RowMajor>(
      out_ptrs,
      a_ptrs,
      b_ptrs,
      a_scales_ptrs,
      b_scales_ptrs,
      stride_a,
      stride_b,
      stride_c,
      layout_sfa,
      layout_sfb,
      problem_sizes,
      expert_offsets,
      workspace);

  // 250610 TODO print after get(create) tensors
  // std::cout << "============start 250610 ==============" << std::endl;

  // // 普通 tensor
  // print_tensor_debug("a", a);
  // print_tensor_debug("b", b);
  // print_tensor_debug("scales_a", scales_a);
  // print_tensor_debug("scales_b", scales_b);
  // print_tensor_debug("layout_sfa", layout_sfa);
  // print_tensor_debug("layout_sfb", layout_sfb); //
  // print_tensor_debug("problem_sizes", problem_sizes);

  // // // pointer tensor address
  // auto a_ptrs_cpu = a_ptrs.to(torch::kCPU);
  // auto b_ptrs_cpu = b_ptrs.to(torch::kCPU); //
  // auto a_scales_ptrs_cpu = a_scales_ptrs.to(torch::kCPU); //
  // auto b_scales_ptrs_cpu = b_scales_ptrs.to(torch::kCPU); //

  // std::cout << "[Pointer] a_ptrs: ";
  // for (int i = 0; i < std::min<int64_t>(20, a_ptrs.numel()); ++i)
  //   std::cout << a_ptrs_cpu[i].item<int64_t>() << " ";
  // std::cout << std::endl;

  // //
  // std::cout << "[Pointer] b_ptrs: ";
  // for (int i = 0; i < std::min<int64_t>(20, b_ptrs.numel()); ++i)
  //   std::cout << b_ptrs_cpu[i].item<int64_t>() << " ";
  // std::cout << std::endl;

  // //
  // std::cout << "[Pointer] a_scales_ptrs: ";
  // for (int i = 0; i < std::min<int64_t>(20, a_scales_ptrs.numel()); ++i)
  //   std::cout << a_scales_ptrs_cpu[i].item<int64_t>() << " ";
  // std::cout << std::endl;

  // //
  // std::cout << "[Pointer] b_scales_ptrs: ";
  // for (int i = 0; i < std::min<int64_t>(20, b_scales_ptrs.numel()); ++i)
  //   std::cout << b_scales_ptrs_cpu[i].item<int64_t>() << " ";
  // std::cout << std::endl;

  // // // deref 第一個實際內容
  // void* a0 = reinterpret_cast<void*>(a_ptrs_cpu[0].item<int64_t>());
  // void* b0 = reinterpret_cast<void*>(b_ptrs_cpu[0].item<int64_t>()); //
  // void* sa0 = reinterpret_cast<void*>(a_scales_ptrs_cpu[0].item<int64_t>()); //
  // void* sb0 = reinterpret_cast<void*>(b_scales_ptrs_cpu[0].item<int64_t>()); //

  // // redundant?
  // print_tensor_from_ptr_debug("a_ptrs[0]", a0, a.scalar_type(), {a.size(1)});
  // print_tensor_from_ptr_debug("b_ptrs[0]", b0, b.scalar_type(), {b.size(1)}); //
  // print_tensor_from_ptr_debug("a_scales_ptrs[0]", sa0, scales_a.scalar_type(), {scales_a.size(1)}); //
  // print_tensor_from_ptr_debug("b_scales_ptrs[0]", sb0, scales_b.scalar_type(), {scales_b.size(1)}); //

  // print_tensor_full_debug("stride_a", stride_a);
  // print_tensor_full_debug("stride_b", stride_b);
  // print_tensor_full_debug("stride_c", stride_c);
  // std::cout << "============done==============" << std::endl;

  // // } else {
  //   // // printf("Jack entered config2\n");
  //   // run_get_group_gemm_starts<MmaConfig2::LayoutSFA, MmaConfig2::LayoutSFB, MmaConfig2::ScaleConfig>(
  //   //     expert_offsets,
  //   //     a_ptrs,
  //   //     b_ptrs,
  //   //     out_ptrs,
  //   //     a_scales_ptrs,
  //   //     b_scales_ptrs,
  //   //     a,
  //   //     b,
  //   //     output,
  //   //     scales_a,
  //   //     scales_b,
  //   //     layout_sfa,
  //   //     layout_sfb,
  //   //     problem_sizes,
  //   //     problem_sizes_transpose);
  //   // launch_sm90_fp8_blockwise_scaled_group_mm<OutType, MmaConfig2, cutlass::layout::RowMajor>(
  //   //     out_ptrs,
  //   //     a_ptrs,
  //   //     b_ptrs,
  //   //     a_scales_ptrs,
  //   //     b_scales_ptrs,
  //   //     stride_a,
  //   //     stride_b,
  //   //     stride_c,
  //   //     layout_sfa,
  //   //     layout_sfb,
  //   //     problem_sizes,
  //   //     expert_offsets,
  //   //     workspace);
  //   // printf("Jack entered config3\n");
  //   run_get_group_gemm_starts<MmaConfig3::LayoutSFA, MmaConfig3::LayoutSFB, MmaConfig3::ScaleConfig>(
  //       expert_offsets,
  //       a_ptrs,
  //       b_ptrs,
  //       out_ptrs,
  //       a_scales_ptrs,
  //       b_scales_ptrs,
  //       a,
  //       b,
  //       output,
  //       scales_a,
  //       scales_b,
  //       layout_sfa,
  //       layout_sfb,
  //       problem_sizes,
  //       problem_sizes_transpose);
  //   launch_sm90_fp8_blockwise_scaled_group_mm<OutType, MmaConfig3, cutlass::layout::RowMajor>(
  //       out_ptrs,
  //       a_ptrs,
  //       b_ptrs,
  //       a_scales_ptrs,
  //       b_scales_ptrs,
  //       stride_a,
  //       stride_b,
  //       stride_c,
  //       layout_sfa,
  //       layout_sfb,
  //       problem_sizes,
  //       expert_offsets,
  //       workspace);
  // }
#else
  if (a.size(0) <= 512 && a.size(1) >= 2048) {
    // printf("Jack entered config1\n");
    run_get_group_gemm_starts<MmaConfig1::LayoutSFA, MmaConfig1::LayoutSFB, MmaConfig1::ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        b_t,
        a_t,
        output_t,
        scales_b_t,
        scales_a_t,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose,
        true);
    launch_sm90_fp8_blockwise_scaled_group_mm<OutType, MmaConfig1, cutlass::layout::ColumnMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes_transpose,
        expert_offsets,
        workspace);
    output = output_t.t();
  } else if (a.size(0) > 512 && a.size(1) >= 2048) {
    // printf("Jack entered config2\n");
    run_get_group_gemm_starts<MmaConfig2::LayoutSFA, MmaConfig2::LayoutSFB, MmaConfig2::ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a,
        b,
        output,
        scales_a,
        scales_b,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose);
    launch_sm90_fp8_blockwise_scaled_group_mm<OutType, MmaConfig2, cutlass::layout::RowMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
        workspace);
  } else {
    // printf("Jack entered config3\n");
    run_get_group_gemm_starts<MmaConfig3::LayoutSFA, MmaConfig3::LayoutSFB, MmaConfig3::ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a,
        b,
        output,
        scales_a,
        scales_b,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose);
    launch_sm90_fp8_blockwise_scaled_group_mm<OutType, MmaConfig3, cutlass::layout::RowMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
        workspace);
  }
#endif
}

// using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
template <typename OutType, typename ScheduleConfig, typename LayoutD>
void launch_sm100_fp8_blockwise_scaled_group_mm(
    torch::Tensor& out_ptrs,
    const torch::Tensor& a_ptrs,
    const torch::Tensor& b_ptrs,
    const torch::Tensor& a_scales_ptrs,
    const torch::Tensor& b_scales_ptrs,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = OutType;
  using ElementD = ElementC;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = LayoutD;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      void,
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutC*,
      AlignmentC,
      typename ScheduleConfig::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutA*, typename ScheduleConfig::LayoutSFA*>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB*, typename ScheduleConfig::LayoutSFB*>,
      AlignmentB,
      ElementAccumulator,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename ScheduleConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  int num_experts = (int)expert_offsets.size(0);
  // Create an instance of the GEMM
  Gemm gemm_op;

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(stride_a.data_ptr()),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(stride_b.data_ptr()),
      static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFB*>(layout_sfb.data_ptr())};

  cutlass::KernelHardwareInfo hw_info;

  hw_info.device_id = 0;
  // sm_count is the number of SMs on the current device, since we only support SM100 blackwell, so we set it to 148
  hw_info.sm_count = 148;
  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      static_cast<StrideC*>(stride_c.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(stride_c.data_ptr())};

  UnderlyingProblemShape* problem_sizes_as_shapes = static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  at::cuda::CUDAGuard device_guard{(char)a_ptrs.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a_ptrs.get_device());

  auto can_implement_status = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess, "Failed to implement GEMM");

  auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <typename OutType>
void sm100_fp8_blockwise_group_mm_dispatch_shape(
    torch::Tensor& output,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  // Check the first matrix size to decide on the configuration
  // Assuming all matrices in the group have similar size characteristics
  // bool use_small_config = a[0].size(0) <= 128;
  struct MmaConfig1 {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _32, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using ScaleConfig =
        cutlass::detail::Sm100BlockwiseScaleConfig<128, 1, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  struct MmaConfig2 {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using ScaleConfig =
        cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  struct MmaConfig3 {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_64, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using ScaleConfig =
        cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  int num_experts = (int)expert_offsets.size(0);
  torch::TensorOptions options_int = torch::TensorOptions().dtype(torch::kInt64).device(a.device());
  torch::Tensor problem_sizes_transpose = torch::empty(num_experts * 3, options_int);
  torch::Tensor output_t = output.t();
  torch::Tensor a_t = a.t();
  torch::Tensor b_t = b.transpose(1, 2);
  torch::Tensor scales_a_t = scales_a.t();
  torch::Tensor scales_b_t = scales_b.transpose(1, 2);

  if (a.size(0) <= 512 && a.size(1) >= 2048) {
    run_get_group_gemm_starts<MmaConfig1::LayoutSFA, MmaConfig1::LayoutSFB, MmaConfig1::ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        b_t,
        a_t,
        output_t,
        scales_b_t,
        scales_a_t,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose,
        true);
    launch_sm100_fp8_blockwise_scaled_group_mm<OutType, MmaConfig1, cutlass::layout::ColumnMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes_transpose,
        expert_offsets,
        workspace);
    output = output_t.t();
  } else if (a.size(0) > 512 && a.size(1) >= 2048) {
    run_get_group_gemm_starts<MmaConfig2::LayoutSFA, MmaConfig2::LayoutSFB, MmaConfig2::ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a,
        b,
        output,
        scales_a,
        scales_b,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose);
    launch_sm100_fp8_blockwise_scaled_group_mm<OutType, MmaConfig2, cutlass::layout::RowMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
        workspace);
  } else {
    run_get_group_gemm_starts<MmaConfig3::LayoutSFA, MmaConfig3::LayoutSFB, MmaConfig3::ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a,
        b,
        output,
        scales_a,
        scales_b,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose);
    launch_sm100_fp8_blockwise_scaled_group_mm<OutType, MmaConfig3, cutlass::layout::RowMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
        workspace);
  }
}

/**
 * @brief Performs blockwise grouped matrix multiplication on FP8 quantized inputs,
 *        with per-block scaling.
 *
 * This function dispatches to hardware-specific implementations (e.g., SM100 FP8)
 * to compute:
 *     C_i = scale_a[i] * A_i * scale_b[i] * B_i
 * for each expert group `i`, using input `problem_sizes` and `expert_offsets`
 * to describe the individual matrix dimensions and their offsets.
 *
 * Input tensors A and B must be quantized to 8-bit formats and dequantized before multiplication.
 * The output tensor is written with bfloat16 or half precision.
 *
 * @param output         Output tensor (must be of type bfloat16 or half).
 * @param a              Input tensor A (must be kFloat8_e4m3fn).
 * @param b              Input tensor B (must be kFloat8_e4m3fn).
 * @param scales_a       Scaling factors for tensor A, float32 per expert group.
 * @param scales_b       Scaling factors for tensor B, float32 per expert group.
 * @param stride_a       Stride information for tensor A (int32).
 * @param stride_b       Stride information for tensor B (int32).
 * @param stride_c       Stride information for output tensor C (int32).
 * @param layout_sfa     Layout descriptor for A (int32), e.g., row-major/column-major.
 * @param layout_sfb     Layout descriptor for B (int32).
 * @param problem_sizes  2D int32 tensor of shape (num_experts, 3), specifying (M, N, K)
 *                       for each grouped matrix multiplication problem.
 * @param expert_offsets 1D int32 tensor of size (num_experts), used to index into
 *                       the grouped input tensors for dispatch.
 *  @note Performance Optimization:
 *       If the batch size (a.size(0)) is smaller than 512, the implementation
 *       will internally transpose input matrices to align with the optimal memory access
 *       pattern for better GPU efficiency. This transformation is done within the kernel.
 */
void fp8_blockwise_scaled_grouped_mm(
    torch::Tensor& output,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0), "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "a must be kFloat8_e4m3fn");
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn, "b must be kFloat8_e4m3fn");
  TORCH_CHECK(
      output.scalar_type() == torch::kBFloat16 || output.scalar_type() == torch::kHalf,
      "output must be bfloat16 or half");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be float32");
  TORCH_CHECK(stride_a.scalar_type() == torch::kInt64, "stride_a must be int64");
  TORCH_CHECK(stride_b.scalar_type() == torch::kInt64, "stride_b must be int64");
  TORCH_CHECK(stride_c.scalar_type() == torch::kInt64, "stride_c must be int64");
  TORCH_CHECK(layout_sfa.scalar_type() == torch::kInt32, "layout_sfa must be int32");
  TORCH_CHECK(layout_sfb.scalar_type() == torch::kInt32, "layout_sfb must be int32");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32, "expert_offsets must be int32");

  TORCH_CHECK(output.dim() == 2, "output must be 2D tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D tensor");
  TORCH_CHECK(b.dim() == 3, "b must be 3D tensor");
  TORCH_CHECK(scales_a.dim() == 2, "scales_a must be 2D tensor");
  TORCH_CHECK(scales_b.dim() == 3, "scales_b must be 3D tensor");
  TORCH_CHECK(stride_a.dim() == 1, "stride_a must be 1D tensor");
  TORCH_CHECK(stride_b.dim() == 1, "stride_b must be 1D tensor");
  TORCH_CHECK(stride_c.dim() == 1, "stride_c must be 1D tensor");
  TORCH_CHECK(layout_sfa.dim() == 2, "layout_sfa must be 1D tensor");
  TORCH_CHECK(layout_sfb.dim() == 2, "layout_sfb must be 1D tensor");
  TORCH_CHECK(a_ptrs.dim() == 1, "a_ptrs must be 1D tensor");
  TORCH_CHECK(b_ptrs.dim() == 1, "b_ptrs must be 1D tensor");
  TORCH_CHECK(out_ptrs.dim() == 1, "out_ptrs must be 1D tensor");
  TORCH_CHECK(a_scales_ptrs.dim() == 1, "a_scales_ptrs must be 1D tensor");
  TORCH_CHECK(b_scales_ptrs.dim() == 1, "b_scales_ptrs must be 1D tensor");
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0), "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32");
  TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be 1D tensor");
  TORCH_CHECK(workspace.dim() == 1, "workspace must be 1D tensor");

  bool can_implement = false;
  auto sm_version = getSMVersion();

// Jack implementation
// 抄襲fp8_clockwise_gemm_kernel.cu對sm90的支持 (因為100寫法兩邊一樣)
  // printf("Jack CUDA_VERSION = %d\n", CUDA_VERSION); // 12040
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) // from cutlass
#if defined CUDA_VERSION && CUDA_VERSION >= 12000
  if (sm_version == 90) {
    // printf("Jack entered sm90 hack start\n");
    // TODO  preprocess
    if (output.scalar_type() == torch::kBFloat16) {
      sm90_fp8_blockwise_group_mm_dispatch_shape<cutlass::bfloat16_t>(
          output,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
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
          workspace);
    } else {
      sm90_fp8_blockwise_group_mm_dispatch_shape<cutlass::half_t>(
          output,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
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
          workspace);
    }
    // printf("Jack entered sm90 hack end\n");
    can_implement = true;
  }
#endif
#endif

#if defined(CUTLASS_ARCH_MMA_SM100A_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#if defined CUDA_VERSION && CUDA_VERSION >= 12080
  if (sm_version == 100) {
    if (output.scalar_type() == torch::kBFloat16) {
      sm100_fp8_blockwise_group_mm_dispatch_shape<cutlass::bfloat16_t>(
          output,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
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
          workspace);
    } else {
      sm100_fp8_blockwise_group_mm_dispatch_shape<cutlass::half_t>(
          output,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
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
          workspace);
    }
    can_implement = true;
  }
#endif
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      can_implement, "No implemented fp8_blockwise_scaled_mm for current compute capability: ", sm_version);
}
