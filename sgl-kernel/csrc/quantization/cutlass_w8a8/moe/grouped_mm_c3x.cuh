#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp" // Jack missing
// => cutlass_extensions/epilogue/*
// -> core/scalar_type.hpp
#include "cutlass_extensions/common.hpp" // Jack missing
#include "get_group_starts.cuh"

using namespace cute;

namespace {

using ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

template <typename ElementAB_, typename ElementC_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_3x_group_gemm {
  using ElementAB = ElementAB_;
  using ElementC = void;
  using ElementD = ElementC_;
  using ElementAccumulator = float;

  using Epilogue = Epilogue_<ElementAccumulator, ElementD, TileShape>;

  using StrideC =
      cute::remove_pointer_t<cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>>;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;

  using EVTCompute = typename Epilogue::EVTCompute;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutC*, AlignmentC, ElementD,
          LayoutC*, AlignmentC, EpilogueSchedule, EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementAB, LayoutA*, AlignmentAB, ElementAB,
          LayoutB*, AlignmentAB, ElementAccumulator, TileShape, ClusterShape,
          Stages, KernelSchedule>::CollectiveOp;

  using KernelType = enable_sm90_only<cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};

template <typename Gemm>
void cutlass_group_gemm_caller(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  int num_experts = static_cast<int>(expert_offsets.size(0));
  int k_size = a_tensors.size(1);
  int n_size = out_tensors.size(1);

  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != num_experts;

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  auto options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(a_tensors.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);

  run_get_group_gemm_starts(expert_offsets, a_ptrs, b_ptrs, out_ptrs,
                            a_scales_ptrs, b_scales_ptrs, a_tensors, b_tensors,
                            out_tensors, a_scales, b_scales);

  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = Stride<int64_t, Int<1>, Int<0>>;
  using StrideB = Stride<int64_t, Int<1>, Int<0>>;
  using StrideC = typename GemmKernel::InternalStrideC;

  ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<ProblemShape::UnderlyingProblemShape*>(
          problem_sizes.data_ptr());
  ProblemShape prob_shape{num_experts, problem_sizes_as_shapes, nullptr};

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementAB**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(a_strides.data_ptr()),
      static_cast<const ElementAB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(b_strides.data_ptr())};

  // Currently, we are only able to do broadcast on either all or none a_scales
  // and on either all or none b_scales
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
          static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
          per_act_token, per_out_ch),
      nullptr, static_cast<StrideC*>(c_strides.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(c_strides.data_ptr())};

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped, prob_shape, mainloop_args,
      epilogue_args};

  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a_tensors.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

void cutlass_group_blockwise_gemm_caller(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
        // TODO
        // from cutlass_w4a8_group_gemm_caller() https://github.com/vllm-project/vllm/compare/main...bytedance-iaas:vllm:feat/w4a8
        // example68 is my goal

        /* note:
         * my out_tensors = yichen's d_tensors
         * I don't have
                torch::Tensor const& d_strides,
                torch::Tensor const& s_strides,
                int64_t chunk_size)
        */



        /* yichen's code TODO TODO */
        using Gemm = cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>;
        using Args = typename Gemm::GemmScaleOnly::Arguments;

        int num_experts = static_cast<int>(expert_offsets.size(0));
        bool per_act_token = a_scales.numel() != 1;
        bool per_out_ch = b_scales.numel() != num_experts;


        // // Check inputs
        // TORCH_CHECK(a_tensors.dim() == 2, "A tensor must be 2D");
        // TORCH_CHECK(b_tensors.dim() == 3, "B tensor must be 3D [E, N, K/2]");
        // TORCH_CHECK(b_scales.dim() == 3, "Scale tensor must be 3D [E, K//512, N*4]");
        // TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be a 1D tensor");
        // TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");

        // // Check tensor shapes
        // TORCH_CHECK(problem_sizes.size(0) == num_experts,
        //             "problem_sizes must have num_experts rows");
        // TORCH_CHECK(problem_sizes.size(1) == 3,
        //             "problem_sizes must have 3 columns (N, M, K)");
        // TORCH_CHECK(b_tensors.size(0) == num_experts,
        //             "B tensor first dimension must match number of groups");
        // TORCH_CHECK(b_scales.size(0) == num_experts,
        //             "Scale tensor first dimension must match number of groups");
        // TORCH_CHECK(b_tensors.size(2) * 2 == a_tensors.size(1),
        //             "B tensor K/2 dimension must match A tensor K dimension");
        // TORCH_CHECK(b_scales.size(1) == a_tensors.size(1) / 512,
        //             "Scale tensor second dimension must be K//512");
        // TORCH_CHECK(b_scales.size(2) == 4 * b_tensors.size(1),
        //             "Scale tensor last dimension must be 4*N");

        // // Check tensor types
        // TORCH_CHECK(a_tensors.scalar_type() == torch::kFloat8_e4m3fn,
        //             "A tensor must be fp8 (float_e4m3_t) type");
        // TORCH_CHECK(b_tensors.scalar_type() == torch::kInt8,
        //             "B tensor must contain packed int4 values (stored as int8)");
        // TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32,
        //             "Expert offsets must be int32 type");
        // TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32,
        //             "Problem sizes must be int32 type");

        auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());
        auto options_int =
            torch::TensorOptions().dtype(torch::kInt64).device(a_tensors.device());

        torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
        torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
        torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
        torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
        torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = a_tensors.device().index();
        hw_info.sm_count =
            cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
                hw_info.device_id);

        // Set up fusion arguments
        Args arguments;
        decltype(arguments.epilogue.thread) fusion_args;
        fusion_args.alpha = 0;
        fusion_args.beta = 0;
        fusion_args.alpha_ptr = nullptr;
        fusion_args.beta_ptr = nullptr;
        fusion_args.alpha_ptr_array =
            static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr());
        fusion_args.beta_ptr_array = nullptr;
        fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
        fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};

        ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
            static_cast<ProblemShape::UnderlyingProblemShape*>(
                problem_sizes.data_ptr());

        // Jack: run_get_group_gemm_starts
        /*
            void run_get_group_gemm_starts(
                torch::Tensor const& expert_offsets,
                torch::Tensor& a_ptrs,
                torch::Tensor& b_ptrs,
                torch::Tensor& out_ptrs,
                torch::Tensor& a_scales_ptrs,
                torch::Tensor& b_scales_ptrs,
                torch::Tensor const& a_tensors,
                torch::Tensor const& b_tensors,
                torch::Tensor& out_tensors,
                torch::Tensor const& a_scales,
                torch::Tensor const& b_scales,
                torch::Tensor const& layout_sfa, // cutlass_group_gemm_caller()沒用...
                torch::Tensor const& layout_sfb, // cutlass_group_gemm_caller()沒用...
                torch::Tensor const& problem_sizes,
                torch::Tensor& problem_sizes_transpose,
                bool transpose = false) {}
        */
        // Jack's: same as cutlass_group_gemm_caller() DONE
        run_get_group_gemm_starts(expert_offsets, a_ptrs, b_ptrs, out_ptrs,
                            a_scales_ptrs, b_scales_ptrs, a_tensors, b_tensors,
                            out_tensors, a_scales, b_scales);
        // yichen's
        // run_int4_fp8_get_group_gemm_starts(expert_offsets, a_ptrs, b_ptrs, out_ptrs,
        //                     a_scales_ptrs, b_scales_ptrs, a_tensors,
        //                     b_tensors, d_tensors, a_scales, b_scales);
    

        // Jack's WIP
        arguments =
            Args{cutlass::gemm::GemmUniversalMode::kGrouped,
                {num_experts, problem_sizes_as_shapes, nullptr},
                {static_cast<const QuantType**>(b_ptrs.data_ptr()),
                    static_cast<typename Gemm::StrideB*>(b_strides.data_ptr()),
                    static_cast<const MmaType**>(a_ptrs.data_ptr()),
                    static_cast<typename Gemm::StrideA*>(a_strides.data_ptr()),
                    static_cast<const ElementScalePacked**>(b_scales_ptrs.data_ptr()),
                    a_scales_ptrs, a_scales, // 我有注意到yichen's scale只傳b的. 是說a應該不需要有scale乘回去精度吧?so我這邊可以簡單nullptr?
                    b_scales_ptrs, b_scales,
                    // static_cast<typename Gemm::StrideS*>(s_strides.data_ptr()),
                    // static_cast<int>(chunk_size)},
                {fusion_args, nullptr, nullptr,
                    static_cast<ElementD**>(out_ptrs.data_ptr()),
                    static_cast<typename Gemm::StrideD*>(d_strides.data_ptr())},
                hw_info};

        // // yichen's
        // arguments =
        //     Args{cutlass::gemm::GemmUniversalMode::kGrouped,
        //         {num_experts, problem_sizes_as_shapes, nullptr},
        //         {static_cast<const QuantType**>(b_ptrs.data_ptr()),
        //             static_cast<typename Gemm::StrideB*>(b_strides.data_ptr()),
        //             static_cast<const MmaType**>(a_ptrs.data_ptr()),
        //             static_cast<typename Gemm::StrideA*>(a_strides.data_ptr()),
        //             static_cast<const ElementScalePacked**>(b_scales_ptrs.data_ptr()),
        //             static_cast<typename Gemm::StrideS*>(s_strides.data_ptr()),
        //             static_cast<int>(chunk_size)},
        //         {fusion_args, nullptr, nullptr,
        //             static_cast<ElementD**>(out_ptrs.data_ptr()),
        //             static_cast<typename Gemm::StrideD*>(d_strides.data_ptr())},
        //         hw_info};
    
        // // vs 68 check args_from_options() in /nvme0n1/jack/vllm-w8a8-cutlass/.deps/cutlass-src/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling_with_sparse_groups.cu
        // GemmArguments arguments{
        //     cutlass::gemm::GemmUniversalMode::kGrouped,
        //     {options.groups, problem_sizes.get(), host_problem_shapes_available ? options.problem_sizes_after_alignment_host.data() : (decltype(options.problem_sizes_after_alignment_host.data())) nullptr},
        //     {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get(),
        //     ptr_blockscale_A.get(), layout_SFA.get(), // Jack猜是 a_scales_ptrs, a_scales
        //     ptr_blockscale_B.get(), layout_SFB.get() // JACK猜是 b_scales_ptrs, b_scales
        //     },
        //     {
        //     {}, // epilogue.thread
        //     ptr_C.get(), stride_C.get(),
        //     ptr_D.get(), stride_D.get()
        //     },
        //     kernel_hw_info
        // };
        // // vs yichen's origin examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm.cu
        // arguments = Args {
        //             cutlass::gemm::GemmUniversalMode::kGrouped,
        //             {options.groups, problem_sizes.get(), nullptr},
        //             {ptr_B.get(), dB, ptr_A.get(), stride_A.get(), ptr_scale_packed.get(), stride_S.get(), options.c},
        //             {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
        //             hw_info
        //         };


        // Instantiate and run GEMM
        typename Gemm::GemmScaleOnly gemm;
        size_t workspace_size = Gemm::GemmScaleOnly::get_workspace_size(arguments);
        auto const workspace_options =
            torch::TensorOptions().dtype(torch::kUInt8).device(a_tensors.device());
        auto workspace = torch::empty(workspace_size, workspace_options);

        cutlass::Status status = gemm.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            TORCH_CHECK(false, "GEMM implementation not supported");
        }

        status = gemm.initialize(arguments, workspace.data_ptr(), stream);
        if (status != cutlass::Status::kSuccess) {
            TORCH_CHECK(false, "GEMM initialization failed");
        }

        status = gemm.run(stream);
        if (status != cutlass::Status::kSuccess) {
            //  if (current_device == debug_node) {
            //      printf("%d GEMM execution failed. Status: %d\n", current_device,
            //      static_cast<int>(status));
            //  }
            TORCH_CHECK(false, "GEMM execution failed");
        }

}

}  // namespace
