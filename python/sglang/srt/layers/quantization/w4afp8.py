# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py

import logging
<<<<<<< HEAD
from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

=======
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

from grouped_gemm.ops import permute

>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)
from sglang.srt.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.utils import (
    is_layer_skipped,
)
from sglang.srt.utils import (
    set_weight_attrs,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod


ACTIVATION_SCHEMES = ["static", "dynamic"]
MAX_SEQ_LEN = 32768

logger = logging.getLogger(__name__)

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

<<<<<<< HEAD
=======
@singleton
class GlobalVar:
    def __init__(self):
        self.inited = False
        self.act_scale = None
        self.a_q_fp8 = None
        self.expert_offsets = None
        self.problem_sizes = []
        self.permute_map = []
        self.inter = None
        self.c1 = None
        self.c2 = None
        self.inter_q = None
        self.remap_q = None
        self.permute_ws_inited = False

    def init_permute_ws(self, top_k, hidden_size):
        if not self.permute_ws_inited:
            errs = [list(range(top_k)) for i in range(MAX_SEQ_LEN)]
            indexes = torch.tensor(errs, dtype=torch.int32, device="cuda")

            input_act = torch.empty(
                (MAX_SEQ_LEN, hidden_size), dtype=torch.float8_e4m3fn, device="cuda"
            )
            _, _ = permute(input_act, indexes, max_token_num=MAX_SEQ_LEN)
            self.permute_ws_inited = True
            del input_act


>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)
class W4AFp8Config(QuantizationConfig):
    """Config class for MIXED_PRECISION W4AFp8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = True,
        is_checkpoint_w4afp8_serialized: bool = True,
        linear_activation_scheme: str = "dynamic",
        moe_activation_scheme: str = "static",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: Optional[List[int]] = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.is_checkpoint_w4afp8_serialized = is_checkpoint_w4afp8_serialized
        if is_checkpoint_w4afp8_serialized:
            logger.warning("Detected w4afp8 checkpoint. Please note that")
        if moe_activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {moe_activation_scheme}")
        self.linear_activation_scheme = linear_activation_scheme
        self.moe_activation_scheme = moe_activation_scheme
        self.ignored_layers = ignored_layers or []
        self.weight_block_size = [128, 128]
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> str:
        return "w4afp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "W4AFp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        is_checkpoint_w4afp8_serialized = "w4afp8" in quant_method
        linear_activation_scheme = "dynamic"
        moe_activation_scheme = "static"
        weight_block_size = [128, 128]
        return cls(
            is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized,
            is_checkpoint_w4afp8_serialized = is_checkpoint_w4afp8_serialized,
            linear_activation_scheme = linear_activation_scheme,
            moe_activation_scheme = moe_activation_scheme,
            weight_block_size = weight_block_size,
        )

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return W4AFp8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class W4AFp8MoEMethod:

    def __init__(self, quant_config: W4AFp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype, # param_dtype = torch.bfloat16
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoeWeightScaleSupported
        assert "weight_loader" in extra_weight_attrs

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                intermediate_size * 2,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts_per_partition,
                2 * intermediate_size,
                hidden_size // self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
<<<<<<< HEAD
=======
        # extra_weight_attrs.update(
        #     {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})
>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts_per_partition,
                hidden_size,
                intermediate_size //
                self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Input scales
        w13_input_scale = torch.nn.Parameter(torch.ones((num_experts_per_partition, 2),
                                                        dtype=torch.bfloat16),
                                             requires_grad=False)
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(torch.ones(num_experts_per_partition,
                                                       dtype=torch.bfloat16),
                                            requires_grad=False)
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        # Pre-populate the strides
<<<<<<< HEAD
=======
        # num_experts_per_partition = layer.w2_weight.shape[0]
>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)
        device = layer.w13_weight.device

        self.a_strides1 = torch.full((num_experts_per_partition, 3),
                                      hidden_size,
                                      device=device,
                                      dtype=torch.int64)
        self.c_strides1 = torch.full((num_experts_per_partition, 3),
                                     2 * intermediate_size,
                                     device=device,
                                     dtype=torch.int64)
        self.a_strides2 = torch.full((num_experts_per_partition, 3),
                                      intermediate_size,
                                      device=device,
                                      dtype=torch.int64)
        self.c_strides2 = torch.full((num_experts_per_partition, 3),
                                     hidden_size,
                                     device=device,
                                     dtype=torch.int64)
        self.b_strides1 = self.a_strides1
        self.s_strides13 = self.c_strides1
        self.b_strides2 = self.a_strides2
        self.s_strides2 = self.c_strides2

<<<<<<< HEAD
=======
        # singleton_var = GlobalVar()
        # # [TODO]: for other moe topk_nums modify here
        # singleton_var.init_permute_ws(8, hidden_size)

>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)
        self.expert_offsets = torch.empty((num_experts_per_partition + 1),
                                 dtype=torch.int32,
                                 device=device)
        self.problem_sizes1 = torch.empty((num_experts_per_partition, 3),
                                 dtype=torch.int32,
                                 device=device)
        self.problem_sizes2 = torch.empty((num_experts_per_partition, 3),
                                 dtype=torch.int32,
                                 device=device)

        return

    def _interleave_scales(self, scales: torch.Tensor) -> torch.Tensor:
        """Interleave scales in groups of 4 similar to TRT-LLM implementation."""
        s_shape = scales.shape
        # Reshape to separate groups of 4
        scales_interleaved = scales.reshape(s_shape[0], s_shape[1],
                                            (s_shape[2] // 4), 4)
        # Permute dimensions to interleave
        scales_interleaved = scales_interleaved.permute(0, 2, 1, 3)
        # Reshape back to original dimensions but with interleaved values
        scales_interleaved = scales_interleaved.reshape(
            s_shape[0], s_shape[2] // 4, s_shape[1] * 4)
        return scales_interleaved.contiguous()

    def process_weights_after_loading(self, layer: Module) -> None:
        dtype = torch.bfloat16
        device = layer.w2_weight.device

        # Interleave w13_weight_scale (gate_up_proj)
        w13_weight_scale = layer.w13_weight_scale_inv.to(dtype)
        w13_weight_scale = self._interleave_scales(w13_weight_scale)
        layer.w13_weight_scale_inv = Parameter(w13_weight_scale,
                                               requires_grad=False)

        # Interleave w2_weight_scale (down_proj)
        w2_weight_scale = layer.w2_weight_scale_inv.to(dtype)
        w2_weight_scale = self._interleave_scales(w2_weight_scale)
        layer.w2_weight_scale_inv = Parameter(w2_weight_scale,
                                              requires_grad=False)

        # Process input scales
        w13_input_scale_max = layer.w13_input_scale.max().to(dtype).item()
        new_w13_input_scale = torch.tensor(
            [w13_input_scale_max],  # Pass as a list to create a 1-D tensor with one element
            dtype=dtype,
            device=device
        )
        layer.w13_input_scale = Parameter(new_w13_input_scale, requires_grad=False)

        w2_input_scale_max = layer.w2_input_scale.max().to(dtype).item()
        new_w2_input_scale = torch.tensor(
            [w2_input_scale_max],
            dtype=dtype,
            device=device
        )
        layer.w2_input_scale = Parameter(new_w2_input_scale, requires_grad=False)
<<<<<<< HEAD
=======



    # def apply(
    #     self,
    #     layer: torch.nn.Module,
    #     x: torch.Tensor,
    #     topk_weights: torch.Tensor,
    #     topk_ids: torch.Tensor,
    #     expert_map: torch.Tensor,
    #     router_logits: torch.Tensor,
    #     top_k: int,
    #     renormalize: bool,
    #     use_grouped_topk: bool,
    #     topk_group: Optional[int] = None,
    #     num_expert_group: Optional[int] = None,
    #     custom_routing_function: Optional[Callable] = None,
    #     correction_bias: Optional[torch.Tensor] = None,
    #     activation: str = "silu",
    #     apply_router_weight_on_input: bool = False,
    #     inplace: bool = True,
    #     no_combine: bool = False,
    #     routed_scaling_factor: Optional[float] = None,
    # ) -> torch.Tensor:
    #     # from sglang.srt.layers.moe.topk import select_experts

    #     # topk_weights, topk_ids = select_experts(
    #     #     hidden_states=x,
    #     #     router_logits=router_logits,
    #     #     top_k=top_k,
    #     #     use_grouped_topk=use_grouped_topk,
    #     #     renormalize=renormalize,
    #     #     topk_group=topk_group,
    #     #     num_expert_group=num_expert_group,
    #     #     correction_bias=self.correction_bias,
    #     #     custom_routing_function=self.custom_routing_function,
    #     #     routed_scaling_factor=self.routed_scaling_factor,
    #     #     expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
    #     #         ep_rank=self.tp_rank,
    #     #         layer_id=self.layer_id,
    #     #     ),
    #     # )

    #     # return cutlass_w4a8_moe(
    #     #     x,
    #     #     layer.w13_weight,  # Alreay transpose
    #     #     layer.w2_weight,  # Alreay transpose
    #     #     layer.w13_weight_scale_inv,  # Already interleaved
    #     #     layer.w2_weight_scale_inv,  # Already interleaved
    #     #     topk_weights,
    #     #     topk_ids,
    #     #     self.a_strides1,
    #     #     self.b_strides1,
    #     #     self.c_strides1,
    #     #     self.a_strides2,
    #     #     self.b_strides2,
    #     #     self.c_strides2,
    #     #     self.s_strides13,
    #     #     self.s_strides2,
    #     #     layer.w13_input_scale,
    #     #     layer.w2_input_scale,
    #     #     expert_map,
    #     #     apply_router_weight_on_input,
    #     # )

    #     return x
>>>>>>> 3230724ba (init: w4a8精度准确版本，copy from w4a8.v0.2镜像)
