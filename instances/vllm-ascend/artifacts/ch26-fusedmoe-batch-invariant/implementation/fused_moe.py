"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/ops/fused_moe/fused_moe.py

全书最大单体 OOT 算子。AscendFusedMoE 继承 vLLM FusedMoE：super().__init__ 复用基类『身体』
（权重创建/路由参数/EP 配置/weight_loader），再把『头』换成昇腾版——
quant_method+base_quant_method→AscendUnquantizedFusedMoEMethod、runner→AscendMoERunner、
覆写 forward_impl 走 prepare→apply→finalize 三段昇腾路径。这是 ch23『换头不换身』压力最大的实证。
"""
from collections.abc import Callable
from dataclasses import dataclass

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner  # type: ignore

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult, setup_moe_comm_method
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.quant_type import QuantType

# SUBTRACTED: vllm_version_is('0.21.0') 兼容分叉与 get_compressed_expert_map 回退定义
#            （原 fused_moe.py:L51-L62）——纯版本兼容代码，与算子语义无关。


@dataclass
class FusedMoEResult:
    # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L65
    routed_out: torch.Tensor
    # SUBTRACTED: before_dispatch_evt/before_gmm2_evt/before_combine_evt/swiglu_limit 事件字段
    #            （原 L67-L71）——shared-expert 并流计时用，本精简版不走并流。


# SUBTRACTED: FusedMoEEvents 数据类、mock_false/mock_true（原 L74-L92）——并流事件与 DBO 旋钮占位。


# SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L92
class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig = None, tid2eid=None):
        # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L93
        super().__init__(moe=moe)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        self.tid2eid = tid2eid

    @property
    def is_monolithic(self) -> bool:
        # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L99
        return False

    # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L101
    def maybe_make_prepare_finalize(self, routing_tables=None):
        # Ascend uses its own MoE communication and forward_impl path.
        # Do not let upstream modular-kernel initialization replace it.
        return None

    # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L105
    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)

        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)
        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

        # fused dispatch_ffn_combine 只吃 NZ 权重格式，故 enable_fused_mc2 时强制 cast 到 NZ（呼应 ch23）。
        if get_ascend_config().enable_fused_mc2:
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
            import torch_npu

            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
        else:
            from vllm_ascend.utils import maybe_trans_nz

            layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
            layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)

    # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L129
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: torch.Tensor | None = None,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SUBTRACTED: zero_expert_num/zero_expert_type 取出 + num_shared_experts/num_logical_experts 计算
        #            （原 L152-L163）——zero-expert 与 shared-expert 旁路；这里直接用 num_experts。
        input_ids = getattr(get_forward_context(), "input_ids", None)
        # 路由器选 topk 专家
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_experts,
            tid2eid=self.tid2eid,
            input_ids=input_ids,
        )
        # SUBTRACTED: enable_return_routed_experts 的 capturer.capture（原 L181-L191）——可观测性。
        # SUBTRACTED: zero_experts_compute 分支（原 L193-L200）——zero-expert 旁路。

        topk_weights = topk_weights.to(x.dtype)
        # naive load balance for profile runs（随机路由打散，避免单卡堆积）
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        # SUBTRACTED: MoECommType.FUSED_MC2 时把 w1/w2 包成 list + dummy scale 的分支（原 L218-L233）——
        #            融合算子吃 list 格式的细节；本精简版只走非量化主线。
        w1 = layer.w13_weight
        w2 = layer.w2_weight

        # 打包成 build_fused_experts_input → 交给 moe_comm_method.fused_experts（进入 token 重分发）
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                w1_bias=layer.w13_bias if self.moe.has_bias else None,
                w2_bias=layer.w2_bias if self.moe.has_bias else None,
                quant_type=QuantType.NONE,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                swiglu_limit=layer.swiglu_limit,
            )
        )
        return final_hidden_states


# SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L265
class AscendMoERunner(MoERunner):
    # 继承 vLLM MoERunner，只覆写两处『行为旋钮』——换头不换身。

    @property
    def use_dp_chunking(self) -> bool:
        # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L266
        """Ascend uses its own forward_impl path, not the FlashInfer Cutlass
        chunked path. Always return False to stay on forward_impl."""
        return False

    @property
    def _fused_output_is_reduced(self) -> bool:
        # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L272
        # MC2/ALLTOALL/FUSED_MC2 的 finalize() 内部已含 TP all-reduce；告诉基类
        # MoERunner.forward() 别再 _maybe_reduce_final_output 二次 reduce（否则 double-count）。
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        return moe_comm_type in {
            MoECommType.ALLTOALL,
            MoECommType.MC2,
            MoECommType.FUSED_MC2,
        } or (moe_comm_type == MoECommType.ALLGATHER and _EXTRA_CTX.flash_comm_v1_enabled)

    # SUBTRACTED: _maybe_reduce_shared_expert_output / forward_impl(转调 layer.forward_impl) /
    #            _forward_impl(sequence_parallel_context 包裹)（原 L286-L331）——shared-expert reduce
    #            与 SP 上下文；保留两个覆写属性即足以呈现『只改两处行为旋钮』。


# SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L335
class AscendFusedMoE(FusedMoE):
    moe_counter = -1

    def __init__(self, *args, **kwargs):
        # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L338
        # Save original routed_scaling_factor before super().__init__ modifies it.
        tid2eid = kwargs.pop("tid2eid") if "tid2eid" in kwargs else None
        self._original_routed_scaling_factor = kwargs.get("routed_scaling_factor", 1.0)
        # 复用 vLLM FusedMoE 的『身体』：权重创建/路由参数/EP/DP/TP 配置/weight_loader。
        super().__init__(*args, **kwargs)
        self._routed_input_transform = kwargs.get("routed_input_transform")

        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter
        self._expert_map = None
        self.log2phy = None
        self.tid2eid = tid2eid

        # 换『头』之一：把 quant_method 换成昇腾非量化版（或量化版）
        if self.quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod(self.moe_config, tid2eid=self.tid2eid)
        else:
            self.quant_method = self.quant_config.get_quant_method(self, self.layer_name, tid2eid=self.tid2eid)
        assert self.quant_method is not None
        # base_quant_method 必须同步换掉，否则 FusedMoE.maybe_init_modular_kernel 会 dispatch 回
        # 上游 UnquantizedFusedMoEMethod.maybe_make_prepare_finalize（设计上会 raise）。
        self.base_quant_method = self.quant_method

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        if self.moe_config.ep_size > 1:
            self.moe_config.ep_group = get_ep_group()
        self.moe_config.supports_eplb = self.quant_method.supports_eplb

        # SUBTRACTED: multistream_overlap_shared_expert / multistream_overlap_gate / gate_stream /
        #            e_score_correction_bias dtype 调整 / enable_sp 共享专家提示（原 L417-L444）——并流性能旋钮。
        # SUBTRACTED: init_eplb_config 冗余专家 log2phy 重映射 + dynamic_eplb 负载表(moe_load/load_counter)
        #            （原 L446-L480）——EPLB 是独立子系统（前面章节伏笔），本章只需 expert_map/log2phy 占位。

        self.n_shared_experts = kwargs.get("n_shared_experts", 0)
        self.global_redundant_expert_num = 0
        self.global_num_experts = kwargs["num_experts"]
        self.dynamic_eplb = False
        self.local_num_experts = self.global_num_experts // self.ep_size
        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.local_num_experts
        self.moe_config.global_redundant_expert_num = self.global_redundant_expert_num
        self.swiglu_limit = getattr(self.vllm_config.model_config.hf_config, "swiglu_limit", 0)

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }
        self.quant_method.create_weights(layer=self, **moe_quant_params)

        self.enable_shared_expert_dp = get_ascend_config().enable_shared_expert_dp
        self.enable_npugraph_ex_static_kernel = (
            get_ascend_config().ascend_compilation_config.enable_static_kernel
        )

        # 换『头』之三：建三选一通信注册表（f10 回收落地）
        setup_moe_comm_method(self.moe_config)
        self.quant_type = self._get_quant_type()

        # 换『头』之二：建昇腾 runner
        self.runner = AscendMoERunner(
            self.layer_name,
            self.moe_config,
            self.router,
            self._routed_input_transform,
            kwargs.pop("gate", None),
            kwargs.pop("shared_experts", None),
            self.quant_method,
            self.vllm_config.parallel_config.enable_dbo,
        )
        # SUBTRACTED: multistream_overlap_shared_expert 时 wrap process_weights_after_loading 做
        #            shared-expert 拆分一致性校验（原 L480-L493）——并流校验旁路。

    # SUBTRACTED: _validate_shared_expert_consistency / _shared_experts_part1 / _shared_experts_part2
    #            （原 L494-L542）——shared expert 拆分计算与校验。

    # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L543
    def _get_quant_type(self) -> QuantType:
        quant_type = QuantType.NONE
        method = getattr(self.quant_method, "quant_method", None)
        if method is not None:
            quant_type = getattr(method, "quant_type", QuantType.NONE)
        return quant_type

    # SUBTRACTED: update_expert_map / get_log2phy_map / clear_moe_load /
    #            maybe_all_reduce_tensor_model_parallel / gate / is_internal_router 属性
    #            （原 L552-L588）——EPLB 回写与 router 旁路访问器。

    @property
    def use_dp_chunking(self) -> bool:
        # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L584
        """Just returning False in vllm-ascend (always走 forward_impl)."""
        return False

    # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L590
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.ensure_moe_quant_config_init()
        # 委托给 runner（vLLM MoERunner.forward 骨架；use_dp_chunking=False → 走 forward_impl）
        return self.runner.forward(hidden_states, router_logits)

    # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L601
    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, return_with_event: bool = False
    ) -> "torch.Tensor | FusedMoEResult":
        assert self.quant_method is not None

        # SUBTRACTED: enable_npugraph_ex_static_kernel 的 moe_layer_index 回绕（原 L605-L617）——
        #            static kernel 编译/capture 两跑的索引保护。
        enable_force_load_balance = _EXTRA_CTX.in_profile_run

        # SUBTRACTED: multistream_overlap_gate 的 gate_stream / shared_experts 并流 + 提前 select_experts
        #            （原 L619-L675）——gate 并流块，不改数值与数据流。

        # ① 通信前置：AllGather 做 DP all-gather；MC2/All2All 做 pad + TP 切片
        prepare_output = _EXTRA_CTX.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=_EXTRA_CTX.flash_comm_v1_enabled,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            quant_type=self.quant_type,
        )
        hidden_states = prepare_output.hidden_states
        router_logits = prepare_output.router_logits
        mc2_mask = prepare_output.mc2_mask
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape
        pertoken_scale = prepare_output.pertoken_scale

        # ② Matrix multiply：select_experts 选专家 → moe_comm_method.fused_experts（dispatch→mlp→combine）
        fused_experts_results: FusedExpertsResult = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            pertoken_scale=pertoken_scale,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            num_experts=self.moe_config.num_experts,
            expert_map=self._expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self._original_routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            mc2_mask=mc2_mask,
        )

        # SUBTRACTED: dynamic_eplb 负载统计（local_load 累加/multi_stage index_add_）（原 L702-L720）。

        # ③ 通信后置：AllGather 做 reduce-scatter + unpad；MC2/All2All 做 all-gather 回拼 + unpad。
        #    reduce_results 仅 AllGather 为 True（其余 finalize 内已 reduce）。
        routed_out = _EXTRA_CTX.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl),
            padded_hidden_states_shape=padded_hidden_states_shape,
        )

        # SUBTRACTED: return_with_event 包成 FusedMoEResult 的分支（原 L724-L737）——并流事件回传；
        #            vLLM FusedMoE forward_impl 不返回事件，直接返回 routed_out。
        return routed_out
