# Subtract-only companion for v3 ch17 — vllm/v1/worker/gpu_model_runner.py
# (pin v0.27.1 / 6e448d0ea), 两段式骨架版。Same names, same structure, same
# control flow; only dossier-approved deletions, each marked `# SUBTRACTED:`.
#
# Deletion #6 (dossier subtraction_plan): GPUModelRunner 内部全部执行细节——
# 除 L437-L451（ExecuteModelState）、L941-L943（单槽字段构造）、L4166-L4178
# （execute_model 入口断言）、L4506-L4535（打包暂存→return None）、L4552-L4592
# （sample_tokens 解包→bitmask→_sample）外的方法体全部退化为注释占位——
# _update_states/_prepare_inputs/_build_attention_metadata/前向/持久缓冲/采样
# bookkeeping/PP 广播分支归 ch18（差量调和/固定地址）与 ch19（编译/捕获）。
# 本章精简版只需两段式状态机骨架：断言→（前向占位）→打包暂存→None；
# 解包即清→bitmask→_sample 调用位。
#
# 全书锚定 V1 实现（7928 行）；vllm/v1/worker/gpu/ 的 V2 仍是 experimental
# （README:L1-L3），由 gpu_worker.init_device 的 use_v2_model_runner 分支引用。

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from .._host_seams import (
    ModelRunnerOutput,
    get_pp_group,
    init_logger,
    record_function_or_nullcontext,
)
from ..structured_output.utils import apply_grammar_bitmask

if TYPE_CHECKING:
    from .._host_seams import GrammarOutput, SchedulerOutput

logger = init_logger(__name__)


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L437-L451 ExecuteModelState —
# 两段之间的暂存协议本体
# SOURCE: (见 impl-notes.md §Source Map——worker/gpu_model_runner.py)
class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: object | None
    spec_decode_common_attn_metadata: object | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: object | None
    cudagraph_stats: object | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None


# SUBTRACTED: 三支 mixin 基类（LoRAModelRunnerMixin / KVConnectorModelRunnerMixin /
#   ECConnectorModelRunnerMixin——LoRA 面与 KV/EC 连接器面，ch16/ch33 域；
#   本章 runner 只保两段式接口，混合基类无从谈起）。
# SOURCE: vllm/v1/worker/gpu_model_runner.py:L454 GPUModelRunner — 第三层
class GPUModelRunner:
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L455-L458 __init__ 签名
    def __init__(
        self,
        vllm_config,
        device: torch.device,
    ):
        # SUBTRACTED: __init__ 主体（gpu_model_runner.py:L459-L940——input_batch/
        #   持久缓冲/attention backend/sampler/offloader 等 900 行装配，
        #   ch18/ch19/ch21 域）。只保留两段式状态机的单槽字段与占位属性。
        # Ephemeral state transferred between execute_model() and sample_tokens().
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L941-L942
        self.execute_model_state: ExecuteModelState | None = None
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L943
        self.kv_connector_output = None
        # SUBTRACTED: L944 起的 mamba/offloader 装配（ch18 域）。
        # HOST SEAM: apply_grammar_bitmask 需要的批消费面（真实 InputBatch 归
        # ch18；此处占位，测试可注入带 req_ids 的替身）。
        self.input_batch = None  # HOST SEAM — ch18 持有 InputBatch
        self.use_async_scheduling = (
            vllm_config.scheduler_config.async_scheduling
        )  # HOST SEAM — 由 config 直读（真实为 __init__ 深水区装配的字段）
        self.broadcast_pp_output = False  # HOST SEAM — ch34 域字段占位

    # SUBTRACTED: get_supported_tasks / load_model / get_model / get_draft_model /
    #   get_kv_cache_spec / initialize_kv_cache / profile_run /
    #   profile_cudagraph_memory / capture_model / _dummy_run /
    #   maybe_remove_all_loras / reset_mm_cache / reset_encoder_cache /
    #   _dummy_sampler_run / _dummy_pooler_run / take_draft_token_ids /
    #   update_config / reload_weights / update_max_model_len / shutdown /
    #   is_pooling_model / lora_config（gpu_model_runner.py 各处——ch18/ch19/
    #   ch29/ch33 域；本章 runner 只保两段式接口，Worker 侧经由占位桩调用）。
    #   以下同签名最小桩维持 Worker/compile 编排骨架可驱动：
    # SOURCE: vllm/v1/worker/gpu_model_runner.py load_model 桩位
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        # SUBTRACTED: 模型加载深水（weight loader 链）——ch18 域
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py initialize_kv_cache 桩位
    def initialize_kv_cache(self, kv_cache_config) -> None:
        # SUBTRACTED: KV 池初始化深水——ch14/ch18 域
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py shutdown 桩位
    def shutdown(self) -> None:
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py get_kv_cache_spec 桩位
    def get_kv_cache_spec(self):
        # SUBTRACTED: KVCacheSpec 剖析——ch14 域
        return {}

    # SOURCE: vllm/v1/worker/gpu_model_runner.py capture_model 桩位
    def capture_model(self) -> int:
        # SUBTRACTED: CUDA Graph 捕获——ch19 域
        return 0

    # SOURCE: vllm/v1/worker/gpu_model_runner.py _dummy_run 桩位
    def _dummy_run(self, *args, **kwargs):
        # SUBTRACTED: 预热前向——ch19 域
        return (None, None)

    # SOURCE: vllm/v1/worker/gpu_model_runner.py maybe_remove_all_loras 桩位
    def maybe_remove_all_loras(self, lora_config) -> None:
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py _dummy_sampler_run 桩位
    def _dummy_sampler_run(self, hidden_states=None) -> None:
        # SUBTRACTED: 采样器预热——ch19/ch29 域
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py reset_mm_cache 桩位
    def reset_mm_cache(self) -> None:
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py reset_encoder_cache 桩位
    def reset_encoder_cache(self) -> None:
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py get_model 桩位
    def get_model(self):
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py get_draft_model 桩位
    def get_draft_model(self):
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py get_supported_tasks 桩位
    def get_supported_tasks(self):
        return ()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py get_encoder_timing_stats 桩位
    def get_encoder_timing_stats(self):
        return {}

    # SOURCE: vllm/v1/worker/gpu_model_runner.py profile_run 桩位
    def profile_run(self) -> None:
        # SUBTRACTED: 显存 profile 前向——ch14 域
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py take_draft_token_ids 桩位
    def take_draft_token_ids(self):
        # SUBTRACTED: spec decode 草稿 token 面板——ch33 域
        return None

    # SUBTRACTED: @torch.inference_mode()（装饰在真实 execute_model 上；
    #   骨架版保留方法面，装饰随前向深水一并裁除）。
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4166-L4178 execute_model 入口
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors=None,
    ):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4167-L4171 双向断言·入口半边
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # SUBTRACTED: routed_experts 缓冲清理（L4177-L4178——ch18 域）。
        # SUBTRACTED: execute_model 前向主体（L4180-L4505——ngram 复制、
        #   _update_states 差量调和、_prepare_inputs 固定地址、attention
        #   metadata 构建、CUDA Graph 重播、前向计算、logits 提取与 PP
        #   broadcast——ch18/ch19/ch34 域；骨架版把打包所需的局部变量以
        #   None 占位绑定，契约行为（打包→None）不变）。
        logits = None
        spec_decode_metadata = None
        spec_decode_common_attn_metadata = None
        hidden_states = None
        sample_hidden_states = None
        aux_hidden_states = None
        ec_connector_output = None
        cudagraph_stats = None
        slot_mappings = None
        kv_connector_output = None
        deferred_state_corrections_fn = None

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4516-L4527 打包暂存
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        )
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4528
        self.kv_connector_output = kv_connector_output

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4529-L4533 异步纠偏回调
        # Now the batch has been launched we can wait for corrections from the
        # previous model forward without breaking async scheduling.
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4535
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4552-L4592 sample_tokens
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4554-L4564 空槽早退分支
        if self.execute_model_state is None:
            kv_connector_output = self.kv_connector_output
            self.kv_connector_output = None
            # receive sampled token ids from the last PP rank.
            if self.use_async_scheduling and not get_pp_group().is_last_rank:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4566-L4577 解包
        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4578-L4580 解包即清
        # Clear ephemeral state.
        self.execute_model_state = None

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4582-L4586 bitmask 施加点
        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4588-L4589 _sample 调用位
        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4591 _update_states 调用位
        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
        # SUBTRACTED: L4592 起——PP 广播 sampled token ids（ch34）、spec
        #   drafter（ch33）、bookkeeping 与返回值装配（ch18）。
        # 骨架版 _sample 返回占位（无 sampled_token_ids 字段时结构洞）。
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:_sample 桩位
    def _sample(self, logits, spec_decode_metadata):
        # SUBTRACTED: 采样栈（ch29/sampler 域）——本章只保调用位；测试经
        #   子类/属性注入观测。真实返回带 sampled_token_ids 的采样输出。
        return _SeamSamplerOutput()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:_update_states_after_model_execute 桩位
    def _update_states_after_model_execute(self, sampled_token_ids, scheduler_output):
        # SUBTRACTED: 批记账（ch18 域）
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:
    #   _pp_receive_prev_sampled_token_ids_to_input_batch 桩位
    # SOURCE: (见 impl-notes.md §Source Map——worker/gpu_model_runner.py)
    def _pp_receive_prev_sampled_token_ids_to_input_batch(self):
        # SUBTRACTED: PP 采样 token 回传（ch34 域）
        return None


# SOURCE: vllm/v1/sample/sampler.py SamplerOutput — HOST SEAM 载体（采样栈归
# ch29；sample_tokens 骨架需要 sampled_token_ids 字段形状）
# SOURCE: (见 impl-notes.md §Source Map——worker/gpu_model_runner.py)
class _SeamSamplerOutput:  # HOST SEAM
    # SOURCE: (见 impl-notes.md §Source Map——worker/gpu_model_runner.py)
    def __init__(self) -> None:
        self.sampled_token_ids: list = []
