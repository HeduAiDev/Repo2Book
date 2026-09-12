# SOURCE: vllm/v1/worker/gpu_model_runner.py
# v3 ch31 脊柱⑧：两段式窗口的 worker 面——ExecuteModelState（L437-L450 十元组
# 暂存态）、execute_model（L4165-L4175 状态防御 + L4484-L4485 logits 产出 +
# L4516-L4535 打包 return None 第一幕）、sample_tokens（L4553-L4589 解包即清
# → apply_grammar_bitmask → _sample 第二幕，『先掩码后采样』钉死的六行）、
# _sample 头段（L3692-L3706）。
# SUBTRACTED（delete[3]）：两幕之外的执行臂——_prepare_inputs/注意力元数据/
#   cudagraph/模型前向/PP 广播/pooling/EC 分支、sample_tokens 尾部 drafter 与
#   bookkeeping（L4591 起——归 ch12/ch18/ch19/ch32/ch33）。头注加
#   from __future__ import annotations 使中段类型名（SpecDecodeMetadata 等）
#   可 TYPE_CHECKING 化（真实文件顶部全量 import，ch18/ch19 的域）。
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput, SamplerOutput
    from vllm.v1.spec_decode.interface import SpecDecodeMetadata
    from vllm.v1.worker.common_attention import CommonAttentionMetadata
    from vllm.v1.worker.kv_connector import ECConnectorOutput
    from vllm.v1.worker.gpu_utils import CUDAGraphStat


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L437-L450 ExecuteModelState —— 逐字
class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None


#   （真实继承 LoRA/KVConnector/ECConnector 三个 Mixin——L454，装配面归 ch17）
# SOURCE: vllm/v1/worker/gpu_model_runner.py:L453 GPUModelRunner —— 本章切面
class GPUModelRunner:
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L456-L460 __init__ 签名 —— 保留
    def __init__(self, vllm_config, device) -> None:
        # SUBTRACTED: 真实 __init__ L456-L940（模型/attention 后端/cudagraph/
        #   采样器/各 CPU-GPU 缓冲区装配——ch17/ch18/ch19）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L941-L942 两段式暂存态单槽 —— 逐字
        # Ephemeral state transferred between execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None
        # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L943
        #   self.kv_connector_output（KV connector 出件直通——ch16；其消费方
        #   L4528 与 sample_tokens 早退分支均已删）。
        # ENGINE SEAM（ch17/ch18 边界）：model（compute_logits 面）/sampler/
        #   input_batch（logits_indices·sampling_metadata 面）由外部注入。
        self.model = None
        self.sampler = None
        self.input_batch = None
        self._seam_hidden_states = None

    # ENGINE SEAM（ch17/ch18 边界，ch12 同款注入位）：真实 L4202-L4483 的
    #   _prepare_inputs→注意力元数据→cudagraph→模型前向 → hidden_states 归
    #   ch18/ch19；精简版以注入的前向产物承载（测试对 self._seam_hidden_states
    #   脚本化赋值——不在环内伪造 forward）。
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4202-L4483 前向注入位 —— ENGINE SEAM（ch17/ch18 边界，ch12 同款：不在环内伪造 forward）
    def _forward_hidden_states(self, scheduler_output: "SchedulerOutput"):
        return self._seam_hidden_states

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4165-L4175 execute_model 头段
    #   （状态防御）+ L4484-L4485（logits 产出）+ L4516-L4535（打包尾段）—— 逐字
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: "IntermediateTensors | None" = None,
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None":
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L4177-L4483 两幕之外的
        #   执行臂（delete[3]）：routed experts 清台/ngram 拷贝/KV connector
        #   抢占处理/_update_states/_prepare_inputs/注意力元数据/cudagraph/
        #   模型前向/PP 中间张量与广播/pooling/EC encoder——前向本体归
        #   ch18/ch19，PP/KV/EC 归 ch12/ch16。
        # ENGINE SEAM：上述中段产出的十元组字段，在无投机、单卡、无 connector
        #   的本章演示部署下真实值即 None；hidden_states 经注入位承载。
        spec_decode_metadata = None
        spec_decode_common_attn_metadata = None
        aux_hidden_states = None
        ec_connector_output = None
        cudagraph_stats = None
        slot_mappings = None
        hidden_states = self._forward_hidden_states(scheduler_output)
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4484-L4485 切采样位 +
        #   compute_logits —— 逐字
        sample_hidden_states = hidden_states[self.input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4516-L4527 打包十元组 —— 逐字
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
        # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L4528
        #   self.kv_connector_output = kv_connector_output（ch16）。
        # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L4530-L4533
        #   deferred_state_corrections_fn()（spec decode 乐观纠偏回调——归
        #   ch32/33；dossier 摘录 elide 注明）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4535 —— 逐字
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4553-L4589 sample_tokens 第二幕
    #   —— 逐字（仅删 None 早退分支与 L4591 起的尾部）
    @torch.inference_mode
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4553-L4589 sample_tokens 第二幕 —— 逐字（仅删早退分支与 L4591 尾部）
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors":
        # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L4556-L4564
        #   execute_model_state 为 None 的早退分支（上拍 execute 失败/KV-only
        #   拍的 kv_connector_output 直通——ch16；配对防御保证两段式流程下
        #   第二幕必在第一幕之后）。

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
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L4591-L4700
        #   _update_states_after_model_execute/PP 广播/drafter 草稿与概率暂存/
        #   bookkeeping/AsyncModelRunnerOutput 组装（delete[3]——归 ch12/ch32/33；
        #   删后本函数直接返回 _sample 的 SamplerOutput，EngineCore 面的
        #   ModelRunnerOutput 组装在真实代码属已删的 bookkeeping 段）。
        return sampler_output

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3692-L3706 _sample 头段 —— 逐字
    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: "SpecDecodeMetadata | None",
    ) -> "SamplerOutput":
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        # Update output token ids with tokens sampled in last step
        # if async scheduling and required by current sampling params.
        self.input_batch.update_async_output_token_ids()
        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        # SUBTRACTED: vllm/v1/worker/gpu_model_runner.py:L3708-L3726 async spec
        #   草稿回填 + rejection sampling（ch32/33 的域；本章两段式流程恒
        #   spec_decode_metadata=None）
