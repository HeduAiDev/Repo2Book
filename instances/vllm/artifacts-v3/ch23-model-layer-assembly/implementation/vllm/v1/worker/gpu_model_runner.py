# SOURCE: vllm/v1/worker/gpu_model_runner.py
# ch23 主角文件之十二（m5/m10 的 runner 侧半边）：GPUModelRunner 薄切面——
# 本章只新增两薄层（delete[13] 明示：_prepare_inputs/_update_states/
# _build_attention_metadata 等复用 ch18 精简版产物，不重刻）：
#   ① load_model 线（L5303-L5390 减法：三段删除 + 两处 delete[10] 明示保留
#      ——L5362 的 _setup_eagle3_aux 调用与 L5376-L5390 EPLB enable 块含头）；
#   ② execute_model 尾段（L4432-L4456 set_forward_context + _model_forward →
#      L4459-L4465 aux 解包 + L4467 主路径头 + L4484-L4485 采样位切片两行）。
# SUBTRACTED：delete[10]（load_model 的 LoRA/drafter/MoE 解析三段——
#   L5327-L5330/L5331-L5360/L5364-L5374 连头删）、delete[11]（execute_model
#   尾段三处——PP 中间 rank return 段 L4469-L4473、pooling 分支 L4475-L4482、
#   PP 广播 else 块 L4486-L4514，均连头删不留悬空 if）、观测面（@instrument/
#   DeviceMemoryProfiler/计时/EplbState 预构——ch34 域）、kv connector/
#   cudagraph 前置面（ch16/ch19 域）。
from __future__ import annotations

from typing import Any, Optional

import torch

from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L453 GPUModelRunner —— 切面构造
#   （ch22/ch18 同款：真实 __init__ 的模型/采样/cudagraph 装配面以装配参数
#   直供；本章消费面：vllm_config/model 与批元数据三件 + load_model/execute_
#   model 尾段的支撑属性五件）
class GPUModelRunner:
    # SOURCE: vllm/v1/worker/gpu_model_runner.py 切面 __init__（装配参数直供）
    def __init__(
        self,
        vllm_config,
        model: Optional[torch.nn.Module] = None,
        query_start_loc: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        slot_mapping: Any = None,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.load_config = vllm_config.load_config
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L468（逐字——EPLB 块
        #   条件位 L5378 的属性来源）
        self.parallel_config = vllm_config.parallel_config
        self.device = torch.device(vllm_config.device_config.device)
        # 站6 契约[0]：输入 GPU 张量归 runner 常驻缓冲所有，模型只读不拥有——
        # 切面以批元数据三件直供（ch18 的持久缓冲域）
        self.model = model
        self.query_start_loc = query_start_loc
        self.attn_metadata = attn_metadata
        self.slot_mapping = slot_mapping
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L512-L515 broadcast_pp_
        #   output —— 减法子集（真实为 external_launcher 分布式后端与 PP
        #   ranks>1 的判断；切面单 rank 恒 False，按默认 distributed_executor_
        #   backend="none" 置 False 语义等价；L4467 头的属性来源）
        self.broadcast_pp_output = False
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L578（逐字——默认 False；
        #   EAGLE3 spec config 才置 True（L633-L650，spec 域删除）；aux 解包
        #   L4459 的判断位）
        self.use_aux_hidden_state_outputs = False
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L550-L551（逐字——
        #   load_model EPLB enable 块（L5376-L5390）的支撑属性：_moe_model
        #   维持 None → 块体恒不进；eplb_state 的 add_model 位）
        self.eplb_state = None
        self._moe_model = None

    # SUBTRACTED: @instrument(span_name="Loading (GPU)") 观测装饰
    #   （gpu_model_runner.py:L5302）——tracing 域

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5303-L5390 load_model
    #   减法子集（delete[10] 三段删除：LoRA 分支 L5327-L5330、drafter 段
    #   L5331-L5360、MoE 解析段 L5364-L5374，均连头删；delete[10] 明示
    #   原样保留：启动线 L5323-L5326、L5362 的 _setup_eagle3_aux 调用、
    #   L5376-L5390 的 EPLB enable 条件块整块含 if 头；观测面/在线量化/
    #   计时另删——tracing/ch27/ch34 域）
    def load_model(self, load_dummy_weights: bool = False) -> None:
        """
        Args:
            load_dummy_weights: load dummy weights instead of real weights.
        """
        # SUBTRACTED: logger.info_once 启动日志（L5308-L5312）与 EplbState
        #   预构（L5314-L5316）、try/DeviceMemoryProfiler/time_before_load
        #   观测上下文（L5318-L5320）——观测/EPLB 域
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5321-L5326 启动线（逐字
        #   ——delete[10] 明示保留：get_model_loader → loader.load_model）
        if load_dummy_weights:
            self.load_config.load_format = "dummy"
        model_loader = get_model_loader(self.load_config)
        self.model = model_loader.load_model(
            vllm_config=self.vllm_config, model_config=self.model_config
        )
        # SUBTRACTED: LoRA 分支（L5327-L5330，连头删）——delete[10]：
        #   lora_config 恒 None，体永不进
        # SUBTRACTED: drafter 装载与 drafter-EPLB 注册段（L5331-L5360，连头
        #   删）——delete[10]：hasattr(self, "drafter") 恒 False
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5362（逐字——delete[10]
        #   明示保留：use_aux_hidden_state_outputs=False 时 no-op 守卫首行即
        #   return（def L5482-L5484），非 EAGLE3 场景零行为；d0 只删 llama.py
        #   侧钩子，不含此调用）
        self._setup_eagle3_aux_hidden_state_outputs()
        # SUBTRACTED: MoE 解析段（L5364-L5374：四行注释 + moe_candidate/
        #   is_mixture_of_experts/SupportsMultiModal 解包 + self._moe_model
        #   赋值）——delete[10]：self._moe_model 维持 __init__ L551 的 None
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5376-L5390 EPLB enable
        #   条件块（逐字——delete[10] 明示原样保留整块含 if 头：条件第一项
        #   self._moe_model is not None 恒假、块体不进；eplb_models 计数位
        #   同体保留，仅在块体内可达（其初始化位 L5316 随 EPLB 预构删——
        #   该行只在 enable_eplb 时执行，与块体同一恒假条件族）
        if (
            self._moe_model is not None
            and self.parallel_config.enable_eplb
            and not load_dummy_weights
        ):
            logger.info_once(
                "EPLB is enabled for model %s.",
                self.model_config.model,
            )
            assert self.eplb_state is not None
            self.eplb_state.add_model(
                self._moe_model,
                self.model_config,
            )
            eplb_models += 1
        # SUBTRACTED: time_after_load/model_memory_usage 计时汇报
        #   （L5392-L5393）与 try/except OutOfMemoryError 收尾（L5394 起）
        #   ——观测/显存域

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5482-L5484 _setup_eagle3_
    #   aux_hidden_state_outputs —— 减法子集（no-op 守卫逐字；spec config 的
    #   aux layers 解析体 L5486 起 → ch32/33 域：use_aux_hidden_state_outputs
    #   =False 时首行即 return，普通生成路径零行为——delete[10] 保留其调用位）
    def _setup_eagle3_aux_hidden_state_outputs(self) -> None:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5483-L5484 no-op 守卫（逐字）
        if not self.use_aux_hidden_state_outputs:
            return
        # SUBTRACTED: supports_eagle3 校验与 aux layers 解析体（L5486 起）
        #   ——EAGLE3 域 ch32/33（本精简版 use_aux 恒 False，此处不可达）

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3879-L3907 _model_forward
    #   （逐字——self.model(...) 一层薄封装：模型域唯一入口）
    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """Helper method to call the model forward pass.

        This method can be overridden by subclasses for model execution.
        Motivation: We can inspect only this method versus
        the whole execute_model, which has additional logic.

        Args:
            input_ids: Input token IDs
            positions: Token positions
            intermediate_tensors: Tensors from previous pipeline stages
            inputs_embeds: Input embeddings (alternative to input_ids)
            **model_kwargs: Additional model arguments

        Returns:
            Model output tensor
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3879-L3907 _model_forward
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1960-L2280 _prepare_inputs
    #   切面子集（delete[13]：ch18 的 token 收集/持久缓冲面不重刻；本章只保
    #   L2232-L2240 的采样位策略段——「哪些位置要 logits」归 runner）
    def _prepare_inputs(
        self,
        scheduler_output,
        query_start_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2232（逐字）
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2239（逐字——每请求
            #   本拍已算 token 的最后一行）
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
        # SUBTRACTED: spec decode 分支（gpu_model_runner.py:L2241-L2262）——
        #   logits_indices = spec_decode_metadata.logits_indices（L2262）；
        #   draft 窗口验证位归 ch33
        return (
            logits_indices,
            spec_decode_metadata,
        )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4165-L4175 execute_model 入口
    #   （逐字——@torch.inference_mode；两段式状态机守卫归 ch18 切面）
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output=None,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        if scheduler_output is None:
            scheduler_output = _EmptySchedulerOutputSeam()

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # SUBTRACTED: kv connector/cudagraph/EPLB 前置面
        #   （gpu_model_runner.py:L4417-L4431）——ch16/ch19/ch34 域
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4432-L4449 with 块签名
        #   （减法子集：kwargs 只保本章消费面 attn_metadata/slot_mapping）
        with (
            set_forward_context(
                self.attn_metadata,
                self.vllm_config,
                slot_mapping=self.slot_mapping,
            ),
        ):
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4450-L4456
            #   metadata 挂进 thread-local ForwardContext 后调模型
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None,
            )

        # SUBTRACTED: with record_function_or_nullcontext("...postprocess")
        #   观测包裹头（L4458）——tracing 域
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4459-L4465 aux 解包段
        #   （逐字——delete[11] 明示保留：use_aux_hidden_state_outputs 恒
        #   False 恒走 else 常路，aux_hidden_states=None）
        if self.use_aux_hidden_state_outputs:
            # True when EAGLE 3 is used.
            hidden_states, aux_hidden_states = model_output
        else:
            # Common case.
            hidden_states = model_output
            aux_hidden_states = None

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4467-L4468 主路径头
        #   （逐字——delete[11] 明示保留头；broadcast_pp_output 恒 False）
        if not self.broadcast_pp_output:
            # Common case.
            # SUBTRACTED: PP 中间 rank return 段（L4469-L4473，连头删不留
            #   悬空 if）——delete[11]①：if not get_pp_group().is_last_rank
            #   及其四行体；单 rank 恒 is_last_rank
            # SUBTRACTED: pooling 分支（L4475-L4482，连头删）——delete[11]②：
            #   if self.is_pooling_model 与 self._pool 六行体
            logits_indices, _spec_decode_metadata = self._prepare_inputs(
                scheduler_output, self.query_start_loc
            )
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4484-L4485 主路径
            #   切片两行（逐字——契约调用侧证据：切片 + compute_logits 都
            #   发生在 runner）
            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)
        # SUBTRACTED: PP 广播 else 块（L4486-L4514，连头连体删）——
        #   delete[11]③：broadcast_pp_output 罕路（send_tensor_dict/
        #   broadcast_tensor_dict 全段），单 rank 恒 False 整块死代码
        return logits


# SOURCE: vllm/v1/worker/gpu_model_runner.py SchedulerOutput 的切面最小承载
#   （真实协议载体归 ch05/ch18 域；本章只读 scheduled_spec_decode_tokens 一面）
class _EmptySchedulerOutputSeam:
    # SOURCE: vllm/v1/core/sched/output.py SchedulerOutput.scheduled_spec_decode_
    #   tokens 字段位 —— HOST SEAM：空批默认
    scheduled_spec_decode_tokens: list = []
