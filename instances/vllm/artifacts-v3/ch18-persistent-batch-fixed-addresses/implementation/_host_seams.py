# HOST SEAMS for the ch18 subtract-only companion (pin vLLM v0.27.1).
#
# vLLM 本体不安装在本 Windows host；本章保留代码所触碰的 vllm.* 名字中，
# **不在本章减法面内**的成员由本文件的 stdlib/torch 站位承载——同一可观测
# 接口子集，每个 seam 带 `# SOURCE:` 锚点到钉版源码；不发明真实模块在本书
# 练习路径上没有的行为。完整清单登记于 impl-notes.md §Seam 清单。
# 章内文件以包相对导入消费（ch17 立下的 v3 惯例）。

from __future__ import annotations

import logging
from enum import IntEnum

import torch

from .outputs import SamplerOutput


# ---------------------------------------------------------------------------
# logger / envs —— vllm/logger.py、vllm/envs.py 的最小消费面
# ---------------------------------------------------------------------------


# SOURCE: vllm/logger.py init_logger —— HOST SEAM（ch12 同款）
def init_logger(name: str):
    # SOURCE: vllm/logger.py init_logger
    return logging.getLogger(name)


# SOURCE: vllm/envs.py 的三个剖面开关（消费点 utils.record_function_or_
#   nullcontext 与 _bookkeeping_sync 的 VLLM_COMPUTE_NANS_IN_LOGITS）——
#   HOST SEAM：默认 False（与真实默认环境一致）
class _Envs:
    # SOURCE: vllm/envs.py VLLM_COMPUTE_NANS_IN_LOGITS
    VLLM_COMPUTE_NANS_IN_LOGITS = False

    # SOURCE: vllm/envs.py VLLM_CUSTOM_SCOPES_FOR_PROFILING
    VLLM_CUSTOM_SCOPES_FOR_PROFILING = False

    # SOURCE: vllm/envs.py VLLM_NVTX_SCOPES_FOR_PROFILING
    VLLM_NVTX_SCOPES_FOR_PROFILING = False


envs = _Envs()


# ---------------------------------------------------------------------------
# distributed group 面 —— vllm/distributed/parallel_state（单机单卡占位）
# ---------------------------------------------------------------------------


# SOURCE: vllm/distributed/parallel_state.py get_pp_group —— HOST SEAM：
# 单进程单卡：is_last_rank/is_first_rank 恒 True、world_size=1（精简配置的
# 真实取值）
class _Group:
    # SOURCE: vllm/distributed/parallel_state.py GroupFacade 属性面
    def __init__(self) -> None:
        self.is_last_rank = True
        self.is_first_rank = True
        self.world_size = 1
        self.ranks = [0]


_GROUP = _Group()


# SOURCE: vllm/distributed/parallel_state.py get_pp_group
def get_pp_group() -> _Group:
    # SOURCE: vllm/distributed/parallel_state.py get_pp_group
    return _GROUP


# SOURCE: vllm/distributed/parallel_state.py get_dcp_group —— HOST SEAM：
# 未初始化即抛 AssertionError（BlockTable L121-L134 的 try/except 正是为
# 「testing 未初始化」准备的真实路径——单卡捕获后 world_size=1）
def get_dcp_group():
    # SOURCE: vllm/distributed/parallel_state.py get_dcp_group 未初始化分支
    raise AssertionError("DCP group is not initialized (host seam)")


# SOURCE: vllm/distributed/parallel_state.py get_pcp_group —— 同上
def get_pcp_group():
    # SOURCE: vllm/distributed/parallel_state.py get_pcp_group 未初始化分支
    raise AssertionError("PCP group is not initialized (host seam)")


# ---------------------------------------------------------------------------
# CUDA 事件 / 流 —— torch.cuda.Event(blocking=True) / torch.cuda.Stream
# ---------------------------------------------------------------------------


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L743 torch.cuda.Event(blocking=True)
#   —— HOST SEAM：CPU host 无 CUDA 事件。契约位：未 record 过的事件
#   synchronize() 立即返回（CUDA 语义）；record()=入队未完成；synchronize()
#   =阻塞至完成——HOST 侧无真 DMA，等待即刻满足（完成时刻是同步的），
#   record/synchronize 的**调用顺序**（m13 防踩协议）以计数器观测。
class HostEvent:
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L741-L743 — HOST SEAM
    def __init__(self):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L741-L743 — HOST SEAM
        self._done = True
        self.records = 0  # HOST SEAM 观测位（防踩协议的调用序）
        self.syncs = 0  # HOST SEAM 观测位

    # SOURCE: vllm/v1/worker/gpu_model_runner.py event.record() 契约位
    def record(self):
        # SOURCE: event.record() — HOST SEAM（入队未完成）
        self.records += 1
        self._done = False

    # SOURCE: vllm/v1/worker/gpu_model_runner.py event.synchronize() 契约位
    def synchronize(self):
        # SOURCE: event.synchronize() — HOST SEAM（阻塞至完成——HOST 侧即刻）
        self.syncs += 1
        self._done = True

    # HOST SEAM 查询面：事件是否已完成
    # SOURCE: vllm/v1/worker/gpu_model_runner.py event 查询面 — HOST SEAM
    def is_set(self) -> bool:
        # SOURCE: event 查询面 — HOST SEAM
        return self._done

    # HOST SEAM test hook：显式完成（模拟异步完成时刻的注入位）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py event 完成位 — HOST SEAM
    def set(self):
        # SOURCE: event 完成位 — HOST SEAM
        self._done = True


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L740 torch.cuda.Stream() ——
#   HOST SEAM：轻量对象承载 stream 面（本章消费点均已随异步包裹协议归 ch12）
class HostCopyStream:
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L740 — HOST SEAM
    def __init__(self):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L740 — HOST SEAM
        pass

    # SOURCE: vllm/v1/worker/gpu_model_runner.py wait_stream 契约位 — HOST SEAM
    def wait_stream(self, other):
        # SOURCE: wait_stream — HOST SEAM
        pass


# ---------------------------------------------------------------------------
# SamplingParams —— vllm/sampling_params.py（全量归 ch08；本章只消费
# add_request 装填与 sampling_type 分派所读的属性面）
# ---------------------------------------------------------------------------


# SOURCE: vllm/sampling_params.py:L64-L67 SamplingType
class SamplingType(IntEnum):
    # SOURCE: vllm/sampling_params.py:L65-L67
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2


# SOURCE: vllm/sampling_params.py:L153- SamplingParams —— HOST SEAM：
# 只承载 InputBatch.add_request / _update_states 消费的属性面（默认值与真实
# L224-L326 一致）；sampling_type 判定逐字（L718-L723）
class SamplingParams:
    # SOURCE: vllm/sampling_params.py:L363- __init__ 关键参数面 — HOST SEAM
    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        logprobs: int | None = None,
        prompt_logprobs: int | None = None,
        logprob_token_ids: list[int] | None = None,
        allowed_token_ids: list[int] | None = None,
        bad_words_token_ids: list[list[int]] | None = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ):
        # SOURCE: vllm/sampling_params.py:L224-L250 默认值面 — HOST SEAM
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.logprobs = logprobs
        self.prompt_logprobs = prompt_logprobs
        self.logprob_token_ids = logprob_token_ids
        self.allowed_token_ids = allowed_token_ids
        self.bad_words_token_ids = bad_words_token_ids
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty

    # SOURCE: vllm/sampling_params.py:L718-L723 sampling_type（逐字）
    @property
    def sampling_type(self) -> SamplingType:
        # SOURCE: vllm/sampling_params.py:L718-L723（_SAMPLING_EPS = 1e-5）
        if self.temperature < 1e-5:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM


# ---------------------------------------------------------------------------
# 各域注解占位（对象回退，sched/output.py else 分支同款惯例）
# ---------------------------------------------------------------------------

# SOURCE: vllm/lora/request.py LoRARequest —— 占位（LoRA 装填/解绑面已随
#   dossier.delete[3] 删除；CachedRequestState.lora_request 字段注解-only）
LoRARequest = object
# SOURCE: vllm/multimodal/inputs.py MultiModalFeatureSpec —— 占位（mm 域）
MultiModalFeatureSpec = object
# SOURCE: vllm/v1/pool/metadata.py PoolingMetadata —— 占位（pooling 域）
PoolingMetadata = object
# SOURCE: vllm/config/reasoning.py ReasoningConfig —— 占位（thinking budget 面）
ReasoningConfig = object
# SOURCE: vllm/v1/sample/thinking_budget_state.py ThinkingBudgetStateHolder ——
#   占位（holder 类体归 ch30；reasoning_config=None 时恒 None，守卫天然不触发）
ThinkingBudgetStateHolder = object


# SOURCE: vllm/pooling_params.py PoolingParams —— HOST SEAM：add_request 的
#   pooling elif 支已删；get_pooling_params 返回面与 __post_init__ 的
#   PoolingStates() 构造保留，承载真实默认值面（L67 requires_token_ids）
class PoolingParams:
    # SOURCE: vllm/pooling_params.py PoolingParams 构造面 — HOST SEAM
    def __init__(
        self,
        task: str | None = None,
        requires_token_ids: bool = False,
        extra_kwargs: dict | None = None,
    ):
        # SOURCE: vllm/pooling_params.py:L67 与 task/extra_kwargs 面 — HOST SEAM
        self.task = task
        self.requires_token_ids = requires_token_ids
        self.extra_kwargs = extra_kwargs


# SOURCE: vllm/v1/pool/metadata.py:L37-L42 PoolingStates —— 逐字（6 行小类）
class PoolingStates:
    # SOURCE: vllm/v1/pool/metadata.py:L38-L41
    def __init__(self) -> None:
        # for chunked prefill with ALL pooling
        self.hidden_states_cache: list[torch.Tensor] = []

    # SOURCE: vllm/v1/pool/metadata.py:L43-L44 clean
    def clean(self) -> None:
        # SOURCE: vllm/v1/pool/metadata.py:L44
        self.hidden_states_cache.clear()


# ---------------------------------------------------------------------------
# thinking budget / late interaction / KV zeroer —— 三面「保留调用位、
# 删实现体」的域对象（delete[6] 明示整组保留调用位防悬空）
# ---------------------------------------------------------------------------


# SOURCE: vllm/v1/sample/thinking_budget_state.py:L18-L26 maybe_create_
#   thinking_budget_state_holder —— None 守卫逐字；holder 类体（L28 起，
#   thinking 段跟踪与预算强制终止）归 ch30
def maybe_create_thinking_budget_state_holder(
    reasoning_config: ReasoningConfig | None,
    max_num_seqs: int,
    num_spec_tokens: int,
    device: torch.device,
    is_pin_memory: bool,
) -> "ThinkingBudgetStateHolder | None":
    # SOURCE: vllm/v1/sample/thinking_budget_state.py:L24-L26
    if reasoning_config is None:
        return None
    # SUBTRACTED: ThinkingBudgetStateHolder 构造（L27-L28——ch30 域；本章
    #   精简配置 reasoning_config=None，恒走 None 支）。
    raise NotImplementedError("ThinkingBudgetStateHolder → ch30 域")


# SOURCE: vllm/v1/pool/late_interaction_runner.py:L16 LateInteractionRunner ——
#   HOST SEAM：_update_states 的 5 处调用位整组保留（delete[6] 防悬空）；
#   迟交互打分本体（query 缓存/doc 计数簿记，ch29 池化域）不在本章面内
class LateInteractionRunner:
    # SOURCE: vllm/v1/pool/late_interaction_runner.py:L19-L25 __init__ — SEAM
    def __init__(self) -> None:
        # SOURCE: vllm/v1/pool/late_interaction_runner.py:L19-L25 — SEAM
        pass

    # SOURCE: vllm/v1/pool/late_interaction_runner.py:L32-L39 register_request
    def register_request(
        self, req_id: str, pooling_params: PoolingParams | None
    ) -> None:
        # SUBTRACTED: 迟交互元数据解析与 doc 登记（L35-L39——ch29 域；
        #   生成式主线 pooling_params=None，真实走 else 摘除支=无操作）
        return None

    # SOURCE: vllm/v1/pool/late_interaction_runner.py:L41-L45 on_requests_finished
    def on_requests_finished(self, finished_req_ids) -> None:
        # SUBTRACTED: query 使用计数回收（L42-L45——ch29 域；无登记时真实
        #   行为即 no-op）
        return None

    # SOURCE: vllm/v1/pool/late_interaction_runner.py:L27-L30 clear
    def clear(self) -> None:
        # SUBTRACTED: 三缓存清空（L28-L30——ch29 域；空缓存 clear 等价 no-op）
        return None


# SOURCE: vllm/v1/worker/utils.py:L93 KVBlockZeroer —— HOST SEAM：
# _init_kv_zero_meta 的构造位保留（gpu_worker 启动期外部调用，delete[6]①
# 明示保留）；块清零 Triton kernel 本体归 ch14
class KVBlockZeroer:
    # SOURCE: vllm/v1/worker/utils.py:L101-L109 __init__ 签名面 — SEAM
    def __init__(
        self,
        device: torch.device,
        attn_groups_iter,
        kernel_block_sizes: list[int],
        cache_dtype: str,
        static_forward_context: dict,
        runner_only_attn_layers: set[str] | None = None,
    ) -> None:
        # SUBTRACTED: 绝对地址段表预计算（L110 起——ch14 域）。
        self.device = device

    # SOURCE: vllm/v1/worker/utils.py zero_block_ids 调用面 — SEAM
    def zero_block_ids(self, block_ids: list[int]) -> None:
        # SUBTRACTED: 段清零 kernel 派发（ch14 域）。
        return None


# ---------------------------------------------------------------------------
# 两个小工具函数（真实纯函数，逐字移植）
# ---------------------------------------------------------------------------


# SOURCE: vllm/utils/__init__.py:L15-L28 length_from_prompt_token_ids_or_embeds
#   —— 逐字（CachedRequestState.__post_init__ 与 add_request 的消费面）
def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | torch.Tensor | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    # SOURCE: vllm/utils/__init__.py:L22-L23
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    # SOURCE: vllm/utils/__init__.py:L25-L28
    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    return prompt_token_len


# SOURCE: vllm/utils/collection_utils.py:L123-L131 swap_dict_values —— 逐字
def swap_dict_values(obj: dict, key1, key2) -> None:
    """Swap values between two keys."""
    # SOURCE: vllm/utils/collection_utils.py:L125-L131
    v1 = obj.get(key1)
    v2 = obj.get(key2)
    if v1 is not None:
        obj[key2] = v1
    else:
        obj.pop(key2, None)
    if v2 is not None:
        obj[key1] = v2
    else:
        obj.pop(key1, None)


# ---------------------------------------------------------------------------
# 配置类 —— vllm/config/cache.py 的 DEFAULT_BLOCK_SIZE 常量面
# ---------------------------------------------------------------------------


# SOURCE: vllm/config/cache.py:L47 DEFAULT_BLOCK_SIZE —— HOST SEAM：常量逐字
#   （真实 ClassVar[int] = 16；block_size=None 时的占位回退，
#   gpu_model_runner.py:L694-L696 消费）
class CacheConfig:
    # SOURCE: vllm/config/cache.py:L47
    DEFAULT_BLOCK_SIZE: int = 16

    # SOURCE: vllm/config/cache.py:L266 block_size 回退 — HOST SEAM
    def __init__(self, block_size: int | None = None):
        # SOURCE: vllm/config/cache.py:L266
        self.block_size = block_size


# ---------------------------------------------------------------------------
# 前向/采样 —— reorder 钩子（ch21/22）与 Sampler greedy 支（ch08）
# ---------------------------------------------------------------------------


# reorder_batch_to_split_decodes_and_prefills —— HOST SEAM：_may_reorder_
#   batch 的调用位保留（m12 重排钩子）；四区重排本体（decode/short_extend/
#   long_extend/prefill 的 numpy 划分与 swap_states 扇出）归 ch21/22
# SOURCE: vllm/v1/attention/backends/utils.py:L665 reorder_batch_to_split_decodes_and_prefills
def reorder_batch_to_split_decodes_and_prefills(
    input_batch,
    scheduler_output,
    decode_threshold: int = 1,
) -> bool:
    # SUBTRACTED: 四区重排本体（L683-L740——ch21/22 域）。
    return False


# SOURCE: vllm/v1/sample/sampler.py Sampler —— HOST SEAM：只承载 greedy 支
#   （本书测试全 greedy；random/topk-topp 支归 ch08）。greedy 主算术 L241
#   与尾段 [n,1] 装配 L143-L151 逐字
class Sampler:
    # SOURCE: vllm/v1/sample/sampler.py:L63-L70 __init__（模式参数面）— SEAM
    def __init__(self, logprobs_mode: str = "raw_logprobs", use_fp64_gumbel: bool = False):
        # SOURCE: vllm/v1/sample/sampler.py:L63-L70 — SEAM
        self.logprobs_mode = logprobs_mode
        self.use_fp64_gumbel = use_fp64_gumbel

    # SOURCE: vllm/v1/sample/sampler.py:L239-L241 greedy_sample（逐字）
    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/sampler.py:L241
        return logits.argmax(dim=-1).view(-1)

    # SOURCE: vllm/v1/sample/sampler.py:L72 forward 的 greedy 支 — SEAM
    def forward(self, logits: torch.Tensor, sampling_metadata) -> SamplerOutput:
        # SOURCE: vllm/v1/sample/sampler.py:L255-L267（all_greedy 短路：
        #   greedy_sampled + processed_logprobs=None 直返）
        assert sampling_metadata.all_greedy
        sampled = self.greedy_sample(logits)
        # SUBTRACTED: penalties/temperature/topk-topp/logprobs 全链（ch08）。
        # SOURCE: vllm/v1/sample/sampler.py:L143-L151 尾段（int32 + [n,1] 装配
        #   逐字——注释原文 'expanded to 2D tensor with shape [num_requests, 1]'）
        sampled = sampled.to(torch.int32)
        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=None,
        )
        return sampler_output

    # SOURCE: vllm/v1/nn/ Sampler 的 __call__ 面（nn.Module 转发 forward）— SEAM
    def __call__(self, logits: torch.Tensor, sampling_metadata) -> SamplerOutput:
        # SOURCE: nn.Module.__call__ → forward — SEAM
        return self.forward(logits, sampling_metadata)
