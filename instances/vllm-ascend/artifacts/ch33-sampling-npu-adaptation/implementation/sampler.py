# ch28 sampler.py —— subtract-only 精简版（采样的 NPU 对位：薄壳子类化 + Gumbel 异步指数随机 + Triton 优雅回退）
#
# 只删 dossier.subtraction_plan.delete 批准项，保留全部 must_keep 符号：
#   - 删 enable_reduce_sample 的 TP all-gather 分布式分支（默认关，需多卡）；
#   - 删 _apply_top_k_top_p_ascendc 整函数与 AscendC 派发（需 A2/A3 + CANN 自定义算子），
#     保留 _apply_top_k_top_p_pytorch 作 apply_top_k_top_p 的唯一实现；
#   - 删 async-exponential 路径（do_async_exponential / set_q_event / async_exponential_event /
#     forward_native 的 enable_async_exponential 分支），默认关、需 torch.npu.Event/Stream；
#   - 删全部 logger.* 观测日志。
# random_sample 逐字保留（含 npu_stream_switch/global_stream/wait_stream —— NPU-only，host 测试
# 经 conftest 桩成 nullcontext / no-op，符号与调用原样在）。
#
# SOURCE: vllm_ascend/sample/sampler.py
import torch
import vllm.envs as envs

# SUBTRACTED: from vllm.distributed.parallel_state import get_tp_group —— 仅 enable_reduce_sample
#   的 TP all-gather 分支用（已删），原 vllm_ascend/sample/sampler.py:L3。
# SUBTRACTED: from vllm.logger import logger —— 全部 logger.* 观测日志已删，原 sampler.py:L4。
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

# SUBTRACTED: from vllm_ascend.ascend_config import get_ascend_config —— 仅 enable_reduce_sample /
#   enable_async_exponential 分支用（均已删），原 sampler.py:L10。
from vllm_ascend.sample.penalties import apply_all_penalties

# SUBTRACTED: AscendDeviceType, get_ascend_device_type —— 仅 AscendC 派发用（已删），原 sampler.py:L12。
from vllm_ascend.utils import global_stream, npu_stream_switch

DEFAULT_LOGPROBS_MODE = "raw_logprobs"

_SAMPLING_EPS = 1e-5


# SOURCE: vllm_ascend/sample/sampler.py:L19
def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-NPU synchronization.
    """
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            # TODO(woosuk): This can be slow because we handle each request
            # one by one. Optimize this.
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(global_stream())
    return probs.div_(q).argmax(dim=-1).view(-1)


# SOURCE: vllm_ascend/sample/sampler.py:L45
class AscendSampler(Sampler):
    @staticmethod
    # SOURCE: vllm_ascend/sample/sampler.py:L46
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """Use Triton-Ascend penalties on NPU when Triton is available; else vLLM default."""
        if not HAS_TRITON:
            # SUBTRACTED: logger.warning_once(...) 告警文案（sampler.py:L54-L57），仅观测，不影响回退控制流。
            return Sampler.apply_penalties(logits, sampling_metadata, output_token_ids)

        if sampling_metadata.no_penalties:
            return logits
        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )

    # SOURCE: vllm_ascend/sample/sampler.py:L72
    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        # TODO: support logprobs_mode in vllm-ascend
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler(logprobs_mode=logprobs_mode)
        # SUBTRACTED: self.async_exponential_event = torch.npu.Event() —— async-exponential 路径
        #   的 NPU Event（默认关），原 sampler.py:L76。
        # SUBTRACTED: logger.debug(...) 初始化日志（sampler.py:L77-L81）。

    # SUBTRACTED: set_q_event / do_async_exponential（sampler.py:L83-L102）—— async-exponential 旁路
    #   预计算入口，默认关（enable_async_exponential=False），需 torch.npu.Stream/Event 重叠。删后走
    #   random_sample 主路，采样分布不变。

    # SOURCE: vllm_ascend/sample/sampler.py:L86
    def prepare_sampling(self, top_k):
        self.topk_topp_sampler.prepare_sampling(top_k)

    @staticmethod
    # SOURCE: vllm_ascend/sample/sampler.py:L104
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        # SUBTRACTED: enable_reduce_sample 分支（sampler.py:L106-L122）—— 词表按 TP 切分时用
        #   all-gather 求全局 argmax，默认关、需 get_tp_group。单卡直接 argmax 与基类一致。
        return logits.argmax(dim=-1).view(-1)


# SOURCE: vllm_ascend/sample/sampler.py:L127
class AscendTopKTopPSampler(TopKTopPSampler):
    # SOURCE: vllm_ascend/sample/sampler.py:L128
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_top_k_top_p = apply_top_k_top_p
        self.top_k = None

    # SUBTRACTED: set_q_event（sampler.py:L133-L137）—— async-exponential 结果回传入口，随该路径删除。

    # SOURCE: vllm_ascend/sample/sampler.py:L139
    def prepare_sampling(self, top_k):
        if top_k is not None:
            self.top_k = top_k
        else:
            self.top_k = None

    # SOURCE: vllm_ascend/sample/sampler.py:L145
    def forward_native(self, logits, generators, k, p):
        """Override pytorch native implementation to torch_npu"""
        # when batch_invariant mode is enabled, we should use vllm's implementation.
        # or it will make batch_invariant mode not working.
        if envs.VLLM_BATCH_INVARIANT:
            # SUBTRACTED: logger.debug_once(...) BATCH_INVARIANT 提示（sampler.py:L150-L153）。
            # 与 ch26 batch-invariant 确定性同源：昇腾 Triton/torch_npu 路径会破坏 batch 不变性，
            # 故确定性模式必须回退基类原生 top-k/top-p 实现。
            return super().forward_native(logits, generators, k, p)

        # SUBTRACTED: enable_reduce_sample 分支（sampler.py:L156-L172）—— TP 候选集采样
        #   (apply_top_k_top_p 带 self.top_k + cand_idx.gather)，默认关、需多卡。保留 else 单卡主路。
        logits = self.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        # SUBTRACTED: enable_async_exponential 分支（sampler.py:L182-L189）—— 复用预算好的 self.q
        #   直接 probs.div_(self.q).argmax，默认关、需 NPU Event。删后走 random_sample 主路。
        return random_sample(probs, generators), logits_to_return


# SOURCE: vllm_ascend/sample/sampler.py:L193
def _apply_top_k_top_p_pytorch(
    logits: torch.Tensor,  # [B, V_local]
    k: torch.Tensor,  # [B] or None
    p: torch.Tensor,  # [B] or None
    top_k: int | None = None,
) -> torch.Tensor:
    # SUBTRACTED: enable_reduce_sample 分支（sampler.py:L199-L235）—— topk→all-gather→top-p 的
    #   TP 分布式截断，默认关、需 get_tp_group。保留 else 单卡纯 torch sort/cumsum/masked_fill 截断。
    if p is None and k is None:
        return logits

    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        elements_to_discard = probs < top_k_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    return logits


# SUBTRACTED: _apply_top_k_top_p_ascendc 整函数（sampler.py:L268-L295）—— A2/A3 走 AscendC 自定义算子
#   torch.ops._C_ascend.npu_apply_top_k_top_p，需芯片 + CANN，host 不可跑。

# SUBTRACTED: 按芯片型号在加载期派发 ascendc/pytorch（sampler.py:L298-L302）。精简版只留 pytorch 实现。
apply_top_k_top_p = _apply_top_k_top_p_pytorch
