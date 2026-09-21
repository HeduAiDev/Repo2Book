# ch34《投机解码 vLLM 落地》测试配置。
# 行为基准 = 真实 vLLM v0.27.1（6e448d0ea，instances/vllm/source 现核行号）：
#   - vllm/v1/sample/rejection_sampler.py:L38-L953（RejectionSampler 组合 Sampler
#     + 双 Triton kernel + 残差 recovered + parse_output）
#   - vllm/v1/spec_decode/metadata.py:L9-L66（SpecDecodeMetadata 契约容器）
#   - vllm/v1/spec_decode/ngram_proposer.py:L12-L293（KMP/LPS 零模型 drafter）
#   - vllm/v1/spec_decode/llm_base_proposer.py:L428-L500/L502-L767/L1848-L1886
#     （V1 模型类 drafter 契约：greedy 草稿 + 自回归多步 + 概率化 draft_probs）
import os
import pathlib
import sys

IMPL_DIR = pathlib.Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL_DIR))

# HOST SEAM 确定性（ch30 同款）：构造期 FlashInfer 裁决默认走显式禁用支路
# （VLLM_USE_FLASHINFER_SAMPLER=0 → forward_native 绑定），host 未装 flashinfer
# 时掷骰不炸、行为与真实禁用支路一致。需要真 flashinfer 的测试自行覆盖。
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    # 三个 Triton kernel 需 CUDA；本工作机有 GPU（RTX PRO 6000）即真跑。
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_sampling_metadata(
    *,
    all_greedy: bool = False,
    all_random: bool = False,
    temperature=None,
    top_k=None,
    top_p=None,
    generators=None,
    no_penalties: bool = True,
    prompt_token_ids=None,
    frequency_penalties=None,
    presence_penalties=None,
    repetition_penalties=None,
    output_token_ids=None,
    allowed_token_ids_mask=None,
    bad_words_token_ids=None,
    logitsprocs=None,
    max_num_logprobs=None,
    spec_token_ids=None,
    batch_size: int = 0,
    device=None,
):
    """按真实 SamplingMetadata（vllm/v1/sample/metadata.py:L14-L55）构造测试快照。

    默认值对齐真实默认路径：no_penalties=True、无 bad_words、无 allowed mask、
    logprobs 不请求（max_num_logprobs=None）、spec_token_ids=None。
    """
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata

    if output_token_ids is None:
        output_token_ids = [[] for _ in range(batch_size)]
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=top_p,
        top_k=top_k,
        generators=generators or {},
        max_num_logprobs=max_num_logprobs,
        no_penalties=no_penalties,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=frequency_penalties
        if frequency_penalties is not None
        else torch.tensor([], device=device),
        presence_penalties=presence_penalties
        if presence_penalties is not None
        else torch.tensor([], device=device),
        repetition_penalties=repetition_penalties
        if repetition_penalties is not None
        else torch.tensor([], device=device),
        output_token_ids=output_token_ids,
        allowed_token_ids_mask=allowed_token_ids_mask,
        bad_words_token_ids=bad_words_token_ids or {},
        logitsprocs=logitsprocs if logitsprocs is not None else LogitsProcessors(),
        logprob_token_ids=None,
        spec_token_ids=spec_token_ids,
        thinking_budget_state_holder=None,
    )


def make_spec_metadata(num_draft_tokens, draft_token_ids_2d, device):
    """按 runner _calc_spec_decode_metadata 的产出形态构造 SpecDecodeMetadata。

    cu_num_* 为含末项累积和（int32，vllm/v1/spec_decode/metadata.py:L44-L48）；
    三组 index 在 kernel 测试里不被消费，给合法形状即可。
    """
    import numpy as np

    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

    num_sampled = [n + 1 for n in num_draft_tokens]
    cu_draft = np.cumsum(num_draft_tokens, dtype=np.int32)
    cu_sampled = np.cumsum(num_sampled, dtype=np.int32)
    flat = [t for row in draft_token_ids_2d for t in row]
    num_tokens = len(flat)
    return SpecDecodeMetadata(
        draft_token_ids=torch.tensor(flat, dtype=torch.int32, device=device),
        num_draft_tokens=list(num_draft_tokens),
        cu_num_draft_tokens=torch.from_numpy(cu_draft).to(device),
        cu_num_sampled_tokens=torch.from_numpy(cu_sampled).to(device),
        target_logits_indices=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        bonus_logits_indices=torch.zeros(
            len(num_draft_tokens), dtype=torch.int32, device=device
        ),
        logits_indices=torch.zeros(
            num_tokens + len(num_draft_tokens), dtype=torch.int32, device=device
        ),
    )
