# ch29 m05 allowed 白名单 + bad_words 前缀匹配掩码 — 驱动脚本（host）。
# 机制：③白名单 allowed_token_ids_mask masked_fill_(-inf)（True=禁位，False=放行，
# gpu_input_batch.py:L282-L283 注释原话）；④bad_words 只在『会补全成被禁短语』时
# 屏蔽末 token：短语 (len-1) 前缀 == 该请求最近输出的同长后缀（bad_words.py:L9-L36，
# slice 赋值避免 cpu→gpu sync）。
# 行为基准：vllm/v1/sample/sampler.py:L391-L396、ops/bad_words.py:L9-L36、
# worker/gpu_input_batch.py:L924-L932/L282-L283（真实 v0.27.1 行号）。
import json
import os
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import _apply_bad_words_single_batch
from vllm.v1.sample.sampler import Sampler

out = {}

# ── A. allowed 白名单：mask 极性 True=禁、False=放行 ────────────────
V = 5
logits_a = torch.tensor([[5.0, 1.0, 4.0, 0.0, 2.0]])
allowed = [1, 2]
mask = torch.ones(1, V, dtype=torch.bool)   # 默认全 True=全禁
mask[0, allowed] = False                     # 只放行 1/2
after_a = logits_a.clone()
after_a.masked_fill_(mask, float("-inf"))
out["allowed_whitelist"] = {
    "logits": [5.0, 1.0, 4.0, 0.0, 2.0],
    "allowed_token_ids": allowed,
    "mask": mask[0].tolist(),
    "mask_polarity": "True=禁位(masked_fill_ 写 -inf)/False=放行——gpu_input_batch.py:L282-L283 注释原话",
    "logits_after": ["-inf", "1.0", "4.0", "-inf", "-inf"],
    "argmax_before": 0,
    "argmax_after": int(after_a[0].argmax()),
    "claim": "头名 token0(5.0) 不在白名单 → -inf → 第二名 token2(4.0) 出门",
}
# 端到端（step3 走真 Sampler）
md_a = dict(
    temperature=None, all_greedy=True, all_random=False, top_p=None, top_k=None,
    generators={}, max_num_logprobs=None, no_penalties=True, prompt_token_ids=None,
    frequency_penalties=torch.zeros(1), presence_penalties=torch.zeros(1),
    repetition_penalties=torch.ones(1), output_token_ids=[[]],
    allowed_token_ids_mask=mask, bad_words_token_ids={}, logitsprocs=LogitsProcessors(),
)
out["allowed_whitelist"]["sampler_end_to_end"] = (
    Sampler()(torch.tensor([[5.0, 1.0, 4.0, 0.0, 2.0]]), SamplingMetadata(**md_a))
    .sampled_token_ids.flatten().tolist()
)

# ── B. bad_words 前缀匹配：4 个短语 4 种分支 ────────────────────────
# 请求 0 已输出（past）：[10, 20, 30, 40]（token id 即玩具词面）。
past = [10, 20, 30, 40]
V2 = 60
logits_b = torch.zeros(V2)
logits_b[51] = 5.0   # 会补全 bw1=[40,51]
logits_b[53] = 4.0   # 单 token 禁词 bw3=[53]
logits_b[52] = 2.0   # 前缀不匹配的 bw2 末位
logits_b[40] = 3.0   # 历史末 token 本身（不在禁表）
logits_b[0] = 1.0
bad_words = [
    [40, 51],              # bw1: 前缀 [40] == past[-1:] → 封 51
    [20, 30, 52],          # bw2: 前缀 [20,30] != past[-2:]=[30,40] → 52 保留
    [53],                  # bw3: 单 token，前缀空恒匹配 → 恒封 53
    [10, 20, 30, 40, 50, 54],  # bw4: len 6 > len(past)+1=5 → continue 跳过
]
decisions = []
for bw in bad_words:
    prefix_length = len(bw) - 1
    too_long = len(bw) > len(past) + 1
    actual = past[-prefix_length:] if prefix_length > 0 else []
    expected = bw[:prefix_length]
    decisions.append({
        "bad_word": bw,
        "len": len(bw),
        "prefix_length": prefix_length,
        "too_long_skipped": too_long,
        "actual_prefix_tail_of_past": actual,
        "expected_prefix": expected,
        "prefix_match": (not too_long) and actual == expected,
        "blocked_token": None if too_long or actual != expected else bw[-1],
    })
after_b = logits_b.clone()
_apply_bad_words_single_batch(after_b, bad_words, past)
masked_b = [i for i in range(V2) if after_b[i].item() == float("-inf")]
out["bad_words_prefix_match"] = {
    "past_output_tokens": past,
    "bad_words": bad_words,
    "decisions": decisions,
    "blocked_tokens": masked_b,
    "logits_51": 5.0, "logits_53": 4.0, "logits_52": 2.0, "logits_40": 3.0,
    "argmax_before": 51,
    "argmax_after": 40,
    "claim": "只在『再走一步就补全成被禁短语』时封末 token：bw1 补全位 51、bw3 恒封 53；前缀不匹配的 52 与过长的 bw4 不动",
}

# 端到端（step4 走真 Sampler.apply_logits_processors 全链）
md_b = dict(
    temperature=None, all_greedy=True, all_random=False, top_p=None, top_k=None,
    generators={}, max_num_logprobs=None, no_penalties=True, prompt_token_ids=None,
    frequency_penalties=torch.zeros(1), presence_penalties=torch.zeros(1),
    repetition_penalties=torch.ones(1), output_token_ids=[past],
    allowed_token_ids_mask=None,
    bad_words_token_ids={0: bad_words}, logitsprocs=LogitsProcessors(),
)
out["bad_words_prefix_match"]["sampler_end_to_end"] = (
    Sampler()(logits_b.clone().unsqueeze(0), SamplingMetadata(**md_b))
    .sampled_token_ids.flatten().tolist()
)

out["table_rows_echo"] = [
    ["③allowed 白名单", "logits=[5.0, 1.0, 4.0, 0.0, 2.0]", "放行 {1, 2}→mask=[True, False, False, True, True]", "masked_fill_(-inf)", "argmax 0 → 2（端到端采样 2）"],
    ["④bw1=[40, 51]", "past=[10, 20, 30, 40]", "前缀 [40]==past[-1:]", "再走一步即成禁语", "封 51（logit 5.0 → -inf）"],
    ["④bw2=[20, 30, 52]", "同上", "前缀 [20, 30]≠[30, 40]", "不会补全成禁语", "52 保留（2.0）"],
    ["④bw3=[53]", "前缀空 []==[]", "单 token 禁词", "无需历史即成禁语", "恒封 53（4.0 → -inf）"],
    ["④bw4=[10, 20, 30, 40, 50, 54]", "len 6 > len(past)+1=5", "continue 跳过", "历史太短不可能匹配", "54 不动"],
    ["④结果", "argmax_before=51", "封 51/53 后", "logits 40=3.0 成头名", "argmax_after=40（端到端采样 40）"],
]

print(json.dumps(out, ensure_ascii=False, indent=1))
with open(pathlib.Path(__file__).parent / "ch29_m05_allowed_badwords.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
