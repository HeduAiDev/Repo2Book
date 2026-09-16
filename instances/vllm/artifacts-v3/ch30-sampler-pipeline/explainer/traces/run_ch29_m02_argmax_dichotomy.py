# ch29 m02 argmax 不变性二分 — 驱动脚本（host python，吃 implementation/ 精简版）。
# 机制：LogitsProcessors 构造期按 is_argmax_invariant() 两列分类；非不变列
# (min_tokens/logit_bias) step5 greedy 前生效、不变列 (min_p) step7c 温度后
# 随机路径；谎报声明 = 静默破坏 greedy 语义（无人强校验）。
# 行为基准：vllm/v1/sample/logits_processor/state.py:L148-L160、
# builtin.py:L47-L49/L102-L116/L130-L133/L159-L162/L183-L186/L229-L233、
# sampler.py:L243-L302（真实 v0.27.1 行号）。
import json
import os
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

R = lambda x: round(float(x), 4)


def make_metadata(batch: int, **over) -> SamplingMetadata:
    fields = dict(
        temperature=torch.full((batch,), 1.0),
        all_greedy=False,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch),
        presence_penalties=torch.zeros(batch),
        repetition_penalties=torch.ones(batch),
        output_token_ids=[[] for _ in range(batch)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    fields.update(over)
    return SamplingMetadata(**fields)


def min_p_proc(min_p_value: float) -> MinPLogitsProcessor:
    """测试自建 state（update_state 归 ch18）：min_p_cpu 落值+计数+设备切片。"""
    cfg = type("C", (), {"scheduler_config": type("S", (), {"max_num_seqs": 4})})()
    mp = MinPLogitsProcessor(cfg, torch.device("cpu"), False)
    mp.min_p_cpu[0] = min_p_value
    mp.min_p_count = 1
    mp.min_p = mp.min_p_device[:1].unsqueeze(1).clone()
    return mp


def min_tokens_proc(min_toks: int, stop_token: int) -> MinTokensLogitsProcessor:
    mt = MinTokensLogitsProcessor(None, torch.device("cpu"), False)
    mt.min_toks = {0: (min_toks, [], {stop_token})}
    mt.logits_slice = (
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([stop_token], dtype=torch.int32),
    )
    return mt


def logit_bias_proc(token: int, bias: float) -> LogitBiasLogitsProcessor:
    lb = LogitBiasLogitsProcessor(None, torch.device("cpu"), False)
    lb.biases = {0: {token: bias}}
    lb.bias_tensor = torch.tensor([bias], dtype=torch.float32)
    lb.logits_slice = (
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([token], dtype=torch.int32),
    )
    return lb


class LyingMinTokens(MinTokensLogitsProcessor):
    """谎报声明模拟：把『封 EOS 能改 argmax』的 MinTokens 声明成 argmax 不变。
    LogitsProcessors 只信声明、无人强校验（interface.py:L87-L94 抽象方法）。"""

    def is_argmax_invariant(self) -> bool:
        return True


out = {}

# ── A. 构造期两列分类（state.py:L148-L160）──────────────────────────
mp = min_p_proc(0.3)
mt = min_tokens_proc(min_toks=3, stop_token=2)
lb = logit_bias_proc(token=3, bias=10.0)
procs = LogitsProcessors([mp, mt, lb])
out["classification"] = {
    "argmax_invariant": [type(p).__name__ for p in procs.argmax_invariant],
    "argmax_invariant_flags": [p.is_argmax_invariant() for p in procs.argmax_invariant],
    "non_argmax_invariant": [type(p).__name__ for p in procs.non_argmax_invariant],
    "non_argmax_invariant_flags": [
        p.is_argmax_invariant() for p in procs.non_argmax_invariant
    ],
}

# ── B. MinTokens（非不变列）能在 greedy 前改 argmax ─────────────────
# 词表 4：token 2 = EOS。原始 logits EOS 领先 → argmax=2。
logits_b = torch.tensor([[1.0, 2.0, 6.0, 0.0]])
argmax_before = int(logits_b.argmax())
md_b = make_metadata(
    1, all_greedy=True, temperature=None,
    logitsprocs=LogitsProcessors([min_tokens_proc(3, 2)]),
)
out_b = Sampler()(logits_b.clone(), md_b)
sampled_b = int(out_b.sampled_token_ids.flatten()[0])
after_b = logits_b.clone()
after_b[0, 2] = float("-inf")
out["min_tokens_flips_argmax"] = {
    "logits": [1.0, 2.0, 6.0, 0.0],
    "argmax_before": argmax_before,
    "eos_token": 2,
    "logit_after_step5": ["1.0", "2.0", "-inf", "0.0"],
    "argmax_after_step5": int(after_b.argmax()),
    "greedy_sampled": sampled_b,
    "claim": "MinTokens 在 step5(greedy 前)封 EOS → argmax 2 → 1",
}

# ── C. MinP（不变列）砍尾不动 argmax — 只在随机路径执行 ─────────────
logits_c = torch.tensor([[3.0, 2.0, 1.0, 0.5]])
probs_c = torch.softmax(logits_c[0], dim=-1)
mp_c = min_p_proc(0.3)
argmax_c_before = int(logits_c.argmax())
logits_c_rand = mp_c.apply(logits_c.clone())
survivors_c = [i for i in range(4) if logits_c_rand[0, i].item() != float("-inf")]
out["min_p_cuts_tail_keeps_argmax"] = {
    "logits": [3.0, 2.0, 1.0, 0.5],
    "probs": [R(p) for p in probs_c],
    "max_prob": R(probs_c.max()),
    "min_p": 0.3,
    "threshold_min_p_times_max_prob": R(0.3 * probs_c.max()),
    "argmax_before": argmax_c_before,
    "logit_after_min_p": ["3.0", "2.0", "-inf", "-inf"],
    "survivors": survivors_c,
    "argmax_after": int(
        torch.where(
            torch.tensor([v != float("-inf") for v in logits_c_rand[0]]),
            logits_c_rand[0],
            torch.tensor(float("-inf")),
        ).argmax()
    ),
    "claim": "阈值=min_p×max_prob 只砍尾部；max_prob≥min_p×max_prob 恒成立 → 最高位恒留 → argmax 不变",
}

# ── D. 反例：谎报 is_argmax_invariant=True 的 MinTokens ──────────────
lying = LyingMinTokens(None, torch.device("cpu"), False)
lying.min_toks = {0: (3, [], {2})}
lying.logits_slice = (
    torch.tensor([0], dtype=torch.int32),
    torch.tensor([2], dtype=torch.int32),
)
procs_lie = LogitsProcessors([lying])
out["lying_declaration"] = {
    "declared_is_argmax_invariant": lying.is_argmax_invariant(),
    "classified_into": "argmax_invariant"
    if any(p is lying for p in procs_lie.argmax_invariant)
    else "non_argmax_invariant",
    "non_argmax_invariant_len": len(procs_lie.non_argmax_invariant),
    "logits": [1.0, 2.0, 6.0, 0.0],
    "greedy_sampled_with_lie": int(
        Sampler()(
            torch.tensor([[1.0, 2.0, 6.0, 0.0]]),
            make_metadata(1, all_greedy=True, temperature=None,
                          logitsprocs=procs_lie),
        ).sampled_token_ids.flatten()[0]
    ),
    "correct_answer_from_B": sampled_b,
    "claim": "谎报 True → 被分进不变列 → all_greedy 早退在 step7c 之前返回 → 封禁从未执行 → 采样出 EOS(token 2)",
}

# ── E. LogitBias（非不变列）稀疏坐标 += 改 argmax ───────────────────
logits_e = torch.tensor([[1.0, 2.0, 3.0, 2.5]])
lb_e = logit_bias_proc(token=3, bias=10.0)
out_e_logits = lb_e.apply(logits_e.clone())
out["logit_bias_flips_argmax"] = {
    "logits": [1.0, 2.0, 3.0, 2.5],
    "bias": {"token": 3, "value": 10.0},
    "argmax_before": int(logits_e.argmax()),
    "logit_after": [R(v) for v in out_e_logits[0]],
    "argmax_after": int(out_e_logits.argmax()),
}

# 表格回显（explainer.json 逐轮表的数字与本 trace 逐字一致）
out["table_rows_echo"] = [
    ["构造期分类(state.py:L148-L160)", "MinP", "True", "argmax_invariant 列", "step7c(温度后·随机路径)"],
    ["构造期分类", "MinTokens", "False", "non_argmax_invariant 列", "step5(greedy 前·全体)"],
    ["构造期分类", "LogitBias", "False", "non_argmax_invariant 列", "step5(greedy 前·全体)"],
    ["B: MinTokens.apply 封 EOS", "logits=[1.0, 2.0, 6.0, 0.0]", "token2: 6.0 → -inf", "argmax 2 → 1", "greedy 采样出 1(第二名)"],
    ["C: MinP.apply 砍尾",
     f"probs=[{R(probs_c[0])}, {R(probs_c[1])}, {R(probs_c[2])}, {R(probs_c[3])}]",
     f"阈值=0.3×{R(probs_c.max())}={R(0.3 * probs_c.max())}",
     f"survivors={survivors_c}", "argmax 前后都=0(不变)"],
    ["D: 谎报 is_argmax_invariant=True", "MinTokens 声明 True", "分进 argmax_invariant 列", "all_greedy 早退(L261-L271)在 step7c 前返回", "采样出 2(EOS)——正确答案 1(B 行)"],
    ["E: LogitBias.apply 稀疏 +=", "logits=[1.0, 2.0, 3.0, 2.5]", "token3: 2.5+10.0=12.5", "argmax 2 → 3", "非不变列,step5 生效"],
]

print(json.dumps(out, ensure_ascii=False, indent=1))
with open(pathlib.Path(__file__).parent / "ch29_m02_argmax_dichotomy.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
