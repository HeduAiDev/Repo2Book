# ch29 m09 top-k/top-p 截断的 pytorch sort 路径 — 驱动脚本（host）。
# 机制：apply_top_k_top_p_pytorch（topk_topp_sampler.py:L367-L408 教学主实现，
# 批<8 或无 Triton 时的路径）：升序 sort → top-k 用第 (V-k) 位值做阈值
# （严格 <，保并列）→ top-p 用 softmax cumsum≤1-p 反向 mask（最末位恒保
# at least one）→ scatter 回原位。
# 行为基准：vllm/v1/sample/ops/topk_topp_sampler.py:L349-L364（分流）、
# L367-L408（sort 实现）（真实 v0.27.1 行号）。
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
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.sampler import Sampler

R = lambda x: round(float(x), 4)
out = {}


def survivors(row: torch.Tensor) -> list:
    return [i for i in range(len(row)) if row[i].item() != float("-inf")]


# ── A. top-k 并列保留：阈值=第 (V-k) 位值、严格 < 才 mask ───────────
# V=5，token1 与 token3 并列 2.0 恰好骑在 k=2 的边界上。
logits_a = torch.tensor([[3.0, 2.0, 0.5, 2.0, 1.0]])
sort_vals, sort_idx = logits_a.sort(dim=-1, descending=False)
k2 = torch.tensor([2])
V = logits_a.shape[1]
thr_idx = V - 2  # logits_sort.size(1) - k = 5-2 = 3
threshold = sort_vals[0, thr_idx]
strict_mask = sort_vals[0] < threshold
final_a = apply_top_k_top_p_pytorch(logits_a.clone(), k2, None)
out["topk_tie_kept"] = {
    "logits": [3.0, 2.0, 0.5, 2.0, 1.0],
    "sorted_values": [R(v) for v in sort_vals[0]],
    "sorted_indices": sort_idx[0].tolist(),
    "k": 2,
    "V_minus_k_index": thr_idx,
    "threshold_logits_sort_at_3": R(threshold),
    "strict_lt_mask": strict_mask.tolist(),
    "final_logits": [("-inf" if v == float("-inf") else R(float(v))) for v in final_a[0]],
    "survivors": survivors(final_a[0]),
    "survivor_count": len(survivors(final_a[0])),
    "claim": "严格 < 保并列：与第 k 名(2.0)并列的 token3 一并保留 → k=2 存活 3 个",
}

# ── B. 无并列对照：k=2 恰好存活 2 个 ────────────────────────────────
logits_b = torch.tensor([[3.5, 1.0, 2.5, 0.5, 3.0]])
sort_b, idx_b = logits_b.sort(dim=-1, descending=False)
thr_b = sort_b[0, 3]
final_b = apply_top_k_top_p_pytorch(logits_b.clone(), k2, None)
out["topk_no_tie_exact_k"] = {
    "logits": [3.5, 1.0, 2.5, 0.5, 3.0],
    "sorted_values": [R(v) for v in sort_b[0]],
    "threshold": R(thr_b),
    "final_logits": [("-inf" if v == float("-inf") else R(float(v))) for v in final_b[0]],
    "survivors": survivors(final_b[0]),
    "survivor_count": len(survivors(final_b[0])),
}

# ── C. top-p（nucleus）：升序 softmax cumsum ≤ 1-p 反向 mask ─────────
logits_c = torch.tensor([[3.0, 2.0, 1.0, 0.5, 0.1]])
sort_c, idx_c = logits_c.sort(dim=-1, descending=False)
probs_c = torch.softmax(logits_c[0], dim=-1)
probs_sort = torch.softmax(sort_c, dim=-1)[0]
cumsum = torch.cumsum(probs_sort, dim=-1)
p09 = torch.tensor([0.9])
mask_c = cumsum <= 1 - 0.9
final_c = apply_top_k_top_p_pytorch(logits_c.clone(), None, p09)
nucleus_mass = sum(probs_c[i] for i in survivors(final_c[0]))
out["topp_cumsum"] = {
    "logits": [3.0, 2.0, 1.0, 0.5, 0.1],
    "probs_descending_by_token": [R(p) for p in probs_c],
    "sorted_values": [R(v) for v in sort_c[0]],
    "probs_sort_ascending": [R(p) for p in probs_sort],
    "cumsum": [R(c) for c in cumsum],
    "one_minus_p": 0.1,
    "mask_cumsum_le_0.1": mask_c.tolist(),
    "final_logits": [("-inf" if v == float("-inf") else R(float(v))) for v in final_c[0]],
    "survivors": survivors(final_c[0]),
    "nucleus_mass": R(nucleus_mass),
    "claim": "核=累积概率质量刚超过 p=0.9 的最小集合：0.6096+0.2243+0.0825=0.9164≥0.9 → 砍掉升序端 cumsum≤0.1 的两位",
}

# ── D. 至少留 1 边界：p→0 角案，最末位由 top_p_mask[:,-1]=False 强保 ──
p0 = torch.tensor([0.0])
mask_d = cumsum <= 1 - 0.0
mask_d_fixed = mask_d.clone()
mask_d_fixed[-1] = False  # top_p_mask[:, -1] = False（L403「at least one」）
final_d = apply_top_k_top_p_pytorch(logits_c.clone(), None, p0)
out["topp_at_least_one"] = {
    "p": 0.0,
    "one_minus_p": 1.0,
    "cumsum": [R(c) for c in cumsum],
    "mask_before_guard": mask_d.tolist(),
    "mask_after_guard_last_forced_false": mask_d_fixed.tolist(),
    "final_logits": [("-inf" if v == float("-inf") else R(float(v))) for v in final_d[0]],
    "survivors": survivors(final_d[0]),
    "claim": "p=0 时 cumsum≤1-p 全位命中（含最末位）——top_p_mask[:, -1]=False 强制保留最高位，至少留 1",
}

# ── E. k+p 联合：先 top-k 再在重归一化分布上 top-p ──────────────────
k3 = torch.tensor([3])
final_e = apply_top_k_top_p_pytorch(logits_c.clone(), k3, p09)
after_k = apply_top_k_top_p_pytorch(logits_c.clone(), k3, None)
renorm = torch.softmax(after_k, dim=-1)[0]
out["topk_topp_combined"] = {
    "logits": [3.0, 2.0, 1.0, 0.5, 0.1],
    "k": 3, "p": 0.9,
    "after_topk3_survivors": survivors(after_k[0]),
    "probs_renormalized_after_topk": [R(p) for p in renorm],
    "final_logits": [("-inf" if v == float("-inf") else R(float(v))) for v in final_e[0]],
    "survivors": survivors(final_e[0]),
    "claim": "联合时先 top-k 砍尾、softmax 在幸存者上重归一化（0.6096/0.9164=0.6652），top-p 再砍到 {0,1}",
}

# ── F. 端到端（批<8 → 本路径）：采样恒落幸存集 ─────────────────────
torch.manual_seed(7)
md_f = dict(
    temperature=torch.full((2,), 1.0), all_greedy=False, all_random=True,
    top_p=None, top_k=torch.tensor([2, 2]), generators={}, max_num_logprobs=None,
    no_penalties=True, prompt_token_ids=None,
    frequency_penalties=torch.zeros(2), presence_penalties=torch.zeros(2),
    repetition_penalties=torch.ones(2), output_token_ids=[[], []],
    allowed_token_ids_mask=None, bad_words_token_ids={}, logitsprocs=LogitsProcessors(),
)
logits_f = torch.stack([logits_b[0], logits_c[0]])
s = Sampler()
if "forward" not in s.topk_topp_sampler.__dict__:
    s.topk_topp_sampler.forward = s.topk_topp_sampler.forward_native
toks = s(logits_f.clone(), SamplingMetadata(**md_f)).sampled_token_ids.flatten().tolist()
out["end_to_end_batch2"] = {
    "batch": 2, "top_k": [2, 2],
    "row0_survivors": survivors(final_b[0]),
    "row1_survivors": survivors(apply_top_k_top_p_pytorch(logits_c.clone(), k2, None)[0]),
    "sampled_tokens": toks,
    "claim": "端到端批=2（<8 分流阈）走本 sort 路径，两行采样 token 均落在各自幸存集内",
}

out["table_rows_echo"] = [
    ["升序 sort(descending=False)", "logits=[3.0, 2.0, 0.5, 2.0, 1.0]", f"sorted={[R(v) for v in sort_vals[0]]}", f"idx={sort_idx[0].tolist()}", "最小位在左"],
    ["top-k 阈值=k 第 (V-k) 位", "k=2, V=5 → 下标 3", "阈值=logits_sort[3]=2.0", "索引算术 logits_sort.size(1)-k", "不排序取第 k 名的值"],
    ["严格 < 保并列", f"mask={strict_mask.tolist()}", "并列 2.0 的 token1/token3 都不 < 2.0", f"final survivors={survivors(final_a[0])}", "k=2 存活 3 个（对照 B 行恰好 2 个）"],
    ["top-p 升序 cumsum", f"probs_sort={[R(p) for p in probs_sort]}", f"cumsum={[R(c) for c in cumsum]}", "mask=cumsum≤1-p=0.1", "砍升序端前两位"],
    ["核=最小覆盖集", f"survivors={survivors(final_c[0])}", "0.6096+0.2243+0.0825=0.9164", "0.9164≥0.9 刚过 p", "scatter 回原位后 -inf 就位"],
    ["至少留 1 角案", "p=0 → cumsum≤1.0 全命中", "top_p_mask[:,-1]=False 强保", f"survivors={survivors(final_d[0])}", "最高位恒留=argmax 不受 top-p 影响"],
    ["k+p 联合", "先 k=3 再 p=0.9", "重归一化 0.6096/0.9164=0.6652", f"final survivors={survivors(final_e[0])}", "两刀都落在同一升序轴上"],
    ["端到端批=2(<8)", f"top_k=[2,2]", f"行0 幸存 {survivors(final_b[0])} / 行1 幸存 {survivors(apply_top_k_top_p_pytorch(logits_c.clone(), k2, None)[0])}", f"sampled={toks}", "采样恒落幸存集"],
]

print(json.dumps(out, ensure_ascii=False, indent=1))
with open(pathlib.Path(__file__).parent / "ch29_m09_topk_topp_sort.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
