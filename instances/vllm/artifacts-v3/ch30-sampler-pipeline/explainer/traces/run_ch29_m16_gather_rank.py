# ch29 m16 gather logprobs 与不排序 rank — 驱动脚本（host）。
# 机制：step8 gather_logprobs（sampler.py:L308-L356）对第 1 步留底的 raw
# logprobs 取 topk 值+下标、被采样 token 的 logprob、
# rank=batched_count_greater_than=(x>=v).sum(-1)（ops/logprobs.py:L11-L29，
# O(V) 计数不排序；torch.compile+mark_unbacked 防批 1→2 重编译）；
# num_logprobs==-1 全词表未排序不排名分支（L120-L126，RejectionSampler
# bonus 位契约——前指 ch32/33）。
# 行为基准：vllm/v1/sample/sampler.py:L120-L136/L252-L306、
# ops/logprobs.py:L10-L27（真实 v0.27.1 行号）。
import json
import os
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.sampler import Sampler

R = lambda x: round(float(x), 4)
out = {}

# ── A. 3-token 词表数一遍 rank：逐元素比较（含自身）────────────────
logits3 = torch.tensor([[2.0, 1.0, 0.5]])
lp3 = torch.log_softmax(logits3, dim=-1)  # = logits - logsumexp
sampled3 = torch.tensor([1], dtype=torch.int64)
v3 = lp3[0, 1]
cmp3 = [bool(x >= v3) for x in lp3[0]]
rank3 = batched_count_greater_than(lp3, v3.reshape(1, 1))
out["rank_3token_counting"] = {
    "logits": [2.0, 1.0, 0.5],
    "logprobs": [R(x) for x in lp3[0]],
    "sampled_token": 1,
    "its_logprob_v": R(v3),
    "elementwise_ge_v": cmp3,
    "count": int(rank3[0]),
    "claim": "rank=#{logprob >= 自身}=2：−0.4644≥v ✓、自身 −1.4644≥v ✓（计入）、−1.9644<v ✗——O(V) 计数、无需排序",
}

# ── B. 并列：相等 logprob 共享名次（含自身、含并列）────────────────
lp_ties = torch.tensor([[0.0, 0.5, 0.5, 1.0]])
v_ties = torch.tensor([[0.5]])
cmp_ties = [bool(x >= 0.5) for x in lp_ties[0]]
out["rank_with_ties"] = {
    "logprobs": [0.0, 0.5, 0.5, 1.0],
    "v": 0.5,
    "elementwise_ge_v": cmp_ties,
    "count": int(batched_count_greater_than(lp_ties, v_ties)[0]),
    "claim": "两个 0.5 并列 + 1.0 + 自身 → count=3：并列 token 共享同一名次，>= 计数天然实现（0 基名次=count-1）",
}

# ── C. 端到端：raw 先行 + topk 拼装（被采样 token 恒第 0 列）────────
# V=5；frequency_penalty=3.0 × token0 出现 3 次 → 惩罚后 greedy 改选 token1；
# logprobs_tensors 取的是第 1 步留底的 raw（未经惩罚）视角。
V = 5
logits_c = torch.tensor([[2.0, 1.5, 1.0, 0.5, 0.0]])
md_c = dict(
    temperature=None, all_greedy=True, all_random=False, top_p=None, top_k=None,
    generators={}, max_num_logprobs=2, no_penalties=False,
    prompt_token_ids=torch.tensor([[4]]),  # 在词表内的 prompt token；repetition=1.0 对它 no-op
    frequency_penalties=torch.tensor([3.0]), presence_penalties=torch.tensor([0.0]),
    repetition_penalties=torch.tensor([1.0]),
    output_token_ids=[[0, 0, 0]], allowed_token_ids_mask=None,
    bad_words_token_ids={}, logitsprocs=LogitsProcessors(),
)
o = Sampler()(logits_c.clone(), SamplingMetadata(**md_c))
lt = o.logprobs_tensors
raw_lp = torch.log_softmax(logits_c, dim=-1)
pen_logits = logits_c.clone()
pen_logits[0, 0] -= 3.0 * 3  # freq × 出现次数
pen_lp = torch.log_softmax(pen_logits, dim=-1)
sampled_c = int(o.sampled_token_ids.flatten()[0])
out["end_to_end_raw_first_topk"] = {
    "raw_logits": [2.0, 1.5, 1.0, 0.5, 0.0],
    "penalty": "frequency=3.0 × token0 出现 3 次 → logit0 2.0−9.0",
    "raw_argmax": 0,
    "sampled_after_penalty": sampled_c,
    "raw_logprobs": [R(x) for x in raw_lp[0]],
    "penalized_logprobs": [R(x) for x in pen_lp[0]],
    "sampled_logprob_raw_view": R(lt.logprobs[0, 0]),
    "sampled_logprob_penalized_view": R(pen_lp[0, sampled_c]),
    "logprob_token_ids_row": lt.logprob_token_ids[0].tolist(),
    "logprobs_row": [R(x) for x in lt.logprobs[0]],
    "token_ranks": lt.selected_token_ranks.tolist(),
    "shape": f"indices/logprobs=[1,3]=k+1、ranks=[1]",
    "claim": "被采样 token 恒第 0 列（indices[0]=[1, 0, 1]）；logprob=−1.3471 是 raw 视角——若用惩罚后视角应是 −0.7874，二者不等即 raw-first（V0 用惩罚后视角，NOTE(woosuk) L80-83）",
}

# ── D. num_logprobs==-1：全词表未排序不排名（RejectionSampler bonus）──
md_d = dict(md_c)
md_d["max_num_logprobs"] = -1
o2 = Sampler()(logits_c.clone(), SamplingMetadata(**md_d))
lt2 = o2.logprobs_tensors
out["full_vocab_unsorted_branch"] = {
    "max_num_logprobs": -1,
    "logprob_token_ids_shape": list(lt2.logprob_token_ids.shape),
    "logprobs_shape": list(lt2.logprobs.shape),
    "selected_token_ranks_shape": list(lt2.selected_token_ranks.shape),
    "logprobs_row": [R(x) for x in lt2.logprobs[0]],
    "equals_raw_log_softmax": bool(torch.allclose(lt2.logprobs, raw_lp)),
    "claim": "L138-L142：−1 → LogprobsTensors(empty(0), raw_logprobs, empty(0))——整行全词表、不排序不排名，供 RejectionSampler 统一算 bonus 位 logprobs（→ch32/33）",
}

out["table_rows_echo"] = [
    ["A: 逐元素计数", "lp=[−0.4644, −1.4644, −1.9644]", "v=−1.4644(token1 自身)", "比较=[✓, ✓(自身), ✗]", "rank=2——不排序"],
    ["B: 并列", "lp=[0.0, 0.5, 0.5, 1.0]", "v=0.5", "比较=[✗, ✓, ✓, ✓]", "count=3：并列共享名次"],
    ["C: 惩罚改变采样", "freq=3.0×3 次 token0", "logit0 2.0→−7.0(argmax 0→1)", f"sampled={sampled_c}", "greedy 吃惩罚后 logits"],
    ["C: raw 先行(step1)", "raw 视角 logprob(token1)=−1.3471", "惩罚后视角=−0.7874", "两者不等=raw-first 实证", "top-k logprobs 反映模型原始分布"],
    ["C: topk 拼装", f"topk2 下标=[0, 1] 值=[−0.8471, −1.3471]", f"cat 后 indices={lt.logprob_token_ids[0].tolist()}", f"ranks={lt.selected_token_ranks.tolist()}", "被采样 token 恒第 0 列（k+1 列）"],
    ["D: −1 分支", "max_num_logprobs=−1", "logprob_token_ids/ranks shape=[0]", "logprobs=[1,5]=整行 raw", "未排序不排名——bonus 位契约(→ch32)"],
]

print(json.dumps(out, ensure_ascii=False, indent=1))
with open(pathlib.Path(__file__).parent / "ch29_m16_gather_rank.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
