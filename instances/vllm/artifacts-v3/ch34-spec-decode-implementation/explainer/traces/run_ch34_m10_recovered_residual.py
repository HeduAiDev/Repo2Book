# ch34 m10 worked example driver — recovered 残差采样：
# prob=max(p_t−p_d,0)、score=prob·inv_q 免归一化 argmax（Gumbel-max）、
# 每请求一份 q 的 v0.27 布局（两个草稿位共享同一 inv_q）、NO_DRAFT_PROBS
# 残差=屏蔽 draft token 的 p_t、边缘分布统计核验、全零残差退化。
# 全程驱动精简版 sample_recovered_tokens 真跑（Triton 残差 kernel）；
# seeded q 用同 seed 重构（= kernel 实际消费值），并与 torch 同算式逐位对拍。
import json
import os
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np
import torch

from vllm.v1.sample.rejection_sampler import sample_recovered_tokens
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def r6(x):
    return round(float(x), 6)


def make_md(num_draft, drafts_flat):
    cu_draft = np.cumsum(num_draft, dtype=np.int32)
    num_tokens = int(cu_draft[-1])
    batch = len(num_draft)
    return SpecDecodeMetadata(
        draft_token_ids=torch.tensor(drafts_flat, dtype=torch.int32, device=DEVICE),
        num_draft_tokens=num_draft,
        cu_num_draft_tokens=torch.from_numpy(cu_draft).to(DEVICE),
        cu_num_sampled_tokens=torch.tensor(
            np.cumsum([n + 1 for n in num_draft], dtype=np.int32)).to(DEVICE),
        target_logits_indices=torch.arange(num_tokens, dtype=torch.int32, device=DEVICE),
        bonus_logits_indices=torch.arange(batch, dtype=torch.int32, device=DEVICE),
        logits_indices=torch.arange(num_tokens + batch, dtype=torch.int32, device=DEVICE),
    )


class SMeta:
    all_greedy = False
    all_random = True
    generators = {}
    temperature = None


def main():
    trace = {
        "mechanism": "m10",
        "what": "recovered 残差采样：残差分布 / Gumbel-max 免归一化 / 每请求一份 q / NO_DRAFT 变体",
        "env": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
        "reconstruction_note": (
            "seeded q 用同 seed 重构（exponential_(generator) 消耗序列一致）"
            "= kernel 实际消费值；kernel 输出与 torch 同算式逐位对拍一致。"
        ),
    }

    V = 6

    # ══ Part A：精确走读——1 请求 2 草稿位共享同一份 q ═════════════════
    p_t_rows = [
        [0.30, 0.25, 0.20, 0.10, 0.10, 0.05],   # pos0 的 target 分布
        [0.10, 0.10, 0.30, 0.25, 0.15, 0.10],   # pos1 的 target 分布
    ]
    p_d_rows = [
        [0.20, 0.35, 0.10, 0.15, 0.10, 0.10],   # pos0 的 draft 分布
        [0.10, 0.05, 0.10, 0.15, 0.30, 0.30],   # pos1 的 draft 分布
    ]
    drafts = [1, 2]
    seed = 2024

    target_probs = torch.tensor(p_t_rows, dtype=torch.float32, device=DEVICE)
    draft_probs = torch.tensor(p_d_rows, dtype=torch.float32, device=DEVICE).contiguous()
    md = make_md([2], drafts)

    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    smeta = SMeta()
    smeta.generators = {0: g}
    recovered = sample_recovered_tokens(
        2, [2], md.cu_num_draft_tokens, md.draft_token_ids,
        draft_probs, target_probs, smeta, DEVICE,
    )
    rec_out = recovered.tolist()

    # 重构 q：fresh seed → exponential_(V)（本次 standalone 调用里 generator
    # 的唯一消耗序列；q.exponential_() 全局 RNG 先写满、再被 generator 行覆写）
    gr = torch.Generator(device=DEVICE); gr.manual_seed(seed)
    q_row = torch.empty(V, dtype=torch.float32, device=DEVICE)
    q_row.exponential_(generator=gr)
    inv_q = q_row.reciprocal()

    # torch 对拍（与 kernel L872-L953 同算式）
    per_pos = []
    for i in range(2):
        residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
        score = residual * inv_q
        per_pos.append({
            "pos": i,
            "draft_token": drafts[i],
            "p_t": [r6(v) for v in target_probs[i].tolist()],
            "p_d": [r6(v) for v in draft_probs[i].tolist()],
            "residual_max_pt_minus_pd_0": [r6(v) for v in residual.tolist()],
            "residual_mass": r6(residual.sum().item()),
            "residual_normalized": [r6(v / residual.sum().item()) for v in residual.tolist()],
            "score_residual_times_inv_q": [r6(v) for v in score.tolist()],
            "argmax": int(score.argmax().item()),
            "kernel_output": rec_out[i],
            "match": int(score.argmax().item()) == rec_out[i],
        })

    resid0 = torch.clamp(target_probs[0] - draft_probs[0], min=0.0)
    resid1 = torch.clamp(target_probs[1] - draft_probs[1], min=0.0)
    part_a = {
        "what": "精确走读：残差=两分布之差的正部、两位共享同一 inv_q（真跑+重构对拍）",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L663-L710 + L872-L953（score=prob·inv_q 免归一化 L931-L932）",
        "params": {
            "vocab": V, "num_draft_tokens": 2, "drafts": drafts,
            "q_layout": "[batch=1, vocab=6] 一行——两位共享（L678 注释 'Create only one distribution for each request'）",
            "noise_saved": "对照逐位一份 [num_tokens=2, vocab=6]：省 1 倍；真实批 B=256、num_tokens=1024 时省 4 倍",
        },
        "q_row": [r6(v) for v in q_row.tolist()],
        "inv_q": [r6(v) for v in inv_q.tolist()],
        "tv_distance_pos0": r6((target_probs[0] - draft_probs[0]).abs().sum().item() / 2),
        "residual_mass_pos0": r6(resid0.sum().item()),
        "residual_mass_pos1": r6(resid1.sum().item()),
        "positions": per_pos,
        "all_match": all(p["match"] for p in per_pos),
        "note": (
            "残差总质量 = Σmax(p_t−p_d,0) = (Σ|p_t−p_d|)/2 = 两分布的 TV 距离"
            "（残差归一化即 ch33 的残差分布 norm(max(0,p_t−p_d))）"
        ),
    }

    # ══ Part B：统计律——免归一化 argmax 的边缘分布 = 残差归一化分布 ════
    # 单草稿位、固定分布、20000 次独立 q（无 seed）→ recovered 频率 ≈ 残差/Σ残差
    p_t_b = torch.tensor([0.30, 0.25, 0.20, 0.10, 0.10, 0.05], device=DEVICE)
    p_d_b = torch.tensor([0.20, 0.35, 0.10, 0.15, 0.10, 0.10], device=DEVICE)
    residual_b = torch.clamp(p_t_b - p_d_b, min=0.0)
    expected = (residual_b / residual_b.sum()).tolist()
    N = 20000
    counts = torch.zeros(V, dtype=torch.long)
    smeta_b = SMeta()
    tp_b = p_t_b.unsqueeze(0)
    dp_b = p_d_b.unsqueeze(0).contiguous()
    for _ in range(N):
        md_b = make_md([1], [1])
        rec = sample_recovered_tokens(
            1, [1], md_b.cu_num_draft_tokens, md_b.draft_token_ids,
            dp_b, tp_b, smeta_b, DEVICE,
        )
        counts[rec[0].item()] += 1
    measured = (counts.float() / N).tolist()
    part_b = {
        "what": "统计律：Gumbel-max 免归一化 argmax 的边缘分布 = 残差归一化分布",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L931-L932 注释 + arXiv:1611.01162（Gumbel-max）",
        "samples": N,
        "expected": [r6(v) for v in expected],
        "measured": [r6(v) for v in measured],
        "max_abs_dev": r6(max(abs(a - b) for a, b in zip(expected, measured))),
        "note": "单看任一位 argmax(残差·inv_q) 都是残差分布的合法样本——共享 q 只造成跨位相关，而只有首个拒绝位被消费",
    }

    # ══ Part C：NO_DRAFT_PROBS（ngram）残差 = 屏蔽 draft token 的 p_t ═══
    p_t_c = [0.50, 0.30, 0.20, 0.0, 0.0, 0.0]
    draft_c = 0
    g_c = torch.Generator(device=DEVICE); g_c.manual_seed(777)
    smeta_c = SMeta(); smeta_c.generators = {0: g_c}
    md_c = make_md([1], [draft_c])
    tp_c = torch.tensor([p_t_c], device=DEVICE)
    rec_c = sample_recovered_tokens(
        1, [1], md_c.cu_num_draft_tokens, md_c.draft_token_ids,
        None, tp_c, smeta_c, DEVICE,
    )
    gr_c = torch.Generator(device=DEVICE); gr_c.manual_seed(777)
    q_c = torch.empty(V, dtype=torch.float32, device=DEVICE)
    q_c.exponential_(generator=gr_c)
    inv_q_c = q_c.reciprocal()
    prob_c = tp_c[0].clone(); prob_c[draft_c] = 0.0
    score_c = prob_c * inv_q_c
    part_c = {
        "what": "NO_DRAFT_PROBS（ngram）：残差 = p_t 屏蔽 draft token（mask 实现 L913-L918）",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L913-L918",
        "params": {"p_t": p_t_c, "draft_token": draft_c},
        "prob_masked": [r6(v) for v in prob_c.tolist()],
        "q_row": [r6(v) for v in q_c.tolist()],
        "score": [r6(v) for v in score_c.tolist()],
        "torch_argmax": int(score_c.argmax().item()),
        "kernel_output": rec_c.tolist(),
        "match": int(score_c.argmax().item()) == rec_c[0].item(),
    }

    # ══ Part D：退化观察——p_t == p_d 全零残差 ═════════════════════════
    p_t_d = [0.20, 0.20, 0.20, 0.20, 0.10, 0.10]
    md_d = make_md([1], [1])
    tp_d = torch.tensor([p_t_d], device=DEVICE)
    dp_d = tp_d.clone().contiguous()
    smeta_d = SMeta()
    rec_d = sample_recovered_tokens(
        1, [1], md_d.cu_num_draft_tokens, md_d.draft_token_ids,
        dp_d, tp_d, smeta_d, DEVICE,
    )
    residual_d = torch.clamp(tp_d[0] - dp_d[0], min=0.0)
    part_d = {
        "what": "退化观察：p_t == p_d → 残差全零 → 所有 score=0 → argmax 取 index 0",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L932-L953（OOV mask −inf + clamp 的数值卫生就在这条全零路径上）",
        "residual": [r6(v) for v in residual_d.tolist()],
        "kernel_output": rec_d.tolist(),
        "note": (
            "工程上到不了这里：p_t==p_d 时接受判据 ratio=1 恒成立、永不拒绝、"
            "recovered 无人消费；OOV mask+clamp（L947-L954）保证就算到了也不越界"
        ),
    }

    trace.update({
        "part_a_exact": part_a,
        "part_b_marginal": part_b,
        "part_c_no_draft": part_c,
        "part_d_degenerate": part_d,
    })

    out_path = pathlib.Path(__file__).resolve().parent / "ch34_m10_recovered_residual.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print(json.dumps(trace, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
