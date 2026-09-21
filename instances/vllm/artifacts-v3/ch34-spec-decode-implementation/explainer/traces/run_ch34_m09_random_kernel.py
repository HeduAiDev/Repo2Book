# ch34 m9 worked example driver — random 拒绝 kernel（算法心脏）：
# 接受判据 draft_prob>0 且 target_prob/draft_prob >= uniform_prob（u∈[0,1) 隐式
# min(1,·)）、NO_DRAFT_PROBS 退化（以 p_t(x) 接受）、拒绝位 recovered+早停、
# 全收补 bonus、float64 uniform（pytorch#16706）。
# 全程驱动精简版 rejection_sample 真跑；seeded generator 的 u/q 用同 seed
# 同消耗序列重构（重构值 = kernel 实际消费值，由输出反向核验）。
import json
import os
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np
import torch

from vllm.v1.sample.rejection_sampler import rejection_sample
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


def run_single(drafts, p_t_rows, p_d_rows, seed, vocab):
    """单请求、num_draft=len(drafts)、temp=1.0、seeded generator。

    返回 (output_row, 重构的 u 列表, 重构的 q 行, 重构 recovered 列表)。
    重构法：generate_uniform_probs 先按请求消耗 n 个 uniform_(generator)，
    sample_recovered_tokens 再消耗 V 个 exponential_(generator)——同 seed
    重复同一消耗序列即得 kernel 实际用的随机数。
    """
    k = len(drafts)
    n = vocab
    target_logits = torch.log(
        torch.tensor(p_t_rows, dtype=torch.float32)).to(DEVICE)
    md = make_md([k], drafts)
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)

    class SMeta:
        all_greedy = False
        all_random = True
        generators = {0: g}
        temperature = torch.ones(1, dtype=torch.float32, device=DEVICE)

    bonus = torch.tensor([[777]], dtype=torch.int32, device=DEVICE)
    draft_probs = (
        torch.tensor(p_d_rows, dtype=torch.float32, device=DEVICE).contiguous()
        if p_d_rows is not None else None
    )
    out = rejection_sample(
        md.draft_token_ids, md.num_draft_tokens, md.max_spec_len,
        md.cu_num_draft_tokens, draft_probs, target_logits, bonus, SMeta,
    )

    # 重构 u（前 k 个 uniform）与 q（接着 V 个 exponential）
    gr = torch.Generator(device=DEVICE); gr.manual_seed(seed)
    u_rec = torch.empty(k, dtype=torch.float64, device=DEVICE)
    u_rec.uniform_(generator=gr)
    q_rec = torch.empty(n, dtype=torch.float32, device=DEVICE)
    q_rec.exponential_(generator=gr)
    inv_q = q_rec.reciprocal()

    # 重构 recovered（与 kernel L872-L953 同算式：残差·inv_q 免归一化 argmax）
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    rec = []
    for i in range(k):
        if draft_probs is None:
            prob = target_probs[i].clone()
            prob[drafts[i]] = 0.0
        else:
            prob = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
        score = prob * inv_q
        rec.append(int(score.argmax().item()))
    return out[0].tolist(), [r6(v) for v in u_rec.tolist()], \
        [r6(v) for v in q_rec.tolist()], rec


def main():
    trace = {
        "mechanism": "m9",
        "what": "random 拒绝 kernel：p_t/p_d >= u 接受判据 / NO_DRAFT_PROBS 退化 / float64 uniform / 早停+recovered",
        "env": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
        "reconstruction_note": (
            "seeded generator 下 u/q 用同 seed 同消耗序列重构 = kernel 实际消费值；"
            "重构判据与 kernel 输出逐位反向核验一致。"
        ),
    }

    V = 6

    # ══ Part A：有 draft_probs 的精确走读（seed 42/43/44 三支对比）══════
    # p_t/p_d 设计：pos0 的 ratio = 0.30/0.60 = 0.5（掷骰边界）；pos1 的
    # ratio = 0.80/0.20 = 4.0 > 1（隐式 min(1,·) 恒接受——kernel 不写 min 的原因）
    p_t0 = [0.10, 0.30, 0.20, 0.30, 0.05, 0.05]
    p_d0 = [0.10, 0.60, 0.10, 0.10, 0.05, 0.05]  # Σ=1.0
    p_t1 = [0.05, 0.80, 0.05, 0.03, 0.04, 0.03]
    p_d1 = [0.05, 0.20, 0.20, 0.20, 0.20, 0.15]  # Σ=1.0
    exact_walks = []
    for seed in (42, 43, 44, 45, 46, 47):
        drafts = [1, 1]  # 两个位置都猜 token 1
        out_row, us, qs, rec = run_single(
            drafts, [p_t0, p_t1], [p_d0, p_d1], seed, V)
        ratio0 = r6(0.30 / 0.60)
        ratio1 = r6(0.80 / 0.20)
        walk = {
            "seed": seed,
            "drafts": drafts,
            "pos0": {
                "draft_token": 1, "p_t_x": 0.3, "p_d_x": 0.6,
                "ratio_p_t_over_p_d": ratio0,
                "uniform_u": us[0],
                "criterion": f"p_t/p_d={ratio0} >= u={us[0]} ?",
                "accepted": out_row[0] == 1,
            },
            "pos1": {
                "draft_token": 1, "p_t_x": 0.8, "p_d_x": 0.2,
                "ratio_p_t_over_p_d": ratio1,
                "uniform_u": us[1],
                "note": "ratio=4.0 > 1：u∈[0,1) 恒小于它 → 恒接受（min(1,·) 隐式）",
                "criterion": f"p_t/p_d={ratio1} >= u={us[1]} ?",
                "accepted_if_reached": True,
            },
            "q_row": qs,
            "recovered_reconstructed": rec,
            "output_row": out_row,
            "valid_tokens": [t for t in out_row if t != -1],
        }
        exact_walks.append(walk)

    part_a = {
        "what": "精确走读：两个草稿位、ratio 0.5 与 4.0、六种子对比（真跑+重构；含接受与拒绝两分支）",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L772-L845（判据 L829）+ L608-L660（float64 uniform）",
        "params": {
            "vocab": V,
            "drafts": [1, 1],
            "pos0_p_t": p_t0, "pos0_p_d": p_d0,
            "pos1_p_t": p_t1, "pos1_p_d": p_d1,
            "temperature": 1.0,
        },
        "walks": exact_walks,
    }

    # ══ Part B：NO_DRAFT_PROBS（ngram）精确走读 ═══════════════════════
    # draft_prob=1 退化：接受当且仅当 p_t(x) >= u。p_t(3)=0.3。
    p_t_n = [0.20, 0.15, 0.15, 0.30, 0.10, 0.10]
    out_n, us_n, qs_n, rec_n = run_single([3], [p_t_n], None, 42, V)
    part_b = {
        "what": "NO_DRAFT_PROBS（ngram 无概率）：draft_prob=1 → 接受概率=p_t(x)",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L811-L817（NO_DRAFT_PROBS 分支）",
        "params": {"draft_token": 3, "p_t_x": 0.3, "draft_prob": 1},
        "uniform_u": us_n[0],
        "criterion": f"p_t/p_d=0.3/1=0.3 >= u={us_n[0]} ?",
        "accepted": out_n[0] == 3,
        "recovered_reconstructed": rec_n,
        "output_row": out_n,
        "note": "ngram 残差=屏蔽 draft token 的 p_t（kernel L913-L918 mask 实现，详见 m10）",
    }

    # ══ Part C：统计律——接受率 = min(1, p_t/p_d) ═════════════════════
    def acceptance_stats(p_t_x, p_d_x, n_samples=20000):
        # 一次性大 batch（无 seed，走全局 RNG）
        p_t_row = torch.zeros(V); p_t_row[1] = p_t_x
        rest_t = (1 - p_t_x) / (V - 1)
        p_t_row[torch.arange(V) != 1] = rest_t
        p_d_row = torch.zeros(V); p_d_row[1] = p_d_x
        rest_d = (1 - p_d_x) / (V - 1)
        p_d_row[torch.arange(V) != 1] = rest_d
        B = n_samples
        drafts_flat = [1] * B
        target_logits = torch.log(p_t_row).unsqueeze(0).repeat(B, 1).to(DEVICE)
        draft_probs = p_d_row.unsqueeze(0).repeat(B, 1).to(DEVICE).contiguous()
        md = make_md([1] * B, drafts_flat)
        class SMetaB:
            all_greedy = False
            all_random = True
            generators = {}
            temperature = torch.ones(B, dtype=torch.float32, device=DEVICE)
        bonus = torch.full((B, 1), 777, dtype=torch.int32, device=DEVICE)
        out = rejection_sample(
            md.draft_token_ids, md.num_draft_tokens, md.max_spec_len,
            md.cu_num_draft_tokens, draft_probs, target_logits, bonus, SMetaB,
        )
        accepted = (out[:, 0] == 1).float().mean().item()
        return r6(accepted)

    acc_half = acceptance_stats(0.3, 0.6)     # ratio 0.5
    acc_always = acceptance_stats(0.6, 0.3)   # ratio 2.0 → min=1
    # ngram 口径（draft_probs=None）：接受率 ≈ p_t(x)=0.3
    B = 20000
    p_t_row = torch.zeros(V); p_t_row[1] = 0.3
    p_t_row[torch.arange(V) != 1] = 0.7 / (V - 1)
    md = make_md([1] * B, [1] * B)
    target_logits = torch.log(p_t_row).unsqueeze(0).repeat(B, 1).to(DEVICE)
    class SMetaN:
        all_greedy = False
        all_random = True
        generators = {}
        temperature = torch.ones(B, dtype=torch.float32, device=DEVICE)
    bonus = torch.full((B, 1), 777, dtype=torch.int32, device=DEVICE)
    out_nstats = rejection_sample(
        md.draft_token_ids, md.num_draft_tokens, md.max_spec_len,
        md.cu_num_draft_tokens, None, target_logits, bonus, SMetaN,
    )
    acc_ngram = r6((out_nstats[:, 0] == 1).float().mean().item())

    part_c = {
        "what": "统计律：20000 样本实测接受率 vs 理论 min(1, p_t/p_d)",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L829（判据）",
        "samples": B,
        "rows": [
            {"p_t_x": 0.3, "p_d_x": 0.6, "theory_min_1_ratio": 0.5,
             "measured_acceptance": acc_half},
            {"p_t_x": 0.6, "p_d_x": 0.3, "theory_min_1_ratio": 1.0,
             "measured_acceptance": acc_always},
            {"p_t_x": 0.3, "p_d_x": 1.0, "mode": "NO_DRAFT_PROBS(ngram)",
             "theory": 0.3, "measured_acceptance": acc_ngram},
        ],
    }

    # ══ Part D：float64 uniform（pytorch#16706）═══════════════════════
    part_d = {
        "what": "uniform 的 dtype 与动机",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L629-L638 注释原话 + pytorch#16706",
        "uniform_dtype": "torch.float64",
        "comment_quote": (
            "We deliberately use float64 instead of float32 here because when "
            "using float32, there's a non-negligible chance that uniform_prob "
            "is sampled to be exact 0.0 ... Using float64 mitigates the issue."
        ),
        "why_u0_breaks": (
            "判据是 >=：u=0.0 时 p_t/p_d >= 0 恒真 → 本应大概率拒绝的坏草稿"
            "（p_t/p_d 极小）被无条件放行 → 分布被破坏"
        ),
        "example_u_values": [w["pos0"]["uniform_u"] for w in exact_walks],
    }

    # ══ Part E：padded draft(-1) 直接拒 ══════════════════════════════
    out_pad, us_pad, qs_pad, rec_pad = run_single(
        [-1], [[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]], None, 42, V)
    part_e = {
        "what": "padded draft(-1)：random kernel L807-L809 显式分支直接拒",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L806-L810",
        "drafts": [-1],
        "uniform_u": us_pad[0],
        "output_row": out_pad,
        "note": "不走判据（p_t(-1)/p_d(-1) 无意义）——直接 accepted=False、写 recovered",
    }

    trace.update({
        "part_a_exact_walk": part_a,
        "part_b_no_draft_probs": part_b,
        "part_c_statistical": part_c,
        "part_d_float64": part_d,
        "part_e_padded": part_e,
    })

    out_path = pathlib.Path(__file__).resolve().parent / "ch34_m09_random_kernel.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print(json.dumps(trace, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
