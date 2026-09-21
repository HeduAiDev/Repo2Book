# ch34 m2 worked example driver — draft 排批的调度器三本账（token 差账 / 拒绝回扣 / lookahead）。
#
# 取证方式（诚实声明）：scheduler.py 不在本章精简版范围（dossier subtraction_plan.delete[5]：
# 跨进程编排+持久批依赖，只作正文内嵌解读）。本驱动把三本账的**算式逐字复刻**自
# v0.27.1 真实源码行（每个算式带 file:Lxxx 锚点写进 trace），并驱动精简版
# rejection_sampler.py 的真实 kernel 产出「验证期采样结果」——账本走读的每一步都是
# 真算式真跑，不是手推：
#   num_new_tokens 公式        vllm/v1/core/sched/scheduler.py:L516-L520
#   num_scheduled_spec_tokens  vllm/v1/core/sched/scheduler.py:L640-L648
#   spec_token_ids 清空        vllm/v1/core/sched/scheduler.py:L654-L656
#   乐观推进 num_computed      vllm/v1/core/sched/scheduler.py:L1326-L1333
#   拒绝回扣                   vllm/v1/core/sched/scheduler.py:L1766-L1790
#   抢占清草稿                 vllm/v1/core/sched/scheduler.py:L1290-L1296
#   lookahead 三态             vllm/v1/core/sched/scheduler.py:L261-L270
# 采样结果（generated/accepted/rejected）来自 implementation/vllm/v1/sample/rejection_sampler.py
# 的 rejection_sample 真跑（RTX PRO 6000 上 Triton greedy kernel），greedy 全确定性零 RNG。
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
from vllm.v1.sample.sampler import Sampler  # noqa: F401 (组合持有面)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V = 100  # 玩具词表：token id 直接用可读数字


def r6(x):
    return round(float(x), 6)


def verify_greedy(drafts, argmax_seq, bonus_id):
    """真跑 rejection_sample（greedy 路径，L715-L769 Triton kernel）。

    drafts: 本拍草稿 [d1..dk]；argmax_seq: k 个草稿位各自的 target argmax
    （target_logits=logits[target_logits_indices] 只有草稿位行；bonus 位由
    外部 Sampler 供给，按真实调用面以 bonus_token_ids 传入）。
    返回 (output_row, generated_list)。
    """
    k = len(drafts)
    assert len(argmax_seq) == k
    num_draft_tokens = [k]
    cu_draft = np.cumsum(num_draft_tokens, dtype=np.int32)
    rows = []
    for a in argmax_seq:
        row = torch.zeros(V, dtype=torch.float32)
        row[int(a)] = 10.0
        rows.append(row)
    target_logits = torch.stack(rows).to(DEVICE)
    md = SpecDecodeMetadata(
        draft_token_ids=torch.tensor(drafts, dtype=torch.int32, device=DEVICE),
        num_draft_tokens=num_draft_tokens,
        cu_num_draft_tokens=torch.from_numpy(cu_draft).to(DEVICE),
        cu_num_sampled_tokens=torch.tensor([k + 1], dtype=torch.int32, device=DEVICE),
        target_logits_indices=torch.zeros(k, dtype=torch.int32, device=DEVICE),
        bonus_logits_indices=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        logits_indices=torch.zeros(k + 1, dtype=torch.int32, device=DEVICE),
    )

    class SM:  # 最小 SamplingMetadata 快照：all_greedy=True（greedy 判定路径）
        all_greedy = True
        all_random = False
        generators = {}
        temperature = None

    bonus = torch.full((1, 1), bonus_id, dtype=torch.int32, device=DEVICE)
    out = rejection_sample(
        md.draft_token_ids, md.num_draft_tokens, md.max_spec_len,
        md.cu_num_draft_tokens, None, target_logits, bonus, SM(),
    )
    output_row = out[0].tolist()
    # parse_output 的 valid_mask（rejection_sampler.py:L252-L287）逻辑同款过滤
    generated = [t for t in output_row if t != -1]
    return output_row, generated


def main():
    trace = {
        "mechanism": "m2",
        "what": "draft 排批的调度器三本账：token 差账 / 乐观推进+拒绝回扣 / lookahead / 抢占清草稿",
        "replication_note": (
            "scheduler 算式逐字复刻自 v0.27.1（scheduler.py:L516-L520/L640-L656/"
            "L1326-L1333/L1766-L1790/L1290-L1296/L261-L270，见各步 provenance 字段）；"
            "验证期采样结果由精简版 rejection_sampler.rejection_sample 真跑产出"
            "（greedy Triton kernel，零 RNG 全确定性）。"
        ),
        "env": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
        "params": {
            "prompt_tokens": 10,
            "K": 3,
            "method": "eagle（num_lookahead_tokens=K，scheduler.py:L262-L264）",
            "num_sampled_tokens_per_step": 1,
            "scheduling": "sync（num_output_placeholders=0；async 变体见 async_rollback 段）",
            "token_budget": 2048,
        },
    }

    # ── lookahead 三态（L261-L270 逐字复刻的分支表）────────────────────
    K = 3
    lookahead = {
        "provenance": "vllm/v1/core/sched/scheduler.py:L261-L270",
        "eagle_or_draft_model": K,       # L262-L264 / L265-L267
        "dflash": K + 1,                 # L268-L272：in-fill 式多一查 → +1
        "dspark": K,                     # L273-L276：anchor 即预测位 → =K
    }

    # ── 账本走读：1 个请求、2 个投机周期 ────────────────────────────────
    prompt = 10
    # 周期 1 起点状态（普通 decode 走到第 5 个 output：o5 刚采样、还没喂进前向）
    num_tokens = prompt + 5       # 10 prompt + o1..o5 已生成（spec 不计入 num_tokens）
    num_computed = prompt + 4     # 喂到 o4 的位置；o5 采样自 o4 位的 logits
    ledgers = []

    def step(name, prov, **kv):
        ledgers.append({"step": name, "provenance": prov, **kv})

    # —— 周期 1：3 草稿，接受 1 个、拒 2 个 ——
    drafts1 = [31, 32, 33]
    nts = num_tokens + len(drafts1)  # update_draft_token_ids 挂账后的 num_tokens_with_spec
    step("c1 挂账 update_draft_token_ids", "scheduler.py:L2146-L2167",
         spec_token_ids=sorted(drafts1), num_tokens=num_tokens,
         num_tokens_with_spec=nts, num_computed_tokens=num_computed,
         num_output_placeholders=0)
    # 排批（L516-L520 + L640-L656）
    num_new = nts + 0 - num_computed                     # L516-L520
    num_sched_spec = num_new + num_computed - num_tokens - 0  # L641-L644
    assert num_sched_spec == len(drafts1)
    step("c1 排批 schedule()", "scheduler.py:L516-L520 + L640-L656",
         num_new_tokens=num_new, num_scheduled_spec_tokens=num_sched_spec,
         scheduled_spec_decode_tokens=sorted(drafts1),
         spec_token_ids_after="[]（L654-L656 排完即清空）",
         note="num_new_tokens=4 = 1(o5 补喂) + 3(草稿)")
    # target 一次前向 + RejectionSampler（真跑）：d1=31 对 argmax 31 接受；
    # d2=32 位 argmax=77 → 拒，写 77（greedy 残差=argmax），早停
    output_row1, generated1 = verify_greedy(drafts1, [31, 77, 0], bonus_id=99)
    step("c1 验证 rejection_sample（真跑）", "rejection_sampler.py:L715-L769",
         output_row=output_row1, generated=generated1, len_generated=len(generated1))
    # 乐观推进（L1326-L1333）+ 回扣（L1766-L1790）
    num_computed_opt = num_computed + num_new
    num_accepted = max(len(generated1) - 1, 0)
    num_rejected = len(drafts1) - num_accepted
    num_computed_after = num_computed_opt - num_rejected
    step("c1 记账 update_from_output", "scheduler.py:L1326-L1333 + L1766-L1790",
         num_computed_tokens_optimistic=num_computed_opt,
         num_accepted=num_accepted, num_rejected=num_rejected,
         num_computed_tokens_after_rollback=num_computed_after,
         note="回扣 2：被拒的 d2/d3 位置退回『未计算』，下拍补喂 recovered")
    num_tokens = num_tokens + len(generated1)  # update_from_output 追加生成 token
    step("c1 收口", "scheduler.py:update_from_output 追加 output_token_ids",
         num_tokens=num_tokens, num_computed_tokens=num_computed_after,
         check="num_computed=16 覆盖 prompt10+o1..o5(5)+31(1)；32/33 两位已退回")

    # —— 周期 2：3 草稿全收 + bonus ——
    drafts2 = [41, 42, 43]
    nts2 = num_tokens + len(drafts2)
    step("c2 挂账 update_draft_token_ids", "scheduler.py:L2146-L2167",
         spec_token_ids=sorted(drafts2), num_tokens=num_tokens,
         num_tokens_with_spec=nts2, num_computed_tokens=num_computed_after,
         num_output_placeholders=0)
    num_new2 = nts2 + 0 - num_computed_after
    num_sched_spec2 = num_new2 + num_computed_after - num_tokens - 0
    step("c2 排批 schedule()", "scheduler.py:L516-L520 + L640-L656",
         num_new_tokens=num_new2, num_scheduled_spec_tokens=num_sched_spec2,
         scheduled_spec_decode_tokens=sorted(drafts2),
         note="num_new_tokens=4 = 1(上拍 recovered=77 补喂) + 3(新草稿)")
    output_row2, generated2 = verify_greedy(drafts2, [41, 42, 43], bonus_id=99)
    step("c2 验证 rejection_sample（真跑）", "rejection_sampler.py:L715-L769",
         output_row=output_row2, generated=generated2, len_generated=len(generated2),
         note="三位全收 + bonus=99 → 4 token/拍（一次前向的白嫖上限 k+1）")
    num_computed_opt2 = num_computed_after + num_new2
    num_accepted2 = max(len(generated2) - 1, 0)
    num_rejected2 = len(drafts2) - num_accepted2
    num_computed2 = num_computed_opt2 - num_rejected2
    num_tokens2 = num_tokens + len(generated2)
    step("c2 记账收口", "scheduler.py:L1326-L1333 + L1766-L1790",
         num_computed_tokens_optimistic=num_computed_opt2,
         num_accepted=num_accepted2, num_rejected=num_rejected2,
         num_computed_tokens_after_rollback=num_computed2,
         num_tokens=num_tokens2,
         note="num_rejected=0 → 不回扣；差账 num_tokens_with_spec−num_computed=4=1+3 下拍照常")

    # ── async 变体：回扣两本账共变（L1780-L1784）──────────────────────
    # async 下占位数与计算账同步回扣（聚焦该行算式，不展开完整 async 状态机）
    async_case = {
        "provenance": "vllm/v1/core/sched/scheduler.py:L1779-L1784",
        "scenario": "async 调度下同一 c1 场景：num_output_placeholders=3（草稿占位）",
        "num_rejected": num_rejected,
        "num_computed_tokens_before": num_computed_opt,
        "num_computed_tokens_after": num_computed_after,
        "num_output_placeholders_before": 3,
        "num_output_placeholders_after": 3 - num_rejected,
    }

    # ── 抢占：清草稿 + 计数归零（L1290-L1296）─────────────────────────
    preempt = {
        "provenance": "vllm/v1/core/sched/scheduler.py:L1290-L1296",
        "spec_token_ids_before": sorted(drafts2),
        "num_computed_tokens_before": num_computed2,
        "after": {"spec_token_ids": [], "num_computed_tokens": 0},
        "note": "stale 输出仍送出但不得回扣（L1777-L1779 output_is_stale 守卫）",
    }

    # ── 产出账（quantified 用）────────────────────────────────────────
    # 普通走到第 5 个 output：5 拍每拍 1 token；随后 c1/c2 两个投机周期
    normal_steps_for_first5 = 5
    total_tokens = 5 + len(generated1) + len(generated2)
    total_steps = normal_steps_for_first5 + 2
    production = {
        "spec_mode": {"tokens": total_tokens, "steps": total_steps,
                      "tokens_per_step": r6(total_tokens / total_steps)},
        "no_spec_mode": {"tokens": total_tokens, "steps": total_tokens},
        "speedup": r6(total_tokens / total_steps),
        "note": "加速=接受率依赖：c1 只产 2 token（拒 2），c2 产 4（全收+bonus）",
    }

    trace.update({
        "lookahead": lookahead,
        "ledger_walk": ledgers,
        "async_rollback": async_case,
        "preemption_clear": preempt,
        "production_account": production,
    })

    out_path = pathlib.Path(__file__).resolve().parent / "ch34_m02_scheduler_ledger.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print(json.dumps(trace, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
