"""ch33 m09 驱动脚本（anchor-as-first：查询位布局 + 调度器 lookahead 账）。

跑法：python explainer/traces/run_m09_layout.py
产出：explainer/traces/run_m09_layout.json

素材来源：
- implementation/dspark.py（query_layout / num_lookahead_slots /
  anchor_as_first_inputs / fill_in_inputs）
- 布局 kernel 的 KV 槽位算术（query_pos//block_size、%block_size）按
  vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:L554-L585 的公式在 numpy 复算；
- dspark_block_size 卫兵口径：vllm/config/speculative.py:L1035-L1058。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import dspark  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


F = lambda x, n=6: round(float(x), n)  # noqa: E731

N = 5
P("== ch33 m09 · anchor-as-first：γ 个输入（anchor+γ−1 mask）→ γ 个 draft logits ==")

# ── A) 查询位布局小表（N=5） ────────────────────────────────────────────────
P("")
P("-- [m09·A] DSpark 布局 N=5：每个查询位都是预测位（sample_off=0、sample_pos=query_pos+1）--")
lay_d = dspark.query_layout("dspark", N)
P(f"    num_queries = {lay_d['num_queries']}")
for r in lay_d["records"]:
    P(f"    query_off={r['query_off']}: input={r['input']:6s} sampled={r['sampled']}  "
          f"target_off={r['target_off']}（预测位 = 查询位 + 1）")
P(f"    输入串 = {dspark.anchor_as_first_inputs(7, N)}（anchor=7 + 4 个 MASK=-1）")

P("")
P("-- [m09·B] DFlash 对照布局：1+N=6 个输入、anchor 是 bonus 槽（sample_off=1）--")
lay_f = dspark.query_layout("dflash", N)
P(f"    num_queries = {lay_f['num_queries']}")
for r in lay_f["records"]:
    to = "—" if r["target_off"] is None else r["target_off"]
    P(f"    query_off={r['query_off']}: input={r['input']:6s} sampled={r['sampled']!s:5s}  "
          f"target_off={to}（mask 位预测自身位置）")
P(f"    输入串 = {dspark.fill_in_inputs(7, N)}（anchor=7 + 5 个 MASK=-1）")

# ── C) KV 槽位算术（kernel 公式复算） ────────────────────────────────────────
P("")
P("-- [m09·C] KV 槽位算术（dflash speculator L557-L573 公式：q_block=query_pos//bs、q_slot=block_id*bs+pos%bs）--")
last_valid_pos, block_size = 9, 4
block_table = [5, 9, 7, 11]  # 该请求的块表（block_num -> 物理 block id）
P(f"    请求已写到 last_valid_pos={last_valid_pos}（0..9 共 10 个 token）、block_size={block_size}、"
      f"block_table={block_table}")
P(f"    {'query_off':>9} {'query_pos':>9} {'q_block_num':>11} {'q_block_id':>10} {'q_slot':>7} {'sample_pos':>10}")
for k in range(N):
    query_pos = last_valid_pos + 1 + k
    q_block_num = min(query_pos // block_size, len(block_table) - 1)
    q_block_id = block_table[q_block_num]
    q_slot = q_block_id * block_size + (query_pos % block_size)
    sample_pos = query_pos + 1
    P(f"    {k:>9} {query_pos:>9} {q_block_num:>11} {q_block_id:>10} {q_slot:>7} {sample_pos:>10}")

# ── D) 调度器 lookahead 账 ──────────────────────────────────────────────────
P("")
P("-- [m09·D] 调度器 lookahead 槽位账（scheduler.py:L261-L270）--")
P(f"    DFlash：num_lookahead = N+1 = {dspark.num_lookahead_slots('dflash', N)}"
      f"（in-fill 式：最后已采 token 的查询 + 每 draft 位查询）")
P(f"    DSpark：num_lookahead = N   = {dspark.num_lookahead_slots('dspark', N)}"
      f"（anchor 本身是第一个预测位，无单独 bonus 查询）")

# ── E) dspark_block_size 卫兵 ───────────────────────────────────────────────
P("")
P("-- [m09·E] dspark_block_size 卫兵（speculative.py:L1035-L1058 的行为口径）--")
dspark_block_size, n_spec = 5, 3
violates = n_spec < dspark_block_size
P(f"    dspark_block_size={dspark_block_size}、num_speculative_tokens={n_spec}："
      f"n_spec < block_size = {violates}")
P("    -> ValueError('DSpark requires num_speculative_tokens >= dspark_block_size (5); got 3.")
P("       Smaller values produce incorrect output. Use num_speculative_tokens=5 or larger (e.g. 7).')")
P("    卫兵性质：小了是『乱码级错误』不是『变慢』——块布局是正确性约束（Markov 头的 W_1/W_2")
P("    与采样循环的步数和 checkpoint 训练时的块布局绑定）")

# ══ 落盘 ═══════════════════════════════════════════════════════════════════
out = {
    "params": {"N": N, "last_valid_pos": last_valid_pos, "block_size": block_size,
               "block_table": block_table, "dspark_block_size": dspark_block_size,
               "n_spec_violating": n_spec},
    "raw_stdout": "\n".join(LINES),
}
jf = Path(__file__).with_suffix(".json")
jf.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"\n[written] {jf}")
