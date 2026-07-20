"""m15/m01 位掩码布局的代价算术（纯算术，无外部依赖）。

行宽公式来自源码：outlines / lm-format-enforcer 显式写
torch.full((max_num_seqs, (vocab_size + 31) // 32), -1, dtype=torch.int32)
（backend_outlines.py:L95-101 / backend_lm_format_enforcer.py:L137-143）；
xgrammar / guidance 由各自库的 allocate_token_bitmask 分配同构张量
（backend_xgrammar.py:L124-125）。
"""
from _fakes import dump

rows = []
raw = {"formula_row_int32": "(vocab_size + 31) // 32", "cases": []}
for vocab in (128, 32000, 128256, 150000):
    words = (vocab + 31) // 32
    mask_bytes = words * 4
    logits_bytes = vocab * 4
    ratio = logits_bytes / mask_bytes
    rows.append([
        str(vocab),
        str(words),
        str(mask_bytes),
        f"{mask_bytes / 1024:.1f}",
        f"{logits_bytes / 1024:.1f}",
        f"{ratio:.2f}",
    ])
    raw["cases"].append({
        "vocab_size": vocab, "row_int32_words": words,
        "mask_row_bytes": mask_bytes,
        "mask_row_kb": round(mask_bytes / 1024, 2),
        "logits_row_bytes": logits_bytes,
        "logits_row_kb": round(logits_bytes / 1024, 2),
        "ratio_logits_over_mask": round(ratio, 4),
    })

# 整张掩码（一批 max_num_seqs 条序列）
for max_num_seqs in (8, 256):
    vocab = 150000
    words = (vocab + 31) // 32
    total = max_num_seqs * words * 4
    raw.setdefault("full_bitmask", []).append({
        "max_num_seqs": max_num_seqs, "vocab_size": vocab,
        "shape": [max_num_seqs, words],
        "total_bytes": total, "total_kb": round(total / 1024, 2),
    })

raw["fill_value_all_allowed"] = -1
raw["note"] = (
    "-1 的 int32 补码是 32 个 1，即『该 32 个 token 全部允许』；掩码张量初值就是它。"
    "|V|=150000 时 ratio 恰好 32.00——每 token 1 bit 对 32 bit float32 的必然结果，"
    "只在 |V| 是 32 的倍数时精确成立（150000 = 32×4687.5，因 ceil 到 4688 而略小于 32）。"
)

dump("bitmask_math.json", {
    "mechanism": "m15-bitmask-layout",
    "columns": ["词表 |V|", "行宽（int32 个数）", "一行字节数", "一行 KB",
                "同一行 logits(float32) KB", "logits/掩码 倍数"],
    "rows": rows,
    "raw": raw,
})
