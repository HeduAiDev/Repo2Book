#!/usr/bin/env python3
"""01-logprobs-assembly-pipeline: 两条平行泳道 sample / prompt 装配总览。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 980, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, lines, fs=14, tcol="#0f172a", weight="normal"):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n = len(lines)
    cy = y + h / 2 - (n - 1) * (fs + 3) / 2 + fs * 0.35
    for i, t in enumerate(lines):
        L.append(f'<text x="{x + w/2}" y="{cy + i*(fs+3)}" font-family="sans-serif" '
                 f'font-size="{fs}" font-weight="{weight}" text-anchor="middle" fill="{tcol}">{esc(t)}</text>')


def arrow(x1, y1, x2, y2, col="#475569", marker="a"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
             f'stroke-width="1.8" marker-end="url(#{marker})"/>')


# 标题
L.append(f'<text x="{W/2}" y="30" font-family="sans-serif" font-size="18" '
         f'font-weight="bold" text-anchor="middle" fill="#0f172a">Logprobs 双路装配：update_from_output 分派</text>')

# 入口框
box(40, 55, 200, 46, "#e0e7ff", "#6366f1",
    ["EngineCoreOutput", "update_from_output"], fs=14, weight="bold")

# 上泳道 sample（蓝）
SY = 150
L.append(f'<rect x="20" y="{SY-28}" width="940" height="150" rx="10" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1"/>')
L.append(f'<text x="34" y="{SY-8}" font-family="sans-serif" font-size="13" font-weight="bold" fill="#1d4ed8">sample logprobs · new_logprobs (LogprobsLists, numpy)</text>')
steps_s = [
    (40, "逐 step", ["rank/logprobs/", "token_ids", ".tolist()"]),
    (200, "去 token", ["convert_ids_", "list_to_tokens", "（非增量）"]),
    (375, "上下文", ["_get_sampled_", "context_ids", "(self.logprobs)"]),
    (685, "累计", ["cumulative", "+= logprobs[0]"]),
    (835, "写入", ["append_logprobs_", "for_next_position"]),
]

# 下泳道 prompt（绿）
PY = 360
L.append(f'<rect x="20" y="{PY-28}" width="940" height="150" rx="10" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1"/>')
L.append(f'<text x="34" y="{PY-8}" font-family="sans-serif" font-size="13" font-weight="bold" fill="#15803d">prompt logprobs · new_prompt_logprobs_tensors (LogprobsTensors, torch)</text>')

# 共享修正框（横跨中间）
box(540, 248, 120, 64, "#fef3c7", "#f59e0b",
    ["_verify_tokens", "→ _correct_", "decoded_token"], fs=12, weight="bold")
L.append(f'<text x="600" y="328" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#92400e">U+FFFD 字节回退修正（共享）</text>')

# sample 步骤
bw, bh = 140, 64
xs_s = [40, 195, 375, 695, 835]
labels_s = [
    ["rank/logprobs/", "token_ids", ".tolist()"],
    ["convert_ids_", "list_to_tokens"],
    ["_get_sampled_", "context_ids", "(self.logprobs)"],
    ["cumulative_logprob", "+= logprobs[0]"],
    ["append_logprobs_", "for_next_position"],
]
for x, lab in zip(xs_s, labels_s):
    box(x, SY, bw, bh, "#dbeafe", "#3b82f6", lab, fs=12)
for i in range(len(xs_s) - 1):
    arrow(xs_s[i] + bw, SY + bh/2, xs_s[i+1], SY + bh/2)
# 入口 → sample 第一步
arrow(140, 101, 110, SY)
# context → 修正框 → 累计
arrow(xs_s[2] + bw, SY + bh/2, 540, 280)
arrow(660, 280, xs_s[3], SY + bh/2, col="#475569")
# self.logprobs 容器
box(835, SY+92, bw, 30, "#bfdbfe", "#2563eb", ["self.logprobs"], fs=12, weight="bold")
arrow(835+bw/2, SY+bh, 835+bw/2, SY+92)

# prompt 步骤
xs_p = [40, 215, 375, 835]
labels_p = [
    ["flatten().tolist()", ".tolist()", "（自行 Pythonize）"],
    ["逐 pos 切片", "offset=", "pos*num_logprobs"],
    ["_get_sampled_", "context_ids", "(self.prompt_logprobs)"],
    ["append_logprobs_", "for_next_position", "（无 cumulative）"],
]
for x, lab in zip(xs_p, labels_p):
    box(x, PY, bw, bh, "#dcfce7", "#22c55e", lab, fs=12)
for i in range(len(xs_p) - 1):
    arrow(xs_p[i] + bw, PY + bh/2, xs_p[i+1], PY + bh/2, col="#15803d", marker="ag")
arrow(140, 101, 110, PY, col="#15803d", marker="ag")
# context → 修正框（下行进，上行出到 append）
arrow(xs_p[2] + bw, PY + bh/2, 540, 280)
arrow(660, 280, xs_p[3], PY + bh/2, col="#15803d", marker="ag")
box(835, PY+92, bw, 30, "#bbf7d0", "#16a34a", ["self.prompt_logprobs"], fs=11, weight="bold")
arrow(835+bw/2, PY+bh, 835+bw/2, PY+92, col="#15803d", marker="ag")

# 底注差异
L.append(f'<text x="{W/2}" y="540" font-family="sans-serif" font-size="12.5" text-anchor="middle" fill="#475569">'
         f'差异：sample 收 numpy 已 tolist · 有 cumulative ｜ prompt 收 torch 张量需自行 Pythonize · 无 cumulative · 上下文取自本次累计的前序 prompt 位置</text>')

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch10-logprobs/diagrams/01-logprobs-assembly-pipeline.svg", "w").write('\n'.join(L))
print("ok")
