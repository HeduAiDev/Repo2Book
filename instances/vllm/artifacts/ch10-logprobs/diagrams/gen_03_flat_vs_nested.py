#!/usr/bin/env python3
"""03-flat-vs-nested: FlatLogprobs vs list[dict[int,Logprob]] 内存布局对比。
3 个位置，每位置 K=2 候选。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 980, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" font-family="sans-serif" font-size="18" font-weight="bold" '
         f'text-anchor="middle" fill="#0f172a">FlatLogprobs vs list[dict[int, Logprob]]：同样 3 位置 × 2 候选</text>')


def txt(x, y, s, fs=12, col="#0f172a", anchor="start", weight="normal", family="sans-serif"):
    L.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{fs}" '
             f'text-anchor="{anchor}" font-weight="{weight}" fill="{col}">{esc(s)}</text>')


def rect(x, y, w, h, fill, stroke, rx=5):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')


# ===== 左：nested =====
txt(40, 62, "nested：list[dict[int, Logprob]]", 15, "#b91c1c", weight="bold")
txt(40, 82, "对象数 O(L × K)：每位置 1 个 dict + 每候选 1 个 Logprob", 12, "#7f1d1d")

lx = 40
ly = 100
for p in range(3):
    by = ly + p * 120
    # dict 框
    rect(lx, by, 360, 100, "#fef2f2", "#fca5a5")
    txt(lx + 10, by + 20, f"位置 {p}  dict 对象", 12, "#991b1b", weight="bold")
    for k in range(2):
        ox = lx + 12 + k * 175
        oy = by + 32
        rect(ox, oy, 165, 56, "#fee2e2", "#ef4444", rx=4)
        tid = 1065 + p * 2 + k
        txt(ox + 8, oy + 18, f"key {tid} →", 11, "#7f1d1d", family="monospace")
        txt(ox + 8, oy + 34, "Logprob(", 11, "#991b1b", weight="bold", family="monospace")
        txt(ox + 8, oy + 48, " logprob,rank,tok)", 10.5, "#991b1b", family="monospace")

txt(lx, ly + 3*120 + 6, "3 dict + 6 Logprob = 9 个 Python 对象（长序列×大 K 会爆炸）", 12, "#b91c1c", weight="bold")

# ===== 右：flat =====
rx0 = 540
txt(rx0, 62, "FlatLogprobs：6 条平行原生列表 + 区间索引", 15, "#15803d", weight="bold")
txt(rx0, 82, "对象数 O(1)：6 个 list，元素是原生 int/float/str", 12, "#166534")

arrays = [
    ("start_indices", ["0", "2", "4"], "#dcfce7", "#22c55e"),
    ("end_indices", ["2", "4", "6"], "#dcfce7", "#22c55e"),
    ("token_ids", ["1065", "1066", "1067", "1068", "1069", "1070"], "#dbeafe", "#3b82f6"),
    ("logprobs", ["-.5", "-1.", "-.3", "-.9", "-.2", "-1."], "#dbeafe", "#3b82f6"),
    ("ranks", ["3", "1", "4", "1", "2", "1"], "#dbeafe", "#3b82f6"),
    ("decoded", ["A", "B", "C", "D", "E", "F"], "#dbeafe", "#3b82f6"),
]
ay = 100
cellw = 56
for ai, (name, vals, bg, st) in enumerate(arrays):
    yy = ay + ai * 50
    txt(rx0, yy + 22, name, 12, "#166534", weight="bold")
    bx = rx0 + 110
    for vi, v in enumerate(vals):
        rect(bx + vi * cellw, yy, cellw - 4, 32, bg, st, rx=3)
        txt(bx + vi * cellw + (cellw - 4)/2, yy + 21, v, 11.5, "#0f172a", anchor="middle", family="monospace")

# 区间箭头：start/end 指回 token_ids 切片
# 位置0 → token_ids[0:2]
tok_y = ay + 2*50
sx = rx0 + 110
txt(rx0, ay + 6*50 + 18, "位置 i 的候选 = token_ids[start_indices[i] : end_indices[i]]", 12, "#15803d", weight="bold")
txt(rx0, ay + 6*50 + 40, "__getitem__(i) 现造 {token_id: Logprob} dict，向后兼容 list[dict] 接口", 11.5, "#166534")
txt(rx0, ay + 6*50 + 62, "→ 不论序列多长、K 多大，常驻对象数恒为 6，GC 扫描成本不随规模增长", 11.5, "#166534")

# 底部对照条
by2 = H - 26
txt(W/2, by2, "GC 压力：nested 随 L×K 线性增长　｜　flat 恒定。访问单位置：nested O(1) 取 dict，flat O(K) 现造 dict。",
    12.5, "#475569", anchor="middle")

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch10-logprobs/diagrams/03-flat-vs-nested.svg", "w").write('\n'.join(L))
print("ok")
