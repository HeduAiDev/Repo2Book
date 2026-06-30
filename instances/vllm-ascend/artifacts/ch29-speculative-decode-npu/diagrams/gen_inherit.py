#!/usr/bin/env python3
"""fig29-2: 薄壳继承 vs 重量级重写 —— 左列单继承覆写几处，右列多继承组合策略类 + 昇腾 base 并重写一批。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1600, 840
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
    '<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=9, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=14, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


def line(x1, y1, x2, y2, color="#64748b", sw=1.6, marker="ar", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#{marker})"{d}/>')


text(W / 2, 40, "薄壳继承 vs 重量级重写：能复用就薄壳，必须特化才重写", 23, "middle", "#0f172a", "bold")
text(W / 2, 66, "灰虚线框 = vLLM 基座类（一行不改）    实线框 = 昇腾子类", 13.5, "middle", "#64748b")

# 分隔线
L.append(f'<line x1="740" y1="92" x2="740" y2="{H-30}" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="4 4"/>')

# ===== 左列：薄壳 =====
text(380, 116, "薄壳：继承 vLLM 单父类 → 覆写几处", 17, "middle", "#047857", "bold")

pairs = [
    ("NgramProposer", "AscendNgramProposer", "propose", "g"),
    ("NgramProposerGPU", "AscendNgramProposerNPU", "三方法 no-op", "a"),
    ("SuffixDecodingProposer", "AscendSuffixDecodingProposer", "propose 转发", "g"),
    ("MedusaProposer", "AscendMedusaProposer", "dummy_run/propose", "a"),
    ("ExtractHiddenStatesProposer", "AscendExtractHiddenStatesProposer", "dummy_run/prepare", "a"),
]
palette = {"g": ("#ecfdf5", "#10b981", "#047857"), "a": ("#fffbeb", "#f59e0b", "#b45309")}

py = 150
for parent, child, ov, cat in pairs:
    # 父类（灰虚线）
    box(60, py, 300, 38, "#f8fafc", "#cbd5e1", 8, 1.4, dash="5 4")
    text(70, py + 24, parent, 13, "start", "#64748b", "bold", mono=True)
    # 箭头
    line(370, py + 19, 410, py + 19)
    text(390, py + 7, "覆写", 10, "middle", "#94a3b8")
    # 子类
    fill, stroke, txt = palette[cat]
    box(420, py, 290, 38, fill, stroke, 8, 1.6)
    text(430, py + 24, child, 11.5, "start", txt, "bold", mono=True)
    py += 56

text(380, py + 16, "新增状态≈只有 self.runner / 一个上下文切换", 12.5, "middle", "#64748b")

# ===== 右列：重量级 =====
cx = 1110  # 右列中心
text(cx, 116, "重量级：多继承 (策略类, 昇腾 base) → 重写一批", 17, "middle", "#b91c1c", "bold")

# vLLM SpecDecodeBaseProposer
box(cx - 150, 150, 300, 38, "#f8fafc", "#cbd5e1", 8, 1.4, dash="5 4")
text(cx, 174, "SpecDecodeBaseProposer  (vLLM)", 12.5, "middle", "#64748b", "bold", mono=True)

# AscendSpecDecodeBaseProposer (red, big)
by = 214
bh = 96
box(cx - 230, by, 460, bh, "#fef2f2", "#ef4444", 10, 2.0)
text(cx, by + 24, "AscendSpecDecodeBaseProposer", 14.5, "middle", "#b91c1c", "bold", mono=True)
text(cx, by + 46, "重写 __init__ / load_model / _propose / prepare_inputs", 11.5, "middle", "#7f1d1d")
text(cx, by + 65, "ACLGraph · 昇腾 Triton spec_decode · MLA · 昇腾并行组", 11, "middle", "#991b1b")
text(cx, by + 84, "2043 行 —— 真正重量级的一处", 11, "middle", "#b91c1c", "bold")
line(cx, 188, cx, by, color="#b91c1c", marker="arr")
text(cx + 70, 206, "重写而非从零", 10.5, "middle", "#b91c1c")

# 子类（多继承）一排：左 Eagle 右 Draft，center 错开让中间留出 base 边的通道
eagle_cx, draft_cx = 905, 1315
# 策略类 (vLLM) 一排：放在各自子类正上方、偏「外侧」，给中间通道让路
sy = 360
box(eagle_cx - 130, sy, 215, 34, "#f8fafc", "#cbd5e1", 8, 1.4, dash="5 4")
text(eagle_cx - 22, sy + 22, "EagleProposer (vLLM)", 11, "middle", "#64748b", "bold", mono=True)
box(draft_cx + 15, sy, 230, 34, "#f8fafc", "#cbd5e1", 8, 1.4, dash="5 4")
text(draft_cx + 130, sy + 22, "DraftModelProposer (vLLM)", 10.5, "middle", "#64748b", "bold", mono=True)

ay = 450
# AscendEagleProposer
box(eagle_cx - 115, ay, 230, 56, "#fef2f2", "#ef4444", 9, 1.7)
text(eagle_cx, ay + 22, "AscendEagleProposer", 12, "middle", "#b91c1c", "bold", mono=True)
text(eagle_cx, ay + 42, "pass_hidden_states=True", 10.5, "middle", "#7f1d1d", mono=True)
# AscendDraftModelProposer
box(draft_cx - 115, ay, 230, 56, "#fef2f2", "#ef4444", 9, 1.7)
text(draft_cx, ay + 22, "AscendDraftModelProposer", 11, "middle", "#b91c1c", "bold", mono=True)
text(draft_cx, ay + 42, "pass_hidden_states=False", 10.5, "middle", "#7f1d1d", mono=True)

# 策略语义边（策略类→子类，外侧竖线）
line(eagle_cx - 22, sy + 34, eagle_cx - 22, ay, color="#94a3b8", dash="4 3")
line(draft_cx + 60, sy + 34, draft_cx + 60, ay, color="#94a3b8", dash="4 3")
text(eagle_cx - 22, sy + 52, "策略语义", 10, "middle", "#94a3b8")
text(draft_cx + 60, sy + 52, "策略语义", 10, "middle", "#94a3b8")

# 构造/前向边（昇腾 base→子类，红虚线，走中间内侧通道）
line(cx - 60, by + bh, eagle_cx + 60, ay, color="#ef4444", marker="arr", dash="5 3")
line(cx + 60, by + bh, draft_cx - 60, ay, color="#ef4444", marker="arr", dash="5 3")
text(cx, by + bh + 24, "__init__ 显式只调昇腾 base 构造", 11, "middle", "#b91c1c")
text(cx, by + bh + 40, "（前向 / 输入准备走 base）", 11, "middle", "#b91c1c")

# AscendDflashProposer（二级继承）
dy = 580
box(eagle_cx - 115, dy, 230, 52, "#fff1f2", "#fb7185", 9, 1.6)
text(eagle_cx, dy + 22, "AscendDflashProposer", 11.5, "middle", "#be123c", "bold", mono=True)
text(eagle_cx, dy + 42, "再叠 DFlash 交叉注意力缓冲", 10.5, "middle", "#9f1239")
line(eagle_cx, ay + 56, eagle_cx, dy, color="#fb7185", marker="arr")
text(eagle_cx + 56, ay + 56 + 24, "二级继承", 10, "middle", "#9f1239")

# 底注
text(W / 2, H - 30, "一个布尔 pass_hidden_states_to_model 区分 eagle(True) / draft(False)；继承链可再延一级（dflash → eagle）", 12.5, "middle", "#475569")

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm-ascend/artifacts/ch29-speculative-decode-npu/diagrams/fig29-2-inherit.svg", "w").write('\n'.join(L))
print("wrote fig29-2-inherit.svg")
