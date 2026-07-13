#!/usr/bin/env python3
"""fig-mla-decode-mqa-576 —— capstone 洞见图:吸收后的 decode 打分,全部 128 个头的
拼接 query [q̃_i(512); q^R_i(64)] 都打向唯一一根共享的 576 维 [c^KV; k^R]——这正是
MQA 的定义,只是 head_dim 从 128 变成 576。

视觉语言呼应 paper-fig-3 的 MQA panel(多 query 扇形收敛到同一份共享 key/value)。
一图一论点:画「许多 query 条指向同一根共享条」这一拳。

色=角色(套本书视觉语言,与 fig-mla-arch-reference 一致):
  amber = 查询内容部 q̃_i(512);blue = 键内容部 c^KV(512,缓存潜向量);
  purple = 位置旋转部(64,解耦 RoPE:query 侧 q^R_i / key 侧 k^R,全头共享)。
数字全部来自 figure-requests.json 的 numbers(溯源论文 §2.1.2/§2.1.3/Table 1/§3.1.2)。
几何全部由常量/循环计算,零手写魔数。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cw(c):
    o = ord(c)
    if o == 0x20:
        return 0.30
    if 0x2E80 <= o <= 0x9FFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F:
        return 1.0
    if c.isascii() and c.isalnum():
        return 0.58
    return 0.5


def tw(s, size):
    return size * sum(cw(c) for c in s)


# ---------------- palette (语义色) ----------------
C_Q_F, C_Q_S, C_Q_T = "#fde68a", "#d97706", "#92400e"   # amber = 查询内容部 q̃_i
C_K_F, C_K_S, C_K_T = "#bfdbfe", "#1d4ed8", "#1e3a8a"   # blue  = 键内容部 c^KV
C_P_F, C_P_S, C_P_T = "#ddd6fe", "#7c3aed", "#5b21b6"   # purple= 位置旋转部(64)
INK, SUB = "#0f172a", "#64748b"
C_FAN = "#94a3b8"

L = []


def rect(x, y, w, h, fill, stroke, sw=1.5, rx=6):
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, s, size, fill, anchor="middle", weight=None, style=None):
    wt = f' font-weight="{weight}"' if weight else ''
    st = f' font-style="{style}"' if style else ''
    L.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="sans-serif" '
             f'font-size="{size}"{wt}{st} fill="{fill}">{esc(s)}</text>')


# ---------------- geometry (零魔数) ----------------
PAD = 40
LEFT_LABEL_W = 52
BAR_W = 340
SEG512 = BAR_W * 512 / 576
SEG64 = BAR_W * 64 / 576
QBAR_H = 42
VGAP = 20
ROW_PITCH = QBAR_H + VGAP

TITLE_Y = 34
SUB_Y = 58
LEGEND_Y = 88
HEADER_Y = 128
TOP = 150            # 第一根 query 条顶

# 5 行:头1/头2/头3/省略号/头128
ROWS = [("头 1", True), ("头 2", True), ("头 3", True), (None, False), ("头 128", True)]
STACK_H = len(ROWS) * ROW_PITCH - VGAP

qbar_x = PAD + LEFT_LABEL_W
qbar_right = qbar_x + BAR_W
FAN_GAP = 196
sh_x = qbar_right + FAN_GAP
SH_W = BAR_W
SH_H = 70
sh_cy = TOP + STACK_H / 2
sh_y = sh_cy - SH_H / 2
sh_right = sh_x + SH_W
sh_cx = sh_x + SH_W / 2

CONTENT_W = sh_right + PAD

# 底部结论条
STRIP_L1 = "decode 形态 = head_dim 576 的 MQA:128 个头的 query 全部打向同一根共享 key"
STRIP_L2 = ("对照  MQA:全头共享 1 份,head_dim 128,每 token 缓存 256 = 2·d_h    "
            "|    MLA-decode:全头共享 1 份,head_dim 576,每 token 缓存 576")
strip_top = TOP + STACK_H + 40
STRIP_H = 74

_strip_right = PAD + 18 + max(tw(STRIP_L1, 13), tw(STRIP_L2, 12.5)) + 18 + PAD
W = max(CONTENT_W, _strip_right)
H = strip_top + STRIP_H + PAD

# ---------------- SVG ----------------
L.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.1f} {H:.1f}">')
L.append('<defs><marker id="fan" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
         f'markerHeight="4.5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_FAN}"/></marker></defs>')
L.append(f'<rect width="{W:.1f}" height="{H:.1f}" fill="white"/>')

# 标题 + 副标题
text(PAD, TITLE_Y, "吸收后的 decode 打分:128 个头共享同一根 576 维 key —— 就是 head_dim 576 的 MQA",
     17, INK, anchor="start", weight="bold")
text(PAD, SUB_Y, "每头把拼接 query [ q̃_i(512) ; q^R_i(64) ] 打向唯一一根共享 [ c^KV ; k^R ],缓存里没有头下标",
     12.5, SUB, anchor="start")

# 图例(3 语义色)
legend = [(C_Q_S, "查询内容部 q̃_i(512)"), (C_K_S, "键内容部 c^KV(512,唯一缓存)"),
          (C_P_S, "位置旋转部(64,解耦 RoPE:q^R / k^R 共享)")]
lx = PAD
for color, lab in legend:
    rect(lx, LEGEND_Y - 11, 14, 14, color, color, sw=0, rx=3)
    text(lx + 20, LEGEND_Y, lab, 11.5, INK, anchor="start")
    lx += 20 + tw(lab, 11.5) + 28

# query 列头
text(qbar_x + BAR_W / 2, HEADER_Y, "每头一份拼接 query = [ q̃_i (512) ; q^R_i (64) ]  →  576 维",
     12.5, INK, weight="bold")

# ---------------- query 条堆叠 + 扇形汇聚 ----------------
def concat_bar(x, y, seg512_fill, seg512_stroke, seg512_txt, seg512_tc,
               seg64_fill, seg64_stroke, seg64_txt, seg64_tc, h, bold=False):
    sw = 2.4 if bold else 1.6
    rect(x, y, SEG512, h, seg512_fill, seg512_stroke, sw=sw)
    rect(x + SEG512, y, SEG64, h, seg64_fill, seg64_stroke, sw=sw)
    text(x + SEG512 / 2, y + h / 2 + 5, seg512_txt, 14 if bold else 13,
         seg512_tc, weight="bold")
    text(x + SEG512 + SEG64 / 2, y + h / 2 + 4, seg64_txt, 10.5, seg64_tc, weight="bold")


for i, (lab, is_bar) in enumerate(ROWS):
    ry = TOP + i * ROW_PITCH
    rcy = ry + QBAR_H / 2
    if not is_bar:
        text(qbar_x + BAR_W / 2, rcy + 5, "⋯   共 128 个头   ⋯", 14, SUB, weight="bold")
        continue
    # 左侧头标签
    text(qbar_x - 10, rcy + 5, lab, 12.5, INK, anchor="end", weight="bold")
    concat_bar(qbar_x, ry, C_Q_F, C_Q_S, "q̃_i", C_Q_T, C_P_F, C_P_S, "q^R", C_P_T, QBAR_H)
    # 扇形虚线:每头 query 右缘 → 共享条左缘中点(带箭头,示「打向同一根」)
    L.append(f'<line x1="{qbar_right:.1f}" y1="{rcy:.1f}" x2="{sh_x - 3:.1f}" y2="{sh_cy:.1f}" '
             f'stroke="{C_FAN}" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#fan)"/>')

# 汇聚点旁小标签
text((qbar_right + sh_x) / 2, sh_cy - 96, "128 根 query", 11.5, SUB)
text((qbar_right + sh_x) / 2, sh_cy - 80, "都打向 ↓", 11.5, SUB)

# ---------------- 唯一共享条 [c^KV; k^R] ----------------
text(sh_cx, sh_y - 14, "全部 128 头共享同一份(无头下标 i)", 13, C_K_T, weight="bold")
concat_bar(sh_x, sh_y, C_K_F, C_K_S, "c^KV", C_K_T, C_P_F, C_P_S, "k^R", C_P_T, SH_H, bold=True)
text(sh_cx, sh_y + SH_H + 24, "[ c^KV ; k^R ] ∈ R^576 = 512 + 64", 13, INK, weight="bold")
text(sh_cx, sh_y + SH_H + 42, "这一根就是缓存里唯一落盘的东西", 11.5, SUB)

# ---------------- 底部结论条 ----------------
rect(PAD, strip_top, W - 2 * PAD, STRIP_H, "#fffbeb", C_Q_S, sw=1.4, rx=10)
text(PAD + 18, strip_top + 28, STRIP_L1, 13, "#78350f", anchor="start", weight="bold")
text(PAD + 18, strip_top + 52, STRIP_L2, 12.5, "#78350f", anchor="start")

L.append('</svg>')
out = Path(__file__).with_name("fig-mla-decode-mqa-576.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W:.0f}x{H:.0f} ratio={W/H:.2f})")
