#!/usr/bin/env python3
"""fig-mla-epiphany: MLA 顿悟头图 —— 纯压缩落差单拳(一图只锚一个洞见)。
那一下:你以为要为每个 token 存一整摞 per-head K+V(2·n_h·d_h=32768),
其实只存一根压缩 latent(d_c+d_h^R=576)——两条同左基线并排,宽度差即压缩比 56.89×。
权重吸收/镜头挪位是另一个洞见,交给它自己的 fig31-3;本图不碰,只留落差。
数字忠实 DeepSeek-V2(n_h=128,d_h=128,d_c=512,d_h^R=64),溯源正文 §一/§三:
32768=2·128·128、576=512+64、56.89=32768÷576。
坐标全由常量/循环计算;无 marker/ellipse,几何门禁零误报。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(str(s))

# ---- palette (色=角色) ----
RED_F, RED_S, RED_D = "#fecaca", "#dc2626", "#991b1b"
REDBG = "#fef2f2"
GRN_F, GRN_S, GRN_D = "#bbf7d0", "#16a34a", "#15803d"
GRNBG = "#f0fdf4"
ORN_S = "#ea580c"
ORN_TINT = "#fff2e8"
SLATE, MUT = "#0f172a", "#64748b"

W, H = 1360, 470

def box(x, y, w, h, fill, stroke, sw=1.5, rx=10, op=None, dash=None):
    a = (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
         f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"')
    if op is not None:
        a += f' fill-opacity="{op}"'
    if dash:
        a += f' stroke-dasharray="{dash}"'
    return a + '/>'

def txt(cx, cy, s, size=13, fill=SLATE, weight="", anchor="middle", op=None):
    wt = f'font-weight="{weight}" ' if weight else ''
    o = f'fill-opacity="{op}" ' if op is not None else ''
    return (f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="{anchor}" font-family="sans-serif" '
            f'font-size="{size}" {wt}{o}fill="{fill}">{esc(s)}</text>')

def line(x1, y1, x2, y2, stroke, sw=1.5, op=None):
    a = f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"'
    if op is not None:
        a += f' stroke-opacity="{op}"'
    return a + '/>'

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>']

# ===== 标题(把「那一下」直接写成 claim,图去证它)=====
L.append(txt(40, 48, "你以为要存一整摞 K 和 V —— 其实只存一根 latent", 27, SLATE, "bold", "start"))
L.append(txt(40, 84, "同一个 token、同一层:MHA 囤一整摞 per-head K + V,MLA 只囤一根压缩 latent"
                     " —— 两条同起点并排,宽度差就是压缩比。", 15, MUT, "", "start"))

# ---- 落差几何:两条同左基线,宽度比精确 = 压缩比 ----
BAR_X = 214
RIGHT = W - 40                      # 1320
RED_W = RIGHT - BAR_X               # 1106,铺满可用宽度
GRN_W = round(RED_W * 576 / 32768)  # ≈19,精确 1/56.89,宽度即压缩比
RED_END = BAR_X + RED_W             # 1320
GRN_END = BAR_X + GRN_W             # 233
BAR_H = 76                          # 红绿等高,只让宽度承载落差

# ================= band A:你以为(MHA)=================
bAy = 104
L.append(box(24, bAy, W - 48, 134, REDBG, "#fecaca", 1, 12))
L.append(txt(44, 166, "你以为", 28, RED_S, "bold", "start"))
L.append(txt(44, 198, "标准 MHA", 15, MUT, "", "start"))
# 胖红条(内部竖线=一摞 per-head 格子),铺满宽度
rby = 134
L.append(box(BAR_X, rby, RED_W, BAR_H, RED_F, RED_S, 2, 6))
for k in range(1, 16):
    xk = BAR_X + RED_W * k / 16
    L.append(line(xk, rby + 4, xk, rby + BAR_H - 4, "#f87171", 1, 0.7))
L.append(txt(BAR_X + RED_W / 2, rby + BAR_H / 2 + 8, "2 · n_h · d_h  =  32768", 24, RED_D, "bold"))
L.append(txt(BAR_X, 230, "每个 token:128 个头,每头存一份 K + 一份 V —— 整整一摞,全部写进 KV 缓存",
             14.5, RED_D, "", "start"))

# ================= band B:其实(MLA)+ 落差主角 =================
bBy = 254
L.append(box(24, bBy, W - 48, 186, GRNBG, "#bbf7d0", 1, 12))
L.append(txt(44, 322, "其实", 28, GRN_S, "bold", "start"))
L.append(txt(44, 354, "MLA 实际缓存", 15, MUT, "", "start"))
# 瘦绿条(细到 1/57),与红条同左基线、同高
gby = 300
L.append(box(BAR_X, gby, GRN_W, BAR_H, GRN_F, GRN_S, 2.5, 3))
L.append(txt(GRN_END + 14, gby + BAR_H / 2 + 6, "d_c + d_h^R  =  512 + 64  =  576",
             20, GRN_D, "bold", "start"))
# 落差主角:占住红条霸占过、绿丝却空着的那一整段留白(软橙底托起单拳)
LBL_END = GRN_END + 300              # 「576」标签右缘,主角落在其右的真空段
badge_cx = (LBL_END + RED_END) / 2   # ≈916,居中于绿丝标签之后、红条之下的空白
L.append(box(badge_cx - 182, gby + 4, 364, BAR_H - 8, ORN_TINT, "#fed7aa", 1, 16))
L.append(txt(badge_cx, gby + BAR_H / 2 - 2, "≈ 57× 更小", 42, ORN_S, "bold"))
L.append(txt(badge_cx, gby + BAR_H / 2 + 28, "56.89  =  32768 ÷ 576", 16.5, MUT))
L.append(txt(BAR_X, 424, "只存一根 latent(压缩摘要)+ 一小撮位置维", 14.5, GRN_D, "", "start"))

L.append('</svg>')
out = Path(__file__).with_name("fig-mla-epiphany.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
