#!/usr/bin/env python3
"""flow 模板(定制):JITFunction.run 一次 launch 的 6 段脊柱 + 命中/未命中分岔。
主干竖排 ①②③→◇判定→(命中)直达⑤⑥ / (未命中)岔入④compile→回填→回到 cache。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def multiline(lines, cx, y0, size=12, weight=False, fill="#0f172a", lh=15):
    out = []
    wattr = 'font-weight="bold" ' if weight else ''
    for k, line in enumerate(lines):
        out.append(f'<text x="{cx}" y="{y0 + k * lh}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="{size}" {wattr}'
                    f'fill="{fill}">{esc(line)}</text>')
    return out

# 主干节点(竖排): 1 2 3 -> diamond -> 5 6 ; 岔支 4 挂在 diamond 右侧
MAIN = [
    ("①", ["driver.active 取 device/stream/target", "+ make_backend(target)"],
     "device=0, target=cuda sm=80 warp=32", "#dbeafe", "#1d4ed8"),
    ("②", ["惰性 binder 得 5 元组", "(create_binder / 直接调用)"],
     "sig_and_spec=[*fp32,*fp32,*fp32,i32,D,D,D,D]; constexpr_vals=(256,)", "#dbeafe", "#1d4ed8"),
    ("③", ["拼 key 查 cache[device]"],
     "key=…((256,), {'debug': False})", "#dbeafe", "#1d4ed8"),
]
DIAMOND_LABEL = "命中?"
AFTER = [
    ("⑤", ["规范化 grid"], "(4,) -> (4, 1, 1)", "#dbeafe", "#1d4ed8"),
    ("⑥", ["kernel.run 跨语言发射"], "需真设备(⚡)", "#fee2e2", "#b91c1c"),
]
BRANCH = ("④", ["未命中 -> compile 支路", "parse_options -> ASTSource -> compile"],
          "回填后此键永久命中", "#fef3c7", "#b45309")

BOX_W, BOX_H, VGAP = 460, 66, 40
DIA = 108
LANE_CX = 300
PAD_L, TOP = 70, 74
BRANCH_CX = LANE_CX + BOX_W / 2 + 260
BRANCH_W = 380

n_main = len(MAIN)
w = PAD_L + BOX_W + 40 + (BRANCH_W - BOX_W / 2 + 40 if True else 0) + 260
w = 1180
h = TOP + n_main * (BOX_H + VGAP) + DIA + VGAP + len(AFTER) * (BOX_H + VGAP) + 120

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
          '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
          '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD_L}" y="40" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">{esc("JITFunction.run 一次 launch 的脊柱：6 段固定工序，只有 ④ 在未命中时展开")}</text>')

# 需真设备 泳道标注(覆盖①区域)
L.append(f'<rect x="{PAD_L - 30}" y="{TOP - 8}" width="18" height="{BOX_H + 16}" rx="4" '
          'fill="#fee2e2" stroke="#b91c1c"/>')
L.append(f'<text x="{PAD_L - 21}" y="{TOP + BOX_H / 2 + 4}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11" fill="#b91c1c" '
          f'transform="rotate(-90 {PAD_L - 21} {TOP + BOX_H / 2 + 4})">{esc("需真设备")}</text>')

y = TOP
main_centers = []
for i, (badge, title_lines, detail, fill, stroke) in enumerate(MAIN):
    cx = LANE_CX
    main_centers.append(y)
    L.append(f'<rect x="{cx - BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<circle cx="{cx - BOX_W/2 + 22}" cy="{y + 20}" r="15" fill="{stroke}"/>')
    L.append(f'<text x="{cx - BOX_W/2 + 22}" y="{y + 25}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" fill="white">{esc(badge)}</text>')
    L += multiline(title_lines, cx + 14, y + 20, size=13, weight=True)
    L.append(f'<text x="{cx}" y="{y + BOX_H - 10}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc(detail)}</text>')
    if i < n_main - 1:
        L.append(f'<line x1="{cx}" y1="{y + BOX_H}" x2="{cx}" y2="{y + BOX_H + VGAP - 4}" '
                  'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
    y += BOX_H + VGAP

# 菱形判定
dia_cx, dia_cy = LANE_CX, y + DIA / 2
L.append(f'<line x1="{dia_cx}" y1="{y - VGAP + BOX_H}" x2="{dia_cx}" y2="{y}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
diamond_pts = f"{dia_cx},{y} {dia_cx + DIA/2},{dia_cy} {dia_cx},{y + DIA} {dia_cx - DIA/2},{dia_cy}"
L.append(f'<polygon points="{diamond_pts}" fill="#e2e8f0" stroke="#475569" stroke-width="1.6"/>')
L.append(f'<text x="{dia_cx}" y="{dia_cy + 4}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc(DIAMOND_LABEL)}</text>')

after_top = y + DIA + VGAP
# "命中" 直下 到 ⑤
L.append(f'<line x1="{dia_cx}" y1="{y + DIA}" x2="{dia_cx}" y2="{after_top}" '
          'stroke="#15803d" stroke-width="2" marker-end="url(#g)"/>')
L.append(f'<text x="{dia_cx + 12}" y="{y + DIA + (VGAP-4)/2 + 4}" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#15803d">{esc("命中 (dict.get 直达)")}</text>')

y2 = after_top
after_centers = []
for i, (badge, title_lines, detail, fill, stroke) in enumerate(AFTER):
    cx = LANE_CX
    after_centers.append(y2)
    L.append(f'<rect x="{cx - BOX_W/2}" y="{y2}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<circle cx="{cx - BOX_W/2 + 22}" cy="{y2 + 20}" r="15" fill="{stroke}"/>')
    L.append(f'<text x="{cx - BOX_W/2 + 22}" y="{y2 + 25}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" fill="white">{esc(badge)}</text>')
    L += multiline(title_lines, cx + 14, y2 + 26, size=13, weight=True)
    L.append(f'<text x="{cx}" y="{y2 + BOX_H - 10}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc(detail)}</text>')
    if i < len(AFTER) - 1:
        L.append(f'<line x1="{cx}" y1="{y2 + BOX_H}" x2="{cx}" y2="{y2 + BOX_H + VGAP - 4}" '
                  'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
    y2 += BOX_H + VGAP

# 需真设备 泳道 覆盖 ⑥
box6_y = after_centers[-1]
L.append(f'<rect x="{PAD_L - 30}" y="{box6_y - 8}" width="18" height="{BOX_H + 16}" rx="4" '
          'fill="#fee2e2" stroke="#b91c1c"/>')
L.append(f'<text x="{PAD_L - 21}" y="{box6_y + BOX_H / 2 + 4}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11" fill="#b91c1c" '
          f'transform="rotate(-90 {PAD_L - 21} {box6_y + BOX_H / 2 + 4})">{esc("需真设备")}</text>')

# 未命中 分支 -> ④
branch_cx = LANE_CX + BOX_W / 2 + 220
branch_y = dia_cy - BOX_H / 2
L.append(f'<line x1="{dia_cx + DIA/2}" y1="{dia_cy}" x2="{branch_cx - BOX_W/2}" y2="{branch_y + BOX_H/2}" '
          'stroke="#b45309" stroke-width="2" marker-end="url(#b)"/>')
L.append(f'<text x="{(dia_cx + DIA/2 + branch_cx - BOX_W/2)/2}" y="{dia_cy - 12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#b45309">{esc("未命中")}</text>')

L.append(f'<rect x="{branch_cx - BOX_W/2}" y="{branch_y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
          f'fill="{BRANCH[3]}" stroke="{BRANCH[4]}" stroke-width="1.8"/>')
L.append(f'<circle cx="{branch_cx - BOX_W/2 + 22}" cy="{branch_y + 20}" r="15" fill="{BRANCH[4]}"/>')
L.append(f'<text x="{branch_cx - BOX_W/2 + 22}" y="{branch_y + 25}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="14" font-weight="bold" fill="white">{esc(BRANCH[0])}</text>')
L += multiline(BRANCH[1], branch_cx + 14, branch_y + 20, size=13, weight=True)
L.append(f'<text x="{branch_cx}" y="{branch_y + BOX_H - 10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#334155">{esc(BRANCH[2])}</text>')

# ④ 回填 -> 回到 ③ 的 cache(回箭到 dia 顶部略上方,示"此键从此命中")
loop_x = branch_cx + BOX_W / 2 + 40
L.append(f'<path d="M {branch_cx + BOX_W/2},{branch_y + BOX_H/2} '
          f'L {loop_x},{branch_y + BOX_H/2} L {loop_x},{main_centers[-1] + BOX_H/2} '
          f'L {LANE_CX + BOX_W/2},{main_centers[-1] + BOX_H/2}" '
          'fill="none" stroke="#b45309" stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#b)"/>')
L.append(f'<text x="{loop_x + 8}" y="{(branch_y + BOX_H/2 + main_centers[-1] + BOX_H/2)/2}" '
          f'font-family="sans-serif" font-size="11" fill="#b45309">{esc("回填 cache[key]")}</text>')

foot_y = h - 46
L.append(f'<text x="{PAD_L}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("cache 条目 miss 后 = 1；第二次同参 launch：k2 is k1 = True（同一 CompiledKernel，无编译）")}</text>')
L.append(f'<text x="{PAD_L}" y="{foot_y + 20}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("红边框 = 需真设备（host 无 GPU 在此断裂）；橙色支路 = 内存 cache 未命中触发的一次性慢路径")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch11-launch-spine.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
