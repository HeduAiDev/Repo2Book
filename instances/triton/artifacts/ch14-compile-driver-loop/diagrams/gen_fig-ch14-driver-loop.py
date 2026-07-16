#!/usr/bin/env python3
"""flow 模板(定制,仿 ch11 launch-spine 主干+分支写法):compile() 未命中缓存时的完整驱动链，
命中则右侧绿色捷径跳过全部 pass。主干 ①拼缓存键→diamond→②选后端→③填 stages→④造起点 module
→⑤逐级 compile_ir(展开 5 级小丸子链，落盘)→⑥写回 metadata+入缓存→END。
改造点：MAIN(主干节点)、STAGE_PILLS(⑤下方 5 级小链)、HIT(命中捷径文案)。全坐标计算，零魔数。"""
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

MAIN = [
    ("①", ["拼缓存键(5 段拼接)", "查磁盘缓存"],
     "key=triton_key()-src.hash()-backend.hash()-options.hash()-env_vars", "#dbeafe", "#1d4ed8"),
]
DIAMOND_LABEL = "磁盘命中?"
AFTER = [
    ("②", ["make_backend(target)", "选唯一后端"], "target=cuda -> CUDABackend(唯一,恰好 1 个)", "#dbeafe", "#1d4ed8"),
    ("③", ["backend.add_stages(stages={})", "填有序 stages 字典"], "插入 5 级:ttir->ttgir->llir->ptx->cubin", "#dbeafe", "#1d4ed8"),
    ("④", ["src.make_ir 造起点 module"], "first_stage=0(ASTSource 从 ttir 起步)", "#dbeafe", "#1d4ed8"),
    ("⑤", ["逐级 compile_ir 降级 + 落盘", "(展开见下方 5 级小链)"], "for ext, compile_ir in stages[first_stage:]", "#e0e7ff", "#4338ca"),
    ("⑥", ["metadata 写回 + put_group"], "编译产物写入磁盘缓存目录", "#dbeafe", "#1d4ed8"),
]
STAGE_PILLS = [
    ("ttir", "make_ttir", "-> name.ttir"),
    ("ttgir", "make_ttgir", "-> name.ttgir"),
    ("llir", "make_llir", "-> name.llir"),
    ("ptx", "make_ptx", "-> name.ptx"),
    ("cubin", "make_cubin", "-> name.cubin(bytes)"),
]
HIT_LABEL = "命中"
MISS_LABEL = "未命中"
HIT_BOX = ("跳过全部 pass", "直接从磁盘反序列化 CompiledKernel")

BOX_W, BOX_H, VGAP = 460, 60, 34
DIA = 96
PAD_L, TOP = 70, 74
PILL_W, PILL_H, PILL_GAP = 128, 58, 14
n_pills = len(STAGE_PILLS)
pills_w = n_pills * PILL_W + (n_pills - 1) * PILL_GAP
LANE_CX = pills_w / 2 + PAD_L  # 主干列 x：留够左边距容纳 5 级小丸子链，不被裁切

HIT_CX = LANE_CX + BOX_W / 2 + 250
HIT_W = 300

n_after = len(AFTER)
w = max(LANE_CX + BOX_W / 2 + 40, HIT_CX + HIT_W / 2 + 60,
        LANE_CX + pills_w / 2 + 60)
h = (TOP + (len(MAIN[0][1]) and BOX_H) + VGAP + DIA + VGAP
     + n_after * (BOX_H + VGAP) + PILL_H + VGAP + 20 + 52 + 100)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
          '<marker id="o" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
          '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD_L}" y="38" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("compile() 驱动主循环：未命中沿 stages 逐级降级并落盘，命中直接跳过全部 pass")}</text>')

y = TOP
main_centers = []
badge, title_lines, detail, fill, stroke = MAIN[0]
cx = LANE_CX
main_centers.append(y)
L.append(f'<rect x="{cx - BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
          f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
L.append(f'<circle cx="{cx - BOX_W/2 + 22}" cy="{y + 20}" r="15" fill="{stroke}"/>')
L.append(f'<text x="{cx - BOX_W/2 + 22}" y="{y + 25}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="14" font-weight="bold" fill="white">{esc(badge)}</text>')
L += multiline(title_lines, cx + 14, y + 20, size=13, weight=True)
L.append(f'<text x="{cx}" y="{y + BOX_H - 10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#334155">{esc(detail)}</text>')
y += BOX_H + VGAP

# 菱形判定
dia_cx, dia_cy = LANE_CX, y + DIA / 2
L.append(f'<line x1="{dia_cx}" y1="{y - VGAP + BOX_H}" x2="{dia_cx}" y2="{y}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
diamond_pts = f"{dia_cx},{y} {dia_cx + DIA/2},{dia_cy} {dia_cx},{y + DIA} {dia_cx - DIA/2},{dia_cy}"
L.append(f'<polygon points="{diamond_pts}" fill="#e2e8f0" stroke="#475569" stroke-width="1.6"/>')
L.append(f'<text x="{dia_cx}" y="{dia_cy + 4}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc(DIAMOND_LABEL)}</text>')

# 命中(绿) -> 右侧捷径框
hit_cy = dia_cy
L.append(f'<line x1="{dia_cx + DIA/2}" y1="{dia_cy}" x2="{HIT_CX - HIT_W/2}" y2="{hit_cy}" '
          'stroke="#15803d" stroke-width="2" marker-end="url(#g)"/>')
L.append(f'<text x="{(dia_cx + DIA/2 + HIT_CX - HIT_W/2)/2}" y="{hit_cy - 10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#15803d">{esc(HIT_LABEL)}</text>')
hit_box_h = 58
L.append(f'<rect x="{HIT_CX - HIT_W/2}" y="{hit_cy - hit_box_h/2}" width="{HIT_W}" height="{hit_box_h}" rx="10" '
          'fill="#dcfce7" stroke="#15803d" stroke-width="1.8"/>')
L += multiline([HIT_BOX[0]], HIT_CX, hit_cy - 6, size=13, weight=True, fill="#15803d")
L += multiline([HIT_BOX[1]], HIT_CX, hit_cy + 14, size=10.5, fill="#166534")

# 未命中(橙) 主线继续向下
after_top = y + DIA + VGAP
L.append(f'<line x1="{dia_cx}" y1="{y + DIA}" x2="{dia_cx}" y2="{after_top}" '
          'stroke="#b45309" stroke-width="2" marker-end="url(#o)"/>')
L.append(f'<text x="{dia_cx + 12}" y="{y + DIA + (VGAP-4)/2 + 4}" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#b45309">{esc(MISS_LABEL)}</text>')

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
    L += multiline(title_lines, cx + 14, y2 + (20 if len(title_lines) == 1 else 16), size=13, weight=True)
    L.append(f'<text x="{cx}" y="{y2 + BOX_H - 9}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#334155">{esc(detail)}</text>')
    if i < len(AFTER) - 1:
        gap = VGAP if i != 3 else VGAP + PILL_H + 14  # ⑤ 下方要给 pill 链留位置
        L.append(f'<line x1="{cx}" y1="{y2 + BOX_H}" x2="{cx}" y2="{y2 + BOX_H + gap - 4}" '
                  'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
    if i == 3:
        # ⑤ 正下方画 5 级小丸子链
        pills_y = y2 + BOX_H + 14
        px0 = cx - pills_w / 2
        prev_edge = None
        for k, (ext, fn, out) in enumerate(STAGE_PILLS):
            px = px0 + k * (PILL_W + PILL_GAP)
            is_last = (k == n_pills - 1)
            pfill, pstroke = ("#fef3c7", "#b45309") if is_last else ("#eef2ff", "#4338ca")
            L.append(f'<rect x="{px}" y="{pills_y}" width="{PILL_W}" height="{PILL_H}" rx="10" '
                      f'fill="{pfill}" stroke="{pstroke}" stroke-width="1.6"/>')
            L.append(f'<text x="{px+PILL_W/2}" y="{pills_y+20}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" font-weight="bold" '
                      f'fill="{pstroke}">{esc(ext)}</text>')
            L.append(f'<text x="{px+PILL_W/2}" y="{pills_y+35}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="10" fill="#334155">{esc(fn)}</text>')
            L.append(f'<text x="{px+PILL_W/2}" y="{pills_y+49}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="9.5" fill="#64748b">{esc(out)}</text>')
            if k > 0:
                L.append(f'<line x1="{prev_edge}" y1="{pills_y+PILL_H/2}" x2="{px}" y2="{pills_y+PILL_H/2}" '
                          'stroke="#94a3b8" stroke-width="1.4" marker-end="url(#a)"/>')
            prev_edge = px + PILL_W
        y2 += BOX_H + 14 + PILL_H + VGAP
    else:
        y2 += BOX_H + VGAP

# 汇合到 END —— END 框与主干同 x 居中；命中捷径经折线拐入 END 顶边右侧，两端都贴元素边缘
end_y = after_centers[-1] + BOX_H + VGAP + 20
end_cx = LANE_CX
end_w = 460
end_box_h = 52
L.append(f'<line x1="{LANE_CX}" y1="{after_centers[-1] + BOX_H}" x2="{LANE_CX}" y2="{end_y - 4}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
elbow_x = end_cx + end_w / 2 - 40
elbow_y = end_y - 36
L.append(f'<path d="M {HIT_CX},{hit_cy + hit_box_h/2} L {HIT_CX},{elbow_y} '
          f'L {elbow_x},{elbow_y} L {elbow_x},{end_y - 4}" '
          'fill="none" stroke="#15803d" stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#g)"/>')
L.append(f'<rect x="{end_cx - end_w/2}" y="{end_y}" width="{end_w}" height="{end_box_h}" rx="12" '
          'fill="#0f172a" stroke="#0f172a"/>')
L.append(f'<text x="{end_cx}" y="{end_y + 22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="white">{esc("返回 CompiledKernel")}</text>')
L.append(f'<text x="{end_cx}" y="{end_y + 40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#cbd5e1">{esc("两条路径唯一交汇点")}</text>')

foot_y = end_y + end_box_h + 34
L.append(f'<text x="{PAD_L}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("5 级降级链固定(ttir/ttgir/llir/ptx/cubin)；first_stage=0(ASTSource 从 ttir 起步)，全 5 级皆跑")}</text>')
L.append(f'<text x="{PAD_L}" y="{foot_y + 20}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("蓝=主线固定工序；紫=展开态(逐级 compile_ir)；橙=末级 cubin(唯一返 bytes)；绿=命中捷径(跳过全部 pass)")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch14-driver-loop.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
