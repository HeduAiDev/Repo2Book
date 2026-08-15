#!/usr/bin/env python3
"""L1 Part 图 ×8 —— L0 的 viewBox 裁切放大（v3 图系 Phase 2 / FIGURE-SYSTEM.md §1）

机制（同源强制，绝不重画布局）：
  - 元素流 = l0_common.build_l0() 原样保留（同一坐标系、同配色、同术语、同图例）；
  - 新 viewBox = Part 区域 + 边距：放大由 SVG 语义自动发生（物理保证的放大镜关系）；
  - 区域外元素包 <g opacity="0.15"> 淡出——逐元素 bbox 与「区域⊕F 余量」求交分区，
    与区域沾边的元素（含跨区接口箭头）保持全不透明；
  - 顶部标题带在 2200 宽画布坐标系设计（Part N — 标题 22px / hook 15px / 章目录 12px），
    按 1/k 缩放落回 L0 坐标——8 张 L1 的标题带视觉规格完全一致。

标题带落位由确定性搜索决定（对 gen_L1 的真正考验——L0 布局是高密度固定坐标）：
  a. 带底边（裁切上缘 y0）不得切在任何 L0 文字身上（半截字是最破相的缺陷）；
  b. 带内三行文字落回 L0 坐标后，不与「条带内」被遮住的 L0 文字 bbox 相撞
     （linter 按文件坐标检查文字相撞，不知道谁被标题带遮住）；
  c. 复刻 linter 的 text-rect 侵入规则。
  bbox 口径：y 用 linter 同款公式（y−0.78fs / y+0.20fs，±1px 余量），x 用保守宽度。
  评分偏好：大 vy（上方淡出视野）> 大 vx（两侧淡出视野）> 行位接近标准 (46,80,110)。

输出：L1-part{N}.svg + L1-part{N}.png（node sharp density=144 → 2x，同 L0 约定）。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import l0_common as lc

HERE = Path(__file__).parent
PLAN = json.loads((HERE / 'pedagogy-plan.json').read_text(encoding='utf-8'))

F = 12.0            # 淡出判定余量：区域⊕F 后仍不相交的元素才淡出（接口箭头不灭）
CANVAS_W = 2200     # 标题带设计坐标系宽（= L0 画布宽）
STD_LINES = (46, 80, 110)
VY_GRID = [170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 12, 6, 0]
# 左右淡出视野偏好（不对称：贴画布边的方向留小边距——那一侧只有空白可显；
# 朝向邻列/接口的方向留大边距——淡出上下文有价值）。落不下时按 30 一档收缩。
VX_PREF = {'I': (0, 0), 'II': (60, 60), 'III': (120, 170), 'IV': (120, 170),
           'V': (170, 170), 'VI': (170, 60), 'VII': (170, 60), 'VIII': (70, 70)}
VX_LADDER = [0, 30, 60, 90, 120, 150]
PART_COLOR = {'I': lc.C_TXT, 'II': lc.C_API_S, 'III': lc.C_ENG_S, 'IV': lc.C_KV_S,
              'V': lc.C_GPU_S, 'VI': lc.C_GPU_S, 'VII': lc.C_SAM_S, 'VIII': lc.C_TXT}
VIII_NOTE = '生产视角：多实例 / 池化 / 弹性 —— 详 L0×N 衍生图（后续章内）'
SHARP = "E:/Laboratory/Repo2Book/node_modules/sharp"

RECT_RE = re.compile(r'<rect x="(-?[\d.]+)" y="(-?[\d.]+)" width="([\d.]+)" height="([\d.]+)"')
TEXT_RE = re.compile(r'<text x="(-?[\d.]+)" y="(-?[\d.]+)" font-family="[^"]+" '
                     r'font-size="([\d.]+)" fill="[^"]+" text-anchor="(\w+)"[^>]*>(.*)</text>')


def ov(a, b):
    """bbox 严格相交（贴边不算）。"""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def expand(r, m):
    return (r[0] - m, r[1] - m, r[2] + m, r[3] + m)


def l0_text_boxes(elems):
    """L0 全部文字的 bbox（y 用 linter 同款公式，x 用保守宽度估计）。"""
    out = []
    for _bb, s in elems:
        m = TEXT_RE.match(s)
        if not m:
            continue
        x, y, fs, anchor, body = float(m[1]), float(m[2]), float(m[3]), m[4], m[5]
        w = lc.tw(body, fs)
        x0 = x - w / 2 if anchor == 'middle' else (x - w if anchor == 'end' else x)
        out.append((x0, y - 0.78 * fs - 1.0, x0 + w, y + 0.20 * fs + 1.0))
    return out


def regions_for(g, pid):
    """Part → L0 区域（语义映射；坐标全部取自 GEO，L0 改版自动联动）。"""
    AX, CW = g['AX'], g['COL_W']
    if pid in ('I', 'VIII'):                       # 全图（VIII 缩小 + 注记，见 build_part）
        return [(0, 0, lc.W, g['H'])]
    if pid == 'II':                                # API 进程框 + ZMQ 边界带
        return [(g['MX'], g['AY'], g['BXR'], g['ZY'] + g['ZH'])]
    if pid == 'III':                               # 五拍循环框（含 io 链标注）+ 调度账本列上半
        return [(g['LOOP_X'], g['LOOP_Y'], g['LOOP_X'] + g['LOOP_W'], g['LOOP_Y'] + g['LOOP_H'] + 28),
                (AX, g['CY0'], AX + CW, g['A2Y'] + g['a2h'])]
    if pid == 'IV':                                # KV 账本列：Scheduler↔KVManager↔BlockPool
        return [(AX, g['CY0'], AX + CW, g['A3Y'] + g['a3h'])]
    if pid == 'V':                                 # GPU 执行臂大框
        return [(g['BX'], g['CY0'], g['BX'] + CW, g['B3Y'] + g['b3h'])]
    if pid == 'VI':                                # 执行臂内的模型层框 + 采样列 compute_logits
        return [(g['BX'], g['B3Y'], g['BX'] + CW, g['B3Y'] + g['b3h']),
                (g['CX'], g['CY0'], g['CX'] + CW, g['C1Y'] + g['c1h'])]
    return [(g['CX'], g['CY0'], g['CX'] + CW, g['C4Y'] + g['c4h'])]   # VII 采样与出口列


def line_specs(pid, part, chapters_s, note):
    """标题带各行（画布坐标系规格）。"""
    specs = [
        ('title', 1100, f"Part {pid} — {part['title']}", 22, lc.C_TXT, True),
        ('hook', 1100, part['hook'], 15, PART_COLOR[pid], False),
        ('chapters', 1100, chapters_s, 12, lc.C_MUTE, False),
    ]
    if note:
        specs.append(('note', 1100, note, 12.5, lc.C_MUTE, False))
    return specs


def line_bbox(spec, v, k, yb0, x0):
    """一行标题带文字的 L0 坐标 bbox（y 用 linter 同款公式 +1px 余量，x 保守宽）。"""
    _, u, s, fs, _fill, bold = spec
    w_c = lc.tw(s, fs, bold)
    fs_l = fs / k
    x, y = x0 + u / k, yb0 + v / k
    x0b = x - w_c / (2 * k)
    return (x0b, y - 0.78 * fs_l - 1.0, x0b + w_c / k, y + 0.20 * fs_l + 1.0)


def text_rect_flag(b, r):
    """复刻 lint_diagram_geometry 的 text-rect 侵入规则（r=(x,y,w,h) 原始框）。"""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    if r[0] <= cx <= r[0] + r[2] and r[1] <= cy <= r[1] + r[3]:
        return False
    ox = min(b[2], r[0] + r[2]) - max(b[0], r[0])
    oy = min(b[3], r[1] + r[3]) - max(b[1], r[1])
    return ox > 0.30 * (b[2] - b[0]) and oy > 0.4 * (b[3] - b[1])


def place_lines(specs, badge_s, text_boxes, rects_xywh, elems_bbs, rx0, ry0, rx1, cw0, std, pref):
    """确定性搜索 (vx_l, vx_r, vy, 行基线)。返回 (vx_l, vx_r, vy, lines, n_violations)。"""
    ladder = []
    for cut in VX_LADDER:
        l, r = max(0, pref[0] - cut), max(0, pref[1] - cut)
        if (l, r) not in [(a, b) for a, b, *_ in ladder]:
            ladder.append((l, r))
    best = None          # (score, vx_l, vx_r, vy, lines)
    fallback = None      # (viol, vx_l, vx_r, vy, lines)
    for vxl, vxr in ladder:
        if best is not None and 0.25 * (vxl + vxr) + VY_GRID[0] <= best[0]:
            break
        x0 = rx0 - vxl
        cw = cw0 + vxl + vxr
        k = CANVAS_W / cw
        bandh = 130 / k
        for vy in VY_GRID:
            if best is not None and vy + 0.25 * (vxl + vxr) <= best[0]:
                break
            y0 = ry0 - vy
            yb0 = y0 - bandh
            # a. 带底边（裁切上缘）不切字（可见缺陷）；带顶边只禁「插入过深」
            #    ——骑在带顶边、插进带内 ≤0.35 倍字高的文字被不透明带遮住且
            #    linter 的 text-rect 侵入规则（0.4 系数）不报，放行。
            if any(b[1] < y0 < b[3] for b in text_boxes):
                continue
            if any(b[1] < yb0 < b[3] and (yb0 - b[1]) > 0.35 * (b[3] - b[1]) for b in text_boxes):
                continue
            # 条带内的文字 / 大框
            strip = (x0, yb0, x0 + cw, y0)
            stexts = [b for b in text_boxes if ov(b, strip)]
            srects = [r for r, bb in zip(rects_xywh, elems_bbs)
                      if r[2] >= 55 and r[3] >= 26 and ov(bb, strip)]
            for l1 in range(22, 104, 2):
                for l2 in range(l1 + 18, min(l1 + 42, 112), 2):
                    for l3 in range(l2 + 14, 121, 2):
                        lines = (l1, l2, l3)
                        bbs = [line_bbox(sp, v, k, yb0, x0) for sp, v in zip(specs, lines)]
                        bw = lc.tw(badge_s, 12) / k
                        bbs.append((x0 + 2166 / k - bw, yb0 + l1 / k - 0.78 * 12 / k - 1.0,
                                    x0 + 2166 / k, yb0 + l1 / k + 0.20 * 12 / k + 1.0))
                        viol = 0
                        clean = True
                        for b in bbs:
                            if any(ov(b, t) for t in stexts):
                                viol += 1
                                clean = False
                                break
                            if any(ov(b, b2) for b2 in bbs if b2 is not b):
                                viol += 1
                                clean = False
                                break
                        if clean:
                            for b in bbs:
                                if any(text_rect_flag(b, r) for r in srects):
                                    viol += 1
                                    clean = False
                                    break
                        if fallback is None or viol < fallback[0]:
                            fallback = (viol, vxl, vxr, vy, lines)
                        if clean:
                            dev = sum(abs(lines[i] - std[i]) for i in range(3))
                            score = vy + 0.25 * (vxl + vxr) - 3.0 * dev
                            if best is None or score > best[0]:
                                best = (score, vxl, vxr, vy, lines)
    if best:
        return best[1], best[2], best[3], best[4], 0
    return fallback[1], fallback[2], fallback[3], fallback[4], fallback[0]


def build_part(pid, part, elems, g, chapters, text_boxes, rects_xywh, elems_bbs):
    regions = regions_for(g, pid)
    rx0 = min(r[0] for r in regions)
    ry0 = min(r[1] for r in regions)
    rx1 = max(r[2] for r in regions)
    ry1 = max(r[3] for r in regions)
    cw0 = rx1 - rx0
    fade = pid not in ('I', 'VIII')
    band = 190 if pid == 'VIII' else 130
    chapters_s = ' · '.join(f"ch{c['no']} {c['title']}" for c in chapters[pid])
    note = VIII_NOTE if pid == 'VIII' else ''

    specs = line_specs(pid, part, chapters_s, note)

    if pid == 'I':
        vxl, vxr, vy, lines = 0, 0, 0, STD_LINES
    elif pid == 'VIII':
        vxl = vxr = vy = 70
        lines = (50, 86, 118, 158)
    else:
        badge_s = f'L1 = L0 ×{CANVAS_W / cw0:.2f}'
        vxl, vxr, vy, lines, viol = place_lines(specs, badge_s, text_boxes, rects_xywh, elems_bbs,
                                                rx0, ry0, rx1, cw0, STD_LINES, VX_PREF[pid])
        if viol:
            print(f'  !! Part {pid}: 无零违例落位，取最小违例 {viol} 处 '
                  f'(vx=({vxl},{vxr}), vy={vy}, lines={lines})')

    x0 = rx0 - vxl
    cw = cw0 + vxl + vxr
    y0 = ry0 - vy
    ch = (ry1 - ry0) + 2 * vy
    k = CANVAS_W / cw
    bandh = band / k
    yb0 = y0 - bandh
    hc = round(band + CANVAS_W * ch / cw)

    to_x = lambda u: x0 + u / k
    to_y = lambda v: yb0 + v / k

    # ---- 淡出分区：区域⊕F 仍不相交的元素进 0.15 组 ----
    if fade:
        frs = [expand(r, F) for r in regions]
        outside = [s for bb, s in elems if not any(ov(bb, fr) for fr in frs)]
        inside = [s for bb, s in elems if any(ov(bb, fr) for fr in frs)]
    else:
        outside, inside = [], [s for _, s in elems]

    # ---- 标题带（画布坐标系设计，1/k 落回 L0 坐标；最后画=压在最上层） ----
    color = PART_COLOR[pid]
    B = []

    def R(u, v, w, h, fill):
        B.append(lc.rect_svg(to_x(u), to_y(v), w / k, h / k, fill, 'none', 0, 0, False))

    def T(u, v, s, fs, fill, anchor='middle', bold=False, maxw=None, tag=''):
        if maxw:
            lc.fit(s, fs, maxw, tag, bold)
        B.append(lc.text_svg(to_x(u), to_y(v), s, fs / k, fill, anchor, bold))

    R(0, 0, CANVAS_W, band, '#ffffff')
    R(0, 0, 12, band, color)                    # 左侧色条
    R(0, band - 3, CANVAS_W, 3, color)          # 底部色带
    for sp, v in zip(specs, lines):
        _, u, s, fs, fill, bold = sp
        maxw = 2040 if sp[0] in ('chapters', 'note') else 1900
        T(u, v, s, fs, fill, bold=bold, maxw=maxw, tag=f'L1-{pid}:{sp[0]}')
    T(2166, lines[0], f'L1 = L0 ×{k:.2f}', 12, lc.C_FAINT, anchor='end')

    # ---- 装配 ----
    # 白底 = L0 整张画布 ∪ viewBox 的外包矩形（不是只盖 viewBox）：视口裁掉超出部分、
    # 视觉不变；但 linter 的「端点在白底内=已连接 / 文字中心在白底内=非侵入」宽容分支
    # 与 L0 完全同语义——裁切层不该因为换了取景框就改变这些判定的真值。
    gx0, gy0 = min(0.0, x0), min(0.0, yb0)
    gx1, gy1 = max(float(lc.W), x0 + cw), max(float(g['H']), y0 + ch)
    # 显式裁剪：sharp/librsvg 对「视口外/骑跨视口边界的元素」存在不裁剪且错位渲染的
    # bug（opacity 组与逐元素 opacity 均复现）——用 clipPath 把 L0 内容钉死在画布内，
    # 不赌渲染器的 viewport 裁剪。标题带在裁剪组之外（自带边界）。
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{hc}" '
           f'viewBox="{x0:.1f} {yb0:.1f} {cw:.1f} {ch + bandh:.1f}" data-zoom="L1">',
           f'<rect x="{gx0:.1f}" y="{gy0:.1f}" width="{gx1 - gx0:.1f}" height="{gy1 - gy0:.1f}" fill="white"/>',
           lc.DEFS,
           f'<clipPath id="vc"><rect x="{x0:.1f}" y="{yb0:.1f}" width="{cw:.1f}" height="{ch + bandh:.1f}"/></clipPath>',
           '<g clip-path="url(#vc)">']
    if outside:
        out.append('<g opacity="0.15">')
        out += outside
        out.append('</g>')
    out += inside
    out.append('</g>')
    out += B
    out.append('</svg>')

    svg = HERE / f'L1-part{pid}.svg'
    svg.write_bytes('\n'.join(out).encode('utf-8'))   # LF 一律（CLAUDE.md 坑 #8）
    return dict(svg=svg, png=svg.with_suffix('.png'), viewBox=(x0, yb0, cw, ch + bandh),
                hc=hc, k=k, vx=(vxl, vxr), vy=vy, lines=lines, n_faded=len(outside), n_full=len(inside))


def render_png(svg_path, png_path):
    js = (f"const s=require('{SHARP}');"
          "const [f,o]=process.argv.slice(1);"
          "s(f,{density:144}).png().toFile(o).then(i=>console.log('  PNG '+o+' '+i.width+'x'+i.height));")
    r = subprocess.run(['node', '-e', js, str(svg_path), str(png_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f'sharp 渲染失败: {svg_path}')
    print(r.stdout.strip())


def main():
    elems, g, warn = lc.build_l0()
    text_boxes = l0_text_boxes(elems)
    rects_xywh, elems_bbs = [], []
    for bb, s in elems:
        m = RECT_RE.match(s) if s.startswith('<rect') else None
        if m:
            rects_xywh.append(tuple(float(v) for v in m.groups()))
            elems_bbs.append(bb)
    chapters = {p['id']: [] for p in PLAN['parts']}
    for c in PLAN['chapters']:
        chapters[c['part']].append(c)
    for pid, chs in chapters.items():
        chs.sort(key=lambda c: c['no'])
    parts = {p['id']: p for p in PLAN['parts']}
    metas = []
    for pid in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']:
        m = build_part(pid, parts[pid], elems, g, chapters, text_boxes, rects_xywh, elems_bbs)
        vb = m['viewBox']
        print(f"Part {pid}: viewBox=({vb[0]:.0f},{vb[1]:.0f},{vb[2]:.0f},{vb[3]:.0f}) "
              f"canvas={CANVAS_W}x{m['hc']} k=×{m['k']:.2f} vx={m['vx']} vy={m['vy']} "
              f"lines={m['lines']} 淡出 {m['n_faded']} / 全亮 {m['n_full']} ")
        metas.append(m)
    if warn:
        print(f'--- {len(warn)} OVERFLOW WARNINGS ---')
        for w in warn:
            print('  ' + w)
    else:
        print('no overflow warnings')
    for m in metas:
        render_png(m['svg'], m['png'])


if __name__ == '__main__':
    main()
