#!/usr/bin/env python3
"""L1 Part 图 ×8 —— minimap + 放大细节（v3 图系 Phase 2 / FIGURE-SYSTEM.md §1）

机制（IDE 概览图模式；2026-08-15 用户裁决，推翻「区域外淡出」旧方案——
「展示局部，放一个缩小版的全局图，框出要放大的部分，边上放大细节」）：

  ┌──────────────────────────────────────────────┐
  │ 标题带（专用高度）：Part N — 标题 · hook · 章目录     │
  ├────────────┬─────────────────────────────────┤
  │ minimap    │   放大细节（本 Part 区域，全亮）      │
  │ L0 ×0.24   │   viewBox 裁切等比放大，零 opacity 混杂 │
  │ ▣高亮框区   │                                 │
  └────────────┴─────────────────────────────────┘

  - 元素流 = l0_common.build_l0() 原样两份（同一坐标系、同配色、同术语）：
    minimap 组 translate+scale 缩小全局；detail 组 translate+scale(k) 放大
    本 Part 区域——同一元素流、两种取景，放大镜关系由几何变换物理保证
    （同源强制，绝不重画布局）；
  - 上下文职责移交 minimap：detail 裁切区外元素直接不可见（clipPath 钉死，
    不赌渲染器 viewport 裁剪——sharp/librsvg 边界 bug 的既有结论）；
  - minimap 上本 Part 区域画高亮描边框（Part 主题色，双层描边=粗线+外发光），
    框外元素整体 opacity 0.45 退后（仍可辨）；框右上/右下角 → detail 左上/
    左下角两条 #94a3b8 虚线，锥形展开锚定放大关系；
  - 锥线锚可见角（2026-09-12 立，ch24 盲审「箭头悬空」后）：上/下根分别锚到
    最上/最下区域各自高亮框的右上/右下角（可见锚点），不再锚多区域外包框
    角（几何锚但无可见元素，虚线像从空处出发）；单区域 Part 两锚同框、几何不变；
  - 裁切护栏（2026-09-12 立，ch24 盲审「切口落在文字行中间/字残」后）：
    detail 裁切边不得落入任何文字墨迹区间。墨迹区间用紧致墨迹模型（CJK=1em、
    字母数字 0.555em、空格 0.30em、其余 ASCII 0.40em、粗体×1.04、锚侧端点精确，
    系数经本机 sharp 渲染像素标定）
    从元素流 <text> 解析重算；发现边切入文字时向两侧走查最近墨迹净空位（±SAFE），
    走查越出 cap 即该向不可行；另把区域自身四界（卡片边框）作结构候选——浅穿透
    （≤TOL2，模型误差带内）即接受：卡片文字内缩 14、间隙标签居中于间隙，结构界
    天然让墨。两向皆不可行且无结构候选 → 保持原位并告警；不许剃进区域环带超过
    SHAVE（= comp() 文字内缩深度，只剃卡片衬边、不碰字）；
  - 标题带在画布顶部专用区（不再与 L0 内容抢位——旧版的确定性落位搜索
    及其 bbox 复刻全部退役）；三行基线 8 张图完全一致；
  - Part→L0 区域映射沿用 regions_for（坐标全部取自 GEO，L0 改版自动联动）；
  - Part I/VIII（全景）：detail=全图、minimap 省——全局即局部，无框可高亮。

linter 协同（lint_diagram_geometry.py）：minimap/detail 组各带 data-minimap/
data-detail 标记。组内元素保留原始 L0 坐标（linter 不解 transform），跨组/
跨画布的碰撞检查无意义——linter 按 ctx 分域检查（data-zoom 豁免模式的扩展）；
每组各含一份完整 L0 元素流 + 白底 rect，组内箭头连通/碰撞判定的真值与
L0 自身逐字一致（端点在白底内=已连接的宽容分支同语义）。

输出：L1-part{N}.svg + L1-part{N}.png（node sharp density=144 → 2x，同 L0 约定）。
"""
import json
import re
import subprocess
from pathlib import Path

import l0_common as lc

HERE = Path(__file__).parent
PLAN = json.loads((HERE / 'pedagogy-plan.json').read_text(encoding='utf-8'))

# ---- 布局常量（画布坐标系；宽 2200 = L0 画布宽） ----
CANVAS_W = 2200
MARGIN = 24                 # 画布外边距
BAND_H = 130                # 标题带高（VIII 多一行注记取 190）
BAND_H_VIII = 190
STD_LINES = (46, 80, 110)   # 标题带三行基线（画布坐标）
VIII_LINES = (50, 86, 118, 158)
CT_PAD = 30                 # 标题带底 → 内容区顶
MM_COL_W = 560              # 左栏宽（含内边距）
MM_PAD = 16
MM_W = MM_COL_W - 2 * MM_PAD        # 528 → S≈0.24（任务带宽 0.22-0.28）
S = MM_W / lc.W
CAP_H = 38                  # 内容区顶 → minimap 框顶（两行小标占位）
GAP = 44                    # minimap 右缘 → detail 左缘（指示线锥形走廊）
CROP_M = 16                 # detail 裁切边距（区域外包⊕后钳回 L0 画布）
F = 12.0                    # minimap 亮/暗分区余量（沾边元素保持全亮，接口箭头不灭）
DIM = 0.45                  # minimap 框外元素透明度
HL_EXP = 10                 # 高亮框相对区域的放大（raw px，给描边留净空）
PART_COLOR = {'I': lc.C_TXT, 'II': lc.C_API_S, 'III': lc.C_ENG_S, 'IV': lc.C_KV_S,
              'V': lc.C_GPU_S, 'VI': lc.C_GPU_S, 'VII': lc.C_SAM_S, 'VIII': lc.C_TXT}
VIII_NOTE = '生产视角：多实例 / 池化 / 弹性 —— 详 L0×N 衍生图（后续章内）'
SHARP = "E:/Laboratory/Repo2Book/node_modules/sharp"


def ov(a, b):
    """bbox 严格相交（贴边不算）。"""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---- 裁切护栏（见模块 docstring）：裁切边不得落入文字墨迹区间 ----
SAFE = 1.5      # 落点离墨迹区间的最小净空
SHAVE = 14.0    # 允许剃进区域环带的最大深度（= comp() 文字内缩 tx-x，只剃卡片衬边）
TOL2 = 3.5      # 结构候选（区域自身四界）的浅穿透容忍——墨迹模型对混排串的误差带

_TEXT_RE = re.compile(
    r'<text x="([-\d.]+)" y="([-\d.]+)" font-family="[^"]+" font-size="([\d.]+)" '
    r'fill="[^"]+" text-anchor="(\w+)"( font-weight="bold")?>(.*)</text>')


def _ink_w(s, fs, bold):
    """紧致墨迹宽度：CJK=1em · 字母数字 0.555em · 空格 0.30em · 其余 ASCII 0.40em · 粗体×1.04。

    系数经本机 sharp 渲染像素标定（2026-09-12：'hidden_states' 实测 64.4px@fs9 ≈0.55em/char，
    0.52 低估会让护栏把切口留在末字符身上）。"""
    w = 0.0
    for ch in s:
        o = ord(ch)
        if o > 0x2E80:
            w += fs
        elif ch == ' ':
            w += 0.30 * fs
        elif ch.isalnum():
            w += 0.555 * fs
        else:
            w += 0.40 * fs
    return w * (1.04 if bold else 1.0)


def ink_boxes(elems):
    """从元素流解析文字的墨迹 bbox（x0,y0,x1,y1）；非 <text> 元素跳过。

    与 l0_common.text() 记录的保守 bbox（±2 padding + ASCII 0.58em）不同——
    那是碰撞检测的保守侧；本墨迹模型是「边切没切到字」的判定侧，过宽会把
    间隙标签↔列头这类 ~3 单位的真间隙判成无隙，导致护栏大幅让位。"""
    out = []
    for _, s_ in elems:
        m = _TEXT_RE.match(s_)
        if not m:
            continue
        x, y, fs = float(m.group(1)), float(m.group(2)), float(m.group(3))
        anchor, bold, body = m.group(4), bool(m.group(5)), m.group(6)
        for a, b in (('&lt;', '<'), ('&gt;', '>'), ('&quot;', '"'),
                     ('&apos;', "'"), ('&amp;', '&')):
            body = body.replace(a, b)
        w = _ink_w(body, fs, bold)
        x0 = x - w / 2 if anchor == 'middle' else (x - w if anchor == 'end' else x)
        out.append((x0 - 0.5, y - 0.86 * fs - 0.5, x0 + w + 0.5, y + 0.23 * fs + 0.5))
    return out


def _walk(edge, ivs, d, lo_cap, hi_cap):
    """从 edge 沿方向 d 走查到第一个墨迹净空位；越出 cap 即该向不可行（返回 None）。"""
    t = edge
    for _ in range(80):
        hit = next(((a, b) for a, b in ivs if a < t < b), None)
        if hit is None:
            return t if lo_cap <= t <= hi_cap else None
        t = (hit[1] + SAFE) if d > 0 else (hit[0] - SAFE)
        if (d > 0 and t > hi_cap) or (d < 0 and t < lo_cap):
            return None
    return None


def _guard_edge(edge, ivs, lo_cap, hi_cap, struct, pid, tag, glog):
    """单条裁切边的护栏：不动(未切字) / 走查净空位 / 结构候选(区域界,浅穿透容忍) / 保持+告警。"""
    if not any(a < edge < b for a, b in ivs):
        return edge
    cands = []
    for d in (-1, 1):
        t = _walk(edge, ivs, d, lo_cap, hi_cap)
        if t is not None:
            cands.append(t)
    for e_ in struct:
        if not (lo_cap <= e_ <= hi_cap):
            continue
        pen = min((min(e_ - a, b - e_) for a, b in ivs if a < e_ < b), default=None)
        if pen is None or pen <= TOL2:
            cands.append(e_)
    if not cands:
        glog.append(f'L1-{pid} {tag} {edge:.1f}: 护栏无净空位，保持原位（需人工核图）')
        return edge
    new = min(cands, key=lambda c: (abs(c - edge), c))
    if abs(new - edge) > 0.05:
        glog.append(f'L1-{pid} {tag} {edge:.1f} -> {new:.1f}（避开文字墨迹）')
    return new


def guard_crop(crop, inks, reg, pid, g, glog):
    """四条裁切边逐条过护栏；三趟迭代收敛（边动过会改变对侧的相关文字过滤范围）。"""
    cx0, cy0, cx1, cy1 = crop
    rx0, ry0, rx1, ry1 = reg
    for _ in range(3):
        before = (cx0, cy0, cx1, cy1)
        ivx = [(b[0], b[2]) for b in inks if b[1] < cy1 and cy0 < b[3]]
        ivy = [(b[1], b[3]) for b in inks if b[0] < cx1 and cx0 < b[2]]
        cy0 = _guard_edge(cy0, ivy, 0.0, ry0 + SHAVE, (ry0,), pid, '上缘', glog)
        cy1 = _guard_edge(cy1, ivy, ry1 - SHAVE, float(g['H']), (ry1,), pid, '下缘', glog)
        cx0 = _guard_edge(cx0, ivx, 0.0, rx0 + SHAVE, (rx0,), pid, '左缘', glog)
        cx1 = _guard_edge(cx1, ivx, rx1 - SHAVE, float(lc.W), (rx1,), pid, '右缘', glog)
        if (cx0, cy0, cx1, cy1) == before:
            break
    return cx0, cy0, cx1, cy1


def expand(r, m):
    return (r[0] - m, r[1] - m, r[2] + m, r[3] + m)


def regions_for(g, pid):
    """Part → L0 区域（语义映射；坐标全部取自 GEO，L0 改版自动联动）。"""
    AX, CW = g['AX'], g['COL_W']
    if pid in ('I', 'VIII'):                       # 全图（VIII + 注记，见 build_part）
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


def build_part(pid, part, elems, g, chapters, inks, glog):
    regions = regions_for(g, pid)
    full = pid in ('I', 'VIII')
    rx0 = min(r[0] for r in regions)
    ry0 = min(r[1] for r in regions)
    rx1 = max(r[2] for r in regions)
    ry1 = max(r[3] for r in regions)
    # detail 裁切区 = 区域外包 ⊕ 边距，钳回 L0 画布（无淡出上下文，不取景画布外）
    cx0, cy0 = max(0.0, rx0 - CROP_M), max(0.0, ry0 - CROP_M)
    cx1 = min(float(lc.W), rx1 + CROP_M)
    cy1 = min(float(g['H']), ry1 + CROP_M)
    if not full:                       # 裁切护栏：切口不得落在文字行中间（半行残字）
        cx0, cy0, cx1, cy1 = guard_crop((cx0, cy0, cx1, cy1), inks,
                                        (rx0, ry0, rx1, ry1), pid, g, glog)
    cw, chh = cx1 - cx0, cy1 - cy0

    band = BAND_H_VIII if pid == 'VIII' else BAND_H
    lines = VIII_LINES if pid == 'VIII' else STD_LINES
    chapters_s = ' · '.join(f"ch{c['no']} {c['title']}" for c in chapters[pid])
    note = VIII_NOTE if pid == 'VIII' else ''
    color = PART_COLOR[pid]

    dx0 = MARGIN if full else MARGIN + MM_COL_W + GAP
    dw = CANVAS_W - MARGIN - dx0
    k = dw / cw
    ct = band + CT_PAD                 # 内容区顶 = detail 上缘
    dh = chh * k
    dy1 = ct + dh
    mmx, mmy = MARGIN + MM_PAD, ct + CAP_H
    mmh = S * g['H']
    H = round(max(dy1, mmy + mmh if not full else 0.0) + MARGIN)

    bg = f'<rect x="0" y="0" width="{lc.W}" height="{g["H"]:.0f}" fill="white"/>'

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{H}" '
           f'viewBox="0 0 {CANVAS_W} {H}" data-zoom="L1">',
           f'<rect width="{CANVAS_W}" height="{H}" fill="white"/>',
           lc.DEFS,
           f'<clipPath id="dc"><rect x="{dx0:.1f}" y="{ct:.1f}" width="{dw:.1f}" height="{dh:.1f}"/></clipPath>']

    n_dim = 0
    n_bright = len(elems)
    if not full:
        # ---- minimap（左栏：全局缩小 + 高亮框 + 框外退后） ----
        frs = [expand(r, F) for r in regions]
        outside = [s_ for bb, s_ in elems if not any(ov(bb, fr) for fr in frs)]
        inside = [s_ for bb, s_ in elems if any(ov(bb, fr) for fr in frs)]
        n_dim, n_bright = len(outside), len(inside)
        out.append(f'<g data-minimap="1" transform="translate({mmx:.1f},{mmy:.1f}) scale({S:.4f})">')
        out.append(bg)
        if outside:
            out.append(f'<g opacity="{DIM}">')
            out += outside
            out.append('</g>')
        out += inside
        out.append('</g>')
        # 框描边 + 顶部小标（画布坐标，画在 minimap 组之上）
        out.append(lc.rect_svg(mmx, mmy, MM_W, mmh, 'none', lc.C_FAINT, rx=4, sw=1.4, dash=False))
        lc.fit('全局（L0）', 11.5, MM_W, f'L1-{pid}:mm-title', True)
        out.append(lc.text_svg(mmx, ct + 13, '全局（L0）', 11.5, lc.C_TXT, 'start', True))
        cap2 = '高亮框内 = 本 Part 区域 · 右侧同区放大'
        lc.fit(cap2, 9.5, MM_W, f'L1-{pid}:mm-sub')
        out.append(lc.text_svg(mmx, ct + 29, cap2, 9.5, lc.C_MUTE, 'start', False))
        # 高亮框：Part 主题色双层描边（外发光 + 粗实线），画布坐标与 minimap 内容对齐
        def mm(x, y):
            return mmx + x * S, mmy + y * S
        ubx0, uby0 = mm(rx0 - HL_EXP, ry0 - HL_EXP)
        ubx1, uby1 = mm(rx1 + HL_EXP, ry1 + HL_EXP)
        for (a0, b0, a1, b1) in regions:
            hx0, hy0 = mm(a0 - HL_EXP, b0 - HL_EXP)
            hx1, hy1 = mm(a1 + HL_EXP, b1 + HL_EXP)
            out.append(f'<rect x="{hx0:.1f}" y="{hy0:.1f}" width="{hx1 - hx0:.1f}" '
                       f'height="{hy1 - hy0:.1f}" rx="5" fill="none" stroke="{color}" '
                       f'stroke-width="9" opacity="0.18"/>')
            out.append(lc.rect_svg(hx0, hy0, hx1 - hx0, hy1 - hy0, 'none', color, rx=5, sw=3.5, dash=False))
        if len(regions) > 1:            # 多区域：细虚线外包框统一锥形指示的锚点
            out.append(lc.rect_svg(ubx0, uby0, ubx1 - ubx0, uby1 - uby0,
                                   'none', lc.C_MUTE, rx=5, sw=1.2, dash=True))

    # ---- detail（右区/全景：本 Part 区域全亮放大，clipPath 钉死视口） ----
    out.append('<g data-detail="1" clip-path="url(#dc)">')
    out.append(f'<g transform="translate({dx0:.1f},{ct:.1f}) scale({k:.4f}) '
               f'translate({-cx0:.1f},{-cy0:.1f})">')
    out.append(bg)
    out += [s_ for _, s_ in elems]
    out.append('</g></g>')
    out.append(lc.rect_svg(dx0, ct, dw, dh, 'none', lc.C_FAINT, rx=4, sw=1.2, dash=False))

    # ---- 指示线：高亮框右缘两角 → detail 左缘两角（锥形展开；锚可见角） ----
    if not full:
        top_r = min(regions, key=lambda r: r[1])       # 最上区域 → 上根锚其高亮框右上角
        bot_r = max(regions, key=lambda r: r[3])       # 最下区域 → 下根锚其高亮框右下角
        for (rx_, ry_), y_to in ((mm(top_r[2] + HL_EXP, top_r[1] - HL_EXP), ct),
                                 (mm(bot_r[2] + HL_EXP, bot_r[3] + HL_EXP), dy1)):
            out.append(f'<line x1="{rx_:.1f}" y1="{ry_:.1f}" x2="{dx0:.1f}" '
                       f'y2="{y_to:.1f}" stroke="{lc.C_FAINT}" stroke-width="1.6" '
                       f'stroke-dasharray="6,4"/>')

    # ---- 标题带（最后画=压在最上层；专用区，绝不与内容重叠） ----
    out.append(lc.rect_svg(0, 0, CANVAS_W, band, '#ffffff', 'none', 0, 0, False))
    out.append(lc.rect_svg(0, 0, 12, band, color, 'none', 0, 0, False))
    out.append(lc.rect_svg(0, band - 3, CANVAS_W, 3, color, 'none', 0, 0, False))
    specs = [('title', f"Part {pid} — {part['title']}", 22, lc.C_TXT, True, 1900),
             ('hook', part['hook'], 15, color, False, 1900),
             ('chapters', chapters_s, 12, lc.C_MUTE, False, 2040)]
    if note:
        specs.append(('note', note, 12.5, lc.C_MUTE, False, 2040))
    for (tag, s_, fs, fill, bold, maxw), v in zip(specs, lines):
        lc.fit(s_, fs, maxw, f'L1-{pid}:{tag}', bold)
        out.append(lc.text_svg(1100, v, s_, fs, fill, 'middle', bold))
    badge = f'L1 = L0 ×{k:.2f}'
    lc.fit(badge, 12, 190, f'L1-{pid}:badge')
    out.append(lc.text_svg(2176, lines[0], badge, 12, lc.C_FAINT, 'end', False))

    out.append('</svg>')
    svg = HERE / f'L1-part{pid}.svg'
    svg.write_bytes('\n'.join(out).encode('utf-8'))   # LF 一律（CLAUDE.md 坑 #8）
    return dict(svg=svg, png=svg.with_suffix('.png'), H=H, k=k,
                crop=(cx0, cy0, cx1, cy1), n_dim=n_dim, n_bright=n_bright)


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
    inks = ink_boxes(elems)
    glog = []
    chapters = {p['id']: [] for p in PLAN['parts']}
    for c in PLAN['chapters']:
        chapters[c['part']].append(c)
    for pid, chs in chapters.items():
        chs.sort(key=lambda c: c['no'])
    parts = {p['id']: p for p in PLAN['parts']}
    metas = []
    for pid in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']:
        m = build_part(pid, parts[pid], elems, g, chapters, inks, glog)
        cp = m['crop']
        mm_s = '省（全景页）' if pid in ('I', 'VIII') else f'×{S:.3f} 暗{m["n_dim"]}/亮{m["n_bright"]}'
        print(f"Part {pid}: canvas={CANVAS_W}x{m['H']} crop=({cp[0]:.0f},{cp[1]:.0f})-"
              f"({cp[2]:.0f},{cp[3]:.0f}) k=×{m['k']:.2f} minimap={mm_s}")
        metas.append(m)
    if glog:
        print(f'--- {len(glog)} CROP-GUARD ADJUSTMENTS ---')
        for w in glog:
            print('  ' + w)
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
