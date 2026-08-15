#!/usr/bin/env python3
"""L2 章图渲染器 —— v3 图系第三层（FIGURE-SYSTEM.md §2，契约见该节）

L1 放大 Part；L2 放大**章**：
  - minimap 保留 L0 全图缩小（×0.2），高亮框框到**本章的 L0 区域**（比 Part 更细，
    如 ch9 = 五拍循环框那一块）；框外元素退后（IDE 概览图模式，沿 gen_L1 裁决）；
  - detail **不再裁切 L0**——L0 没有方法级/站号级密度，detail 是用 l0_common 的
    原语/配色/框风格**新画**的本章组件展开：框内方法签名（v0.27.1 已核锚点）
    + 站号徽标「第 N 站」（= 请求流经顺序，吃 spec 的 stations 账本）；
  - 顶部窄条：ch{N} 标题 + hook 一句 + 「L0 位置：{l0_zoom}」；
  - 底部：读图一行 + 前置依赖章 + 本章埋/收的伏笔（pedagogy-plan 自动带出）。

数据契约 l2-spec/1（输入 = l2-specs/ch{N}.json，Phase 3 起随章 dossier 顺产）：
    chapter / part / title / hook / l0_zoom / l0_region / depends_on / reading
    frame{title, file}          本章舞台外框（进程/系统边界）
    center{name, title, where}  核心机制区外框（可选；center 区组件按序成拍片+回环）
    components[{name, role, zone, kind, file, methods, stations, note}]
        zone  ∈ north|center|south   north=请求进出条（左→右）· center=主角拍片 · south=支撑/why 注
        role  ∈ engine|gpu|kv|api|zmq|sample|io|plain|beat —— 映射 l0_common 配色常量（同源强制，spec 不带色值）
        kind  ∈ comp|queue|note
    flows[{from, to, label, up, dash, color_role}]   from/to 填组件名 / frame / center.name
    loop{label}                 center 末拍片 → 首拍片 回环（可选）
    stations[{n, where, what}]  本章站号账本（渲染器校验：徽标 ⊆ 账本、账本有挂点）

linter 协同：根元素带 data-zoom="L2"（缩放层，画布坐标 overflow 只查 ctx=None）；
minimap 组 data-minimap / detail 组 data-detail（组内坐标自洽，跨组不查——沿 gen_L1 模式）。
输出：L2-ch{N}.svg + L2-ch{N}.png（node sharp density=144 → 2x）。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import l0_common as lc

PLAN = json.loads((HERE / 'pedagogy-plan.json').read_text(encoding='utf-8'))

# ---- 布局常量（画布坐标系；宽 2200 = L0/L1 家族画布宽） ----
CANVAS_W = 2200
MARGIN = 24
BAND_H = 96                  # 顶部窄条（标题/hook/位置徽标 两行）
CT_PAD = 26                  # 窄条底 → 内容区顶
MM_COL_W = 470               # 左栏宽（minimap + 站号轨道）
MM_PAD = 15
MM_W = MM_COL_W - 2 * MM_PAD         # 440 → S = 0.2（任务带宽 ~0.2x）
S = MM_W / lc.W
CAP_H = 38                   # 内容区顶 → minimap 框顶（两行小标）
GAP = 44                     # minimap 右缘 → detail 左缘（锥形指示线走廊）
INSET = 16                   # 进程外框内衬
ROW_GAP = 18                 # 行距（north ↔ center ↔ south）
DIM = 0.45                   # minimap 框外元素透明度（沿 gen_L1）
F = 12.0                     # minimap 亮/暗分区余量
HL_EXP = 10                  # 高亮框相对区域的放大
CHIP_GAP = 36                # center 拍片间距
ROLE = {'engine': lc.C_ENG_S, 'gpu': lc.C_GPU_S, 'kv': lc.C_KV_S,
        'api': lc.C_API_S, 'zmq': lc.C_ZMQ_S, 'sample': lc.C_SAM_S,
        'io': lc.C_MUTE, 'plain': lc.C_MUTE, 'beat': lc.C_BEAT_S}
QUEUE_W = 110                # queue_glyph 固定宽（queue_glyph 自身 h=96）
SHARP = "E:/Laboratory/Repo2Book/node_modules/sharp"

_TOK = re.compile(r'[ -~→·]+|.')       # ascii 连串成 token，其余逐字（CJK 友好断行）


def wrap_cn(s, fs, maxw):
    """贪心断行：CJK 逐字、ascii 连串；超长 token 硬拆。"""
    lines, cur = [], ''
    for tk in _TOK.findall(s):
        if not tk:
            continue
        while lc.tw(tk, fs) > maxw and len(tk) > 1:
            k = max(1, int(len(tk) * maxw / max(1.0, lc.tw(tk, fs))))
            lines.append(tk[:k])
            tk = tk[k:]
        if cur and lc.tw(cur + tk, fs) > maxw:
            lines.append(cur.rstrip())
            cur = tk.lstrip()
        else:
            cur += tk
    if cur.strip():
        lines.append(cur.rstrip())
    # 孤行合并：续行只有收尾标点（如「）」）时并回上一行——2px 溢出不可见，孤儿行刺眼
    merged = []
    for ln in lines:
        if merged and len(ln) <= 3 and not re.search(r'[0-9A-Za-z]', ln):
            merged[-1] += ln
        else:
            merged.append(ln)
    return merged or [' ']


def ov(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def expand(r, m):
    return (r[0] - m, r[1] - m, r[2] + m, r[3] + m)


def anchor_regions(g):
    """L0 锚点名 → 区域（坐标全部取自 GEO，L0 改版自动联动）。"""
    CW = g['COL_W']
    return {
        'full':          [(0, 0, lc.W, g['H'])],
        'api_band':      [(g['MX'], g['AY'], g['BXR'], g['AY'] + g['AH'])],
        'zmq_band':      [(g['MX'], g['ZY'], g['BXR'], g['ZY'] + g['ZH'])],
        'engine_band':   [(g['MX'], g['EY'], g['BXR'], g['EY'] + g['EH'])],
        'loop_box':      [(g['LOOP_X'], g['LOOP_Y'], g['LOOP_X'] + g['LOOP_W'],
                           g['LOOP_Y'] + g['LOOP_H'] + 28)],      # +28 含 io 链标注（同 L1 Part III 口径）
        'kv_column':     [(g['AX'], g['CY0'], g['AX'] + CW, g['A3Y'] + g['a3h'])],
        'gpu_column':    [(g['BX'], g['CY0'], g['BX'] + CW, g['B3Y'] + g['b3h'])],
        'sample_column': [(g['CX'], g['CY0'], g['CX'] + CW, g['C4Y'] + g['c4h'])],
    }


def resolve_region(spec, g):
    lr = spec.get('l0_region') or {}
    if isinstance(lr, dict) and lr.get('anchors'):
        A = anchor_regions(g)
        rs = []
        for a in lr['anchors']:
            rs += A[a]
        return rs
    rects = lr if isinstance(lr, list) else lr.get('rects', [])
    return [tuple(r) for r in rects]


# ---------- spec 校验（渲染前置闸：站号账本 / 流引用 / 词表） ----------
def validate(spec):
    errs = []
    comps = spec.get('components', [])
    pseudo = {'frame'}
    if spec.get('center', {}).get('name'):
        pseudo.add(spec['center']['name'])
    known = {c['name'] for c in comps} | pseudo
    st_ledger = {s['n'] for s in spec.get('stations', [])}
    st_badges = set()
    for c in comps:
        if c.get('zone') not in ('north', 'center', 'south'):
            errs.append(f"组件 {c['name']}: zone 非法 {c.get('zone')}")
        if c.get('role') not in ROLE:
            errs.append(f"组件 {c['name']}: role 非法 {c.get('role')}")
        if c.get('kind') not in (None, 'comp', 'queue', 'note'):
            errs.append(f"组件 {c['name']}: kind 非法 {c.get('kind')}")
        st_badges |= set(c.get('stations', []))
    for f in spec.get('flows', []):
        for k in ('from', 'to'):
            if f[k] not in known:
                errs.append(f"flow {f.get('label', '')[:12]}: {k}={f[k]} 无此组件")
    if not st_badges <= st_ledger:
        errs.append(f"站号徽标超出账本: {sorted(st_badges - st_ledger)}")
    for n in sorted(st_ledger - st_badges):
        errs.append(f"警告: 第{n}站无挂点（账本有、组件无）")
    return errs


def badge_text(ns):
    ns = sorted(ns)
    if not ns:
        return ''
    if len(ns) == 1:
        return f'第{ns[0]}站'
    if ns == list(range(ns[0], ns[-1] + 1)):
        return f'第{ns[0]}-{ns[-1]}站'
    return '第' + '/'.join(str(n) for n in ns) + '站'


# ---------- 组件绘制（原语全部来自 l0_common；高度=wrap 纯函数，可先算后排） ----------
def comp_style(c):
    beat = c.get('role') == 'beat'
    if beat:
        return dict(fill=lc.C_BEAT_F, stroke=lc.C_BEAT_S, tcol=lc.C_BEAT_T,
                    lcol=lc.C_BEAT_T, tfs=10, lfs=8.8, lh=13, top=30)
    return dict(fill='#ffffff', stroke=ROLE[c.get('role', 'plain')], tcol=lc.C_TXT,
                lcol='#334155', tfs=11.5, lfs=9.5, lh=17, top=34)


def comp_lines(c, w):
    st = comp_style(c)
    out = []
    for m in c.get('methods', []):
        out += wrap_cn(m, st['lfs'], w - 26)
    return out


def comp_h(c, w):
    st = comp_style(c)
    return st['top'] + len(comp_lines(c, w)) * st['lh'] + (18 if c.get('file') else 6)


def comp_demand(c):
    """期望宽（行布局加权用；queue 固定 110；note 正文按 8.5 估宽防虚胖抢宽）。"""
    if c.get('kind') == 'queue':
        return QUEUE_W
    st = comp_style(c)
    tfs = 9.5 if c.get('kind') == 'note' else st['tfs']
    lfs = 8.5 if c.get('kind') == 'note' else st['lfs']
    d = lc.tw(c['name'], tfs, True) + 40
    for m in c.get('methods', []):
        d = max(d, lc.tw(m, lfs) + 30)
    return d + 46


def note_h(c, w):
    ls = []
    for m in c.get('methods', []):
        ls += wrap_cn(m, 8.5, w - 22)
    return 26 + len(ls) * 12.5 + 8


def comp_draw(x, y, w, c, R):
    st = comp_style(c)
    if c.get('kind') == 'queue':
        lc.queue_glyph(x, y, w, c['name'], c.get('sub', 'queue.Queue'))
        R[c['name']] = dict(x=x, y=y, w=w, h=96)
        return 96
    if c.get('kind') == 'note':
        h = note_h(c, w)
        lc.rect(x, y, w, h, 'none', lc.C_FAINT, rx=7, sw=1.1, dash=True)
        lc.text(x + 10, y + 14, c['name'], 9.5, lc.C_MUTE, 'start', True,
                maxw=w - 20, tag='L2n:' + c['name'][:12])
        yy = y + 28
        for m in c.get('methods', []):
            for ln in wrap_cn(m, 8.5, w - 22):
                lc.text(x + 10, yy, ln, 8.5, lc.C_MUTE, 'start', maxw=w - 18, tag='L2nl:' + ln[:10])
                yy += 12.5
        R[c['name']] = dict(x=x, y=y, w=w, h=h)
        return h
    h = comp_h(c, w)
    lc.rect(x, y, w, h, st['fill'], st['stroke'], rx=7, sw=1.4)
    bw = 0
    badge = badge_text(c.get('stations', []))
    if badge:
        bw = 14 + 10 * len(badge)
        lc.rect(x + w - bw - 6, y + 5, bw, 17, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.0)
        lc.text(x + w - bw / 2 - 6, y + 17.5, badge, 8.5, lc.C_ENG_S, 'middle', True)
    lc.text(x + 12, y + (17 if st['top'] == 30 else 21), c['name'], st['tfs'], st['tcol'],
            'start', True, maxw=w - 24 - (bw + 12 if bw else 0), tag='L2:' + c['name'][:14])
    yy = y + st['top'] + 4
    for ln in comp_lines(c, w):
        lc.text(x + 12, yy, ln, st['lfs'], st['lcol'], 'start', maxw=w - 22, tag='L2m:' + ln[:12])
        yy += st['lh']
    if c.get('file'):
        lc.text(x + 12, y + h - 7, c['file'], 8.5, lc.C_FAINT, 'start', maxw=w - 22,
                tag='L2f:' + c['name'][:10])
    R[c['name']] = dict(x=x, y=y, w=w, h=h)
    return h


def plain_note_draw(x, y, w, lines):
    """组件旁 why 小注（spec 组件的 note 字段）：虚线框、无标题。"""
    ls = []
    for m in lines:
        ls += wrap_cn(m, 8.5, w - 20)
    h = 12 + len(ls) * 12.5 + 6
    lc.rect(x, y, w, h, 'none', lc.C_FAINT, rx=7, sw=1.1, dash=True)
    yy = y + 16
    for ln in ls:
        lc.text(x + 10, yy, ln, 8.5, lc.C_MUTE, 'start', maxw=w - 18, tag='L2pn:' + ln[:10])
        yy += 12.5
    return h


def drain():
    out = [s for _, s in lc.ELEMS]
    lc.reset()
    return out


def row_layout(comps, x0, w_total):
    """行内布局：queue 固定宽、其余按内容需求分宽；gap 吃剩余（钳 24..240）。"""
    n = len(comps)
    dem = [comp_demand(c) for c in comps]
    gap = (w_total - sum(dem)) / (n - 1) if n > 1 else 0.0
    if gap > 240:
        gap = 240.0
        extra = w_total - sum(dem) - gap * (n - 1)
        ds = sum(dem) or 1.0
        dem = [d + extra * d / ds for d in dem]
    elif gap < 24:
        gap = 24.0
        deficit = sum(dem) + gap * (n - 1) - w_total
        ds = sum(dem) or 1.0
        dem = [max(96.0, d - deficit * d / ds) for d in dem]
    xs, x = [], x0
    for d in dem:
        xs.append(x)
        x += d + gap
    return xs, dem, gap


def draw_flow(f, R, zone_of, captions):
    a, b = R[f['from']], R[f['to']]
    color = ROLE.get(f.get('color_role') or 'plain', lc.C_MUTE)
    dash = bool(f.get('dash'))
    oy0, oy1 = max(a['y'], b['y']), min(a['y'] + a['h'], b['y'] + b['h'])
    label = f.get('label')
    if oy1 - oy0 > 0.4 * min(a['h'], b['h']):          # 同行 → 横向
        if b['x'] > a['x']:
            x1, x2 = a['x'] + a['w'], b['x']
        else:
            x1, x2 = a['x'], b['x'] + b['w']
        y = (oy0 + oy1) / 2
        lc.seg(x1, y, x2, y, color, 1.8, 'std', dash)
        if label:
            if zone_of.get(f['from']) == 'center':     # 拍片间距窄：标签下沉到拍片下说明行
                captions.append(((x1 + x2) / 2, label))
            else:
                lc.text((x1 + x2) / 2, y - 5, label, 8.5, lc.C_MUTE, 'middle',
                        maxw=(x2 - x1) + 80, tag='L2fl:' + label[:10])
    else:                                              # 跨行 → 纵向
        x = (max(a['x'], b['x']) + min(a['x'] + a['w'], b['x'] + b['w'])) / 2
        if b['y'] > a['y']:
            y1, y2 = a['y'] + a['h'], b['y']
        else:
            y1, y2 = a['y'], b['y'] + b['h']
        if f.get('up'):
            color = lc.C_ENG_S
            lc.seg(x, y1, x, y2, color, 1.8, 'up', dash)
        else:
            lc.seg(x, y1, x, y2, color, 1.8, 'std', dash)
        if label:
            # 贴下侧框顶边之上 6px（行间净空带）——箭头中点可能穿进旁注框的 y 带
            mid = max(y1, y2) - 6
            if f.get('up'):
                lc.text(x + 7, mid, label, 8.5, color, 'start', maxw=640, tag='L2fl:' + label[:10])
            else:
                lc.text(x - 7, mid, label, 8.5, color, 'end', maxw=640, tag='L2fl:' + label[:10])


def build(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    for e in validate(spec):
        print(f'  [{spec_path.name}] {e}')
    no = spec['chapter']
    part = spec.get('part') or next((c['part'] for c in PLAN['chapters'] if c['no'] == no), 'I')
    color = lc.PART_COLOR[part]

    elems, g, w0 = lc.build_l0()
    elems = list(elems)            # 快照（build_l0 返回的就是 ELEMS 本体，reset 会清空）
    warns = list(w0)
    lc.reset()
    regions = resolve_region(spec, g)
    rx0 = min(r[0] for r in regions)
    ry0 = min(r[1] for r in regions)
    rx1 = max(r[2] for r in regions)
    ry1 = max(r[3] for r in regions)

    ct = BAND_H + CT_PAD
    mmx, mmy = MARGIN + MM_PAD, ct + CAP_H
    mmh = S * g['H']
    dx0 = MARGIN + MM_COL_W + GAP
    dw = CANVAS_W - MARGIN - dx0
    ix, iw = dx0 + INSET, dw - 2 * INSET

    comps = spec.get('components', [])
    R, zone_of, captions = {}, {}, []    # captions: [(gap_center_x, label)] 拍片间标签（下沉说明行）

    # ================= detail（新画；ctx=data-detail） =================
    north = [c for c in comps if c.get('zone') == 'north']
    center = [c for c in comps if c.get('zone') == 'center']
    south = [c for c in comps if c.get('zone') == 'south']
    ccfg = spec.get('center') or {}

    # ---- north 行（含组件旁 note 挂尾） ----
    xs, ws, _gap = row_layout(north, ix, iw)
    north_h = 0
    for c, x, w in zip(north, xs, ws):
        h = comp_draw(x, ct + 40, w, c, R)
        zone_of[c['name']] = 'north'
        col_h = h
        if c.get('note'):
            col_h += 8 + plain_note_draw(x, ct + 40 + h + 8, w, c['note'])
        north_h = max(north_h, col_h)
    north_bottom = ct + 40 + north_h
    s_north = drain()

    # ---- center 区几何（先算后排：框高取决于拍片行高） ----
    cfy = north_bottom + ROW_GAP
    nb = len(center)
    cw_each = (iw - (nb - 1) * CHIP_GAP) / nb if nb else 0
    chip_h = max((comp_h(c, cw_each) for c in center), default=0)
    cf_pad_top = 28 if ccfg else 8
    cf_h = cf_pad_top + chip_h + 56
    chips_y = cfy + cf_pad_top
    chips_bottom = chips_y + chip_h

    # ---- center 拍片 ----
    for i, c in enumerate(center):
        comp_draw(ix + i * (cw_each + CHIP_GAP), chips_y, cw_each, c, R)
        zone_of[c['name']] = 'center'
    s_center = drain()

    # ---- south 行 ----
    south_h = 0
    if south:
        sx, sw_, _ = row_layout(south, ix, iw)
        south_y = cfy + cf_h + ROW_GAP
        for c, x, w in zip(south, sx, sw_):
            south_h = max(south_h, comp_draw(x, south_y, w, c, R))
            zone_of[c['name']] = 'south'
        south_bottom = south_y + south_h
    else:
        south_bottom = cfy + cf_h
    s_south = drain()

    frame_h = south_bottom + INSET - ct
    R['frame'] = dict(x=dx0, y=ct, w=dw, h=frame_h)
    if ccfg.get('name'):
        R[ccfg['name']] = dict(x=ix, y=cfy, w=iw, h=cf_h)

    # ---- 流 + 回环（全部 rect 已知后画，浮于组件之上） ----
    for f in spec.get('flows', []):
        draw_flow(f, R, zone_of, captions)
    if nb > 1 and spec.get('loop'):
        c1 = ix + cw_each / 2
        cn = ix + (nb - 1) * (cw_each + CHIP_GAP) + cw_each / 2
        ch_y = chips_bottom + 32
        lc.parrow([(cn, chips_bottom), (cn, ch_y), (c1, ch_y), (c1, chips_bottom)],
                  lc.C_MUTE, 1.4, 'std')
        lc.text((c1 + cn) / 2, ch_y + 14, spec['loop']['label'], 8.5, lc.C_MUTE,
                'middle', maxw=iw - 40, tag='L2:loop')
    for gx, lab in captions:
        lc.text(gx, chips_bottom + 14, lab, 8.5, lc.C_MUTE, 'middle',
                maxw=cw_each + CHIP_GAP, tag='L2cap:' + lab[:10])
    s_flows = drain()

    # ---- 容器框（画序=层序：进程框最底 → center 框 → 组件 → 流 → 外框标题） ----
    cf_rect = lc.rect_svg(ix, cfy, iw, cf_h, '#ffffff', lc.C_ENG_S, rx=8, sw=1.8, dash=False)
    if ccfg:
        lc.text(ix + 12, cfy + 19, ccfg.get('title', ''), 11.5, lc.C_ENG_S, 'start', True,
                maxw=iw - 360, tag='L2:center-title')
        if ccfg.get('where'):
            lc.text(ix + iw - 12, cfy + 19, ccfg['where'], 9, lc.C_FAINT, 'end',
                    maxw=320, tag='L2:center-where')
    s_cf_hdr = drain()
    ef_rect = lc.rect_svg(dx0, ct, dw, frame_h, lc.C_ENG_F, lc.C_ENG_S, rx=12, sw=2.4, dash=False)
    lc.text(dx0 + INSET, ct + 24, spec['frame']['title'], 13.5, lc.C_ENG_S, 'start', True,
            maxw=dw - 500, tag='L2:frame-title')
    if spec['frame'].get('file'):
        lc.text(dx0 + dw - INSET, ct + 24, spec['frame']['file'], 9.5, lc.C_MUTE, 'end',
                maxw=460, tag='L2:frame-file')
    s_ef_hdr = drain()

    detail_strings = ([ef_rect] + s_north + [cf_rect] + s_cf_hdr + s_center
                      + s_south + s_flows + s_ef_hdr)
    warns += lc.WARN
    lc.reset()

    # ================= chrome（画布坐标 ctx=None） =================
    chrome = [lc.rect_svg(mmx, mmy, MM_W, mmh, 'none', lc.C_FAINT, rx=4, sw=1.4, dash=False)]
    lc.text(mmx, ct + 13, '全局（L0）', 11.5, lc.C_TXT, 'start', True, maxw=MM_W, tag='L2:mm-title')
    lc.text(mmx, ct + 29, '高亮框 = 本章 L0 区域 · 右侧 = 该块的展开', 9.5, lc.C_MUTE,
            'start', maxw=MM_W, tag='L2:mm-sub')
    chrome += drain()
    # 高亮框（双层描边）+ 锥形指示线
    def mm(x, y):
        return mmx + x * S, mmy + y * S
    ubx0, uby0 = mm(rx0 - HL_EXP, ry0 - HL_EXP)
    ubx1, uby1 = mm(rx1 + HL_EXP, ry1 + HL_EXP)
    for (a0, b0, a1, b1) in regions:
        hx0, hy0 = mm(a0 - HL_EXP, b0 - HL_EXP)
        hx1, hy1 = mm(a1 + HL_EXP, b1 + HL_EXP)
        chrome.append(f'<rect x="{hx0:.1f}" y="{hy0:.1f}" width="{hx1 - hx0:.1f}" '
                      f'height="{hy1 - hy0:.1f}" rx="5" fill="none" stroke="{color}" '
                      f'stroke-width="9" opacity="0.18"/>')
        chrome.append(lc.rect_svg(hx0, hy0, hx1 - hx0, hy1 - hy0, 'none', color, rx=5, sw=3.5, dash=False))
    for y_from, y_to in ((uby0, ct), (uby1, ct + frame_h)):
        chrome.append(f'<line x1="{ubx1:.1f}" y1="{y_from:.1f}" x2="{dx0:.1f}" '
                      f'y2="{y_to:.1f}" stroke="{lc.C_FAINT}" stroke-width="1.6" '
                      f'stroke-dasharray="6,4"/>')
    # 站号轨道（minimap 下方）
    sts = spec.get('stations', [])
    if sts:
        rail_y = mmy + mmh + 18
        lc.text(mmx, rail_y + 12, f'本章站号 = 请求流经顺序（共 {len(sts)} 站）', 10,
                lc.C_TXT, 'start', True, maxw=MM_W, tag='L2:rail-cap')
        yy = rail_y + 30
        for st in sts:
            bd = f"第{st['n']}站"
            pw = 12 + lc.tw(bd, 8, True) + 8
            lc.rect(mmx, yy - 9, pw, 13, lc.C_BADGE_F, lc.C_ENG_S, rx=6, sw=1.0)
            lc.text(mmx + pw / 2, yy + 1, bd, 8, lc.C_ENG_S, 'middle', True)
            lc.text(mmx + pw + 8, yy + 1, f"{st['where']} · {st['what']}", 8.5, lc.C_MUTE,
                    'start', maxw=MM_W - pw - 10, tag=f"L2:rail{st['n']}")
            yy += 16.5
        rail_bottom = yy - 10
    else:
        rail_bottom = mmy + mmh
    # 顶部窄条
    lc.rect(0, 0, CANVAS_W, BAND_H, '#ffffff', 'none', 0, 0)
    lc.rect(0, 0, 12, BAND_H, color, 'none', 0, 0)
    lc.rect(0, BAND_H - 3, CANVAS_W, 3, color, 'none', 0, 0)
    lc.text(1100, 46, f"ch{no} · {spec['title']}", 21, lc.C_TXT, 'middle', True,
            maxw=1500, tag='L2:band-title')
    lc.text(2176, 44, f"L2 · L0 位置：{spec.get('l0_zoom', '')}", 11, lc.C_FAINT, 'end',
            maxw=420, tag='L2:band-badge')
    lc.text(1100, 76, spec.get('hook', ''), 13.5, color, 'middle', maxw=1900, tag='L2:band-hook')
    chrome += drain()
    # 底部：读图 + 依赖 + 伏笔
    fy = max(ct + frame_h, rail_bottom) + 24
    dep = ' · '.join(f"ch{d} {next((c['title'] for c in PLAN['chapters'] if c['no'] == d), '')}"
                     for d in spec.get('depends_on', []))
    fs_ = []
    for fo in PLAN.get('foreshadows', []):
        if fo['planted'] == no:
            paid = fo['paid'] if isinstance(fo['paid'], list) else [fo['paid']]
            fs_.append(f"埋 {fo['id']}（{fo['text']} → ch{'/'.join(map(str, paid))} 回收）")
        elif (fo['paid'] == no) or (isinstance(fo['paid'], list) and no in fo['paid']):
            fs_.append(f"回收 {fo['id']}（ch{fo['planted']} 埋的 {fo['text']}）")
    lc.text(MARGIN, fy, spec.get('reading', ''), 10, lc.C_MUTE, 'start', maxw=CANVAS_W - 2 * MARGIN,
            tag='L2:ft-read')
    line2 = (f'前置依赖：{dep}　' if dep else '') + ('　'.join(fs_) if fs_ else '')
    if line2:
        lc.text(MARGIN, fy + 20, line2, 10, lc.C_MUTE, 'start', maxw=CANVAS_W - 2 * MARGIN,
                tag='L2:ft-dep')
    chrome += drain()
    warns += lc.WARN
    lc.reset()
    H = round(fy + (44 if line2 else 26))

    # ================= minimap 组（L0 原坐标，data-minimap ctx） =================
    frs = [expand(r, F) for r in regions]
    outside = [s for bb, s in elems if not any(ov(bb, fr) for fr in frs)]
    inside = [s for bb, s in elems if any(ov(bb, fr) for fr in frs)]
    bg = f'<rect x="0" y="0" width="{lc.W}" height="{g["H"]:.0f}" fill="white"/>'
    mm_group = [f'<g data-minimap="1" transform="translate({mmx:.1f},{mmy:.1f}) scale({S:.4f})">', bg]
    if outside:
        mm_group.append(f'<g opacity="{DIM}">')
        mm_group += outside
        mm_group.append('</g>')
    mm_group += inside
    mm_group.append('</g>')

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{H}" '
           f'viewBox="0 0 {CANVAS_W} {H}" data-zoom="L2">',
           f'<rect width="{CANVAS_W}" height="{H}" fill="white"/>',
           lc.DEFS]
    svg += mm_group
    svg.append('<g data-detail="1">')
    svg += detail_strings
    svg.append('</g>')
    svg += chrome
    svg.append('</svg>')

    out = HERE / f'L2-ch{no}.svg'
    out.write_bytes('\n'.join(svg).encode('utf-8'))       # LF 一律
    return dict(svg=out, png=out.with_suffix('.png'), H=H, frame_h=frame_h,
                n_dim=len(outside), n_bright=len(inside), warns=warns)


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


def render_one(spec_path):
    m = build(spec_path)
    print(f"{spec_path.stem}: canvas={CANVAS_W}x{m['H']} frame_h={m['frame_h']:.0f} "
          f"minimap ×{S:.2f} 暗{m['n_dim']}/亮{m['n_bright']}")
    if m['warns']:
        print(f'--- {len(m["warns"])} OVERFLOW WARNINGS ---')
        for w in m['warns']:
            print('  ' + w)
    else:
        print('  no overflow warnings')
    render_png(m['svg'], m['png'])
    return m


def main():
    args = sys.argv[1:]
    if not args:
        specs = sorted((HERE / 'l2-specs').glob('*.json'))
    else:
        specs = []
        for a in args:
            p = Path(a)
            if not p.is_absolute():
                p = (HERE / 'l2-specs' / a).with_suffix('.json') if not a.endswith('.json') else HERE / a
            specs.append(p)
    for p in specs:
        render_one(p)


if __name__ == '__main__':
    main()
