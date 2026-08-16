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

_TOK = re.compile(r'[!-~→·]+|\s+|.')   # ascii 词连串成 token（空格处断开），空白自成 token，其余逐字（CJK 友好断行）


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
    # 孤行合并：续行只有收尾标点（如「）」）或极短尾词时并回上一行——孤儿行刺眼。
    # exp-2026-08-16（ch2）：并回**不许超 maxw**（原实现假设溢出 ~2px 不可见，实测
    # 「）」并回超 8.9px、「忙循环」并回超 19px，直接触发渲染告警）；并不回时把上一行
    # 从最后一个非收尾 CJK 起截下陪它，避免孤悬标点独占一行。
    merged = []
    for ln in lines:
        if merged and len(ln) <= 3 and not re.search(r'[0-9A-Za-z]', ln):
            if lc.tw(merged[-1] + ln, fs) <= maxw:
                merged[-1] += ln
                continue
            prev = merged[-1]
            cut = max((i for i, ch in enumerate(prev)
                       if ord(ch) > 0x2E80 and ch not in '，、；：）」』》—…'), default=-1)
            if cut > 0 and lc.tw(prev[cut:] + ln, fs) <= maxw:
                merged[-1] = prev[:cut]
                merged.append(prev[cut:] + ln)
            else:
                merged.append(ln)
        else:
            merged.append(ln)
    # 尾行孤词拉回（ch2 盲审：⑦ 拍片「…socket → 前端 / PULL」——PULL 孤行刺眼）：
    # 尾行只剩一个短 token（≤30px、不含空格）时，把上一行末尾的整词（空格界定；
    # 无空格则末位 CJK 字）挪下来陪它。只往下挪：上行变短、下行加宽但钳在 maxw 内，
    # 不会引入新溢出（沿 exp-2026-08-16「并回不许超 maxw」同一条红线）。
    if (len(merged) >= 2 and lc.tw(merged[-1], fs) <= 30
            and ' ' not in merged[-1].strip()):
        prev = merged[-2]
        cut = prev.rfind(' ')
        if cut <= 0:
            cut = max((i for i, ch in enumerate(prev) if ord(ch) > 0x2E80), default=-1)
        if cut > 0 and prev[:cut].strip():
            moved = prev[cut:].strip()
            # 拼接分隔符：CJK↔CJK 边界原文本无空格（ch2 why 注「班车不是|专车」），
            # 插空格=改动真相源文字；其余（词界）补一个空格还原贪心断行吃掉的词距
            sep = '' if (ord(moved[-1]) > 0x2E80 and ord(merged[-1][0]) > 0x2E80) else ' '
            if moved and lc.tw(moved + sep + merged[-1], fs) <= maxw:
                merged[-2] = prev[:cut].rstrip()
                merged[-1] = moved + sep + merged[-1]
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
        # 请求生命线整条（ch2 的 L0 锚）：API 带→ZMQ 带→EngineCore 带三带合一成**单框**
        # （含带间 16px 过线走廊）——若给三带各画一框，16px 带缝×0.2 缩放后出三条贴边平行
        # 描边；单框才是「你在这里」的一个视觉对应物，锥形线从生命线两角出发（ch2 盲审②，
        # 对齐 ch3 已建立的高亮惯例）。框外（标题/users/页脚/启动视角/多实例视角/图例）退后。
        'lifeline':      [(g['MX'], g['AY'], g['BXR'], g['EY'] + g['EH'])],
        'boot':          [g['BOOT_R']],   # 启动视角块（l0_common GEO 注明「ch3 的 L0 锚」）
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
    badge = badge_text(c.get('stations', []))
    if badge:
        bw = 14 + 10 * len(badge)
        # 站号徽标骑框顶（tab 式，跨在框边上）：不再占标题行宽度——ch2 HTTP 入口这类长名
        # 组件标题 239px 与框内右上徽标 64px 相压（tag-on-title），骑边后标题吃满 w-24
        lc.rect(x + w - bw - 6, y - 8, bw, 17, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.0)
        lc.text(x + w - bw / 2 - 6, y + 3, badge, 8.5, lc.C_ENG_S, 'middle', True)
    fit_draw(x + 12, y + (17 if st['top'] == 30 else 21), c['name'], st['tfs'], st['tcol'],
             'start', True, w - 24, 'L2:' + c['name'][:14], floor=0.83)
    yy = y + st['top'] + 4
    for ln in comp_lines(c, w):
        lc.text(x + 12, yy, ln, st['lfs'], st['lcol'], 'start', maxw=w - 22, tag='L2m:' + ln[:12])
        yy += st['lh']
    if c.get('file'):
        fit_draw(x + 12, y + h - 7, c['file'], 8.5, lc.C_FAINT, 'start', False, w - 22,
                 'L2f:' + c['name'][:10], floor=0.85)
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
    # 只清 ELEMS、不清 WARN：WARN 由 build() 各阶段末 `warns += lc.WARN; lc.reset()` 统一收割
    # （exp-2026-08-16：此前 drain 连 WARN 一起清，detail 期全部溢出告警被吞——"no overflow warnings"
    #  是假阴性，标题压角标/文件名出血这类真碰撞因此漏网）
    out = [s for _, s in lc.ELEMS]
    lc.ELEMS.clear()
    return out


def fit_draw(x, y, s, fs, fill, anchor, bold, maxw, tag, floor=0.85):
    """lc.text 的 fit() 只告警不缩字——这里真缩：超宽先等比缩字号（下限 floor*fs），仍超再截断加 …。
    用于组件标题/文件行这类「内容是真相源、不许改 spec 文字」的场景（ch2 HTTP 入口标题
    239px > 框宽-24=241px 边缘、文件路径 261px > 243px——砍文字=动真相，缩字才是正解）。"""
    w = lc.tw(s, fs, bold)
    if w > maxw:
        fs2 = max(floor * fs, fs * maxw / w)
        if lc.tw(s, fs2, bold) <= maxw:
            fs, w = fs2, lc.tw(s, fs2, bold)
        else:
            while s and lc.tw(s + '…', fs, bold) > maxw:
                s = s[:-1]
            s += '…'
    lc.text(x, y, s, fs, fill, anchor, bold, maxw=maxw, tag=tag)


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


def draw_flow(f, R, zone_of, captions, badges=(), note_names=()):
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
            # 标签留在间隙内须两端各 ≥3px 净空（框线不许切字）；装不下 → 下沉到行下方说明行，
            # 与拍片 captions 同一模式。旧阈值 gap+30 过松（ch2「msgpack 帧」47.9px 落在
            # 24px 间隙、两条框线从字中穿过——lint rect-rect 盲区，靠亲眼看兜底）
            if zone_of.get(f['from']) == 'center' or lc.tw(label, 8.5) > (x2 - x1) - 6:
                captions.append(((x1 + x2) / 2, label, zone_of.get(f['from'])))
            else:
                lc.text((x1 + x2) / 2, y - 5, label, 8.5, lc.C_MUTE, 'middle',
                        maxw=(x2 - x1) + 80, tag='L2fl:' + label[:10])
    else:                                              # 跨行 → 纵向
        x = (max(a['x'], b['x']) + min(a['x'] + a['w'], b['x'] + b['w'])) / 2
        # 虚线框落点相位（ch3 盲审④顺带）：目标为 note（框线 dasharray 6,4）且下行时，
        # 箭头尖恰落进框顶虚线的「洞」里会看似与框线悬空数 px（ch3：compute_hash 盒顶
        # 小箭头）。落点 x 微移 ≤5px 对到 dash 段实在墨迹上——rx=7 圆角使相位起点有
        # (x+rx) 与 x 两解（SVG 规范 vs 宽松实现），取两解皆实的 [0.5,2.5] 段（p 按
        # 落点相对 note 左缘的相位算）。
        if f['to'] in note_names and b['y'] > a['y']:
            p = (x - b['x']) % 10
            if not 0.5 <= p <= 2.5:
                d = (1.5 - p) % 10
                if d > 5:
                    d -= 10
                x += d
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
            # 贴下侧框顶边之上 6px（行间净空带；north→center 行距在有下沉说明行时已加宽，
            # 两拨文字各占一带不再相撞——箭头中点可能穿进旁注框的 y 带，不用中点）
            mid = max(y1, y2) - 6
            if f.get('up'):
                lc.text(x + 7, mid, label, 8.5, color, 'start', maxw=640, tag='L2fl:' + label[:10])
            elif x - 7 - lc.tw(label, 8.5) < R['frame']['x'] + 12:
                # 'end' 锚会把标签左端伸出 detail 外框左线、伸进 minimap 走廊——框线穿字、
                # 压缩略图框角、锥形虚线贴着字形过（ch2 回程紫色标签盲审三连）→ 改锚流线
                # 右侧 'start'。y 不用 max-6（目标行框顶带）：站号徽标骑框顶上探 8px
                # （tab 式），y2-6 的字底正好落进徽标矩形里（ch2 第16站徽标穿 PULL 字腰）——
                # 改贴**源框底边之内** 6px（center 框底自带 56px 内衬净空，无内容冲突）
                lc.text(x + 7, y1 - 6, label, 8.5, color, 'start', maxw=640, tag='L2fl:' + label[:10])
            else:
                # 'end' 锚左伸同样可能扫进**目标行**的站号徽标（ch3 盲审②：worker 标签
                # 左端压进工厂②顶上第16站徽标下部 ~7px——徽标骑框顶 y-8..y+9，y2-6 基线
                # 的降部正落该带，与左溢出分支的 PULL 字腰同一类）。沟内标签必须贴箭头
                # 某一侧（x 区间跨过箭头的标签会被箭头线从字中穿过）：左端撞徽标 → 改贴
                # 右侧 'start'；右侧也撞徽标或出框右缘 → 退到「贴源框底边之内」兜底。
                lw = lc.tw(label, 8.5)
                bb_end = (x - 9 - lw, mid - 8.8, x - 5, mid + 3.7)
                if not any(ov(bb_end, bd) for bd in badges):
                    lc.text(x - 7, mid, label, 8.5, color, 'end', maxw=640, tag='L2fl:' + label[:10])
                else:
                    bb_start = (x + 5, mid - 8.8, x + 9 + lw, mid + 3.7)
                    if (bb_start[2] <= R['frame']['x'] + R['frame']['w'] - 12
                            and not any(ov(bb_start, bd) for bd in badges)):
                        lc.text(x + 7, mid, label, 8.5, color, 'start', maxw=640, tag='L2fl:' + label[:10])
                    else:
                        lc.text(x + 7, y1 - 6, label, 8.5, color, 'start', maxw=640, tag='L2fl:' + label[:10])


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
    # 行下方下沉说明行的专属净空：north 行有标签装不进间隙要下沉时，north→center 行距
    # 加宽 16px——否则 18px 行间带同时挤「下沉说明行」和「跨行纵向流标签」两拨文字
    # （ch2：下沉 ROUTER→DEALER × 纵向流长标签 input_queue→scheduler… 相撞）
    # 判据与 draw_flow 的下沉阈值同一条（tw > gap-6），不再用 >60 的粗代理——
    # 否则短标签（msgpack 帧 47.9px）下沉了、行距却不加宽，两拨文字贴上。
    north_pos = {c['name']: (x, w) for c, x, w in zip(north, xs, ws)}

    def _sinks(pos, f):
        if not (f.get('label') and f['from'] in pos and f['to'] in pos):
            return False
        (ax, aw), (bx, bw) = pos[f['from']], pos[f['to']]
        x1, x2 = (ax + aw, bx) if bx > ax else (ax, bx + bw)
        return lc.tw(f['label'], 8.5) > (x2 - x1) - 6

    has_sunk_north = any(_sinks(north_pos, f) for f in spec.get('flows', []))
    cfy = north_bottom + ROW_GAP + (16 if has_sunk_north else 0)
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

    # 南行下沉说明行 vs frame 底边的净空（ch2 盲审：collector.put 落在 generate() 框下方，
    # 「p」下伸笔止于 y=1455、橙色外框线起于 y=1456——0px 贴线，违全图 6px 净空纪律）。
    # 标签上移无解：字顶距行底本就只剩 ~6.5px，上移 6-8px 换成上方贴线 → 让 frame 底边
    # 让位 +8px——字底距框线 8.5px、字顶距行底 6.5px，双向达标。
    south_pos = {c['name']: (x, w) for c, x, w in zip(south, sx, sw_)} if south else {}
    has_sunk_south = any(_sinks(south_pos, f) for f in spec.get('flows', []))
    frame_h = south_bottom + INSET + (8 if has_sunk_south else 0) - ct
    R['frame'] = dict(x=dx0, y=ct, w=dw, h=frame_h)
    if ccfg.get('name'):
        R[ccfg['name']] = dict(x=ix, y=cfy, w=iw, h=cf_h)

    # ---- 流 + 回环（全部 rect 已知后画，浮于组件之上） ----
    # 站号徽标矩形（含 ±2 余量，与 lc.rect 发射的 bbox 同口径）：draw_flow 的沟内标签
    # 避撞用。徽标只画在 comp（queue/note 无）右上、骑框顶 y-8..y+9（tab 式）。
    badges = []
    for c in comps:
        # 徽标由 stations 驱动（exp-2026-08-16 ch3 盲审：note 类带 stations 也必须画——
        # 站号轨道断轨=读者沿账本找不到挂点；无 stations 才不画，kind 只控框样式）
        bd = badge_text(c.get('stations', []))
        if bd and c['name'] in R:
            r = R[c['name']]
            bw = 14 + 10 * len(bd)
            badges.append((r['x'] + r['w'] - bw - 8, r['y'] - 10, r['x'] + r['w'] - 4, r['y'] + 11))
    note_names = {c['name'] for c in comps if c.get('kind') == 'note'}
    for f in spec.get('flows', []):
        draw_flow(f, R, zone_of, captions, badges, note_names)
    if nb > 1 and spec.get('loop'):
        # 回环两端锚到首/末拍片的**实际**底边（R 里是真实 rect）——此前统一用 chips_bottom
        # （=最高拍片底），拍片高矮不齐时矮片那端悬空在框内（arrow-inside，ch2 ②-⑦ 高 75..108）
        b1 = R[center[0]['name']]['y'] + R[center[0]['name']]['h']
        bn = R[center[-1]['name']]['y'] + R[center[-1]['name']]['h']
        c1 = ix + cw_each / 2
        cn = ix + (nb - 1) * (cw_each + CHIP_GAP) + cw_each / 2
        ch_y = max(b1, bn) + 32
        lc.parrow([(cn, bn), (cn, ch_y), (c1, ch_y), (c1, b1)],
                  lc.C_MUTE, 1.4, 'std')
        lc.text((c1 + cn) / 2, ch_y + 14, spec['loop']['label'], 8.5, lc.C_MUTE,
                'middle', maxw=iw - 40, tag='L2:loop')
    # 下沉说明行：按源行落位（拍片行沉到拍片底+14；north/south 行沉到该行底+13，
    # 均在行间净空带内、不越 frame 底边 frame_h=south_bottom+INSET）
    cap_y = {'center': chips_bottom + 14, 'north': north_bottom + 13,
             'south': south_bottom + 13}
    for gx, lab, zone in captions:
        lc.text(gx, cap_y[zone], lab, 8.5, lc.C_MUTE, 'middle',
                maxw=(cw_each + CHIP_GAP) if zone == 'center' else 260,
                tag='L2cap:' + lab[:10])
    s_flows = drain()

    # ---- 容器框（画序=层序：进程框最底 → center 框 → 组件 → 流 → 外框标题） ----
    cf_rect = lc.rect_svg(ix, cfy, iw, cf_h, '#ffffff', lc.C_ENG_S, rx=8, sw=1.8, dash=False)
    if ccfg:
        lc.text(ix + 12, cfy + 19, ccfg.get('title', ''), 11.5, lc.C_ENG_S, 'start', True,
                maxw=iw - 360, tag='L2:center-title')
        if ccfg.get('where'):
            # where 图注基线 cfy+19 时，字底（+3.75）与首排拍片徽标顶（chips_y-10=cfy+18）
            # 的 y 带必叠 4.75px——x 区间相触即相撞（ch3 盲审②轻碰：第12站徽标上边框切
            # 「py:L972」降部 3-4px）。相触时整体上移 7px 进框顶内留白：字底距徽标顶
            # 2.25px+、字顶距框顶线 1.95px+，双向达标（干净位相不动，ch2/ch9 存量稳定）。
            wh_y = cfy + 19
            wh_bb = (ix + iw - 14 - lc.tw(ccfg['where'], 9), cfy + 9.85,
                     ix + iw - 10, cfy + 22.75)
            if any(ov(wh_bb, bd) for bd in badges):
                wh_y = cfy + 12
            lc.text(ix + iw - 12, wh_y, ccfg['where'], 9, lc.C_FAINT, 'end',
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
    # 站号轨道（minimap 下方）——先于锥形线画：轨道文字 bbox 供下支穿行判定
    sts = spec.get('stations', [])
    if sts:
        rail_y = mmy + mmh + 18
        # 标题 spec 驱动（exp-2026-08-16 ch3 盲审：启动装配章的站号语义 ≠ 请求流经，
        # 硬编码与读图行打架；无 rail_header 字段回落默认措辞）
        rail_cap = spec.get('rail_header') or f'本章站号 = 请求流经顺序（共 {len(sts)} 站）'
        lc.text(mmx, rail_y + 12, rail_cap, 10,
                lc.C_TXT, 'start', True, maxw=MM_W, tag='L2:rail-cap')
        yy = rail_y + 30
        for st in sts:
            bd = f"第{st['n']}站"
            pw = 12 + lc.tw(bd, 8, True) + 8
            lc.rect(mmx, yy - 9, pw, 13, lc.C_BADGE_F, lc.C_ENG_S, rx=6, sw=1.0)
            lc.text(mmx + pw / 2, yy + 1, bd, 8, lc.C_ENG_S, 'middle', True)
            # 站号账本是 dossier 真相源不许删字——超宽换行续写（ch2 第 16 站 439px > 383px）
            rls = wrap_cn(f"{st['where']} · {st['what']}", 8.5, MM_W - pw - 12)[:3]
            for k, rln in enumerate(rls):
                lc.text(mmx + pw + 8, yy + 1 + k * 10.5, rln, 8.5, lc.C_MUTE,
                        'start', maxw=MM_W - pw - 10, tag=f"L2:rail{st['n']}.{k}")
            yy += 16.5 + 10.5 * (len(rls) - 1)
        rail_bottom = yy - 10
    else:
        rail_bottom = mmy + mmh
    rail_bbs = [bb for bb, _ in lc.ELEMS if bb is not None]
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
    # 锥形指示线两支：高亮框右缘 → detail 外框左缘，读作「这块放大成右边」。
    #   ① 贴角（ch3 盲审④顺带）：外框 rx=12 圆角使数学角点悬在可见弧外 ~5px，线端直连
    #     数学角会留缺口（上支实测离框角悬 ~10px@2x）——端点改瞄圆角弧的角向切点、再向
    #     角内（弧心方向）探 TUCK px，视觉钉进角里。
    #   ② 下支穿账本（ch3 盲审②）：站号轨道铺满 minimap 正下方左栏，右下角→外框左下角
    #     的直连线沿途斜切账本正文 5-6 行。撞上轨道文字 → 改走「轨道右缘 ↔ 外框左边框」
    #     的空白竖走廊：自高亮框右缘**中点**出发、更陡斜率落到外框左边框（南行中点高度，
    #     无南行退化为中心框中点）；仍撞（病态长行）把落点 y 钳到「过轨道右缘时已在
    #     文字区顶之上」的安全界再试。
    RX, TUCK = 12, 2.0
    Q = 0.70710678
    k = RX * (1 - Q)

    def _seg_hits(bb, x1, y1, x2, y2):
        # Liang-Barsky 布尔：线段与矩形相交？
        dx, dy = x2 - x1, y2 - y1
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x1 - bb[0]), (dx, bb[2] - x1), (-dy, y1 - bb[1]), (dy, bb[3] - y1)):
            if p == 0:
                if q < 0:
                    return False
            else:
                r = q / p
                if p < 0:
                    t0 = max(t0, r)
                else:
                    t1 = min(t1, r)
                if t0 > t1:
                    return False
        return True

    def _cone(x1, y1, x2, y2):
        chrome.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'stroke="{lc.C_FAINT}" stroke-width="1.6" stroke-dasharray="6,4"/>')

    # 上支：高亮框右上角 → 外框左上角（贴角）
    _cone(ubx1, uby0, dx0 + k + TUCK * Q, ct + k + TUCK * Q)
    # 下支：右下角 → 外框左下角（贴角）；穿账本则改走廊线
    cx2, cy2 = dx0 + k + TUCK * Q, ct + frame_h - k - TUCK * Q
    if not rail_bbs or not any(_seg_hits(bb, ubx1, uby1, cx2, cy2) for bb in rail_bbs):
        _cone(ubx1, uby1, cx2, cy2)
    else:
        ubym = (uby0 + uby1) / 2
        ty = (south_y + south_h / 2) if south else (cfy + cf_h / 2)
        ex = dx0 + TUCK
        if any(_seg_hits(bb, ubx1, ubym, ex, ty) for bb in rail_bbs):
            # 走廊线仍撞实际文字（病态长行）→ 把落点 y 上钳到「线过轨道文字区右缘
            # （mmx+MM_W，bbox 再 +2）时已在区顶之上」的安全界再试。
            xm = mmx + MM_W + 2
            if xm > ubx1:
                ty = min(ty, ubym + (rail_y - 2 - ubym) * (ex - ubx1) / (xm - ubx1))
            if any(_seg_hits(bb, ubx1, ubym, ex, ty) for bb in rail_bbs):
                print('  [warn] L2 锥形下支走廊线仍穿站号轨道文字，需人工核图')
        _cone(ubx1, ubym, ex, ty)
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
