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
NOTE_FIT_TIGHT = 0.928       # note 行 fit_line 有效宽收紧系数（见 comp_draw note 分支头注）
SHARP = "E:/Laboratory/Repo2Book/node_modules/sharp"

_TOK = re.compile(r'[!-~→·]+|\s+|.')   # ascii 词连串成 token（空格处断开），空白自成 token，其余逐字（CJK 友好断行）


def wrap_cn(s, fs, maxw):
    """贪心断行：CJK 逐字、ascii 连串；超长 ascii token **不断词**（见下）。"""
    lines, cur = [], ''
    for tk in _TOK.findall(s):
        if not tk:
            continue
        if lc.tw(tk, fs) > maxw and len(tk) > 1:
            # ASCII 长 token（代码标识符/路径）禁止中间切断——整体独占一行，绘制层按需
            # 缩字号（fit_line）。硬拆曾把 EngineCoreRequestType(bytes(type_frame.buffer))
            # 劈成 bu|ffer、残段再拼进下一行中文注，读作「字节标签判型： ffer))」（ch5 盲审，
            # 方法名完全失去可读性）。CJK 逐字 token 恒单字、宽 ≤ fs，进不了本分支。
            if cur.strip():
                lines.append(cur.rstrip())
            lines.append(tk)
            cur = ''
            continue
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
            # moved 可为空（cut 落在尾部空白上）：先判空再取 moved[-1]——此前 sep 行
            # 在 `if moved` 之前索引 moved[-1]，长 ASCII 串窄框换行时（ch7 why 注
            # 「SSE/accept/add_request」独占行 + 尾行短 token）直接 IndexError 崩渲染。
            # 判空短路 = 非崩溃输入输出逐字节不变。
            if moved:
                # 拼接分隔符：CJK↔CJK 边界原文本无空格（ch2 why 注「班车不是|专车」），
                # 插空格=改动真相源文字；其余（词界）补一个空格还原贪心断行吃掉的词距
                sep = '' if (ord(moved[-1]) > 0x2E80 and ord(merged[-1][0]) > 0x2E80) else ' '
                if lc.tw(moved + sep + merged[-1], fs) <= maxw:
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


def file_lines(c, w):
    """file 行折行判定：floor 缩字后仍放不下（现状=截断丢字，ch7 盲审：「同步面」框底
    灰字路径尾被裁成 vllm/entrypoint…）→ 按词折行保完整路径。判定与 fit_draw 的缩字
    路径逐字镜像（放得下=单行原样/缩字；放不下=wrap_cn 折行），comp_h/comp_draw 共用。
    ASCII 长 token wrap_cn 不切词，两行各按 fit_draw 自缩——只有窄框双路径会进来。"""
    s = c.get('file')
    if not s:
        return []
    maxw = w - 22
    w0 = lc.tw(s, 8.5)
    if w0 <= maxw:
        return [s]
    fs2 = max(0.85 * 8.5, 8.5 * maxw / w0 * 0.999)
    if lc.tw(s, fs2) <= maxw + 0.5:
        return [s]
    return wrap_cn(s, 8.5, maxw)


def comp_h(c, w):
    st = comp_style(c)
    fl = len(file_lines(c, w))
    return st['top'] + len(comp_lines(c, w)) * st['lh'] + (18 + 11.5 * (fl - 1) if fl else 6)


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
        # note 行 fit_line 有效宽收紧 7%（×0.928）：tw 估宽 ASCII=0.58*fs，YaHei 实测全大写+
        # 下划线串 ≈0.61*fs（ch6 盲审：VLLM_DISABLE_REQUEST_ID_RANDOMIZATION 估 182.4 < maxw 184
        # 不进缩字档，实际渲染右缘 1879.0 顶到虚线框线 1878.8 零净空——估宽富余 9.6px 被 ASCII
        # 实际偏宽吃光；rect-rect 盲区、linter 照不到，只能靠亲眼看）。CJK 主导行实测比估宽还窄
        # ~5%、本就够净空；收紧只让估宽贴上限的 ASCII 行提前缩字到与其余行一致内边距（右缘目标
        # ≈ wrap 宽 w-22 = 180 → 12px 净空），不换行、不动 note_h/布局。系数推导：实测该行
        # 实际宽 192.2 / 估宽 182.41 = 1.054 → 需缩到 180/192.2 = 0.936 → eff = 182.41×0.936
        # = 170.8 = 184×0.928。
        for m in c.get('methods', []):
            for ln in wrap_cn(m, 8.5, w - 22):
                fit_line(x + 10, yy, ln, 8.5, lc.C_MUTE, 'start', (w - 18) * NOTE_FIT_TIGHT,
                         'L2nl:' + ln[:10])
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
    # 标题缩字下限 0.80（ch7 盲审：②「output_handler 拉批分块」10px 宽 148.3 > 拍片
    # maxw 122.4，0.83 下限=8.3px 仍 123.1 差 0.2px → 截断成「② output_handler 拉…」
    # 九张站卡唯一残名；0.80 下限放它缩到 8.24px 完整放下，其余章标题放得下 8.3 者不变）。
    fit_draw(x + 12, y + (17 if st['top'] == 30 else 21), c['name'], st['tfs'], st['tcol'],
             'start', True, w - 24, 'L2:' + c['name'][:14], floor=0.80)
    yy = y + st['top'] + 4
    for ln in comp_lines(c, w):
        fit_line(x + 12, yy, ln, st['lfs'], st['lcol'], 'start', w - 22,
                 'L2m:' + ln[:12])
        yy += st['lh']
    fls = file_lines(c, w)
    for k, fl in enumerate(fls):
        fit_draw(x + 12, y + h - 7 - 11.5 * (len(fls) - 1 - k), fl, 8.5, lc.C_FAINT,
                 'start', False, w - 22, 'L2f:' + c['name'][:10], floor=0.85)
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
        fit_line(x + 10, yy, ln, 8.5, lc.C_MUTE, 'start', w - 18,
                 'L2pn:' + ln[:10])
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
        # 比例 ×0.999：留 0.1% 净空。fs*maxw/w 的浮点往返会得到比 maxw 大 ~1e-13 的宽度
        # （ch6 盲审：出港 file 行「…core_client.py」算得缩到 8.23px 恰好放得下、却因
        #  1-ULP 误差判定失败，退回 8.5px 截断成「core_client…」——文件名残缺最扎眼；
        #  另 lc.text 的 fit() 对 ≥maxw 才告警，0.1% 净空同时免掉「need 210 > 210」假告警）。
        fs2 = max(floor * fs, fs * maxw / w * 0.999)
        if lc.tw(s, fs2, bold) <= maxw + 0.5:
            fs, w = fs2, lc.tw(s, fs2, bold)
        else:
            while s and lc.tw(s + '…', fs, bold) > maxw:
                s = s[:-1]
            s += '…'
    lc.text(x, y, s, fs, fill, anchor, bold, maxw=maxw, tag=tag)


def fit_line(x, y, ln, fs, fill, anchor, maxw, tag, floor=0.8):
    """wrap_cn 改为「超长 ascii token 不断词、整体独占一行」后的配套绘制层：
    该行超宽就真缩字号（下限 floor*fs；仍超才走 lc.text 的 fit() 告警）——
    与 fit_draw 同一原则：内容是真相源不许删字，缩字才是正解。"""
    w = lc.tw(ln, fs)
    if w > maxw:
        # ×0.999 留 0.1% 净空（沿 fit_draw 的 1-ULP 修法）：fs*maxw/w 的浮点往返会得到比
        # maxw 大 ~1e-13 的估宽，lc.text 的 fit() 对 ≥maxw 才告警 → 假告警「need 171 > 171」
        # （ch6 note 收紧系数上线后两行命中；与 ch6 出港 file 行的 fit_draw 同病同方）。
        fs = max(floor * fs, fs * maxw / w * 0.999)
    lc.text(x, y, ln, fs, fill, anchor, maxw=maxw, tag=tag)


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


def _contains(outer, inner):
    """outer 矩形严格含 inner（±0.5 容差）——frame/center 容器 vs 其内组件。"""
    return (outer['x'] - 0.5 <= inner['x'] and outer['y'] - 0.5 <= inner['y']
            and inner['x'] + inner['w'] <= outer['x'] + outer['w'] + 0.5
            and inner['y'] + inner['h'] <= outer['y'] + outer['h'] + 0.5)


def _elbow_flow(f, a, b, lane_y, R, badges, note_names, vsegs=None):
    """x 区不重叠的跨行流 → 肘形路由（ch6 盲审立的规矩）。

    旧的「重叠中点垂直直落」在两框 x 区不重叠时，x 落在两框之间的空档——线从
    「正下方无关框」的底边水平处出线、箭头戳进「目标行正下方的另一个无关框」顶边：
    spec 的 from/to 三元组被静默偷换（ch6：⑨→出港 渲成了 ⑤→why·tokenize、
    ⑧→线格式 渲成 ⑥→why·不许阻塞、⑦→mm 特征 渲成 ⑧→why·双轨，出港框零入向箭头）。
    改为按 spec 的真实源/目标框走肘形：源框边中心下探 → 行间通道横走（lane_y，
    build 侧按目标 x 分道）→ 入目标框对边中心。"""
    down = b['y'] > a['y']
    color = lc.C_ENG_S if f.get('up') else ROLE.get(f.get('color_role') or 'plain', lc.C_MUTE)
    dash = bool(f.get('dash'))
    ex = a['x'] + a['w'] / 2                    # 源框（下探方向的）边中心出线
    tx = b['x'] + b['w'] / 2                    # 目标框对边中心入线
    if down:
        y0, y1 = a['y'] + a['h'], b['y']
        # 入点避让目标框顶的站号徽标（骑框顶 y-8..y+9 tab 式，build 侧 badges 同口径）
        for bd in badges:
            if b['y'] - 12 < bd[3] and bd[1] < b['y'] + 12 and bd[0] - 6 <= tx <= bd[2] + 6:
                if bd[0] - 8 >= b['x'] + 10:
                    tx = bd[0] - 8
                elif bd[2] + 8 <= b['x'] + b['w'] - 10:
                    tx = bd[2] + 8
        if f['to'] in note_names:               # 虚线框落点相位（与直落分支同一招）
            p = (tx - b['x']) % 10
            if not 0.5 <= p <= 2.5:
                d = (1.5 - p) % 10
                if d > 5:
                    d -= 10
                tx += d
    else:                                       # 上行（南→中）：源框顶中心出、目标框底入
        y0, y1 = a['y'], b['y'] + b['h']
    if vsegs is not None:                       # 两竖段记档——下沉说明行避让用（见 draw_flow 头注）
        vsegs.append((ex, y0, lane_y))
        vsegs.append((tx, lane_y, y1))
    lc.parrow([(ex, y0), (ex, lane_y), (tx, lane_y), (tx, y1)],
              color, 1.8, 'up' if f.get('up') else 'std', dash)
    label = f.get('label')
    if label:
        # 标签优先贴「源框出线竖段」左（段够长 ≥40px 时，读作『从这出、往那去』）；
        # 出线段太短（行间带窄）则退到通道横线的目标端上方。左越 frame 内界 → 改右侧。
        if abs(lane_y - y0) >= 40:
            lx, ly, anchor = ex - 7, (y0 + lane_y) / 2 + 3, 'end'
            if lx - lc.tw(label, 8.5) < R['frame']['x'] + 12:
                lx, anchor = ex + 7, 'start'
        else:
            lx, ly, anchor = tx - 7, lane_y - 5, 'end'
            if lx - lc.tw(label, 8.5) < R['frame']['x'] + 12:
                lx, anchor = tx + 7, 'start'
        lc.text(lx, ly, label, 8.5, color, anchor, maxw=640, tag='L2fl:' + label[:10])


def draw_flow(f, R, zone_of, captions, badges=(), note_names=(), lanes=None, vsegs=None):
    a, b = R[f['from']], R[f['to']]
    color = ROLE.get(f.get('color_role') or 'plain', lc.C_MUTE)
    dash = bool(f.get('dash'))
    label = f.get('label')
    # 容器流（frame/center 容器 ⇄ 其内组件）不得横穿组件框——ch5 盲审：⑥→frame 命中下方
    # 「同行→横向」分类，线段从 beat 框左缘 (x=1922.3) 横穿整框、从框内方法名文字正中间
    # 穿过（删除线效果）、再穿出右框线落到 frame 边 (x=2176)。改为**出场/入场短刺**：从
    # 组件框离容器最近的边出发、刺到容器对应边，箭头指向流方向（组件→容器 = 交还给容器
    # 边上的下一站，如 ⑥ PULL 到家 → output_handler 出拍）。
    if _contains(b, a) or _contains(a, b):
        inner, outer = (a, b) if _contains(b, a) else (b, a)
        gaps = {'right': outer['x'] + outer['w'] - (inner['x'] + inner['w']),
                'left': inner['x'] - outer['x'],
                'top': inner['y'] - outer['y'],
                'bottom': outer['y'] + outer['h'] - (inner['y'] + inner['h'])}
        side = min(gaps, key=gaps.get)
        yc, xc = inner['y'] + inner['h'] / 2, inner['x'] + inner['w'] / 2
        p_in, p_out = {'right':  ((inner['x'] + inner['w'], yc), (outer['x'] + outer['w'], yc)),
                       'left':   ((inner['x'], yc), (outer['x'], yc)),
                       'top':    ((xc, inner['y']), (xc, outer['y'])),
                       'bottom': ((xc, inner['y'] + inner['h']),
                                  (xc, outer['y'] + outer['h']))}[side]
        # 贴边零长刺防护（ch7 盲审④：①→uplink 首拍片左缘与容器左线 gap=0，p_in==p_out
        # 的零长线只剩 marker-end 渲染——白缝里凭空一枚无杆深蓝箭簇，系列左缘扫描五图
        # 均无此件）。刺长 <3px 连箭簇身位（std 箭簇 ≈10.8px）都撑不满，画了=悬空箭簇，
        # 不画；caption 照发（语义注记不依赖这根刺）。
        if abs(p_in[0] - p_out[0]) >= 3 or abs(p_in[1] - p_out[1]) >= 3:
            if inner is a:                  # 组件 → 容器：刺向容器边
                lc.seg(p_in[0], p_in[1], p_out[0], p_out[1],
                       lc.C_ENG_S if f.get('up') else color, 1.8,
                       'up' if f.get('up') else 'std', dash)
            else:                           # 容器 → 组件：从容器边刺入
                lc.seg(p_out[0], p_out[1], p_in[0], p_in[1], color, 1.8, 'std', dash)
            if side in ('top', 'bottom') and vsegs is not None:
                vsegs.append((xc, p_in[1], p_out[1]))
        if label:
            captions.append(((p_in[0] + p_out[0]) / 2, label,
                             zone_of.get(f['from'] if inner is a else f['to'])))
        return
    oy0, oy1 = max(a['y'], b['y']), min(a['y'] + a['h'], b['y'] + b['h'])
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
        # 两框 x 区不重叠 → 落点必在框外空档，直落=偷换 from/to（见 _elbow_flow 头注）→ 肘形
        if lanes and id(f) in lanes and not (
                a['x'] <= x <= a['x'] + a['w'] and b['x'] <= x <= b['x'] + b['w']):
            _elbow_flow(f, a, b, lanes[id(f)], R, badges, note_names, vsegs)
            return
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
        if vsegs is not None:               # 直落竖段记档——下沉说明行避让用（见下方 captions 块）
            vsegs.append((x, y1, y2))
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
                        # 上行流（目标在源上方）的 y1-6 = 源框顶带，正是站号徽标骑框顶的
                        # y 带——兜底位照旧=与徽标字形相压（ch7：「yield RequestOutput
                        # （DELTA）」压「第13站」下半）→ 改贴目标框底之下 13px 的行间
                        # 走廊（走廊预算见 build 的直落流加宽；下行流 y1-6=源框底内衬，
                        # 原位本就安全，不动）。
                        ly = y2 + 13 if y2 < y1 else y1 - 6
                        lc.text(x + 7, ly, label, 8.5, color, 'start', maxw=640, tag='L2fl:' + label[:10])


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
    # zone_of 先行（肘形通道预检要在画拍片/南行**之前**跑，那时还没到逐框填 zone 的时点）
    R, zone_of, captions = {}, {c['name']: c.get('zone') for c in comps}, []    # captions: [(gap_center_x, label)] 拍片间标签（下沉说明行）

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

    # ---- 跨行肘形流预检（画拍片/南行之前：车道分道 + 必要时加高行距） ----
    # 检测口径与 draw_flow 肘形分支同一条：跨行且「重叠中点」不在两框 x 区内。
    flows = spec.get('flows', [])
    chip_pos = {c['name']: (ix + i * (cw_each + CHIP_GAP), cw_each)
                for i, c in enumerate(center)}
    south_pos0 = {}
    if south:
        sx, sw_, _ = row_layout(south, ix, iw)
        south_pos0 = {c['name']: (x, w) for c, x, w in zip(south, sx, sw_)}

    def _xw(name):
        if name in chip_pos:
            return chip_pos[name]
        if name in south_pos0:
            return south_pos0[name]
        r = R.get(name)
        return (r['x'], r['w']) if r else (None, None)

    def _elbow_q(f):
        if f['from'] not in zone_of or f['to'] not in zone_of:
            return False                     # frame/center 容器伪节点 → 出/入场短刺分支管
        if zone_of[f['from']] == zone_of[f['to']]:
            return False                     # 同行 → 横向
        (ax, aw), (bx, bw) = _xw(f['from']), _xw(f['to'])
        if ax is None or bx is None:
            return False
        x = (max(ax, bx) + min(ax + aw, bx + bw)) / 2
        return not (ax <= x <= ax + aw and bx <= x <= bx + bw)

    lanes = {}

    def _stack(pool, bottom_lane, step=16):
        """目标 x 升序分道：目标最靠目标行起始侧（下行=最左）走最贴近目标行的道，
        其余逐层抬高——远目标走低道，别的流的入线竖段不会横穿它的道。"""
        for i, f in enumerate(sorted(pool, key=lambda t: _xw(t['to'])[0] + _xw(t['to'])[1] / 2)):
            lanes[id(f)] = bottom_lane - step * i

    # north↔center 肘形：车道压进行间带 [north_bottom, cfy]；多道时把行距撑成车道带
    nc = [f for f in flows if _elbow_q(f)
          and {zone_of[f['from']], zone_of[f['to']]} == {'north', 'center'}]
    if len(nc) > 1:
        cfy = max(cfy, north_bottom + 15 + 16 * (len(nc) - 1))
        chips_y = cfy + cf_pad_top
        chips_bottom = chips_y + chip_h
    if nc:
        _stack(nc, cfy - 9)
    # 直落跨行流的标签走廊预算：north↔center 直落（非肘形）流带标签时，其标签若两侧
    # 都被站号徽标占住（draw_flow 沟内标签三连兜底的最后一档），唯一去处是 north↔center
    # 行间走廊——但走廊默认 18px 放不下一行 8.5px 标签（ch7 盲审：「yield RequestOutput
    # （DELTA）」旧兜底位贴源框顶 y1-6，正落站号徽标 y 带、与「第13站」字形相压；改落
    # 走廊后须 ≥31px 才留得下上下各 3px 净空）。无此形态的章 cfy 不动（触发式，逐字节不变）。
    if any(f.get('label') and not _elbow_q(f)
           and {zone_of[f['from']], zone_of[f['to']]} == {'north', 'center'}
           for f in flows if f['from'] in zone_of and f['to'] in zone_of):
        cfy = max(cfy, north_bottom + 31)
        chips_y = cfy + cf_pad_top
        chips_bottom = chips_y + chip_h

    # ---- center 拍片 ----
    for i, c in enumerate(center):
        comp_draw(ix + i * (cw_each + CHIP_GAP), chips_y, cw_each, c, R)
    s_center = drain()

    # ---- south 行（center↔south 有肘形流时，行距加高成泳道通道） ----
    cs = [f for f in flows if _elbow_q(f)
          and {zone_of[f['from']], zone_of[f['to']]} == {'center', 'south'}]
    south_h = 0
    if south:
        south_y = cfy + cf_h + ROW_GAP
        if cs:
            # 泳道带 = 中心容器底边(+8，避开下沉说明行) … 南行框顶(-14，让过骑框顶徽标
            # 的 y-10 带并留 4px)；行距不足以容纳时加高
            south_y = max(south_y, cfy + cf_h + 8 + 16 * (len(cs) - 1) + 28)
            _stack(cs, south_y - 14)
        for c, x, w in zip(south, sx, sw_):
            south_h = max(south_h, comp_draw(x, south_y, w, c, R))
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
    vsegs = []      # 跨行流竖段 (x, y0, y1)——直落竖段/肘形两竖段/上下向容器刺，供下沉说明行避让
    for f in flows:
        draw_flow(f, R, zone_of, captions, badges, note_names, lanes, vsegs)
    loop_vs = []          # 回环两竖段 (x, y0, y1)——下沉说明行避让用（下方 captions 块）
    if nb > 1 and spec.get('loop'):
        # 回环两端锚到首/末拍片的**实际**底边（R 里是真实 rect）——此前统一用 chips_bottom
        # （=最高拍片底），拍片高矮不齐时矮片那端悬空在框内（arrow-inside，ch2 ②-⑦ 高 75..108）
        b1 = R[center[0]['name']]['y'] + R[center[0]['name']]['h']
        bn = R[center[-1]['name']]['y'] + R[center[-1]['name']]['h']
        c1 = ix + cw_each / 2
        cn = ix + (nb - 1) * (cw_each + CHIP_GAP) + cw_each / 2
        ch_y = max(b1, bn) + 32
        # 回环横线还须让过肘形流标签：标签贴「源框底→车道」竖段中腰（_elbow_flow），
        # 拍片高矮不齐时 max(b1,bn)+32 可能落进标签字带正中（ch7：横线 y539 从
        # 「list.append 分支（queue=None）」整行字中横穿、上邻「put 的合并语义」下缘相触）
        # → 抬到全部肘形标签字底 +5px 之上。y0/ly 与 _elbow_flow 的标签落位逐字镜像。
        for f in flows:
            if id(f) not in lanes or not f.get('label'):
                continue
            a = R[f['from']]
            y0 = (a['y'] + a['h']) if R[f['to']]['y'] > a['y'] else a['y']
            lane = lanes[id(f)]
            ly = (y0 + lane) / 2 + 3 if abs(lane - y0) >= 40 else lane - 5
            ch_y = max(ch_y, ly + 3.7 + 5)
        # 回环横线+label 对容器底边钳位（exp-2026-08-18 ch7 盲审：ch_y+14 的 label 落到
        # 容器底边线下 3.7px，橙线从字腰穿过呈删除线效果）。整体上移（竖段随 ch_y 变短），
        # 保证 label baseline 距最近的容器底边 ≥6px 净空。
        _bot = ct + frame_h - 6                      # frame 底边内 6px
        if ccfg.get('name'):
            _bot = min(_bot, cfy + cf_h - 6)         # center 容器底边（更近者）
        if ch_y + 14 > _bot:
            ch_y = _bot - 14
        lc.parrow([(cn, bn), (cn, ch_y), (c1, ch_y), (c1, b1)],
                  lc.C_MUTE, 1.4, 'std')
        lc.text((c1 + cn) / 2, ch_y + 14, spec['loop']['label'], 8.5, lc.C_MUTE,
                'middle', maxw=iw - 40, tag='L2:loop')
        loop_vs = [(cn, bn, ch_y), (c1, ch_y, b1)]
    # 下沉说明行：按源行落位（拍片行沉到拍片底+14；north/south 行沉到该行底+13，
    # 均在行间净空带内、不越 frame 底边 frame_h=south_bottom+INSET）。
    # 同行防撞（ch5 盲审：「PUSH→PULL 回程」与「get_output_async → output_handler…」两条
    # middle 标签首尾相接、字形相碰，整行融合成一句胡话）+ 出场短刺标签出血（居中
    # x=2168 越出 frame 右缘）：同行标签先各自钳进 frame 内，再自右向左保证相邻 ≥18px
    # （≈2 个汉字宽）净空；frame 内实在放不下才把后者沉到第二行（说明行带 56px 内衬容得下）。
    # 无越界/无相撞的章输出逐字节不变（钳位/推移仅在触发条件命中时改坐标）。
    cap_y = {'center': chips_bottom + 14, 'north': north_bottom + 13,
             'south': south_bottom + 13}
    caps = [[gx, lab, zone, 1] for gx, lab, zone in captions]   # [center_x, label, zone, row]
    lo, hi = dx0 + 10, dx0 + dw - 10
    zidx = {}
    for i, c in enumerate(caps):
        zidx.setdefault(c[2], []).append(i)
    # 竖段避让清单 = 回环两竖段 + 跨行流竖段（直落/肘形/上下向刺）：后者是 R3 补的
    # （ch7 盲审②：卡④→detokenizer 工厂的虚线直落竖段 x≈1108 正穿 ③→④ 说明行
    # 「new_token_ids + finish_reason」字腰——R2 只给回环竖段立了避让，跨行流竖段
    # 漏网，与回环竖段同一类「竖线穿说明行字带」）。y 带过滤在循环内按 zone 判。
    avoid_vs = list(loop_vs) + list(vsegs)
    for zone, idxs in zidx.items():
        for i in idxs:                       # ① 各自钳进 frame 内
            w = lc.tw(caps[i][1], 8.5)
            caps[i][0] = (min(max(caps[i][0], lo + w / 2), hi - w / 2)
                          if w <= hi - lo else (lo + hi) / 2)
            # 回环竖段避让（ch7 盲审：右竖段 x≈2087 从「ABORT 帧过线停算（→ ch5）」的
            # 「帧」字正中穿过；左竖段同理可穿行首说明行）：说明行字带与竖段 y 带相交且
            # span 跨线 → 让到线某一侧（优先标签原侧；原侧出 frame 内界才换侧），净距 9px。
            for vx, va, vb in avoid_vs:
                vy0, vy1 = min(va, vb) - 2, max(va, vb) + 2
                if not (vy0 <= cap_y[zone] + 4.7 and cap_y[zone] - 9 <= vy1):
                    continue
                if caps[i][0] - w / 2 >= vx - 9 or caps[i][0] + w / 2 <= vx + 9:
                    continue                 # 本就不跨线（含 9px 净距已够）
                right_c, left_c = vx + 9 + w / 2, vx - 9 - w / 2
                if caps[i][0] >= vx and right_c + w / 2 <= hi:
                    caps[i][0] = max(caps[i][0], right_c)
                elif left_c - w / 2 >= lo:
                    caps[i][0] = min(caps[i][0], left_c)
                elif right_c + w / 2 <= hi:
                    caps[i][0] = right_c
                else:
                    caps[i][0] = left_c
        order = sorted(range(len(idxs)), key=lambda k: caps[idxs[k]][0])
        for _ in range(8):                   # ② 自右向左级联左推，保相邻 ≥18px
            moved = False
            for j in range(len(order) - 2, -1, -1):
                i, i2 = idxs[order[j]], idxs[order[j + 1]]
                w0 = lc.tw(caps[i][1], 8.5)
                lim = caps[i2][0] - lc.tw(caps[i2][1], 8.5) / 2 - 18
                if caps[i][0] + w0 / 2 > lim + 0.01:
                    nc = max(lo + w0 / 2, lim - w0 / 2)
                    if nc < caps[i][0] - 0.01:
                        caps[i][0] = nc
                        moved = True
            if not moved:
                break
        for j in range(1, len(order)):       # ③ 仍相撞（frame 放不下）→ 后者沉第二行
            i, ip = idxs[order[j]], idxs[order[j - 1]]
            if (caps[i][0] - lc.tw(caps[i][1], 8.5) / 2
                    < caps[ip][0] + lc.tw(caps[ip][1], 8.5) / 2 + 4 and caps[ip][3] == 1):
                caps[i][3] = 2
                print(f'  [warn] L2 下沉说明行仍相撞，『{caps[i][1][:16]}』沉第二行——需人工核图')
    for gx, lab, zone, row in caps:
        lc.text(gx, cap_y[zone] + 11 * (row - 1), lab, 8.5, lc.C_MUTE, 'middle',
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
            # 站号账本是 dossier 真相源不许删字——超宽换行续写（ch2 第 16 站 439px > 383px）。
            # exp-2026-08-16（ch5 第 3 站）：原 [:3] 截断让超长账本尾部静默丢失（该站英文引文
            # 掉了 'input back' 尾 10 字符）——与本行注释「不许删字」自相矛盾。改为全量换行：
            # 轨高与画布 H 本就随行数自增；既有章（ch2/3/4/9 全部 ≤3 行）输出逐字节不变。
            rls = wrap_cn(f"{st['where']} · {st['what']}", 8.5, MM_W - pw - 12)
            for k, rln in enumerate(rls):
                fit_line(mmx + pw + 8, yy + 1 + k * 10.5, rln, 8.5, lc.C_MUTE,
                         'start', MM_W - pw - 10, f"L2:rail{st['n']}.{k}")
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
