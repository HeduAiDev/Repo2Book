#!/usr/bin/env python3
"""渐进式架构模型图 —— 每章开篇的「模型长到哪一步了」。

figure-spec（本渲染器对每一章生成的图都遵守同一份 spec 骨架）:
  claim     : 「读到第 N 章，你心里的模型已读 K 个子系统；本章新增 <子系统>——
               它在 <阶段> → <组> 之下，走 M 站源码。」
  template  : layout（三层下钻:生命周期主线 → 组 → 子系统+本章走线）
  numbers   : 章号与站数均来自 arch-model.json（← outline-final.json + 各章 dossier.code_spine），
              **零即兴数字**。
  elements  : ①主线阶段 ②当前阶段的组 ③当前组的子系统 ④本章**模块交互图**
              (泳道=目录/节点=文件/边=相邻站的跨文件跳转,站号标在所属模块上)
  caption   : 由 --caption 输出，给结论不描述画面。

认知约束（用户 2026-07-26 定，写死在渲染器里）:
  · 任一层同时可见的兄弟节点 ≤7 —— 超了就必须先在 arch_model.GROUPS 里再抽一层。
  · 走线不是散点、也不是站号清单:画成**模块交互图**——泳道分目录、节点是文件、
    箭头是控制权移交(相邻两站的跨文件跳转),站号标在所属模块上(用户 2026-07-26)。
  · 累积:已读的节点带「第 N 章已读」回指，让新知识接在旧节点上；未讲的留虚线占位。
  · ⚠️ 图面用词:对读者说「已读/未读」,**不说「挂/挂靠」**——挂靠是我们内部的树隐喻,
    读者看到的是「这章我读过没有」(用户 2026-07-26)。

用法:
  python3 scripts/arch_model_figure.py --chapter ch31 --instance vllm --out /path/arch-model.svg
"""
import argparse
import json
import re
import sys
import xml.sax.saxutils as xs
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instance  # noqa: E402
from arch_model import GROUPS, _chapter_index  # noqa: E402


def esc(s):
    return xs.escape(str(s))


def tw(s, fs, bold=False):
    """粗略文本宽度:CJK 按 1.0em，ASCII 按 0.58em；粗体再加 7%（实测不计会溢出框）。"""
    n_cjk = sum(1 for c in str(s) if ord(c) > 0x2E80)
    w = n_cjk * fs + (len(str(s)) - n_cjk) * fs * 0.58
    return w * (1.07 if bold else 1.0)


# 语义色（>2 种语义色 → 必须画图例，见 legend()）
C_BUILT_F, C_BUILT_S = '#dbeafe', '#3b82f6'      # 已读:读者在前面章节读过的
C_CUR_F, C_CUR_S = '#ffedd5', '#f97316'          # 本章展开
C_TODO_F, C_TODO_S = '#f8fafc', '#cbd5e1'        # 未读
C_TXT, C_MUTE = '#0f172a', '#64748b'


def box(L, x, y, w, h, fill, stroke, dash=False, r=7, sw=1.6):
    d = ' stroke-dasharray="5,4"' if dash else ''
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(L, x, y, s, fs=12, fill=C_TXT, anchor='middle', bold=False):
    b = ' font-weight="bold"' if bold else ''
    L.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="{fs}" '
             f'fill="{fill}" text-anchor="{anchor}"{b}>{esc(s)}</text>')


def fit(L, cx, y, s_, box_w, base, fill, bold=False, minfs=7.6):
    """把文本塞进 box_w:先缩字号(到 minfs),仍不够才截断加省略号。杜绝文字溢出框。"""
    fs = base
    while fs > minfs and tw(s_, fs, bold) > box_w:
        fs -= 0.4
    if tw(s_, fs, bold) > box_w:
        keep = max(3, int(len(s_) * box_w / max(1e-6, tw(s_, fs, bold))) - 1)
        s_ = s_[:keep] + '…'
    text(L, cx, y, s_, round(fs, 1), fill, bold=bold)


def state_of(sub, cur_sub, idx):
    if sub['id'] == cur_sub:
        return 'cur'
    return 'built' if _chapter_index(sub['opened_in']) < idx else 'todo'


def colors(st):
    return {'built': (C_BUILT_F, C_BUILT_S), 'cur': (C_CUR_F, C_CUR_S),
            'todo': (C_TODO_F, C_TODO_S)}[st]


def interaction_graph(spine, max_nodes=12):
    """把本章走线变成**模块交互图**的数据（用户 2026-07-26：不要只列站号，要画各模块/类之间
    怎么交互，再把站点标到对应模块上）。

    节点 = 走线经过的**文件**（⚠️ 用完整路径做身份：ch31 里 `vllm/v1/request.py` 与
           `vllm/v1/structured_output/request.py` 同名不同物，按 basename 会被错误合并）。
    边   = **相邻两站跨文件的跳转** —— code_spine 本身就是请求流经代码的顺序，
           所以「站 N → 站 N+1」天然是一次控制权移交，不是我们编的关系。
    泳道 = 节点所在目录（读者先认模块层次，再看交互）。
    """
    nodes = OrderedDict()          # path -> {stations, symbols}
    for i, u in enumerate(spine, 1):
        n = nodes.setdefault(u['path'], {'stations': [], 'symbols': []})
        n['stations'].append(i)
        if u['symbol'] and u['symbol'] not in n['symbols']:
            n['symbols'].append(u['symbol'])
    if len(nodes) > max_nodes:     # 超预算:只保留站数最多的前 N 个（并如实标注，见 dropped）
        keep = sorted(nodes, key=lambda p: -len(nodes[p]['stations']))[:max_nodes]
        dropped = [p for p in nodes if p not in keep]
        nodes = OrderedDict((p, v) for p, v in nodes.items() if p in keep)
    else:
        dropped = []
    edges = OrderedDict()          # (src,dst) -> [发生跳转的站号]
    for a, b in zip(spine, spine[1:]):
        if a['path'] != b['path'] and a['path'] in nodes and b['path'] in nodes:
            edges.setdefault((a['path'], b['path']), []).append(spine.index(b) + 1)
    lanes = OrderedDict()
    for p in nodes:
        lanes.setdefault(p.rsplit('/', 1)[0] if '/' in p else p, []).append(p)
    return nodes, edges, lanes, dropped


def rng(ids):
    """[1,2,3,7] → '1–3, 7'"""
    out, s, p = [], ids[0], ids[0]
    for i in ids[1:]:
        if i == p + 1:
            p = i
            continue
        out.append(f'{s}–{p}' if p > s else f'{s}')
        s = p = i
    out.append(f'{s}–{p}' if p > s else f'{s}')
    return ', '.join(out)


def build(model, cid):
    idx = _chapter_index(cid)
    ch = model['chapters'][cid]
    cur_sub = ch['subsystem']
    cur_stage = ch['parent_stage']
    subs = model['levels']['L2_subsystems']
    stages = model['levels']['L1_stages']
    spine = ch['spine']

    W = 1180
    M = 26
    L = []

    # ---------- Tier 1: 7 个主线阶段 ----------
    y1, h1 = 90, 50
    n = len(stages)
    gap = 7
    bw = (W - 2 * M - gap * (n - 1)) / n
    stage_cx = {}
    for i, st in enumerate(stages):
        x = M + i * (bw + gap)
        mine = [s for s in subs if s['parent_stage'] == st['id']]
        if st['id'] == cur_stage:
            stt = 'cur'
        elif any(_chapter_index(s['opened_in']) < idx for s in mine):
            stt = 'built'
        else:
            stt = 'todo'
        f, s_ = colors(stt)
        box(L, x, y1, bw, h1, f, s_, dash=(stt == 'todo'), sw=2.2 if stt == 'cur' else 1.6)
        text(L, x + bw / 2, y1 + 21, st['name'], 12.5, C_TXT, bold=(stt == 'cur'))
        cnt = len([s for s in mine if _chapter_index(s['opened_in']) < idx])
        tot = len(mine)
        if tot:
            text(L, x + bw / 2, y1 + 38, f'{cnt}/{tot} 已读', 10, C_MUTE)
        stage_cx[st['id']] = x + bw / 2
        if i < n - 1:
            ax = x + bw + 1
            L.append(f'<path d="M{ax:.1f},{y1 + h1 / 2} L{ax + gap - 2:.1f},{y1 + h1 / 2}" '
                     f'stroke="{C_MUTE}" stroke-width="1.2" marker-end="url(#a)"/>')

    text(L, M, y1 - 14, '① 请求生命周期主线（全书不变，读者每章都见）', 11.5, C_MUTE, anchor='start')

    # ---------- Tier 2: 当前阶段的组 ----------
    y2, h2 = 182, 44
    grouped = GROUPS.get(cur_stage, [])
    cur_group = next((g for g in grouped for s in subs
                      if s['id'] == cur_sub and s['id'] in g[2]), None)
    group_cx = {}
    if grouped:
        gs = [g for g in grouped if any(s['id'] in g[2] for s in subs)]
        gw = min(176, (W - 2 * M - 10 * (len(gs) - 1)) / max(1, len(gs)))
        total = len(gs) * gw + 10 * (len(gs) - 1)
        x0 = (W - total) / 2
        for i, (gid, gname, members) in enumerate(gs):
            x = x0 + i * (gw + 10)
            ms = [s for s in subs if s['id'] in members]
            if cur_group and gid == cur_group[0]:
                stt = 'cur'
            elif any(_chapter_index(s['opened_in']) < idx for s in ms):
                stt = 'built'
            else:
                stt = 'todo'
            f, s_ = colors(stt)
            box(L, x, y2, gw, h2, f, s_, dash=(stt == 'todo'), sw=2.2 if stt == 'cur' else 1.4)
            text(L, x + gw / 2, y2 + 19, gname, 12, C_TXT, bold=(stt == 'cur'))
            nb = len([s for s in ms if _chapter_index(s['opened_in']) < idx])
            text(L, x + gw / 2, y2 + 34, f'{nb}/{len(ms)} 已读', 9.5, C_MUTE)
            group_cx[gid] = x + gw / 2
        # 主线 → 组 的下钻连线
        if cur_stage in stage_cx and cur_group:
            L.append(f'<path d="M{stage_cx[cur_stage]:.1f},{y1 + h1} '
                     f'L{stage_cx[cur_stage]:.1f},{y2 - 16} L{group_cx[cur_group[0]]:.1f},{y2 - 16} '
                     f'L{group_cx[cur_group[0]]:.1f},{y2}" fill="none" stroke="{C_CUR_S}" '
                     f'stroke-width="1.8" marker-end="url(#a2)"/>')
        text(L, M, y2 - 26, f'② 「{next(s["name"] for s in stages if s["id"] == cur_stage)}」内部分组'
                            f'（子系统多到一眼看不过来时，先按这层认）', 11.5, C_MUTE, anchor='start')

    # ---------- Tier 3: 当前组的子系统芯片 ----------
    y3, h3 = 268, 40
    members = cur_group[2] if cur_group else [s['id'] for s in subs if s['parent_stage'] == cur_stage]
    ms = [s for s in subs if s['id'] in members]
    ms.sort(key=lambda s: _chapter_index(s['opened_in']))
    cw = 200
    total = len(ms) * cw + 12 * (len(ms) - 1)
    x0 = (W - total) / 2
    cur_cx = W / 2
    for i, s in enumerate(ms):
        x = x0 + i * (cw + 12)
        stt = state_of(s, cur_sub, idx)
        f, s_ = colors(stt)
        box(L, x, y3, cw, h3, f, s_, dash=(stt == 'todo'), sw=2.2 if stt == 'cur' else 1.4)
        text(L, x + cw / 2, y3 + 17, s['name_cn'], 12, C_TXT, bold=(stt == 'cur'))
        tag = ('← 本章展开' if stt == 'cur'
               else (f"第 {s['opened_in'].replace('ch', '')} 章已读" if stt == 'built'
                     else f"第 {s['opened_in'].replace('ch', '')} 章才讲"))
        text(L, x + cw / 2, y3 + 32, tag, 9.5, C_CUR_S if stt == 'cur' else C_MUTE)
        if stt == 'cur':
            cur_cx = x + cw / 2
    if cur_group and cur_group[0] in group_cx:
        L.append(f'<path d="M{group_cx[cur_group[0]]:.1f},{y2 + h2} '
                 f'L{group_cx[cur_group[0]]:.1f},{y3 - 12} L{cur_cx:.1f},{y3 - 12} '
                 f'L{cur_cx:.1f},{y3}" fill="none" stroke="{C_CUR_S}" stroke-width="1.8" '
                 f'marker-end="url(#a2)"/>')

    # ---------- Tier 3': 本章模块交互图（不是站号清单）----------
    # 泳道=目录，节点=文件(完整路径为身份)，边=相邻两站的跨文件跳转，站号标在所属模块上。
    nodes, edges, lanes, dropped = interaction_graph(spine)
    y4 = 350
    text(L, M, y4 - 12,
         f'③ 本章展开：{len(nodes)} 个模块怎么交互（箭头＝控制权移交，边上数字＝第几站；'
         f'站号＝请求流经代码的顺序）', 11.5, C_MUTE, anchor='start')

    LANE_W = 150          # 左侧目录标签宽
    slot_of = {p: i for i, p in enumerate(nodes)}      # 首站顺序即执行顺序 → 左到右
    n_slot = max(1, len(nodes))
    avail = W - 2 * M - LANE_W - 16
    slot_w = avail / n_slot
    nw = min(slot_w - 22, 150)
    lane_h = 78
    panel_h = len(lanes) * lane_h + 34
    box(L, M, y4, W - 2 * M, panel_h, '#fffbeb', C_CUR_S, r=9, sw=1.6)
    L.append(f'<path d="M{cur_cx:.1f},{y3 + h3} L{cur_cx:.1f},{y4}" stroke="{C_CUR_S}" '
             f'stroke-width="1.8" marker-end="url(#a2)"/>')

    pos = {}
    for li, (lane, members) in enumerate(lanes.items()):
        ly_ = y4 + 16 + li * lane_h
        if li % 2 == 0:
            L.append(f'<rect x="{M + 8:.1f}" y="{ly_:.1f}" width="{W - 2 * M - 16:.1f}" '
                     f'height="{lane_h - 6:.1f}" rx="5" fill="#ffffff" opacity="0.62"/>')
        text(L, M + 18, ly_ + lane_h / 2 - 2, lane, 10.5, C_MUTE, anchor='start', bold=True)
        for p in members:
            sl = slot_of[p]
            x = M + LANE_W + 8 + sl * slot_w + (slot_w - nw) / 2
            ny = ly_ + 6
            nh = 44
            box(L, x, ny, nw, nh, '#ffffff', C_CUR_S, r=6, sw=1.5)
            fn = p.rsplit('/', 1)[-1]
            fit(L, x + nw / 2, ny + 15, fn, nw - 12, 10.5, C_TXT, bold=True)
            info = nodes[p]
            sym = info['symbols'][0] if info['symbols'] else ''
            if sym:
                fit(L, x + nw / 2, ny + 27, sym, nw - 12, 8.8, C_MUTE)
            fit(L, x + nw / 2, ny + 40, f'第 {rng(info["stations"])} 站', nw - 12, 9.2, C_CUR_S, bold=True)
            pos[p] = (x, ny, nw, nh, ly_)

    # 边：前向=实线走上沿；回边(回到更早的模块)=虚线走下沿，避免与前向线纠缠
    for (a, b), ids in edges.items():
        if a not in pos or b not in pos:
            continue
        ax, ay, aw, ah, _ = pos[a]
        bx, by, bw_, bh, _ = pos[b]
        lbl = str(ids[0]) if len(ids) == 1 else ','.join(map(str, ids))
        fwd = slot_of[b] > slot_of[a]
        if fwd:
            sx, sy = ax + aw, ay + ah / 2
            ex, ey = bx, by + bh / 2
            midx = (sx + ex) / 2
            cross = abs(sy - ey) > 2
            d = (f'M{sx:.1f},{sy:.1f} L{midx:.1f},{sy:.1f} L{midx:.1f},{ey:.1f} L{ex:.1f},{ey:.1f}'
                 if cross else f'M{sx:.1f},{sy:.1f} L{ex:.1f},{ey:.1f}')
            L.append(f'<path d="{d}" fill="none" stroke="{C_CUR_S}" stroke-width="1.5" '
                     f'marker-end="url(#a2)"/>')
            # 标签放在**竖直段中点右侧**(泳道之间的空隙),不再压到节点标题上
            if cross:
                text(L, midx + 5, (sy + ey) / 2 + 3, lbl, 9, C_CUR_S, anchor='start', bold=True)
            else:
                text(L, midx, sy - 26, lbl, 9, C_CUR_S, bold=True)   # 抬到节点框上方空白
        else:
            sx, sy = ax + aw / 2, ay + ah
            ex, ey = bx + bw_ / 2, by + bh
            same_lane = pos[a][4] == pos[b][4]
            chy = (max(sy, ey) + 14) if same_lane else (y4 + panel_h - 12)
            L.append(f'<path d="M{sx:.1f},{sy:.1f} L{sx:.1f},{chy:.1f} L{ex:.1f},{chy:.1f} '
                     f'L{ex:.1f},{ey:.1f}" fill="none" stroke="{C_CUR_S}" stroke-width="1.3" '
                     f'stroke-dasharray="4,3" marker-end="url(#a2)"/>')
            text(L, (sx + ex) / 2, chy + 10, f'{lbl} 回', 8.8, C_CUR_S)

    if dropped:
        text(L, W - M - 16, y4 + panel_h - 7,
             f'（另有 {len(dropped)} 个次要模块未画出）', 9, C_MUTE, anchor='end')

    H = y4 + panel_h + 58
    # ---------- 图例 ----------
    ly = H - 34
    items = [(C_BUILT_F, C_BUILT_S, '已读:前面章节讲过的（带章号）', False),
             (C_CUR_F, C_CUR_S, '本章展开', False),
             (C_TODO_F, C_TODO_S, '未读:后续章节才讲', True)]
    lx = M
    for f, s_, lab, dash in items:
        box(L, lx, ly - 11, 17, 14, f, s_, dash=dash, r=3, sw=1.3)
        text(L, lx + 23, ly, lab, 10.5, C_MUTE, anchor='start')
        lx += 30 + tw(lab, 10.5)

    head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H:.0f}">',
            '<defs>'
            f'<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
            f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_MUTE}"/></marker>'
            f'<marker id="a2" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
            f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_CUR_S}"/></marker>'
            '</defs>',
            f'<rect width="{W}" height="{H:.0f}" fill="white"/>']
    n_built = len([s for s in subs if _chapter_index(s['opened_in']) < idx])
    title = (f"你的架构模型读到第 {idx} 章：已读 {n_built} 个子系统，本章新增「"
             f"{next(s['name_cn'] for s in subs if s['id'] == cur_sub)}」")
    head.append(f'<text x="{M}" y="34" font-family="sans-serif" font-size="16" fill="{C_TXT}" '
                f'font-weight="bold">{esc(title)}</text>')
    head.append(f'<text x="{M}" y="53" font-family="sans-serif" font-size="11" fill="{C_MUTE}">'
                f'{esc("每章只展开一个子系统：先看它在主线哪一站，再下钻看模块之间怎么交互")}</text>')
    return '\n'.join(head + L + ['</svg>'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chapter', required=True)
    ap.add_argument('--instance', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--caption', action='store_true')
    a = ap.parse_args()
    bd = Path(instance.book_dir(a.instance))
    model = json.load(open(bd / 'cartography' / 'arch-model.json', encoding='utf-8'))
    if a.chapter not in model['chapters']:
        raise SystemExit(f'{a.chapter} 不在模型里')
    svg = build(model, a.chapter)
    Path(a.out).write_text(svg, encoding='utf-8')
    print(f'✓ {a.out}')
    if a.caption:
        ch = model['chapters'][a.chapter]
        subs = model['levels']['L2_subsystems']
        idx = _chapter_index(a.chapter)
        nb = len([s for s in subs if _chapter_index(s['opened_in']) < idx])
        nm = next(s['name_cn'] for s in subs if s['id'] == ch['subsystem'])
        print(f'CAPTION: 读到这里，模型上已读 {nb} 个子系统；本章新增「{nm}」，'
              f'它在「{ch["parent_stage"]}」之下，沿 {len(ch["spine"])} 站源码走一遍。')


if __name__ == '__main__':
    main()
