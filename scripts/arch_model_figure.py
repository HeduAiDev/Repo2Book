#!/usr/bin/env python3
"""渐进式架构模型图 —— 每章开篇的「模型长到哪一步了」。

figure-spec（本渲染器对每一章生成的图都遵守同一份 spec 骨架）:
  claim     : 「读到第 N 章，你心里的模型已经建成 K 个子系统；本章把 <子系统> 挂上去——
               它落在 <阶段> → <组> 之下，走 M 站源码。」
  template  : layout（三层下钻:生命周期主线 → 组 → 子系统+本章走线）
  numbers   : 章号与站数均来自 arch-model.json（← outline-final.json + 各章 dossier.code_spine），
              **零即兴数字**。
  elements  : ①7 个主线阶段 ②当前阶段的组 ③当前组的子系统芯片 ④本章走线按目录归并的站组
  caption   : 由 --caption 输出，给结论不描述画面。

认知约束（用户 2026-07-26 定，写死在渲染器里）:
  · 任一层同时可见的兄弟节点 ≤7 —— 超了就必须先在 arch_model.GROUPS 里再抽一层。
  · 走线不是散点:按**目录**归并成站组，读者先看到「在哪个模块」，再看第几站。
  · 累积:已建成的节点带「第 N 章挂上」回指，让新知识挂在旧节点上；未展开的留虚线占位。

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


def tw(s, fs):
    """粗略文本宽度:CJK 按 1.0em，ASCII 按 0.55em。"""
    n_cjk = sum(1 for c in str(s) if ord(c) > 0x2E80)
    return n_cjk * fs + (len(str(s)) - n_cjk) * fs * 0.55


# 语义色（>2 种语义色 → 必须画图例，见 legend()）
C_BUILT_F, C_BUILT_S = '#dbeafe', '#3b82f6'      # 已建成:读者在前面章节挂上的
C_CUR_F, C_CUR_S = '#ffedd5', '#f97316'          # 本章展开
C_TODO_F, C_TODO_S = '#f8fafc', '#cbd5e1'        # 待建
C_TXT, C_MUTE = '#0f172a', '#64748b'


def box(L, x, y, w, h, fill, stroke, dash=False, r=7, sw=1.6):
    d = ' stroke-dasharray="5,4"' if dash else ''
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(L, x, y, s, fs=12, fill=C_TXT, anchor='middle', bold=False):
    b = ' font-weight="bold"' if bold else ''
    L.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="{fs}" '
             f'fill="{fill}" text-anchor="{anchor}"{b}>{esc(s)}</text>')


def state_of(sub, cur_sub, idx):
    if sub['id'] == cur_sub:
        return 'cur'
    return 'built' if _chapter_index(sub['opened_in']) < idx else 'todo'


def colors(st):
    return {'built': (C_BUILT_F, C_BUILT_S), 'cur': (C_CUR_F, C_CUR_S),
            'todo': (C_TODO_F, C_TODO_S)}[st]


def spine_groups(spine, max_groups=7):
    """把本章走线按**目录**归并成站组（读者先认模块，再认站号）。"""
    od = OrderedDict()
    for i, u in enumerate(spine, 1):
        d = u['path'].rsplit('/', 1)[0] if '/' in u['path'] else u['path']
        od.setdefault(d, []).append(i)
    items = list(od.items())
    if len(items) > max_groups:            # 超预算:把尾部并成「其余」
        head = items[:max_groups - 1]
        rest = [i for _, ids in items[max_groups - 1:] for i in ids]
        head.append(('（其余）', rest))
        items = head
    return items


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

    W = 1000
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
            text(L, x + bw / 2, y1 + 38, f'{cnt}/{tot} 已挂', 10, C_MUTE)
        stage_cx[st['id']] = x + bw / 2
        if i < n - 1:
            ax = x + bw + 1
            L.append(f'<path d="M{ax:.1f},{y1 + h1 / 2} L{ax + gap - 2:.1f},{y1 + h1 / 2}" '
                     f'stroke="{C_MUTE}" stroke-width="1.2" marker-end="url(#a)"/>')

    text(L, M, y1 - 14, '① 请求生命周期主线（全书固定 7 个，读者每章都见）', 11.5, C_MUTE, anchor='start')

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
            text(L, x + gw / 2, y2 + 34, f'{nb}/{len(ms)} 已挂', 9.5, C_MUTE)
            group_cx[gid] = x + gw / 2
        # 主线 → 组 的下钻连线
        if cur_stage in stage_cx and cur_group:
            L.append(f'<path d="M{stage_cx[cur_stage]:.1f},{y1 + h1} '
                     f'L{stage_cx[cur_stage]:.1f},{y2 - 16} L{group_cx[cur_group[0]]:.1f},{y2 - 16} '
                     f'L{group_cx[cur_group[0]]:.1f},{y2}" fill="none" stroke="{C_CUR_S}" '
                     f'stroke-width="1.8" marker-end="url(#a2)"/>')
        text(L, M, y2 - 26, f'② 「{next(s["name"] for s in stages if s["id"] == cur_stage)}」内部分组'
                            f'（>7 个子系统必须再抽一层，否则认知过载）', 11.5, C_MUTE, anchor='start')

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
               else (f"第 {s['opened_in'].replace('ch', '')} 章挂上" if stt == 'built'
                     else f"第 {s['opened_in'].replace('ch', '')} 章才展开"))
        text(L, x + cw / 2, y3 + 32, tag, 9.5, C_CUR_S if stt == 'cur' else C_MUTE)
        if stt == 'cur':
            cur_cx = x + cw / 2
    if cur_group and cur_group[0] in group_cx:
        L.append(f'<path d="M{group_cx[cur_group[0]]:.1f},{y2 + h2} '
                 f'L{group_cx[cur_group[0]]:.1f},{y3 - 12} L{cur_cx:.1f},{y3 - 12} '
                 f'L{cur_cx:.1f},{y3}" fill="none" stroke="{C_CUR_S}" stroke-width="1.8" '
                 f'marker-end="url(#a2)"/>')

    # ---------- Tier 4: 本章走线（按目录归并成站组）----------
    sg = spine_groups(spine)
    y4 = 350
    text(L, M, y4 - 12, f'③ 本章走线共 {len(spine)} 站，落在这些模块里（站号即正文出现顺序）',
         11.5, C_MUTE, anchor='start')
    rh, rgap = 34, 7
    panel_h = len(sg) * rh + (len(sg) - 1) * rgap + 20
    box(L, M, y4, W - 2 * M, panel_h, '#fffbeb', C_CUR_S, r=9, sw=1.6)
    L.append(f'<path d="M{cur_cx:.1f},{y3 + h3} L{cur_cx:.1f},{y4}" stroke="{C_CUR_S}" '
             f'stroke-width="1.8" marker-end="url(#a2)"/>')
    for i, (d, ids) in enumerate(sg):
        ry = y4 + 10 + i * (rh + rgap)
        box(L, M + 12, ry, W - 2 * M - 24, rh, '#ffffff', '#fed7aa', r=6, sw=1.2)
        text(L, M + 24, ry + 15, d, 11.5, C_TXT, anchor='start', bold=True)
        first = spine[ids[0] - 1]
        sym = first.get('symbol') or first['path'].rsplit('/', 1)[-1]
        text(L, M + 24, ry + 28, f'第 {rng(ids)} 站 · 起于 {sym}', 10, C_MUTE, anchor='start')
        text(L, W - M - 24, ry + 22, f'{len(ids)} 站', 11, C_CUR_S, anchor='end', bold=True)

    H = y4 + panel_h + 58
    # ---------- 图例 ----------
    ly = H - 34
    items = [(C_BUILT_F, C_BUILT_S, '已建成:前面章节挂上的（带章号）', False),
             (C_CUR_F, C_CUR_S, '本章展开', False),
             (C_TODO_F, C_TODO_S, '待建:后续章节才挂', True)]
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
    title = (f"你的架构模型长到第 {idx} 章：已建成 {n_built} 个子系统，本章挂上「"
             f"{next(s['name_cn'] for s in subs if s['id'] == cur_sub)}」")
    head.append(f'<text x="{M}" y="34" font-family="sans-serif" font-size="16" fill="{C_TXT}" '
                f'font-weight="bold">{esc(title)}</text>')
    head.append(f'<text x="{M}" y="53" font-family="sans-serif" font-size="11" fill="{C_MUTE}">'
                f'{esc("每章只展开一个子系统:先看它挂在主线哪一站,再下钻到源码走线")}</text>')
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
        print(f'CAPTION: 读到这里，模型上已挂好 {nb} 个子系统；本章把「{nm}」挂到'
              f'「{ch["parent_stage"]}」之下，沿 {len(ch["spine"])} 站源码走一遍。')


if __name__ == '__main__':
    main()
