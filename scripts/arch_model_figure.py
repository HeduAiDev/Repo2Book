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
from arch_model import _chapter_index  # noqa: E402


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


def fit(L, cx, y, s_, box_w, base, fill, bold=False, minfs=7.6, anchor='middle'):
    """把文本塞进 box_w:先缩字号(到 minfs),仍不够才截断加省略号。杜绝文字溢出框。"""
    fs = base
    while fs > minfs and tw(s_, fs, bold) > box_w:
        fs -= 0.4
    if tw(s_, fs, bold) > box_w:
        keep = max(3, int(len(s_) * box_w / max(1e-6, tw(s_, fs, bold))) - 1)
        s_ = s_[:keep] + '…'
    text(L, cx, y, s_, round(fs, 1), fill, anchor=anchor, bold=bold)


def state_of(sub, cur_sub, idx):
    if sub['id'] == cur_sub:
        return 'cur'
    return 'built' if _chapter_index(sub['opened_in']) < idx else 'todo'


def colors(st):
    return {'built': (C_BUILT_F, C_BUILT_S), 'cur': (C_CUR_F, C_CUR_S),
            'todo': (C_TODO_F, C_TODO_S)}[st]


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


def pick_classes(model, cid, sub, spine, cap=12):
    """本章展开的组件：**前面章节已铺垫的结构** + **本章新增的结构**。

    取 (a) 本章首讲的类（新结构），(b) 早前章节已讲、但本章走线又停靠到的类（已铺垫、本章要用），
    这样图上天然呈现「在旧结构之上长出新结构」。超 cap 时按「本章站数多寡」保留。
    """
    idx = _chapter_index(cid)
    mine = [c for c in model.get('classes', []) if c['subsystem'] == sub
            and _chapter_index(c['introduced_in']) <= idx]
    st = station_of_classes(mine, spine)
    keep = [c for c in mine if c['introduced_in'] == cid or st.get(c['name'])]
    if len(keep) > cap:
        keep.sort(key=lambda c: (-len(st.get(c['name'], [])),
                                 _chapter_index(c['introduced_in'])))
        keep = keep[:cap]
    keep.sort(key=lambda c: (min(st.get(c['name'], [999])), _chapter_index(c['introduced_in'])))
    return keep, st


def station_of_classes(classes, spine):
    """把本章站点落到**组件**上（不是落到文件上——用户 2026-07-26：不必给每个站点标具体文件、
    把每个站点做成独立模块，那太细、不适合架构图）。

    匹配优先级：符号的类名部分精确命中 > 同文件且符号是该类的方法 > 同文件兜底。
    """
    out = {}
    by_file = {}
    for c in classes:
        by_file.setdefault(c['file'], []).append(c)
    for i, u in enumerate(spine, 1):
        sym = u.get('symbol') or ''
        base = sym.split('.')[0] if sym else ''
        hit = None
        for c in classes:                       # 1) 类名精确/包含命中
            names = [n.strip() for n in re.split(r'[/（(]', c['name']) if n.strip()]
            if base and any(base == n or base in n.split() for n in names):
                hit = c
                break
        if not hit and u['path'] in by_file:    # 2) 同文件兜底
            cands = by_file[u['path']]
            hit = next((c for c in cands
                        if base and base in c['name']), cands[0])
        if hit:
            out.setdefault(hit['name'], []).append(i)
    return out


def short(name, n=30):
    """组件显示名：去掉 (ABC) 之类的括注，'A / B' 只留首个 + 省略号。"""
    s = re.sub(r'\s*\((ABC|旧[^)]*)\)', '', name).strip()
    if '/' in s:
        parts = [p.strip() for p in s.split('/')]
        s = parts[0] + ' 等' if len(parts) > 1 else parts[0]
    s = re.split(r'（', s)[0].strip()
    return s if len(s) <= n else s[:n - 1] + '…'


def cross_links(model, cid, cur_sub):
    """本章组件与**前面章节已读组件**之间的控制权移交 —— 图上唯一的关系箭头。

    只采信 file_owner 里「证据决定性」的文件；证据弱的一律当作属于当前章，宁可不画，
    也不画错一条架构关系（实测按「文件里第一个类」推断会把 engine/core.py 判给
    config-and-wiring，那样画出来的箭头是假的）。
    """
    fo = model.get('file_owner', {})
    ch = model['chapters'][cid]

    def own(u):
        v = fo.get(u['path'])
        return v['subsystem'] if (v and v['decisive']) else cur_sub

    seq = [(i, own(u)) for i, u in enumerate(ch['spine'], 1)]
    out = OrderedDict()
    for (_, a), (j, b) in zip(seq, seq[1:]):
        if a != b and cur_sub in (a, b):
            other = b if a == cur_sub else a
            out.setdefault(other, {'stations': [], 'both': False})
            out[other]['stations'].append(j)
            out[other]['both'] = out[other]['both'] or (a != cur_sub)
    for v in out.values():
        v['stations'] = sorted(set(v['stations']))
    return out


def build(model, cid):
    """渲染第 cid 章的架构剖面。

    形态（用户 2026-07-26 指定，参照其给出的 vLLM 架构图）：**自顶向下的架构分层 + 组件盒**，
    全书同一张图；本章只是它的一个**更细粒度的切片**——本章所属组件就地展开成若干类，
    本章站点标在这些类上。不是文件级调用图。
    """
    idx = _chapter_index(cid)
    ch = model['chapters'][cid]
    cur_sub = ch['subsystem']
    subs = {s['id']: s for s in model['levels']['L2_subsystems']}
    layers = model['levels']['layers']
    spine = ch['spine']

    W, M = 1180, 24
    LBL = 104                       # 左侧层标签宽
    CW, CH_, CG = 132, 34, 9        # 折叠组件盒
    L = []
    y = 92
    boxpos = {}

    cls, st_map = (pick_classes(model, cid, cur_sub, spine) if cur_sub else ([], {}))
    covered = sorted({i for v in st_map.values() for i in v})

    text(L, M, y - 16, '架构分层（自顶向下）｜蓝＝前面章节已读，橙＝本章展开，虚线＝后续章节才讲',
         11.5, C_MUTE, anchor='start')

    RG = 40                         # 右侧关系箭头通道
    inner_w = W - 2 * M - LBL - RG
    for lay in layers:
        members = [subs[s] for s in lay['subsystems'] if s in subs]
        if not members:
            continue
        others = [s for s in members if s['id'] != cur_sub]
        has_cur = any(s['id'] == cur_sub for s in members)
        per_row = max(1, int((inner_w + CG) // (CW + CG)))
        rows = (len(others) + per_row - 1) // per_row if others else 0
        exp_h = 0
        if has_cur and cls:
            ccols = 2 if len(cls) > 5 else 1
            crows = (len(cls) + ccols - 1) // ccols
            exp_h = 26 + crows * 30 + 10
        band_h = max(CH_ + 16, rows * (CH_ + CG) + 8 + exp_h + (8 if exp_h and rows else 0))

        L.append(f'<rect x="{M}" y="{y:.1f}" width="{W - 2 * M}" height="{band_h:.1f}" rx="8" '
                 f'fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>')
        text(L, M + 12, y + 20, lay['name'], 11.5, C_TXT, anchor='start', bold=True)
        text(L, M + 12, y + 34, lay['code'], 9.5, C_MUTE, anchor='start')

        x0 = M + LBL
        for i, s in enumerate(others):
            r, c = divmod(i, per_row)
            x = x0 + c * (CW + CG)
            cy = y + 8 + r * (CH_ + CG)
            built = _chapter_index(s['opened_in']) < idx
            f_, k_ = (C_BUILT_F, C_BUILT_S) if built else (C_TODO_F, C_TODO_S)
            box(L, x, cy, CW, CH_, f_, k_, dash=not built, r=6, sw=1.3)
            fit(L, x + CW / 2, cy + 15, s['name_cn'], CW - 10, 10.5, C_TXT, bold=False)
            n_ch = s['opened_in'].replace('ch', '')
            fit(L, x + CW / 2, cy + 27, (f'第 {n_ch} 章已读' if built else f'第 {n_ch} 章才讲'),
                CW - 10, 8.8, C_MUTE)
            boxpos[s['id']] = (x, cy, CW, CH_, y + band_h)

        if has_cur and cls:
            ey = y + 8 + rows * (CH_ + CG) + (8 if rows else 0)
            ccols = 2 if len(cls) > 5 else 1
            crows = (len(cls) + ccols - 1) // ccols
            chip_w = (inner_w - (ccols - 1) * 10) / ccols
            ew = inner_w
            box(L, x0, ey, ew, exp_h, '#fff7ed', C_CUR_S, r=7, sw=2.0)
            boxpos[cur_sub] = (x0, ey, ew, exp_h, y + band_h)
            cur_name = subs[cur_sub]['name_cn']
            text(L, x0 + 12, ey + 17, f'{cur_name}　← 本章展开', 11.5, C_CUR_S,
                 anchor='start', bold=True)
            text(L, x0 + ew - 12, ey + 17,
                 f'本章 {len(spine)} 站，其中 {len(covered)} 站落在下列组件上',
                 9.5, C_MUTE, anchor='end')
            for i, c in enumerate(cls):
                cc, cr = (i % ccols, i // ccols) if ccols > 1 else (0, i)
                cx = x0 + 10 + cc * (chip_w + 10)
                cy2 = ey + 26 + cr * 30
                isnew = c['introduced_in'] == cid
                f_, k_ = (C_CUR_F, C_CUR_S) if isnew else (C_BUILT_F, C_BUILT_S)
                box(L, cx, cy2, chip_w - 20, 26, f_, k_, r=5, sw=1.3)
                ids = st_map.get(c['name'], [])
                badge = f'第 {rng(ids)} 站' if ids else ''
                bw_ = tw(badge, 9, True) + 10 if badge else 0
                fit(L, cx + 8, cy2 + 17, short(c['name']), chip_w - 40 - bw_, 10, C_TXT,
                    bold=isnew, anchor='start')
                if badge:
                    text(L, cx + chip_w - 28, cy2 + 17, badge, 9, C_CUR_S, anchor='end', bold=True)
                if not isnew:
                    text(L, cx + chip_w - 28 - bw_, cy2 + 17,
                         f"第 {c['introduced_in'].replace('ch','')} 章", 8.2, C_MUTE, anchor='end')
        y += band_h + 9

    # ---- 关系箭头：本章结构如何接到前面已读的结构上 ----
    links = cross_links(model, cid, cur_sub) if cur_sub else {}
    if cur_sub in boxpos:
        tx, ty, tw_, th, _tb = boxpos[cur_sub]
        tright = tx + tw_
        for k, (other, info) in enumerate(links.items()):
            if other not in boxpos:
                continue
            ax, ay, aw, ah, abot = boxpos[other]
            chx = tright + 10 + k * 11       # 右侧通道，逐条错开，避开层标签与组件盒
            # 从源盒**底边**出发、走**层带之间的空隙**再上/下到目标 ——
            # 不能在组件行的中线上横穿:实测那条线会直接穿过「IPC 边界」「ModelRunner 执行」
            # 等无关组件盒,读者会误以为它们参与其中。
            sx = ax + aw / 2
            gy = abot + 4.5
            ey2 = ty + th / 2
            m_end = ' marker-end="url(#a2)"'
            m_st = ' marker-start="url(#a2s)"' if info['both'] else ''
            L.append(f'<path d="M{sx:.1f},{ay + ah:.1f} L{sx:.1f},{gy:.1f} L{chx:.1f},{gy:.1f} '
                     f'L{chx:.1f},{ey2:.1f} L{tright:.1f},{ey2:.1f}" fill="none" '
                     f'stroke="{C_CUR_S}" stroke-width="1.5"{m_st}{m_end}/>')
            # 说「交接」而非只给站号:这些站号是**控制权移交发生的那一站**,
            # 与「某组件内部停了哪几站」是两回事,不加限定词读者会混淆。
            text(L, chx - 7, gy - 4, f'第 {rng(info["stations"])} 站交接', 8.8, C_CUR_S,
                 anchor='end', bold=True)

    H = y + 46
    ly = H - 26
    items = [(C_BUILT_F, C_BUILT_S, '前面章节已读（带章号）', False),
             (C_CUR_F, C_CUR_S, '本章新增的结构', False),
             (C_TODO_F, C_TODO_S, '后续章节才讲', True)]
    lx = M
    for f_, k_, lab, dash in items:
        box(L, lx, ly - 11, 16, 13, f_, k_, dash=dash, r=3, sw=1.2)
        text(L, lx + 22, ly, lab, 10.2, C_MUTE, anchor='start')
        lx += 28 + tw(lab, 10.2)
    if cur_sub and len(covered) < len(spine):
        text(L, W - M, ly, f'（另有 {len(spine) - len(covered)} 站落在其他章已讲的组件上）',
             9.5, C_MUTE, anchor='end')

    n_built = len([s for s in model['levels']['L2_subsystems']
                   if _chapter_index(s['opened_in']) < idx])
    title = (f"你的架构模型读到第 {idx} 章：{n_built} 个组件已读，本章展开「"
             f"{subs[cur_sub]['name_cn']}」" if cur_sub
             else f"你的架构模型读到第 {idx} 章：全景导览")
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H:.0f}">',
            '<defs>'
            f'<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
            f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_MUTE}"/></marker>'
            f'<marker id="a2" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
            f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_CUR_S}"/></marker>'
            f'<marker id="a2s" viewBox="0 0 10 6" refX="1" refY="3" markerWidth="6" markerHeight="4" '
            f'orient="auto"><path d="M10,0 L0,3 L10,6 Z" fill="{C_CUR_S}"/></marker>'
            '</defs>',
            f'<rect width="{W}" height="{H:.0f}" fill="white"/>']
    head.append(f'<text x="{M}" y="32" font-family="sans-serif" font-size="16" fill="{C_TXT}" '
                f'font-weight="bold">{esc(title)}</text>')
    head.append(f'<text x="{M}" y="51" font-family="sans-serif" font-size="11" fill="{C_MUTE}">'
                f'{esc("整本书共用这一张架构图：每章在已铺好的结构上长出新的一块，并标出本章站点落在哪些组件上")}</text>')
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
