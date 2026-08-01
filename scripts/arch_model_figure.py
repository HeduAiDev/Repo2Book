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
    这样图上天然呈现「在旧结构之上长出新结构」。超 cap 时优先保留「结构锚点」——
    本章站点停靠的类 + is_a/has_a 关系的参与者。ch03 的 CompilationConfig 曾被纯
    「按站数多寡」的截断挤掉：它没有站点，却同时被 EngineArgs 与 VllmConfig 持有
    （源码 arg_utils.py:627 / vllm.py:312 都有 `compilation_config: CompilationConfig`
    注解），类被挤掉后 canon() 找不到它，图上两处 has_a 持有关系整个消失，架构是错的。
    """
    idx = _chapter_index(cid)
    mine = [c for c in model.get('classes', []) if c['subsystem'] == sub
            and _chapter_index(c['introduced_in']) <= idx]
    st = station_of_classes(mine, spine, model['chapters'][cid].get('relations', {}))
    keep = [c for c in mine if c['introduced_in'] == cid or st.get(c['name'])]
    if len(keep) > cap:
        rel = model['chapters'][cid].get('relations', {})
        parts = set()
        for a, b in (rel.get('is_a') or []) + (rel.get('has_a') or []):
            parts.add(a)
            parts.add(b)

        def essential(c):
            return bool(st.get(c['name'])) or c['name'] in parts

        keep.sort(key=lambda c: (not essential(c), -len(st.get(c['name'], [])),
                                 _chapter_index(c['introduced_in'])))
        n_ess = sum(1 for c in keep if essential(c))
        keep = keep[:max(cap, n_ess)]
    keep.sort(key=lambda c: (min(st.get(c['name'], [999])), _chapter_index(c['introduced_in'])))
    return keep, st


def station_of_classes(classes, spine, relations=None):
    """把本章站点落到**组件**上（不是落到文件上——用户 2026-07-26：不必给每个站点标具体文件、
    把每个站点做成独立模块，那太细、不适合架构图）。

    匹配优先级（**全部按同文件判定**，杜绝跨文件张冠李戴——ch32 曾把
    structured_outputs.py 里 `apply_grammar_bitmask` 的站挂到 utils.py 的
    `apply_grammar_bitmask（旧 gpu_model_runner 路径）` 上，同名不同文件）：
      1) 符号的类名部分精确命中（限定同文件）；
      2) 符号是 relations.methods 里某类的自有方法，且站路径 == 该类文件
         （arch_model.py 从 AST 抽出的「方法-文件」证据，比名字子串可靠）；
      3) 同文件兜底：类名词元**精确**比对（允许点后缀，如
         'StructuredOutputManager.should_fill_bitmask' ← 'should_fill_bitmask'）；
         禁裸子串命中——'apply_grammar_bitmask' 不得挂到
         '_apply_grammar_bitmask_kernel'；
      4) **无符号的站，落给它所在文件归属的类**（ch28 第 5–7 站是 MLA 注意力内部实现，
        在 deepseek_v4_attention.py 里却没有符号，若不兜底就成了图上无主的盲区）。
    """
    out = {}
    by_file = {}
    for c in classes:
        by_file.setdefault(c['file'], []).append(c)
    rel = relations or {}
    mkeys = rel.get('methods') or {}
    mfiles = rel.get('files') or {}
    plain_to_entry = {}
    for c in classes:
        for n in [x.strip() for x in re.split(r'[/（(]', c['name']) if x.strip()]:
            plain_to_entry.setdefault(n, c['name'])
    for i, u in enumerate(spine, 1):
        sym = u.get('symbol') or ''
        base = sym.split('.')[0] if sym else ''
        hit = None
        for c in classes:                       # 1) 类名精确/包含命中（限定同文件）
            if c['file'] != u['path']:
                continue
            names = [n.strip() for n in re.split(r'[/（(]', c['name']) if n.strip()]
            if base and any(base == n or base in n.split() for n in names):
                hit = c
                break
        if not hit and base:                    # 2) 符号是某类的自有方法（AST 证据）
            for rn, meths in mkeys.items():
                if base not in meths:
                    continue
                rf = mfiles.get(rn)
                if rf and rf == u['path'] and rn in plain_to_entry:
                    hit = next((c for c in classes
                                if c['name'] == plain_to_entry[rn]), None)
                    if hit:
                        break
        if not hit and u['path'] in by_file:    # 3) 同文件词元精确命中
            cands = by_file[u['path']]
            hit = next((c for c in cands if base and
                        any(base == t or ('.' in t and base == t.rsplit('.', 1)[1])
                            for t in [x.strip() for x in re.split(r'[/（(]', c['name'])
                                      if x.strip()])),
                       cands[0])
        if not hit and u['path'] in by_file and not sym:   # 4) 无符号 → 该文件归属的类
            hit = by_file[u['path']][0]
        if hit:
            out.setdefault(hit['name'], []).append(i)
    return out


def short(name, n=64):
    """组件显示名：去掉 (ABC) 之类的括注，'A / B' 只留首个 + 省略号。

    n=64（2026-08-01 起）：ch32 的 'StructuredOutputManager.should_fill_bitmask /
    should_advance' 按 30 截断成了 'StructuredOutputManager.shoul…'，读者认不出
    是哪个方法；实际盒宽放得下完整名。真正防溢出的是 fit()，short() 只负责
    去掉括注/合条，不必提前截那么狠。
    """
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


def plan_panel(cls, st_map, rel, width):
    """把本章组件排成**有组织关系的架构**，而不是一列类名。

    用户 2026-07-26：参考图表达的是「各模块的相互作用与组织关系」，靠列举类名表现不出来。
    从源码抽到的真实关系里认出两种结构，分别对应参考图里的两种画法：
      · 契约(contract)：被 >=2 个类继承的基类 → **容器盒**，实现类并排嵌在里面
        （参考图里 Backend 底下并排挂 Flash/Blocksparse/Ipex/Rocm）
      · 组合(containment)：X 持有 Y（has_a）→ **盒套盒**，Y 嵌进 X
        （参考图里 SequenceGroup ⊃ Sequence ⊃ SequenceData；ch28 的
         ForCausalLM ⊃ Model ⊃ DecoderLayer ⊃ {Attention, MoE} ⊃ {MLP, Experts}）
    两者都没有 → 诚实地并列平铺，并在标题标「这几块彼此独立」（ch32 那种确实无层级的章）。
    """
    is_a, has_a = rel.get('is_a', []), rel.get('has_a', [])
    names = {c['name']: c for c in cls}

    def canon(n):
        for k in names:
            if n == k or n in [x.strip() for x in re.split(r'[/（(]', k)]:
                return k
        return None

    # ---- 契约：>=2 个实现的基类 ----
    kids = OrderedDict()
    for a, b in is_a:
        ca, cb = canon(a), canon(b)
        if ca and cb and ca != cb:
            kids.setdefault(cb, [])
            if not any(x[0] == a for x in kids[cb]):
                kids[cb].append((a, ca))
    contracts = OrderedDict((k, v) for k, v in kids.items() if len(v) >= 2)
    in_contract = {ck for v in contracts.values() for _, ck in v} | set(contracts)

    # ---- 组合森林：X has-a Y（只在 key_classes 集合内，且不与契约成员重叠）----
    children = OrderedDict()
    parent = {}
    for a, b in has_a:
        ca, cb = canon(a), canon(b)
        if not (ca and cb) or ca == cb or ca in in_contract or cb in in_contract:
            continue
        children.setdefault(ca, [])
        if cb not in children[ca] and cb not in _ancestors(ca, parent):   # 防环
            children[ca].append(cb)
            parent[cb] = ca
    roots = [c['name'] for c in cls
             if c['name'] not in in_contract and c['name'] not in parent
             and c['name'] in children]
    expanded = set()      # 已在某处展开过的节点 → 再次被持有时只作叶子引用,不重复整棵子树
    nested = {r: _subtree(r, children, set(), expanded, depth=0, maxd=5) for r in roots}
    consumed = set()
    for r in roots:
        _collect(r, children, consumed)

    left = [c for c in cls if c['name'] not in in_contract and c['name'] not in consumed]
    return {'left': left, 'contracts': contracts, 'nested': nested, 'roots': roots,
            'in_contract': in_contract}


def _ancestors(n, parent):
    out, cur = set(), parent.get(n)
    while cur:
        if cur in out:
            break
        out.add(cur)
        cur = parent.get(cur)
    return out


def _subtree(n, children, seen, expanded, depth, maxd):
    # 环、超深、或已在别处展开过 → 收成叶子(只画盒名,不再往里钻)
    if n in seen or depth >= maxd or n in expanded:
        return {'name': n, 'children': [], 'ref': n in expanded}
    seen = seen | {n}
    expanded.add(n)
    return {'name': n, 'ref': False,
            'children': [_subtree(c, children, seen, expanded, depth + 1, maxd)
                         for c in children.get(n, [])]}


def _collect(n, children, acc):
    acc.add(n)
    for c in children.get(n, []):
        if c not in acc:
            _collect(c, children, acc)


def build(model, cid):
    """渲染第 cid 章的架构剖面 —— **第 1 章那张「一个请求的端到端旅程」长大后的样子**。

    骨架与 ch01 图一一对应（入口两扇门 → InputProcessor → IPC → EngineCore 大框 → OutputProcessor），
    EngineCore 是容器，后续章节把新组件装进去；本章的组件就地展开成源码里的真实组织关系
    （契约 + 实现 + 持有），站点标在组件上。
    """
    idx = _chapter_index(cid)
    ch = model['chapters'][cid]
    cur_sub = ch['subsystem']
    subs = {s['id']: s for s in model['levels']['L2_subsystems']}
    rows = model['levels']['layers']
    cgroups = model['levels']['core_groups']
    spine = ch['spine']

    W, M = 1180, 24
    L, boxpos, bands = [], {}, []
    y = 96
    CW, CH_, CG = 138, 34, 9

    cls, st_map = (pick_classes(model, cid, cur_sub, spine) if cur_sub else ([], {}))
    covered = sorted({i for v in st_map.values() for i in v})
    pan = plan_panel(cls, st_map, ch.get('relations', {}), W) if cls else None
    in_core = bool(cur_sub) and cur_sub in {s for g in cgroups for s in g['subsystems']}

    def state(sid):
        s_ = subs.get(sid)
        if not s_:
            return None
        if sid == cur_sub:
            return 'cur'
        return 'built' if _chapter_index(s_['opened_in']) < idx else 'todo'

    def comp(x, cy, sid, w=CW):
        st = state(sid)
        f_, k_ = colors(st)
        box(L, x, cy, w, CH_, f_, k_, dash=(st == 'todo'), r=6, sw=1.3)
        fit(L, x + w / 2, cy + 15, subs[sid]['name_cn'], w - 10, 10.5, C_TXT)
        n_ch = subs[sid]['opened_in'].replace('ch', '')
        fit(L, x + w / 2, cy + 27, (f'第 {n_ch} 章已读' if st == 'built' else f'第 {n_ch} 章才讲'),
            w - 10, 8.8, C_MUTE)
        boxpos[sid] = (x, cy, w, CH_)

    # ---- 组件站号：站归属的唯一真相 = station_of_classes（含类名前缀/方法证据/
    # 同文件兜底四级判定）。曾有两套逻辑（comp_stations 精确匹配 + st_map 兜底），
    # 两套结果对不上：无符号的站（如 ch32 第 5–7、11 站）在 badge 上消失、同一站号
    # 挂到两个盒上。统一委托 st_map，badge 与「N 站落在下列组件上」的计数天然一致。
    def comp_stations(real):
        return st_map.get(real, [])

    def node_h(node):
        """嵌套盒高度：与 draw_node 的列高跟踪（colh[col] += kh + 5）严格一致。

        旧公式只按 krows*24 估算且只数第 0 列——较高的容器排在**第 1 列**时少算：
        ch03 的 VllmConfig 容器（2 列 5 个叶子）按 26+3*24+6=104 画,末行叶子
        实际戳到 463、容器底边 449,叶子压出容器 14px。这里逐子模拟两列累加,
        取 max 列高减掉尾空隙再加底垫。
        """
        kids = node['children']
        if not kids:
            return 36
        two = len(kids) > 1
        colh = [24, 24]
        for i, k in enumerate(kids):
            colh[i % 2 if two else 0] += node_h(k) + 5
        return max(colh) - 5 + 6

    def draw_node(node, nx, ny, nw):
        """递归画组合盒：容器在外、被持有者嵌在内（参考图 SequenceGroup ⊃ Sequence ⊃ …）。"""
        nm = node['name']
        kids = node['children']
        h = node_h(node)
        isnew = names_meta.get(nm, {}).get('introduced_in') == cid
        f_, k_ = (C_CUR_F, C_CUR_S) if isnew else (C_BUILT_F, C_BUILT_S)
        if kids:
            box(L, nx, ny, nw, h, '#fffaf3', C_CUR_S, r=6, sw=1.5)
        else:
            box(L, nx, ny, nw, 36, f_, k_, r=5, sw=1.3)
        ids = comp_stations(nm)
        bd = f'第 {rng(ids)} 站' if ids else ''
        # **叶子盒标签与徽标分行**(名称一行、站号徽标一行):实测同挤一行,长类名截到
        # 「Dee…」后徽标直接压上去(text-text 相撞,几何 linter 实锤)。分行后不再挤。
        # 名称一行、站号徽标单独一行(无论容器还是叶子)：
        # 实测名称与徽标同挤一行时,长类名截断后徽标直接压上去(几何 linter 实锤「Dee…×第 11 站」)
        if kids:
            fit(L, nx + 8, ny + 16, short(nm), nw - 16, 10, C_CUR_S, bold=True, anchor='start')
        else:
            fit(L, nx + 8, ny + 16, short(nm), nw - 16, 9.2, C_TXT, anchor='start')
        if bd:
            if kids:
                fit(L, nx + nw - 8, ny + 30, bd, nw - 14, 8.6, C_CUR_S, bold=True, anchor='end')
            else:
                fit(L, nx + nw - 7, ny + 30, bd, nw - 14, 8.6, C_CUR_S, bold=True, anchor='end')
        if not kids:
            return h
        two = len(kids) > 1
        kcols = 2 if two else 1
        kw = (nw - 16 - (kcols - 1) * 8) / kcols
        colh = [ny + 24, ny + 24]
        for i, kd in enumerate(kids):
            col = i % kcols if two else 0
            kx = nx + 8 + col * (kw + 8)
            kh = draw_node(kd, kx, colh[col], kw)
            colh[col] += kh + 5
        return h

    names_meta = {c['name']: c for c in cls}

    def panel_h():
        if not pan:
            return 0
        flat = not pan['contracts'] and not pan['roots']
        lcols = 2 if flat and len(pan['left']) > 4 else 1
        lh = ((len(pan['left']) + lcols - 1) // lcols) * 30
        rh = sum(node_h({'name': r, 'children': pan['nested'][r]['children']}) + 8
                 for r in pan['roots'])
        rh += sum(24 + ((len(v) + 1) // 2) * 22 + 8 for v in pan['contracts'].values())
        return 30 + max(lh, rh) + 10

    def draw_panel(px, py, pw):
        ph = panel_h()
        box(L, px, py, pw, ph, '#fff7ed', C_CUR_S, r=7, sw=2.0)
        boxpos[cur_sub] = (px, py, pw, ph)
        flat = not pan['contracts'] and not pan['roots']
        tag = '（这几块彼此独立，按功能并列）' if flat else ''
        text(L, px + 12, py + 18, f"{subs[cur_sub]['name_cn']}　← 本章展开{tag}", 11.5, C_CUR_S,
             anchor='start', bold=True)
        text(L, px + pw - 12, py + 18,
             f'本章 {len(spine)} 站，其中 {len(covered)} 站落在下列组件上', 9.3, C_MUTE, anchor='end')
        has_r = bool(pan['contracts']) or bool(pan['roots'])
        lcols = 2 if flat and len(pan['left']) > 4 else 1
        LW = (pw - 30) if not has_r else (pw - 34) * 0.40
        RW = (pw - 30) - LW - (14 if has_r else 0)
        lx0, rx0 = px + 12, px + 12 + LW + 14
        lcw = (LW - (lcols - 1) * 10) / lcols
        for i, c in enumerate(pan['left']):
            lc, lr = (i % lcols, i // lcols) if lcols > 1 else (0, i)
            cx_l = lx0 + lc * (lcw + 10)
            cy2 = py + 30 + lr * 30
            isnew = c['introduced_in'] == cid
            f_, k_ = (C_CUR_F, C_CUR_S) if isnew else (C_BUILT_F, C_BUILT_S)
            box(L, cx_l, cy2, lcw, 26, f_, k_, r=5, sw=1.3)
            ids = comp_stations(c['name'])
            bd = f'第 {rng(ids)} 站' if ids else ''
            fit(L, cx_l + 8, cy2 + 17, short(c['name']),
                lcw - 20 - (tw(bd, 9, True) + 8 if bd else 0), 10, C_TXT, bold=isnew, anchor='start')
            if bd:
                text(L, cx_l + lcw - 8, cy2 + 17, bd, 9, C_CUR_S, anchor='end', bold=True)
        ry = py + 30
        # 组合森林（盒套盒）
        for r in pan['roots']:
            hh = draw_node({'name': r, 'children': pan['nested'][r]['children']}, rx0, ry, RW)
            ry += hh + 8
        # 契约容器（并排实现）
        for ct, members in pan['contracts'].items():
            mrows = (len(members) + 1) // 2
            hh = 24 + mrows * 22 + 8
            box(L, rx0, ry, RW, hh, '#ffffff', C_CUR_S, r=6, sw=1.5)
            ids = comp_stations(ct)
            fit(L, rx0 + 8, ry + 16, short(ct) + '（契约）', RW - 96, 10, C_CUR_S, bold=True,
                anchor='start')
            if ids:
                text(L, rx0 + RW - 8, ry + 16, f'第 {rng(ids)} 站', 9, C_CUR_S, anchor='end', bold=True)
            mw = (RW - 24) / 2
            for j, (real, key) in enumerate(members):
                mx = rx0 + 8 + (j % 2) * (mw + 8)
                my = ry + 24 + (j // 2) * 22
                box(L, mx, my, mw, 19, '#fff7ed', '#fdba74', r=4, sw=1.1)
                mids = comp_stations(real)
                mb = f'第 {rng(mids)} 站' if mids else ''
                fit(L, mx + 6, my + 13, real,
                    mw - 14 - (tw(mb, 8.2, True) + 6 if mb else 0), 9, C_TXT, anchor='start')
                if mb:
                    text(L, mx + mw - 5, my + 13, mb, 8.2, C_CUR_S, anchor='end', bold=True)
            ry += hh + 8

    text(L, M, y - 16, '这张图是第 1 章那张「一个请求的端到端旅程」长大后的样子：'
                       '蓝＝前面章节已读，橙＝本章新增，虚线＝后续章节才讲', 11.2, C_MUTE, anchor='start')

    for row in rows:
        members = [s for s in row['subsystems'] if s in subs]
        is_core = row['id'] == 'core'
        rh = CH_ + 22
        if is_core:
            gh = []
            for g in cgroups:
                ms = [s for s in g['subsystems'] if s in subs]
                exp_here = in_core and cur_sub in ms
                grid = [s for s in ms if not (exp_here and s == cur_sub)]
                cols = max(1, min(4, len(grid))) if grid else 1
                gr = (len(grid) + cols - 1) // cols if grid else 0
                h = 20 + gr * (CH_ + 6) + (8 if gr else 2) + (panel_h() + 8 if exp_here else 0)
                gh.append((g, ms, h, exp_here, cols, grid))
            rh = 30 + sum(h for _, _, h, _, _, _ in gh) + 8 * len(gh) + 4
        elif any(s == cur_sub for s in members):
            rh = CH_ + 22 + panel_h() + 8

        box(L, M, y, W - 2 * M, rh, '#f8fafc' if not is_core else '#fffdf7',
            '#e2e8f0' if not is_core else '#f5c98a', r=8, sw=1.0 if not is_core else 1.8)
        fit(L, M + 12, y + 18, row['name'], W - 2 * M - 24, 11.5,
            C_TXT if not is_core else '#b45309', bold=True, anchor='start')

        if is_core:
            gy = y + 30
            for g, ms, h, exp_here, cols, grid in gh:
                box(L, M + 14, gy, W - 2 * M - 28, h, '#ffffff', '#e2e8f0', r=6, sw=1.1)
                text(L, M + 24, gy + 14, g['name'], 10, C_MUTE, anchor='start')
                gw = (W - 2 * M - 28 - 20 - (cols - 1) * 8) / cols
                for i, sid in enumerate(grid):
                    r_, c_ = divmod(i, cols)
                    comp(M + 24 + c_ * (gw + 8), gy + 20 + r_ * (CH_ + 6), sid, gw)
                if exp_here:
                    draw_panel(M + 24, gy + h - panel_h() - 6, W - 2 * M - 48)
                gy += h + 8
        else:
            x = M + 14
            for sid in members:
                if sid == cur_sub and pan:
                    continue
                comp(x, y + 26, sid)
                x += CW + CG
            if any(s == cur_sub for s in members) and pan:
                draw_panel(M + 14, y + 26 + CH_ + 8, W - 2 * M - 28)

        bands.append((row['id'], y, rh))
        if row is not rows[-1]:
            cx = M + (W - 2 * M) / 2
            L.append(f'<path d="M{cx},{y + rh:.1f} L{cx},{y + rh + 13:.1f}" stroke="#94a3b8" '
                     f'stroke-width="1.6" marker-end="url(#a)"/>')
        y += rh + 18

    H = y + 40
    ly = H - 22
    lx = M
    items = [(C_BUILT_F, C_BUILT_S, '前面章节已读（带章号）', False),
             (C_CUR_F, C_CUR_S, '本章新增的结构', False),
             (C_TODO_F, C_TODO_S, '后续章节才讲', True)]
    for f_, k_, lab, dash in items:
        box(L, lx, ly - 11, 16, 13, f_, k_, dash=dash, r=3, sw=1.2)
        text(L, lx + 22, ly, lab, 10.2, C_MUTE, anchor='start')
        lx += 28 + tw(lab, 10.2)
    if cur_sub and len(covered) < len(spine):
        text(L, W - M, ly, f'（另有 {len(spine) - len(covered)} 站落在其他章已讲的组件上）',
             9.5, C_MUTE, anchor='end')

    n_built = len([s for s in model['levels']['L2_subsystems']
                   if _chapter_index(s['opened_in']) < idx])
    title = (f"你的架构模型读到第 {idx} 章：{n_built} 个组件已读，本章展开「{subs[cur_sub]['name_cn']}」"
             if cur_sub else f"你的架构模型读到第 {idx} 章：全景导览")
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H:.0f}">',
            '<defs>'
            f'<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
            f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
            f'<marker id="a2" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
            f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_CUR_S}"/></marker>'
            '</defs>',
            f'<rect width="{W}" height="{H:.0f}" fill="white"/>']
    head.append(f'<text x="{M}" y="34" font-family="sans-serif" font-size="16" fill="{C_TXT}" '
                f'font-weight="bold">{esc(title)}</text>')
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
