#!/usr/bin/env python3
"""渐进式架构模型 (Progressive Architecture Model) —— 全书唯一的「读者心智模型」真相源。

为什么存在（问题）：
  书里每章内嵌几十段真源码，每段都标了 `path:Lx-Ly`。但**路径是地址、不是路线**——
  读者不知道这段何时跑、谁调它、属于哪个大模块。全书原本有两头：
    · 全局有 roadmap（章粒度的「你在这里」），
    · 局部有 chapter-map（文件/符号粒度的本章走线），
  **中间没有桥**：没有任何工件说「这个文件属于子系统 X，X 挂在主线阶段 Y 下，
  Y 读者在第 Z 章已经见过」。于是每章几十个节点无处挂靠 → 读完即忘。

本模型补的就是这座桥，三层 + 累积状态：
  L1 stages      —— 请求生命周期主线，**固定 7 个**（7±2 认知上限；复用 roadmap.py 的
                    STAGES，读者每章都见，零新词汇 → 挂靠成本最低）
  L2 subsystems  —— 子系统，每个**恰好挂在一个 L1 之下**（来自 outline-final.json 的
                    chapter.subsystem，全书 100% 覆盖）
  L3 units       —— 各章真正走过的代码单元（来自各章 dossier.code_spine / embed_excerpts）

累积（accretion）：每个 L2/L3 节点记 `opened_in`（首次展开它的章）。渲染第 N 章时：
  · opened_in < N  → 已建成（实心，带章号回指：读者在第几章挂上去的）
  · opened_in == N → 本章展开（高亮 + 下钻到 L3 走线）
  · 尚未 opened    → 待建（虚线淡色）
=> 第 1 章几乎空白，逐章长出，到末章成完整体系。这就是「从 1 个节点开始搭建、
   后续逐步挂靠」的可视化；也保证任一时刻**只有 1 个 L2 展开**，其余滚起到 7 个 L1。

用法：
  python3 scripts/arch_model.py build              # 构建/刷新 book/cartography/arch-model.json
  python3 scripts/arch_model.py show --chapter ch31  # 打印第 N 章的累积视图（文字版，供核对）
  python3 scripts/arch_model.py check              # 一致性自检（孤儿文件 / 挂靠倒序 / L1 超 7）
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instance  # noqa: E402

# ---- L1：请求生命周期主线（与 roadmap.py 的 STAGES 对齐，固定 7 个）----
# 顺序即读者心中的「一个请求从进到出」的顺序。
L1_STAGES = [
    ("entrypoints", "入口", "LLM.generate / OpenAI server"),
    ("input-processor", "Stage 1 输入", "tokenize → EngineCoreRequest"),
    ("async-engine", "AsyncLLM 解耦", "三段式 / output_handler"),
    ("ipc", "IPC 边界", "ZMQ + msgpack 跨进程"),
    ("engine-core", "EngineCore 循环", "schedule → execute → sample"),
    ("output-processor", "Stage 3 输出", "detokenize → RequestOutput"),
    ("stream", "流式返回", "SSE / generate() 产出"),
]
L1_KEYS = [k for k, _, _ in L1_STAGES]

# ---- L2 → L1 归属。前 7 个是「子系统即阶段本身」；其余是挂在某阶段下的子系统。----
# 依据：roadmap.py 的 SUBSYS（已有 13 条）+ 补齐 outline 里出现但 SUBSYS 缺的（scheduler）。
SUBSYS_PARENT = {
    # 与主线阶段同名者：自身即该阶段
    "entrypoints": "entrypoints",
    "input-processor": "input-processor",
    "async-engine": "async-engine",
    "engine-core": "engine-core",
    "output-processor": "output-processor",
    # 挂靠型子系统
    "config-and-wiring": "entrypoints",
    "scheduler": "engine-core",          # roadmap 注释明确：调度器并入「EngineCore 循环」框
    "kv-cache": "engine-core",
    "worker-and-executor": "engine-core",
    "model-runner": "engine-core",
    "distributed-parallelism": "engine-core",
    "model-definitions": "engine-core",
    "custom-ops-and-compilation": "engine-core",
    "attention": "engine-core",
    "model-architecture": "engine-core",
    "sampling": "engine-core",
    "structured-output": "engine-core",
    "spec-decode": "engine-core",
    "pd-disaggregation": "ipc",
}

# ---- L1.5：组。**只对 children > 7 的阶段启用**（认知上限）。----
# 实测：engine-core 下挂了 12 个子系统，到第 31 章时该框是一堵 12 枚芯片的墙 —— 正是
# 「一次展开超过 7 个模块就认知疲惫」。故按「读者关心的问题」再抽一层，每组 ≤4：
GROUPS = {
    "engine-core": [
        ("loop", "循环本体", ["engine-core"]),
        ("sched-mem", "调度与显存", ["scheduler", "kv-cache"]),
        ("exec", "执行与并行", ["worker-and-executor", "model-runner", "distributed-parallelism"]),
        ("model", "模型与算子", ["model-definitions", "model-architecture",
                                 "custom-ops-and-compilation", "attention"]),
        ("decode", "解码策略", ["sampling", "structured-output", "spec-decode"]),
    ],
}
GROUP_OF = {s: (st, gid, gname)
            for st, gs in GROUPS.items()
            for gid, gname, subs in gs
            for s in subs}

# 子系统中文名（图上用；避免图面出现生硬英文键）
SUBSYS_CN = {
    "entrypoints": "入口层", "input-processor": "输入处理", "async-engine": "异步引擎",
    "engine-core": "引擎核心", "output-processor": "输出处理", "config-and-wiring": "配置与装配",
    "scheduler": "调度器", "kv-cache": "分页 KV 缓存", "worker-and-executor": "Worker 与执行器",
    "model-runner": "ModelRunner 执行", "distributed-parallelism": "分布式并行",
    "model-definitions": "模型定义层", "custom-ops-and-compilation": "自定义算子与编译",
    "attention": "注意力后端", "model-architecture": "模型架构", "sampling": "采样",
    "structured-output": "结构化输出", "spec-decode": "投机解码", "pd-disaggregation": "P/D 分离",
}

SPINE_RE = re.compile(r'^\s*([\w/\.\-]+\.(?:py|cc|cpp|h|hpp|cu|pyi|td|mlir))\s*:\s*([\dL\-–,\s]+)?\s*[—\-–]\s*(.*)$')


def _chapter_index(cid):
    m = re.search(r'(\d+)', cid or '')
    return int(m.group(1)) if m else 9999


def _evidence_dirs(units, top=3):
    from collections import Counter
    c = Counter(u['path'].rsplit('/', 1)[0] for u in units)
    return [{'dir': d, 'steps': n} for d, n in c.most_common(top)]


def build(inst=None):
    bd = Path(instance.book_dir(inst))
    ad = Path(instance.artifacts_dir(inst))
    outline_p = bd / 'cartography' / 'outline-final.json'
    if not outline_p.exists():
        raise SystemExit(f'缺少 {outline_p}')
    outline = json.load(open(outline_p, encoding='utf-8'))

    # 章 → (subsystem, part, title)
    ch_meta = {}
    for part in outline:
        for c in part.get('chapters', []):
            cid = (c.get('id') or c.get('chapter_id') or '').strip()
            if cid:
                ch_meta[cid] = {
                    'subsystem': (c.get('subsystem') or '').strip(),
                    'part': part.get('part', ''),
                    'title': c.get('title', ''),
                }

    # 章目录（slug）↔ chapter_id
    slug_by_cid = {}
    for d in sorted(p.name for p in ad.iterdir() if p.is_dir() and p.name.startswith('ch')):
        m = re.match(r'(ch\d+)', d)
        if m:
            slug_by_cid.setdefault(m.group(1), d)

    prefixes = instance.canonical_prefixes(inst)
    skipped = []
    l3_by_ch = {}
    file_first_open = {}
    for cid, slug in slug_by_cid.items():
        dp = ad / slug / 'dossier' / 'dossier.json'
        if not dp.exists():
            continue
        try:
            d = json.load(open(dp, encoding='utf-8'))
        except Exception:
            continue
        units = []
        for step in (d.get('code_spine') or []):
            if not isinstance(step, str):
                continue
            m = SPINE_RE.match(step)
            if m:
                path, lines, what = m.group(1), (m.group(2) or '').strip(), m.group(3).strip()
            else:
                pm = re.search(r'([\w/\.\-]+\.(?:py|cc|cpp|h|cu))', step)
                path, lines, what = (pm.group(1) if pm else ''), '', step.strip()
            if not path:
                continue
            # ⚠️ 只收**真源码**路径。dossier 里混进过书内脚手架路径（实测 ch29 的 code_spine
            # 有一条 instances/<inst>/book/assets/roadmap/roadmap.py）——那不是被解读的源码，
            # 放进模型会污染「涉及文件数」并在图上冒出内部路径（违反 HARD RULE 3 零脚手架泄漏）。
            if not any(path == pre or path.startswith(pre + '/') for pre in prefixes):
                skipped.append((cid, path))
                continue
            sym = ''
            sm = re.match(r'^([A-Za-z_][\w\.]*)\s*[:：]', what)
            if sm:
                sym = sm.group(1)
            units.append({'path': path, 'lines': lines, 'symbol': sym, 'what': what[:160]})
            idx = _chapter_index(cid)
            if path not in file_first_open or idx < file_first_open[path][0]:
                file_first_open[path] = (idx, cid)
        l3_by_ch[cid] = units

    # L2 累积：某子系统首次被哪一章展开
    sub_first = {}
    for cid, meta in ch_meta.items():
        s = meta['subsystem']
        if not s:
            continue
        idx = _chapter_index(cid)
        if s not in sub_first or idx < sub_first[s][0]:
            sub_first[s] = (idx, cid)

    subsystems = []
    for s in sorted(set(m['subsystem'] for m in ch_meta.values() if m['subsystem'])):
        chs = sorted((c for c, m in ch_meta.items() if m['subsystem'] == s), key=_chapter_index)
        g = GROUP_OF.get(s)
        subsystems.append({
            'id': s,
            'name_cn': SUBSYS_CN.get(s, s),
            'parent_stage': SUBSYS_PARENT.get(s),
            'group': g[1] if g else None,
            'group_cn': g[2] if g else None,
            'opened_in': sub_first.get(s, (9999, ''))[1],
            'chapters': chs,
        })

    model = {
        'instance': inst or instance.active_name(),
        'levels': {
            'L1_stages': [{'id': k, 'name': n, 'sub': s} for k, n, s in L1_STAGES],
            'L2_subsystems': subsystems,
        },
        'chapters': {
            cid: {
                'subsystem': m['subsystem'],
                # ⚠️ subsystem 来自 outline-final.json 的**声明**,不是证据。
                # 用户 2026-07-26 明确:不要预设既有章节地图/声明成立并据此行动。
                # 故同时记录 evidence_dirs(本章走线真正落在的目录,来自 dossier 真源码路径),
                # 并把 verified 默认置 false —— 声明与证据一致才可置 true。
                'evidence_dirs': _evidence_dirs(l3_by_ch.get(cid, [])),
                'verified': False,
                'parent_stage': SUBSYS_PARENT.get(m['subsystem']),
                'part': m['part'],
                'title': m['title'],
                'slug': slug_by_cid.get(cid, ''),
                'spine': l3_by_ch.get(cid, []),
            } for cid, m in sorted(ch_meta.items(), key=lambda kv: _chapter_index(kv[0]))
        },
        'file_first_open': {p: c for p, (i, c) in sorted(file_first_open.items(), key=lambda kv: kv[1][0])},
        'skipped_non_source': [{'chapter': c, 'path': p} for c, p in skipped],
    }
    out = bd / 'cartography' / 'arch-model.json'
    out.write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding='utf-8')
    return model, out


def check(model):
    issues = []
    if len(model['levels']['L1_stages']) > 7:
        issues.append(f"L1 阶段 {len(model['levels']['L1_stages'])} 个 > 7（认知上限）")
    # 认知上限：任一框的直接子节点不得 > 7（>7 就该再抽一层组）
    for st in model['levels']['L1_stages']:
        subs = [s for s in model['levels']['L2_subsystems'] if s['parent_stage'] == st['id']]
        kids = {s['group'] for s in subs if s['group']} | {s['id'] for s in subs if not s['group']}
        if len(kids) > 7:
            issues.append(f"阶段「{st['name']}」直接子节点 {len(kids)} 个 > 7 —— 需在 GROUPS 里再抽一层组")
        for gid in {s['group'] for s in subs if s['group']}:
            n = len([s for s in subs if s['group'] == gid])
            if n > 7:
                issues.append(f"组 {gid} 下 {n} 个子系统 > 7 —— 需再拆")
    for s in model['levels']['L2_subsystems']:
        if not s['parent_stage']:
            issues.append(f"子系统 {s['id']} 没有归属的 L1 阶段（图上会成孤儿）")
        elif s['parent_stage'] not in L1_KEYS:
            issues.append(f"子系统 {s['id']} 的父阶段 {s['parent_stage']} 不在 7 个主线阶段里")
    for cid, c in model['chapters'].items():
        if not c['subsystem']:
            issues.append(f"{cid} 未声明 subsystem（无法挂靠）")
        if not c['spine']:
            issues.append(f"{cid} 无 code_spine（本章走线无来源）")
    return issues


def show(model, chapter):
    idx = _chapter_index(chapter)
    print(f"=== 第 {chapter} 章时的累积架构模型 ===\n")
    cur = model['chapters'].get(chapter, {})
    cur_sub = cur.get('subsystem')
    def mark_of(s):
        if s['id'] == cur_sub:
            return '◆'
        return '●' if _chapter_index(s['opened_in']) < idx else '○'

    def line(s, pad):
        m = mark_of(s)
        if m == '◆':
            return f"{pad}◆ {s['name_cn']}  ← 本章展开"
        if m == '●':
            return f"{pad}● {s['name_cn']}（第 {s['opened_in'].replace('ch','')} 章挂上）"
        return f"{pad}○ {s['name_cn']}（第 {s['opened_in'].replace('ch','')} 章才展开）"

    for st in model['levels']['L1_stages']:
        subs = [s for s in model['levels']['L2_subsystems'] if s['parent_stage'] == st['id']]
        if not subs:
            continue
        stm = '◆' if any(s['id'] == cur_sub for s in subs) else (
            '●' if any(_chapter_index(s['opened_in']) < idx for s in subs) else '○')
        print(f"{stm} [{st['name']}] {st['sub']}")
        grouped = [g for g in GROUPS.get(st['id'], [])]
        if grouped:
            for gid, gname, members in grouped:
                gs = [s for s in subs if s['id'] in members]
                if not gs:
                    continue
                gm = '◆' if any(s['id'] == cur_sub for s in gs) else (
                    '●' if any(_chapter_index(s['opened_in']) < idx for s in gs) else '○')
                print(f"    {gm} 〈{gname}〉")
                for s in sorted(gs, key=lambda x: _chapter_index(x['opened_in'])):
                    print(line(s, '        '))
                    if s['id'] == cur_sub:
                        for i, u in enumerate(cur.get('spine', [])[:40], 1):
                            nm = f"{u['path'].split('/')[-1]}:{u['symbol']}" if u['symbol'] else u['path'].split('/')[-1]
                            print(f"             {i:>2}. {nm:<46} {u['what'][:50]}")
        else:
            for s in sorted(subs, key=lambda x: _chapter_index(x['opened_in'])):
                print(line(s, '      '))
                if s['id'] == cur_sub:
                    for i, u in enumerate(cur.get('spine', [])[:40], 1):
                        nm = f"{u['path'].split('/')[-1]}:{u['symbol']}" if u['symbol'] else u['path'].split('/')[-1]
                        print(f"           {i:>2}. {nm:<46} {u['what'][:50]}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'show', 'check'])
    ap.add_argument('--chapter', default=None)
    ap.add_argument('--instance', default=None)
    a = ap.parse_args()
    inst = a.instance
    model, out = build(inst)
    if a.cmd == 'build':
        n_sub = len(model['levels']['L2_subsystems'])
        n_ch = len(model['chapters'])
        n_units = sum(len(c['spine']) for c in model['chapters'].values())
        print(f"✓ 已构建 {out}")
        print(f"  L1 主线阶段 {len(model['levels']['L1_stages'])} 个（7±2 上限内）")
        print(f"  L2 子系统 {n_sub} 个 / 章 {n_ch} 个 / L3 走线单元 {n_units} 条 / 涉及文件 {len(model['file_first_open'])} 个")
        iss = check(model)
        print(('  ⚠ 一致性问题 %d 条:' % len(iss)) if iss else '  ✓ 一致性自检通过')
        for i in iss[:12]:
            print('    -', i)
    elif a.cmd == 'check':
        iss = check(model)
        print('\n'.join('- ' + i for i in iss) if iss else '✓ 一致性自检通过')
        sys.exit(1 if iss else 0)
    else:
        show(model, a.chapter or 'ch01')


if __name__ == '__main__':
    main()
