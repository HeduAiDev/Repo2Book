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
  L1 stages      —— 请求生命周期主线（复用 roadmap.py 的 STAGES，读者每章都见，零新词汇
                    → 挂靠成本最低）。7 是**经验参考不是硬阈**（用户 2026-07-26）：确有必要
                    就多放，别为凑数硬拆出牵强的组——假分类比多一两个节点更伤认知。
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
from collections import OrderedDict
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
    "ipc": "ipc",                    # 核实 ch07 后新增:本身即主线一站
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
    "quantization": "engine-core",   # 核实 ch26 后新增(原被误挂 sampling)
}

# ---- L1.5：组。仅当某阶段子节点多到一眼看不过来时才启用（7 为经验参考，非硬阈）。----
# 实测：engine-core 下挂了 12 个子系统，到第 31 章时该框是一堵 12 枚芯片的墙 —— 正是
# 「一次展开超过 7 个模块就认知疲惫」。故按「读者关心的问题」再抽一层，每组 ≤4：
GROUPS = {
    "engine-core": [
        ("loop", "循环本体", ["engine-core"]),
        ("sched-mem", "调度与显存", ["scheduler", "kv-cache"]),
        ("exec", "执行与并行", ["worker-and-executor", "model-runner", "distributed-parallelism"]),
        ("model", "模型与算子", ["model-definitions", "model-architecture",
                                 "custom-ops-and-compilation", "attention", "quantization"]),
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
    "structured-output": "结构化输出", "spec-decode": "投机解码", "pd-disaggregation": "P/D 分离", "quantization": "量化", "ipc": "IPC 边界",
}

# ---- 核实结果:outline-final.json 的 subsystem 声明**不可尽信**,逐章按证据核过的修正在此。----
# 核实方法:章标题 + dossier.code_spine 真正走过的目录(evidence_dirs),两者对不上就查实后改。
# 每条都写明理由 —— 这张表是「已核」的凭据,不是又一次拍脑袋。
OVERRIDES = {
    # 鸟瞰/导览章:不「展开某一个子系统」,而是把 7 站主线整体介绍一遍。
    # 原声明会让它抢走真正开该子系统那一章的 opened_in(实测:ch01 抢了 ch03 的
    # config-and-wiring、ch02 抢了 ch37 的 entrypoints),使图上回指指向错误章号。
    'ch01': {'kind': 'overview', 'subsystem': None,
             'why': '全书导览章(鸟瞰),不专开某子系统;原声明 config-and-wiring 抢了 ch03 的首开权'},
    'ch02': {'kind': 'overview', 'subsystem': None,
             'why': "鸟瞰式全链路 trace(证据 v1/engine 12 步远多于 entrypoints 2 步);"
                    '原声明 entrypoints 抢了 ch37 的首开权'},
    # 章标题就叫 "The IPC Boundary",roadmap.py 注释亦明确「IPC 章(ch07)用 'ipc'」——
    # outline 声明 engine-core 与两者都冲突,且导致 7 站主线里的「IPC 边界」全书无人开启。
    'ch07': {'kind': 'source', 'subsystem': 'ipc',
             'why': '本章即 IPC 边界(ZMQ/msgpack);原声明 engine-core 令主线「IPC 边界」站全书 0 章开启,'
                    '且抢了 ch11 的 engine-core 首开权'},
    # 量化原理章,走线全在 model_executor/layers/quantization/*,与采样无关。
    'ch26': {'kind': 'primer', 'subsystem': 'quantization',
             'why': '量化数学 primer(证据 layers/quantization 9 步),与采样无关;'
                    '原声明 sampling 会把「量化」画到「解码策略→采样」下并抢走 ch30 的首开权'},
}


# ---- 架构骨架：**取自第 1 章教给读者的那张「一个请求的端到端旅程」**。----
# 用户 2026-07-26：「也许你应该先从第一章开始测绘架构图，再写」——这条点破了前几版的根子。
# 前一版按 cartography 的 L0–L6 结构分层画，那是**读者从没见过的另一套分解**：ch01 教的是
# 入口 → InputProcessor → EngineCore(内含逐拍循环/调度器/KV) → OutputProcessor → 出口，
# 到 ch31 却换成七条结构层带，图不是在长大，是换了一张。现按 ch01 的骨架重定，全书据此生长：
#   · EngineCore 是**容器**(ch01 图里就是个大框，里面装 schedule→execute_model→update
#     与 Scheduler / 分页 KV cache)，后续章节往这个框里加东西；
#   · 入口两扇门、Stage1/Stage3、IPC 边界，都与 ch01 图一一对应。
# 行 = 请求经过的环节(与 roadmap.py 的 STAGES 同源，读者每章都见)。
SKELETON = [
    ('entry',    '入口',        ['entrypoints', 'config-and-wiring']),
    ('stage1',   'Stage 1 输入处理', ['input-processor']),
    ('ipc',      'IPC 边界',    ['async-engine', 'ipc', 'pd-disaggregation']),
    ('core',     'EngineCore（逐拍循环：schedule → execute_model → update）', ['engine-core']),
    ('stage3',   'Stage 3 输出处理', ['output-processor']),
]
# EngineCore 这个**容器**里装的分组（ch01 图里已经画了调度器与 KV cache 两块，其余章节陆续加入）
CORE_GROUPS = [
    ('loop', '循环本体', ['engine-core']),
    ('sched-mem', '调度与显存', ['scheduler', 'kv-cache']),
    ('exec', '执行与并行', ['worker-and-executor', 'model-runner', 'distributed-parallelism']),
    ('model', '模型与算子', ['model-definitions', 'model-architecture',
                             'custom-ops-and-compilation', 'attention', 'quantization']),
    ('decode', '解码策略', ['sampling', 'structured-output', 'spec-decode']),
]
LAYERS = [(r[0], '', r[1], r[2]) for r in SKELETON]
LAYER_OF = {s: lid for lid, _, _, subs in LAYERS for s in subs}
CORE_GROUP_OF = {s: (gid, gname) for gid, gname, subs in CORE_GROUPS for s in subs}
for _s in CORE_GROUP_OF:
    LAYER_OF.setdefault(_s, 'core')


SPINE_RE = re.compile(r'^\s*([\w/\.\-]+\.(?:py|cc|cpp|h|hpp|cu|pyi|td|mlir))\s*:\s*([\dL\-–,\s]+)?\s*[—\-–]\s*(.*)$')


def _chapter_index(cid):
    m = re.search(r'(\d+)', cid or '')
    return int(m.group(1)) if m else 9999


def _evidence_dirs(units, top=3):
    from collections import Counter
    c = Counter(u['path'].rsplit('/', 1)[0] for u in units)
    return [{'dir': d, 'steps': n} for d, n in c.most_common(top)]


def _split_names(n):
    """key_classes 里常写成 'XgrammarBackend / XgrammarGrammar'、'StructuredOutputGrammar (ABC)'，
    拆成可与源码 ClassDef 对齐的裸类名。"""
    n = re.sub(r'\s*\((?:ABC|旧[^)]*)\)', '', n)
    n = re.split(r'（', n)[0]
    return [x.strip() for x in n.split('/') if x.strip() and re.match(r'^[A-Za-z_]\w*$', x.strip())]


_SKIP_BASES = {'object', 'ABC', 'Enum', 'IntEnum', 'str', 'Exception', 'Generic', 'Protocol',
               'NamedTuple', 'TypedDict', 'Struct', 'BaseModel', 'Module', 'MutableSequence',
               'Sequence', 'dict', 'list', 'tuple', 'Mapping'}


def extract_relations(src_root, key_classes):
    """从**真实源码**抽取组件之间的组织关系 —— 架构图要表达的是相互作用与组织关系，
    这些关系必须来自源码，不能靠列类名或凭印象编。

      is_a   : class X(Y)          —— 继承/实现契约（参考图里 Backend 下挂四种实现那种结构）
      has_a  : 带注解的属性类型      —— 持有/组合（SequenceGroup ⊃ Sequence 那种嵌套）
      uses   : 类体内引用到的其他类  —— 创建/调用

    只在本章 key_classes 的集合内连边：架构图不是全量类图，越界只会糊。
    """
    want = {}
    for kc in key_classes:
        for nm in _split_names(kc.get('name') or ''):
            want[nm] = (kc.get('file') or '').split(':')[0]
    files = sorted({f for f in want.values() if f})
    is_a, has_a, uses = [], [], []
    methods = {}      # 类 → 它自己的方法名(用于把「裸方法名」的站点精确归到某个类)
    cfiles = {}       # 类 → 它所在的文件(方法重名时必须靠文件区分:compile_grammar 四个后端都有)
    real_file = {}    # 类 → **实际定义所在的文件**(AST 里 class X 出现的那个文件)。
                      # ⚠️ key_classes 的 file 字段不可靠:DeepseekV4Attention 声明在 deepseek_v4.py,
                      # 实际定义在 layers/deepseek_v4_attention.py —— 站点按声明文件兜底会归错类。
    import ast as _ast
    for f in files:
        fp = Path(src_root) / f
        if not fp.exists():
            continue
        try:
            tree = _ast.parse(fp.read_text(encoding='utf-8'))
        except Exception:
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef) or node.name not in want:
                continue
            for b in node.bases:
                bn = _ast.unparse(b).split('[')[0].strip().rsplit('.', 1)[-1]
                if bn == node.name:
                    continue
                # 子类必在本章 key_classes;基类在集合内→内部契约,不在→**具名外部契约**(如
                # Fp8Config/SupportsPP/KVConnectorBase_V1),也保留——它正是「本章新结构接到哪个已有抽象上」。
                # 只滤掉纯语言/框架基类,它们不是架构信息。
                if bn in _SKIP_BASES:
                    continue
                is_a.append([node.name, bn])
            # 类属性别名:  model_cls = DeepseekV4Model  →  self.model_cls(...) 视同构造 DeepseekV4Model
            # (vLLM 常见写法:ForCausalLM 里 model_cls=XxxModel; self.model=self.model_cls(...))
            alias = {}
            for x in node.body:
                if isinstance(x, _ast.Assign) and isinstance(x.value, _ast.Name) and x.value.id in want:
                    for t in x.targets:
                        if isinstance(t, _ast.Name):
                            alias[t.id] = x.value.id
            own, ref = set(), set()
            for x in _ast.walk(node):
                # self.model_cls(...) / model_cls(...) —— 经别名解析
                if isinstance(x, _ast.Call):
                    fn = x.func
                    aname = fn.attr if isinstance(fn, _ast.Attribute) else (fn.id if isinstance(fn, _ast.Name) else None)
                    if aname in alias and alias[aname] != node.name:
                        own.add(alias[aname])
                if isinstance(x, _ast.AnnAssign) and x.annotation is not None:
                    ann = _ast.unparse(x.annotation)
                    for k in want:
                        if k != node.name and re.search(r'\b' + re.escape(k) + r'\b', ann):
                            own.add(k)
                elif isinstance(x, _ast.Call) and isinstance(x.func, _ast.Name) \
                        and x.func.id in want and x.func.id != node.name:
                    own.add(x.func.id)          # 类体内构造另一个组件 = 组合/持有(ch10 的 LogprobsTensors)
                elif isinstance(x, _ast.Name) and x.id in want and x.id != node.name:
                    ref.add(x.id)
            for k in sorted(own):
                has_a.append([node.name, k])
            for k in sorted(ref - own):
                uses.append([node.name, k])
            methods[node.name] = sorted({x.name for x in node.body
                                         if isinstance(x, (_ast.FunctionDef, _ast.AsyncFunctionDef))})
            cfiles[node.name] = f
            real_file[node.name] = f
    return {'is_a': is_a, 'has_a': has_a, 'uses': uses, 'methods': methods, 'files': cfiles,
            'real_files': real_file}


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
                ov = OVERRIDES.get(cid, {})
                ch_meta[cid] = {
                    'subsystem': ov.get('subsystem', (c.get('subsystem') or '').strip()),
                    'declared': (c.get('subsystem') or '').strip(),
                    'kind': ov.get('kind', 'source'),
                    'override_why': ov.get('why', ''),
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
            # ⚠️ what 可能以 'L1565' 这类残留行号打头(SPINE_RE 只吃进 lines 的前一段),
            # 先剥掉再取符号,否则 symbol 全空、站号全并到该文件首个类(盲审抓出的 ch28 归错)
            w2 = re.sub(r'^L?\d+(?:\s*[—\-–]\s*L?\d+)?\s*', '', what, count=1).lstrip()
            sm = re.match(r'^([A-Za-z_][\w\.]*)\s*[:：]', w2)
            if sm:
                sym = sm.group(1)
            units.append({'path': path, 'lines': lines, 'symbol': sym, 'what': what[:160]})
            idx = _chapter_index(cid)
            if path not in file_first_open or idx < file_first_open[path][0]:
                file_first_open[path] = (idx, cid)
        l3_by_ch[cid] = units

    src_root = Path(instance.source_dir(inst))
    rel_by_ch = {}
    # ---- 组件（类）注册表：架构图的**细粒度节点**。每章 dossier.key_classes 首次出现即登记，
    # introduced_in = 首次讲它的那一章 → 这就是「前面章节铺垫的结构」的来源。----
    classes = OrderedDict()
    for cid in sorted(slug_by_cid, key=_chapter_index):
        meta = ch_meta.get(cid)
        if not meta or not meta['subsystem']:
            continue
        dp = ad / slug_by_cid[cid] / 'dossier' / 'dossier.json'
        if not dp.exists():
            continue
        try:
            d = json.load(open(dp, encoding='utf-8'))
        except Exception:
            continue
        rel_by_ch[cid] = extract_relations(src_root, d.get('key_classes') or [])
        # 用真实定义文件修正 key_class 的 file 字段(declared 不可靠)
        rfiles = rel_by_ch[cid].get('real_files') or {}
        for kc in (d.get('key_classes') or []):
            for _nm in _split_names(kc.get('name') or ''):
                if _nm in rfiles:
                    kc['file'] = rfiles[_nm]
                    break
        for kc in (d.get('key_classes') or []):
            nm = (kc.get('name') or '').strip()
            if not nm:
                continue
            f = (kc.get('file') or '').split(':')[0].strip()
            if nm not in classes:
                classes[nm] = {'name': nm, 'file': f, 'subsystem': meta['subsystem'],
                               'introduced_in': cid,
                               'responsibility': (kc.get('responsibility') or '')[:200]}

    # L2 累积：某子系统首次被哪一章展开
    sub_first = {}
    for cid, meta in ch_meta.items():
        s = meta['subsystem']
        if not s or meta['kind'] == 'overview':   # 导览章不抢首开权
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
            'layer': LAYER_OF.get(s),
            'group': g[1] if g else None,
            'group_cn': g[2] if g else None,
            'opened_in': sub_first.get(s, (9999, ''))[1],
            'chapters': chs,
        })

    # ---- 文件 → 子系统「归属」：画组件间关系箭头的依据。----
    # ⚠️ 不能用「该文件里第一个被登记的类属于谁」——实测会把 vllm/v1/engine/core.py 判给
    # config-and-wiring（因为 ch03 恰好先提到该文件里的某个类），据此画出的架构箭头是错的。
    # 改用**全书证据多数票**：哪个子系统的章在这个文件上停的站最多就归谁；且只在票数有
    # 决定性优势时才采信（>=3 站且占比 >=60%），否则视为「归当前章」——宁可不画箭头，
    # 也不画一条错的架构关系。
    from collections import Counter as _C
    _votes = {}
    for _cid, _u in ((c, u) for c, cc in l3_by_ch.items() for u in cc):
        sub = ch_meta.get(_cid, {}).get('subsystem')
        if sub:
            _votes.setdefault(_u['path'], _C())[sub] += 1
    file_owner = {}
    for f, v in _votes.items():
        mc = v.most_common(2)
        (win, n), tot = mc[0], sum(v.values())
        second = mc[1][1] if len(mc) > 1 else 0
        # 判据用「对亚军的优势」而非绝对占比：vllm/v1/core/sched/scheduler.py 是 23:15:7 的
        # 多数(非过半)，按 60% 占比会被判不决定性，但它显然就是调度器的文件。
        file_owner[f] = {'subsystem': win, 'stations': n, 'share': round(n / tot, 2),
                         'runner_up': second,
                         'decisive': bool(n >= 3 and (second == 0 or n >= 1.4 * second))}

    model = {
        'instance': inst or instance.active_name(),
        'file_owner': file_owner,
        'levels': {
            'L1_stages': [{'id': k, 'name': n, 'sub': s} for k, n, s in L1_STAGES],
            'L2_subsystems': subsystems,
            'layers': [{'id': i, 'code': c, 'name': n, 'subsystems': subs} for i, c, n, subs in LAYERS],
            'core_groups': [{'id': g, 'name': n, 'subsystems': subs} for g, n, subs in CORE_GROUPS],
        },
        'classes': list(classes.values()),
        'chapters': {
            cid: {
                'subsystem': m['subsystem'],
                # ⚠️ subsystem 来自 outline-final.json 的**声明**,不是证据。
                # 用户 2026-07-26 明确:不要预设既有章节地图/声明成立并据此行动。
                # 故同时记录 evidence_dirs(本章走线真正落在的目录,来自 dossier 真源码路径),
                # 并把 verified 默认置 false —— 声明与证据一致才可置 true。
                'evidence_dirs': _evidence_dirs(l3_by_ch.get(cid, [])),
                'relations': rel_by_ch.get(cid, {'is_a': [], 'has_a': [], 'uses': []}),
                'kind': m['kind'],
                'declared_subsystem': m['declared'],
                'override_why': m['override_why'],
                'verified': True,          # vllm 39 章已逐章按证据核过(2026-07-26)

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
    issues = []   # 真问题:会让模型自相矛盾
    notes = []    # 参考提示:7±2 这类经验值,不判失败
    if len(model['levels']['L1_stages']) > 7:
        notes.append(f"L1 阶段 {len(model['levels']['L1_stages'])} 个(参考上限 7)")
    # 认知上限：任一框的直接子节点不得 > 7（>7 就该再抽一层组）
    for st in model['levels']['L1_stages']:
        subs = [s for s in model['levels']['L2_subsystems'] if s['parent_stage'] == st['id']]
        kids = {s['group'] for s in subs if s['group']} | {s['id'] for s in subs if not s['group']}
        # 7 是**经验参考不是硬阈**(用户 2026-07-26):确有必要多放几个就多放,
        # 别为了凑数硬拆出牵强的组——硬拆出的假分类比多一两个节点更伤认知。
        # 故这里只提示、不判失败。
        if len(kids) > 7:
            notes.append(f"阶段「{st['name']}」直接子节点 {len(kids)} 个(参考上限 7)——"
                         f"若分组自然可再抽一层,若牵强则维持现状")
        for gid in {s['group'] for s in subs if s['group']}:
            n = len([s for s in subs if s['group'] == gid])
            if n > 7:
                notes.append(f"组 {gid} 下 {n} 个子系统(参考上限 7)")
    for s in model['levels']['L2_subsystems']:
        if not s['parent_stage']:
            issues.append(f"子系统 {s['id']} 没有归属的 L1 阶段（图上会成孤儿）")
        elif s['parent_stage'] not in L1_KEYS:
            issues.append(f"子系统 {s['id']} 的父阶段 {s['parent_stage']} 不在 7 个主线阶段里")
    for cid, c in model['chapters'].items():
        if not c['subsystem'] and c.get('kind') != 'overview':
            issues.append(f"{cid} 未声明 subsystem（无法挂靠）")   # 导览章无子系统是正常的
        if not c['spine']:
            issues.append(f"{cid} 无 code_spine（本章走线无来源）")
    return issues, notes


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
        iss, notes = check(model)
        print(('  ⚠ 一致性问题 %d 条:' % len(iss)) if iss else '  ✓ 一致性自检通过')
        for i in iss[:12]:
            print('    -', i)
        for n in notes[:6]:
            print('    · 提示(非失败):', n)
    elif a.cmd == 'check':
        iss, notes = check(model)
        print('\n'.join('- ' + i for i in iss) if iss else '✓ 一致性自检通过')
        for n in notes:
            print('· 提示(非失败):', n)
        sys.exit(1 if iss else 0)
    else:
        show(model, a.chapter or 'ch01')


if __name__ == '__main__':
    main()
