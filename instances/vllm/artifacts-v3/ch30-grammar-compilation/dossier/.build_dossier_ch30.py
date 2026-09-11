# -*- coding: utf-8 -*-
"""ch30 dossier builder — embed_excerpts 逐字取自 v0.27.1 pin 源文件（切片保证逐字）。
临时构建脚本：产出 dossier.json 后删除。"""
import json
from pathlib import Path

SRC = Path("E:/Laboratory/Repo2Book/instances/vllm/source")
OUT = Path("E:/Laboratory/Repo2Book/instances/vllm/artifacts-v3/ch30-grammar-compilation/dossier/dossier.json")


def excerpt(rel, start, end, what, elide=None, expect=None):
    """读源文件 [start, end] 闭区间行，逐字作为 excerpt.code。"""
    path = SRC / rel
    lines = path.read_text(encoding="utf-8").splitlines()
    code = "\n".join(lines[start - 1:end])
    assert code.strip(), f"empty excerpt {rel} L{start}-L{end}"
    if expect:
        first = lines[start - 1].strip()
        assert first.startswith(expect), f"{rel} L{start} 首行={first!r} 期望前缀={expect!r}"
    return {
        "path": "vllm/" + rel.replace("vllm/", "", 1) if rel.startswith("vllm/") else "vllm/" + rel,
        "lines": f"L{start}-L{end}",
        "code": code,
        "what": what,
        "elide": elide or [],
    }


E = []

# ---------- 用户面与前端校验（API 进程） ----------
E.append(excerpt("vllm/sampling_params.py", 72, 111,
    "用户面入口：六种约束形态六选一（__post_init__ 双向互斥校验）+ 三个开关 + 私有 _backend/_backend_was_auto（docstring 明示只许 Processor 校验期写）",
    ["L113-L127 的 all_constraints_none/all_non_structural_tag_constraints_none 两个判定函数可另段引（from_sampling_params 判『无约束』的依据）"]))
E.append(excerpt("vllm/sampling_params.py", 949, 966,
    "引擎配置后端与请求级 _backend 的冲突裁决：请求级指定后端被拒；_backend_was_auto 记账防 params 复用误报"))
E.append(excerpt("vllm/sampling_params.py", 1043, 1082,
    "auto 回退阶梯：先试 xgrammar（最优），ValueError 降级——非 tekken Mistral 或 schema 含 guidance 不支持的特性（patternProperties）落 outlines，否则落 guidance；_backend_was_auto=True 记账。注意：阶梯只在 xgrammar→guidance→outlines 三家降级，从不选 lm-format-enforcer",
    ["L1000-L1010 的四个 validate_* 惰性 import 与 L1012-L1042 显式后端分派（xgrammar/guidance/outlines/lm-format-enforcer 各自 validate + Mistral 拒绝）形状直白，可一句话带过"]))
E.append(excerpt("vllm/v1/engine/input_processor.py", 91, 109,
    "前端校验的调用点：API 进程 InputProcessor._validate_params → params.verify(structured_outputs_config, tokenizer)——后端选择发生在请求过线之前的校验期"))

# ---------- 引擎侧：请求构造与阻塞态 ----------
E.append(excerpt("vllm/v1/request.py", 106, 114,
    "带结构化约束的生成请求初始状态直接置 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR（不是 WAITING）——门从出生就关",
    ["L116-L127 的 extra_args/kv_transfer_params 提取与约束解码无关"]))
E.append(excerpt("vllm/v1/request.py", 348, 364,
    "RequestStatus IntEnum：WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 与 WAITING_FOR_REMOTE_KVS/WAITING_FOR_STREAMING_REQ 同族阻塞态；PREEMPTED 之后皆 finished（L357-L358 注释）"))

# ---------- IO 线程与异步编译门 ----------
E.append(excerpt("vllm/v1/engine/core.py", 969, 991,
    "preprocess_add_request：跑在输入处理 IO 线程（docstring 原话 allow request initialization running in parallel with Model forward）；req.use_structured_output 为真即调 grammar_init——线程安全注释原话『grammar_init is only invoked in input processing thread』",
    ["L978-L981 的多模态 mm_receiver_cache 分支与本章无关"]))
E.append(excerpt("vllm/v1/structured_output/__init__.py", 46, 58,
    "external_launcher 同步回退开关：torchrun 每 rank 一个调度器，异步编译会让 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR→WAITING 的跃迁时刻在各 rank 漂移、破坏确定性假设；_grammar_bitmask/_full_mask 预分配字段"))
E.append(excerpt("vllm/v1/structured_output/__init__.py", 70, 80,
    "编译线程池：max_workers=半数 CPU（注释明说编译是 CPU-bound 非 IO-bound，默认 CPU×5 太高）+ tokenizer 缓存构造",
    ["L60-L68 的 fill_bitmask 并行线程池（批>128 且无 spec 才开）与 L81-L97 的 reasoning parser 装配分别归 ch31 与 ch31/thinking 门控"]))
E.append(excerpt("vllm/v1/structured_output/__init__.py", 114, 138,
    "grammar_init 首段：首个结构化请求惰性构造全引擎唯一后端——注释原话『We only support a single backend. We do NOT support different backends on a per-request basis』；_backend 由前端校验期落定",
    ["L139-L164 guidance/outlines/lm-format-enforcer 三个 elif 分支形状完全相同（只换类名，后两者函数内惰性 import），末尾 else 抛 ValueError"]))
E.append(excerpt("vllm/v1/structured_output/__init__.py", 166, 175,
    "编译提交：异步路径 executor.submit 得 Future；同步路径（external_launcher）也把异常 set 进 Future 再存——两条路径对下游同形",
    []))
E.append(excerpt("vllm/v1/structured_output/__init__.py", 177, 192,
    "_create_grammar（线程池工作函数）：取 structured_output_key=(枚举,规格) 调 backend.compile_grammar；注释明示请求已在前端校验过、但编译仍可能失败——Future 携错给调度器、只杀这一个请求",
    []))

# ---------- 请求级容器：Future 三态 ----------
E.append(excerpt("vllm/v1/structured_output/request.py", 21, 48,
    "StructuredOutputRequest：请求级容器。_grammar 三态（Future | 成品 | Exception）；reasoning_* 字段服务思考模型门控（归 ch31）",
    ["L28-L33 reasoning_end_token_index 注释与 L34-L37 reasoner 缓存注释归 ch31/thinking 门控"]))
E.append(excerpt("vllm/v1/structured_output/request.py", 50, 79,
    "门控核心：_check_grammar_completion 以 result(timeout=0.0001) 做 100µs 非阻塞探测；超时返 False；编译抛错把 Exception 存进 _grammar（三态闭环）；grammar property 对外抹平 Future/成品/Exception 差别",
    []))
E.append(excerpt("vllm/v1/structured_output/request.py", 82, 103,
    "get_structured_output_key：六种约束归一成 (枚举, 字符串) 二元组——dict/list 形态的 json/choice 先 json.dumps 归一成可哈希键",
    []))

# ---------- 调度侧状态门 ----------
E.append(excerpt("vllm/v1/core/sched/scheduler.py", 2050, 2062,
    "_is_blocked_waiting_status + _enqueue_waiting_request：三个阻塞态复用同一套『skipped_waiting 侧队』机制——与 WAITING_FOR_REMOTE_KVS/P/D、流式会话同一族",
    []))
E.append(excerpt("vllm/v1/core/sched/scheduler.py", 694, 711,
    "schedule() 遍历 waiting/skipped 队列时的晋级探测：每拍对队首请求试 _try_promote_blocked_waiting_request，未就绪则 pop 出来 prepend 进 step_skipped_waiting（本拍跳过、不挡别人）",
    []))
E.append(excerpt("vllm/v1/core/sched/scheduler.py", 2678, 2703,
    "_try_promote_blocked_waiting_request 的语法分支：grammar property 为 None（未就绪）返 False；isinstance(grammar, Exception)（编译失败）记入 grammar_compile_error_reqs 返 False；否则 status→WAITING 晋级",
    ["L2682-L2693 的 WAITING_FOR_REMOTE_KVS 晋级分支（P/D 章）与 L2705-L2707 流式分支形状直白"]))
E.append(excerpt("vllm/v1/core/sched/scheduler.py", 1954, 1962,
    "编译失败收尾：update_from_output 末尾收割 grammar_compile_error_reqs → finish_requests(FINISHED_ERROR)——只杀编译失败的那一个请求，引擎与同批其他请求无恙",
    []))

# ---------- 契约两层 ABC ----------
E.append(excerpt("vllm/v1/structured_output/backend_types.py", 19, 28,
    "StructuredOutputOptions 六形态枚举 + StructuredOutputKey 类型别名（(枚举, 规格) 可哈希键）",
    []))
E.append(excerpt("vllm/v1/structured_output/backend_types.py", 31, 95,
    "StructuredOutputGrammar ABC——请求级契约六方法：accept_tokens（推进）/validate_tokens（试走不推进）/rollback/fill_bitmask（与采样的唯一接口）/is_terminated/reset",
    []))
E.append(excerpt("vllm/v1/structured_output/backend_types.py", 98, 136,
    "StructuredOutputBackend ABC——引擎级契约三方法（compile_grammar/allocate_token_bitmask/destroy）+ vllm_config/tokenizer/vocab_size 三字段：编译器与词表这类重资源全引擎共享，状态机逐请求独立",
    []))

# ---------- xgrammar 深挖 ----------
E.append(excerpt("vllm/v1/structured_output/backend_xgrammar.py", 60, 76,
    "XgrammarBackend.__post_init__ 尾段：TokenizerInfo.from_huggingface（词表进编译器——FSM 直接在 token 词表上答『下一步谁合法』）→ GrammarCompiler(max_threads=8, cache_enabled=True, cache_limit_bytes=VLLM_XGRAMMAR_CACHE_MB·1MB 默认 512MB) → num_speculative_tokens",
    ["L42-L59 的 is_mistral_tokenizer 分支手工拼 TokenizerInfo（RAW/BYTE_FALLBACK），属 tokenizer 兼容细节"]))
E.append(excerpt("vllm/v1/structured_output/backend_xgrammar.py", 78, 126,
    "compile_grammar 五分派：JSON/JSON_OBJECT/GRAMMAR/REGEX（带 ReDoS 超时）/STRUCTURAL_TAG——**无 CHOICE 分支**（choice 已在前端校验期被原地改写成 EBNF grammar）；产物包成 XgrammarGrammar(matcher=GrammarMatcher(ctx, max_rollback_tokens=num_speculative_tokens))",
    ["L96-L110 STRUCTURAL_TAG 的 deprecated 分支（structures/triggers 拆 StructuralTagItem 列表）是旧格式兼容，只留『新旧两条编译路径』一句"]))
E.append(excerpt("vllm/v1/structured_output/backend_xgrammar.py", 128, 132,
    "allocate_token_bitmask / destroy：xgr.allocate_token_bitmask(max_num_seqs, vocab_size)——掩码张量形状与所有权在这里定（ch31 首对象）",
    []))
E.append(excerpt("vllm/v1/structured_output/backend_xgrammar.py", 135, 203,
    "XgrammarGrammar：六方法的参考实现。accept_tokens 逐 token accept、失败即 False；validate_tokens 先真 accept 一遍再 rollback（试走=回退的等价性在 num_speculative_tokens 以内成立）；fill_bitmask 一行委托 matcher.fill_next_token_bitmask",
    []))
E.append(excerpt("vllm/v1/structured_output/backend_xgrammar.py", 293, 303,
    "校验期的原地改写：choice 先 choice_as_grammar 转 EBNF、from_ebnf 试编通过后 so_params.choice=None; so_params.grammar=choice_grammar——六种入口枚举到引擎侧只剩五种的原因",
    []))
E.append(excerpt("vllm/v1/structured_output/utils.py", 553, 561,
    "choice_as_grammar：choice 列表 → `root ::= \"a\" | \"b\"` 的 EBNF 串（转义引号/反斜杠）",
    []))
E.append(excerpt("vllm/v1/structured_output/utils.py", 48, 83,
    "compile_regex_with_timeout：单线程池 + future.result(timeout=VLLM_REGEX_COMPILATION_TIMEOUT_S 默认 5s)——防嵌套量词 (a+)+ 类 pattern 的 DFA 状态空间指数爆炸把编译线程永久挂死",
    []))

# ---------- 每拍语法活（F6 回收）与交棒 ----------
E.append(excerpt("vllm/v1/structured_output/__init__.py", 194, 205,
    "_fill_bitmasks：一请求一行——apply_bitmask 且未 terminated 则 grammar.fill_bitmask 写行；否则整行 fill_(-1) 全允许（非语法请求行的初始化语义）",
    []))
E.append(excerpt("vllm/v1/structured_output/__init__.py", 212, 234,
    "grammar_bitmask 头段：惰性预分配 [max_batch×(1+max_num_spec_tokens)] 行掩码（每个 spec 位一行、再加 bonus/非 spec 位一行）；行序以 structured_output_request_ids 为准（批装配/并行填充/spec 窗口预推进+rollback 归 ch31）",
    ["L236-L350 的批装配主循环（并行分支/串行分支+spec 窗口+thinking 探测）与 L352-L359 numpy 转换归 ch31，本章只引到『每请求 fill 自己那一行』为止"]))
E.append(excerpt("vllm/v1/core/sched/scheduler.py", 1646, 1668,
    "get_grammar_bitmask（本章终点/交棒点）：has_structured_output_requests 快返 None；过滤掉 is_prefill_chunk（prefill 中段不算掩码）；交给 manager.grammar_bitmask 后包成 GrammarOutput(request_ids, bitmask)",
    ["本章只作交棒点展示：掩码跨进程 ndarray 序列化、worker 侧重排+pinned H2D+apply_token_bitmask_inplace 归 ch31"]))
E.append(excerpt("vllm/v1/core/sched/scheduler.py", 1817, 1843,
    "每拍⑤·谁推进 FSM：update_from_output 里 should_advance 检查后 trim_reasoning_for_advance 裁掉思考 token（门控归 ch31），grammar.accept_tokens 吃本步真采样出的 token——FSM 状态的唯一写者是调度器线程；拒绝即 FINISHED_ERROR 终止请求",
    []))

# ---------- 对照后端 ----------
E.append(excerpt("vllm/v1/structured_output/backend_guidance.py", 203, 217,
    "GuidanceGrammar 对照：rollback 要扣 rollback_lag（EOS 后回滚少退一格）——同契约、实现细节分歧的最好例证；fill_bitmask 委托 llguidance_torch（matcher stopped 自动返 EOS 掩码）",
    ["L158-L201 的 accept_tokens/validate_tokens 与 L219-L221 reset 形状同契约"]))
E.append(excerpt("vllm/v1/structured_output/backend_outlines.py", 73, 97,
    "OutlinesBackend.compile_grammar 对照：一切先转正则（JSON→build_regex_from_schema、CHOICE→escape+join），再编 DFA Index（带自建缓存）；Guide(index, max_rollback=max_rollback_tokens)——能力矩阵藏在分派里：GRAMMAR 直接 ValueError",
    []))
E.append(excerpt("vllm/v1/structured_output/backend_outlines.py", 99, 105,
    "outlines 的 allocate_token_bitmask：torch.full((n, ceil(V/32)), -1, int32, pin_memory)——位掩码内存布局的最直白证据（-1=全 1=全允许；xgrammar/guidance 由各自库分配同构张量）",
    []))
E.append(excerpt("vllm/v1/structured_output/backend_lm_format_enforcer.py", 124, 139,
    "lm-format-enforcer 对照：撑不住回滚干脆拒绝——max_rollback_tokens>0 直接 ValueError『does not support speculative tokens』；rollback 契约不是免费的",
    []))

EXCERPTS = E

# ---------- 校验：切片行数自洽（N 行 join = N-1 个换行） ----------
for e in EXCERPTS:
    lo, hi = (int(x[1:]) for x in e["lines"].split("-"))
    assert e["code"].count("\n") == hi - lo, f"{e['path']} {e['lines']} 行数不自洽"
    assert not e["code"].endswith("\n"), e["lines"]

DOSSIER = {
    "chapter_id": "ch30-grammar-compilation",
    "part": "VII",
    "title": "约束解码 I：语法编译",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "chapter_kind": "code",
    "note_on_kind": "标准代码章（非 primer）：implementer 按 subtraction_plan 只删不增、lint_fidelity 全跑。Part VII 第 2 章（l0_zoom=采样列结构化输出组，L0 的 C3 块『结构化输出位掩码』）。回收 F6（ch9 埋的 grammar bitmask 窗口）：ch9 站 4 只讲掩码在这拍的位置、『表怎么算出来是约束解码两章的整章主场』——本章交出前半（语法→FSM→每步一行开关表怎么算），后半（批装配/跨进程/worker 应用/两段式窗口）归 ch31。",
    "hook_suggestion": "「保证输出是合法 JSON」为什么不靠生成后校验重试，而要先把 schema 编译成逐 token 状态机？编译一张复杂 schema 是毫秒到百毫秒级的 CPU 活，引擎忙循环一拍才几十毫秒——编译塞在哪个线程、没编译完的请求挡不挡别人？六种约束入口、四个后端（xgrammar/guidance/outlines/lm-format-enforcer）同台，auto 按什么阶梯选、谁在校验期就被原地改写？",
    "scope_note": "章界（与 ch31 的切分，写作红线）：本章=『语法怎么变成每步一张合法开关表』的前半——①前端校验期选后端（auto 阶梯 + choice→EBNF 原地改写，sampling_params.py）②请求置 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR ③IO 线程 grammar_init 异步提交编译（线程池+Future）④状态门：skipped_waiting→100µs 探测晋级 / 编译失败 Exception→FINISHED_ERROR 只杀一个请求 ⑤契约两层 ABC + xgrammar 深挖 + 四后端对照 ⑥每拍③ fill_next_token_bitmask（F6 回收·一请求一行）+ 每拍⑤ accept_tokens 推进（FSM 唯一写者=调度器）。ch31=bitmask 落地：grammar_bitmask 批装配全貌/并行填充/spec 窗口预推进+rollback/ndarray 跨进程序列化/worker 侧按批序+spec 偏移重排+pinned H2D+apply_token_bitmask_inplace/两段式 execute_model GPU 窗口/thinking 门控展开（should_fill_bitmask/should_advance/trim_reasoning_for_advance/__init__.py:L236-L350）。本章对这些只点名+给锚点，不展开。",
    "code_spine": [
        "vllm/sampling_params.py:L72-L127  StructuredOutputsParams：用户面六选一（json/regex/choice/grammar/json_object/structural_tag）+ _backend/_backend_was_auto 私有字段",
        "vllm/v1/engine/input_processor.py:L91-L109  前端校验入口（API 进程）：_validate_params → SamplingParams.verify(structured_outputs_config, tokenizer)",
        "vllm/sampling_params.py:L923-L1086  _validate_structured_outputs：后端确定与请求内容校验；auto 走 xgrammar→guidance→outlines 降级阶梯（L1043-L1082）",
        "vllm/v1/structured_output/backend_xgrammar.py:L272-L363  validate_xgrammar_grammar：前端校验期真试编一遍 + choice→EBNF 原地改写（L293-L303）",
        "vllm/v1/request.py:L87-L114  Request 构造：StructuredOutputRequest.from_sampling_params 挂载；生成请求初始状态直接置 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR",
        "vllm/v1/engine/core.py:L969-L991  preprocess_add_request（IO 线程）→ grammar_init：编译不占忙循环",
        "vllm/v1/structured_output/__init__.py:L114-L175  grammar_init：惰性建全引擎唯一后端 + executor.submit(_create_grammar) 得 Future",
        "vllm/v1/structured_output/__init__.py:L177-L192  _create_grammar：structured_output_key → backend.compile_grammar；失败经 Future 携错",
        "vllm/v1/structured_output/request.py:L50-L79  _check_grammar_completion：100µs 非阻塞探测；Future→成品/Exception 原地替换",
        "vllm/v1/core/sched/scheduler.py:L2050-L2062  _is_blocked_waiting_status + _enqueue_waiting_request：阻塞态进 skipped_waiting 侧队",
        "vllm/v1/core/sched/scheduler.py:L694-L711 + L2678-L2703  每拍晋级探测 _try_promote_blocked_waiting_request：就绪晋 WAITING / Exception 记 grammar_compile_error_reqs",
        "vllm/v1/core/sched/scheduler.py:L1954-L1962  update_from_output 收尾：grammar_compile_error_reqs → FINISHED_ERROR（只杀一个请求）",
        "vllm/v1/structured_output/backend_xgrammar.py:L35-L76  XgrammarBackend.__post_init__：TokenizerInfo + GrammarCompiler(cache_enabled=True)",
        "vllm/v1/structured_output/backend_xgrammar.py:L78-L126  compile_grammar 五分派 → GrammarMatcher(ctx, max_rollback_tokens=num_speculative_tokens) → XgrammarGrammar",
        "vllm/v1/core/sched/scheduler.py:L1646-L1668  get_grammar_bitmask（每拍③、本章终点）：has_structured_output_requests 快返 + is_prefill_chunk 过滤 → GrammarOutput 交棒",
        "vllm/v1/structured_output/__init__.py:L194-L205 + L212-L234  _fill_bitmasks（一请求一行）+ grammar_bitmask 预算分配 [max_batch×(1+spec), ceil(V/32)]",
        "vllm/v1/core/sched/scheduler.py:L1817-L1843  每拍⑤：update_from_output → should_advance → trim_reasoning_for_advance → grammar.accept_tokens（FSM 唯一写者）",
    ],
    "stations": [
        {
            "n": 1,
            "where": "vllm/sampling_params.py:L923-L1086",
            "what": "API 进程校验期选后端：InputProcessor._validate_params（input_processor.py:L91-L109）→ verify → _validate_structured_outputs——显式后端各自 validate_*；auto 走 xgrammar→guidance→outlines 降级阶梯（L1043-L1082）落 params._backend（docstring 明示只许此处写）。xgrammar 校验里 choice 被原地改写成 EBNF grammar（backend_xgrammar.py:L293-L303）——六种入口、引擎侧只见五种；regex 校验带 ReDoS 超时（utils.py:L48-L83，默认 5s）"
        },
        {
            "n": 2,
            "where": "vllm/v1/request.py:L87-L114",
            "what": "Request 构造：StructuredOutputRequest.from_sampling_params 挂上请求（无约束返回 None）；生成请求初始状态直接置 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR（不是 WAITING）——门从出生就关，与 WAITING_FOR_REMOTE_KVS/WAITING_FOR_STREAMING_REQ 同族阻塞态复用同一套侧队机制"
        },
        {
            "n": 3,
            "where": "vllm/v1/engine/core.py:L983-L990",
            "what": "EngineCore IO 线程收件：preprocess_add_request（L1718 调用点，docstring 原话 allow request initialization running in parallel with Model forward）里 use_structured_output 为真即调 grammar_init——线程安全注释原话『grammar_init is only invoked in input processing thread』，编译在忙循环之外启动"
        },
        {
            "n": 4,
            "where": "vllm/v1/structured_output/__init__.py:L114-L192",
            "what": "grammar_init：首个结构化请求惰性构造全引擎唯一后端（xgrammar/guidance/outlines/lm-format-enforcer 四选一，注释明示不支持逐请求）；executor.submit(_create_grammar) 把 Future 存进请求（external_launcher 模式退同步、异常也 set 进 Future）；_create_grammar 取 structured_output_key=(枚举,规格) 调 backend.compile_grammar，失败 logger.exception 后由 Future 携错"
        },
        {
            "n": 5,
            "where": "vllm/v1/structured_output/backend_xgrammar.py:L78-L126",
            "what": "xgrammar 编译（默认后端主线）：五分派 compile_json_schema/compile_grammar/compile_regex(带超时)/compile_structural_tag——无 CHOICE 分支（站 1 已改写）；GrammarCompiler(cache_enabled=True, cache_limit_bytes=VLLM_XGRAMMAR_CACHE_MB·1MB 默认 512MB) 同 schema 重编命中库内缓存；产物 GrammarMatcher(ctx, max_rollback_tokens=num_speculative_tokens) 包成 XgrammarGrammar——FSM 在 token 词表上（TokenizerInfo 进编译器），回答『下一步哪些 token 合法』"
        },
        {
            "n": 6,
            "where": "vllm/v1/core/sched/scheduler.py:L2058-L2062 · L2695-L2703 · L1954-L1962",
            "what": "状态门（本章命门）：阻塞态分流 skipped_waiting 侧队；每拍 schedule 遍历 waiting/skipped 时 _try_promote_blocked_waiting_request 经 grammar property（内部 result(timeout=0.0001) 100µs 探测）——就绪 status→WAITING 晋级可调度；编译抛错（Exception 存在 _grammar）记入 grammar_compile_error_reqs，本拍 update_from_output 收尾 finish_requests(FINISHED_ERROR) 只杀这一个请求、不挡别人"
        },
        {
            "n": 7,
            "where": "vllm/v1/core/sched/scheduler.py:L1646-L1668 → structured_output/__init__.py:L194-L234 → backend_xgrammar.py:L195-L196",
            "what": "F6 回收·表怎么算出来：每拍③ get_grammar_bitmask（has_structured_output_requests 快返；is_prefill_chunk 过滤——prefill 中段不算）→ manager.grammar_bitmask 在预分配 [max_batch×(1+max_num_spec), ceil(V/32)] int32 掩码上逐请求一行：grammar.fill_bitmask 即 matcher.fill_next_token_bitmask，用 FSM 当前状态写『下一步谁合法』；is_terminated 则整行填 -1 全允许；非语法请求行初始化 -1（全允许、不会被采到）"
        },
        {
            "n": 8,
            "where": "vllm/v1/core/sched/scheduler.py:L1817-L1843",
            "what": "每拍⑤·谁推进 FSM：update_from_output 里 should_advance 检查（thinking 门控归 ch31）→ trim_reasoning_for_advance 裁掉思考 token → grammar.accept_tokens 吃本步真采样出的 token 推进状态机——FSM 的唯一写者是调度器线程；FSM 拒绝即 FINISHED_ERROR 终止请求。validate_tokens 只喂 spec draft 做『不推进的试走』（L2163-L2166/L2191-L2194）。交棒：GrammarOutput(req_ids, ndarray) 经 sample_tokens RPC 去 worker——批装配/并行填充/spec 窗口 rollback/重排+H2D+apply_token_bitmask_inplace/两段式 GPU 窗口全部归 ch31"
        },
    ],
    "stations_note": "站号 = 一个带 JSON schema 的请求流经代码的顺序：第 1 站在 API 进程校验期（请求过线之前）；第 2-6 站在 EngineCore 进程（构造→IO 线程提交→编译线程→状态门晋级——编译期间请求持 Future 挂在侧队，不占批位）；第 7-8 站是晋级后每个 decode 拍的语法活（③ 算表→④ 掩码落地+采样[ch29/ch31]→⑤ 推进 FSM），循环到 is_terminated。正文按讲解需要编排。",
    "key_classes": [
        {"name": "StructuredOutputsParams", "file": "vllm/sampling_params.py:L72", "responsibility": "用户面的约束描述：六种约束形态六选一（__post_init__ 双向互斥）+ disable_any_whitespace/disable_additional_properties/whitespace_pattern 三开关 + 私有 _backend/_backend_was_auto（docstring 明示只许 Processor 校验期写）"},
        {"name": "StructuredOutputRequest", "file": "vllm/v1/structured_output/request.py:L21", "responsibility": "请求级容器：持有 params、_grammar 三态（Future|成品|Exception）、structured_output_key 缓存与 reasoning 字段（ch31）；grammar property 对外抹平三态差别，调度器只读它"},
        {"name": "StructuredOutputManager", "file": "vllm/v1/structured_output/__init__.py:L35", "responsibility": "引擎级编排者：惰性构造唯一后端、把编译提交线程池（grammar_init/_create_grammar）、每步装配 bitmask（grammar_bitmask，批装配归 ch31）、推理模型门控（should_advance/should_fill_bitmask，归 ch31）"},
        {"name": "StructuredOutputGrammar (ABC)", "file": "vllm/v1/structured_output/backend_types.py:L31", "responsibility": "请求级契约六方法：accept_tokens / validate_tokens / rollback / fill_bitmask / is_terminated / reset——『语法在这里变成掩码』的唯一接口是 fill_bitmask"},
        {"name": "StructuredOutputBackend (ABC)", "file": "vllm/v1/structured_output/backend_types.py:L98", "responsibility": "引擎级契约三方法：compile_grammar / allocate_token_bitmask / destroy；dataclass 字段 vllm_config/tokenizer/vocab_size 固定所有后端的构造签名——编译器/词表这类重资源全引擎共享"},
        {"name": "StructuredOutputOptions / StructuredOutputKey", "file": "vllm/v1/structured_output/backend_types.py:L19", "responsibility": "六种约束形态的枚举 + (枚举, 规格字符串) 可哈希键类型——编译分派的开关"},
        {"name": "XgrammarBackend / XgrammarGrammar", "file": "vllm/v1/structured_output/backend_xgrammar.py:L35 / L135", "responsibility": "默认后端：GrammarCompiler(cache_enabled=True) 编译出 CompiledGrammar，包一个 GrammarMatcher(max_rollback_tokens=num_speculative_tokens) 作为逐 token 状态机；六方法的参考实现"},
        {"name": "GuidanceBackend / GuidanceGrammar", "file": "vllm/v1/structured_output/backend_guidance.py:L87 / L142", "responsibility": "llguidance 后端：LLMatcher 消费 token；rollback_lag（EOS 后回滚少退一格）与『matcher stopped 自动返 EOS 掩码』是同契约不同实现的例证；编译无缓存"},
        {"name": "OutlinesBackend / OutlinesGrammar", "file": "vllm/v1/structured_output/backend_outlines.py:L52 / L111", "responsibility": "outlines_core 后端：一切先转正则再编 DFA Index（自建 dict/可选 SQLite 缓存）；Guide(max_rollback=…) 走状态；is_terminated 故意延迟一步让 EOS 还能发出；能力矩阵藏在 compile_grammar 分派（GRAMMAR 直接 ValueError）"},
        {"name": "LMFormatEnforcerBackend / LMFormatEnforcerGrammar", "file": "vllm/v1/structured_output/backend_lm_format_enforcer.py:L94 / L43", "responsibility": "不持状态机、持 current_tokens_prefix 前缀列表按需查 allowed_tokens；显式拒绝投机解码（max_rollback_tokens>0 抛错）——rollback 契约不是免费的活证"},
        {"name": "RequestStatus", "file": "vllm/v1/request.py:L348", "responsibility": "请求状态枚举；WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 是本章的『门』（真实枚举名，不是 WAITING_FOR_FSM）"},
        {"name": "GrammarOutput", "file": "vllm/v1/core/sched/output.py:L287", "responsibility": "(structured_output_request_ids, grammar_bitmask ndarray) 二元组——每拍③的产物、跨进程交棒 ch31 的载体"},
    ],
    "data_flow": [
        "deepread 卡对应：model-sample.json data_flow 的步骤 1/3（调度→bitmask 重叠）与步骤 8（accept_tokens 推进）是本章每拍侧；本章补充『请求到达→编译→晋级』的入场侧（engine-loop.json contracts[0] 输入链的 grammar 切片）。以下按一个带 JSON schema 的请求走（行号全部对 v0.27.1 现核）：",
        "1. 前端（API 进程）：用户给 sampling_params.structured_outputs 六选一 → InputProcessor._validate_params（vllm/v1/engine/input_processor.py:L91-L109）→ SamplingParams.verify（L755-L770）→ _validate_structured_outputs（L923-L1086）：auto 阶梯先 validate_xgrammar_grammar（真试编一遍），失败降 guidance/outlines，落 params._backend；choice 在校验期被原地改写成 grammar（backend_xgrammar.py:L293-L303）。",
        "2. 过线：EngineCoreRequest 携 sampling_params（内嵌 _backend 已定）经 ZMQ 到 EngineCore IO 线程。",
        "3. IO 线程：preprocess_add_request（core.py:L969-L991）构造 Request——structured_output_request 挂载、初始 status=WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR（request.py:L113-L114）；随即 grammar_init（__init__.py:L114-L175）：惰性建唯一后端 + executor.submit(_create_grammar) 得 Future 存进请求。",
        "4. 编译线程：_create_grammar（L177-L192）取 structured_output_key → backend.compile_grammar（xgrammar 分派 backend_xgrammar.py:L78-L126）→ GrammarMatcher 包成 XgrammarGrammar；同 schema 命中 GrammarCompiler 库内缓存（L65-L70）。失败：logger.exception + raise，Future 携错。",
        "5. 忙循环：Scheduler.add_request 经 _enqueue_waiting_request 把阻塞态请求放进 skipped_waiting 侧队（scheduler.py:L2058-L2062）——不进正常 waiting、不占批位。",
        "6. 每拍 schedule()：遍历 waiting/skipped 时对队首试 _try_promote_blocked_waiting_request（L694-L711 → L2678-L2703）——grammar property 触发 100µs 探测；就绪 status→WAITING 本拍可调度；Exception 记入 grammar_compile_error_reqs，本拍收尾 FINISHED_ERROR（L1954-L1962）只杀这一个请求。",
        "7. 晋级后的每个 decode 拍：③ get_grammar_bitmask（scheduler.py:L1646-L1668，EngineCore.step 在 execute_model 发起后调——core.py:L597）→ manager.grammar_bitmask 逐请求一行 fill_next_token_bitmask（__init__.py:L194-L205；预算分配 L225-L234）→ GrammarOutput 过 sample_tokens RPC 去 worker（worker 侧落地归 ch31）。",
        "8. ⑤ update_from_output（scheduler.py:L1817-L1843）：should_advance → trim_reasoning_for_advance（ch31）→ grammar.accept_tokens 用真采样 token 推进 FSM——下一拍的③ 因此算出新的合法集。FSM 拒绝 → FINISHED_ERROR。",
        "9. 终态：matcher.is_terminated() 为真后 _fill_bitmasks 走 else 分支整行 -1（不再约束），请求由常规停止判定收尾。",
    ],
    "why_chains": [
        {
            "id": "WC1",
            "card": "deepread/model-sample.json why_chains[4]（结构化输出=语法编译成 FSM+位掩码预过滤）——本章取其前半（编译侧+单请求 fill），位掩码的批装配/传输/worker 应用归 ch31。锚点已对 v0.27.1 逐行复核一致。",
            "decision": "结构化输出 = 语法编译成 FSM + 位掩码预过滤（采样前把非法 token 的 logits 打成 -inf），而不是采样后重试或逐请求 CPU 枚举",
            "old_design": "朴素做法：a) 生成→校验→重采样循环（每步多轮 GPU 前向，延迟成倍）；b) 每步在 CPU 枚举合法 token id 列表传给采样器（v0 早期 outlines 集成形态，合法集是 list[int]）。",
            "pain": "vocab 13 万，枚举/序列化合法 token 集合本身就有 O(V) 开销且无法与批处理 GPU 管线拼合；逐请求 CPU masking 逐 token 串行；重试路径根本不保证有限步终止。",
            "solution": "语法→FSM：JSON Schema/regex/EBNF/structural_tag 编译成 xgrammar FSM（vllm/v1/structured_output/backend_xgrammar.py:L78-L126，GrammarCompiler 带 LRU+字节预算缓存 cache_limit_bytes=VLLM_XGRAMMAR_CACHE_MB·1MB，L65-L70）；每步 matcher.fill_next_token_bitmask 在预分配 [max_batch*(1+max_spec), ceil(V/32)] int32 位掩码上标记下一步不合法 token（vllm/v1/structured_output/__init__.py:L232-L234 分配预算、L194-L205 逐请求填充与全禁兜底）；掩码以 numpy 跨进程传 worker（L356-L359 注释：ndarray 序列化比 tensor 高效）；worker 侧重排+pinned H2D+apply_token_bitmask_inplace 归 ch31。v0.27 后端从 xgrammar 独苗扩成 4 选 1：xgrammar/guidance/outlines/lm-format-enforcer（__init__.py:L129-L164，引擎级单后端）；regex 编译加 ReDoS 超时（utils.py:L48-L83，防嵌套量词指数爆炸）；outlines 加 SQLite 磁盘索引缓存（utils.py:L218-L301，防 pickle 任意代码执行）。",
            "cost": "① 引擎级单后端，全引擎只能一种（L124-L127 注释原话）；② FSM 活在调度器进程，每步一次 CPU FSM walk + H2D 传输（批>128 且无 spec 时并行填充归 ch31）；③ 位掩码常驻 max_num_seqs*(1+num_spec)*ceil(V/32)*4B（128k vocab 下每行 16KB）；④ xgrammar 不支持的 JSON 特性（multipleOf/uniqueItems/非标 format 等）直接拒单（backend_xgrammar.py:L225-L269）；⑤ 掩码对 logits 的 -inf 写入发生在采样管线之前——与惩罚/bad_words 的先后顺序是隐式契约（ch29 站 1 的交接点）。"
        },
        {
            "id": "WC2",
            "card": "deepread/model-sample.json why_chains[5]（语法编译异步化+调度状态门）——本章主 why 链。锚点已对 v0.27.1 逐行复核（accept 推进卡内引 L1817-L1845，实测 L1817-L1843）。",
            "decision": "语法编译异步化 + 调度状态门：grammar 没编译完的请求不进批、不挡别人",
            "old_design": "同步编译：请求带 schema 到达 → 阻塞编译完 → 才能调度；编译期间引擎空转。",
            "pain": "复杂 schema 编译是 CPU 密集（毫秒到百毫秒级，随嵌套深度），阻塞在关键路径直接打爆 TTFT，且一个慢 schema 请求拖住整个引擎循环。",
            "solution": "grammar_init 在输入处理线程把编译提交 ThreadPool（vllm/v1/engine/core.py:L985-L990 线程安全注释：'grammar_init is only invoked in input processing thread'；structured_output/__init__.py:L166-L174；线程数=半数 CPU L70-L77）；StructuredOutputRequest._grammar 存 Future，_check_grammar_completion 用 result(timeout=0.0001) 非阻塞探测（vllm/v1/structured_output/request.py:L50-L59）；未就绪请求置 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 进 skipped_waiting 侧队（vllm/v1/core/sched/scheduler.py:L2051-L2056 _is_blocked_waiting_status），就绪后晋升回 WAITING（L2695-L2703）。采完样由调度器推进 FSM：update_from_output 里 should_advance 检查后 grammar.accept_tokens（scheduler.py:L1817-L1843）——FSM 状态的唯一写者是调度器线程。v0.27 两条新线：①编译失败经 Future 闭环——Exception 存进 _grammar（request.py:L57-L58）→ grammar_compile_error_reqs（scheduler.py:L2699-L2701）→ FINISHED_ERROR 只杀一个请求（L1954-L1962）；②thinking 模型联动（should_fill_bitmask/should_advance 判『思考段结束后语法才生效』、trim_reasoning_for_advance 处理一步内混思考+语法 token 的边界，修复 #44006/#42452）与 spec 窗口预推进+rollback（__init__.py:L294-L350）都归 ch31 展开。",
            "cost": "100µs 轮询是忙等变体（每步每请求一次探测）；external_launcher（torchrun 每 rank 一个调度器）下各 rank 的门迁移时间会漂移、破坏确定性假设 → 该模式回退同步编译（__init__.py:L46-L55 注释原话）；thinking 联动的 mid-window 边界逻辑复杂，git 上连串 bugfix 实证其脆弱性；Future 三态（Future|成品|Exception）让 grammar property 的类型面变宽（StructuredOutputGrammar | Exception | None），每个读点都要防 Exception。"
        },
        {
            "id": "WC3",
            "card": "deepread/engine-loop.json why_chains[4]（循环四段顺序、慢操作全部挪出循环体）——本章只消费其 grammar 切片（编译在 IO 线程 + 掩码塞进前向窗口）。",
            "decision": "慢操作全部挪出忙循环：语法编译在 input socket IO 线程完成，位掩码计算塞进 GPU 前向窗口",
            "old_design": "朴素单线程引擎（v0 LLMEngine.step 的形态）：tokenize/调度/前向/采样/后处理全在一个同步循环里，做一步等一步。",
            "pain": "忙循环是单线程（core.py:L1378-L1389 run_busy_loop 只有一个 Python 线程驱动一切），任何阻塞都让 GPU 空转。一步 forward 只有几十毫秒（2048-8192 token 预算量级）；10ms 串行 CPU 杂务 ≈ 20%+ 吞吐损失；结构化输出的 grammar 编译/掩码计算若串行在关键路径上，每拍白加一段 CPU 时间。",
            "solution": "core.py:L584-L614 step() 精确排布：L596 execute_model(non_block=True) 发起前向立即返回 Future → L597 趁 GPU 在算，CPU 算 get_grammar_bitmask → L602-L604 future.result() 后 sample_tokens(grammar_output)。同类设计贯穿循环外缘：Request 构造 + grammar 编译在 input socket IO 线程完成（core.py:L1715-L1721 调 preprocess_add_request L969-L991，docstring 原话 'allow request initialization running in parallel with Model forward'）——本章站 3 的线程归属即此。",
            "cost": "execute/sample 两段式引入跨调用暂存态 ExecuteModelState——顺序错误直接 RuntimeError（'State error'，gpu_model_runner.py:L4171-L4175）；两段式契约自认技术债（worker_base.py 注释 'this design may be changed in future if/when structured outputs parallelism is re-architected'）——归 ch31 展开。"
        },
    ],
    "embed_excerpts": EXCERPTS,
    "theory": [
        "为什么必须『采样前掩码』而不是『生成后修复』：设词表 V，语法在状态 s 下的合法集 A(s) ⊆ V。自由采样从 softmax 分布 p 里抽样，落在 V\\A(s) 的概率一般非零，一旦落错，后续所有 token 都在一条不可能回到合法态的轨迹上；后处理修复要解的是『最近合法串』问题，既无唯一解也不保证保持语义。掩码法把分布替换成在 A(s) 上重整的条件分布——等价于对语言模型分布做逐步条件化，天然保证输出串一定在语法定义的语言里（前提是 A(s) 非空）。",
        "位掩码的表示与代价：掩码按每 token 一位打包成 int32 行，行宽 ceil(|V|/32)。outlines / lm-format-enforcer 显式写成 torch.full((n, (vocab_size + 31) // 32), -1, dtype=torch.int32)（-1 即全 1，默认全允许，backend_outlines.py:L99-L105）；xgrammar / guidance 由各自库的 allocate_token_bitmask 分配同构张量。预算 [max_num_seqs×(1+num_spec), ceil(V/32)]（__init__.py:L225-L234）：spec 每个位置一行、再给 bonus/非 spec 位一行。对 |V| 约 13 万的词表（DeepSeek 129280），一行约 16KB（ceil(129280/32)=4040 个 int32 × 4B），比 |V| 个 float32 的 logits（≈505KB）恰好小 32 倍——这正是掩码能在每步热路径上跑的原因。（数值为静态推演；正文不得给未实测的毫秒数。）",
        "编译期 vs 运行期的代价分离：语法编译是 O(语法规模 × 词表规模) 量级的一次性代价（要为每个状态求出词表上的可接受集合），而 fill_bitmask 是每步 O(行宽) 的常数级写入。异步编译 + 编译缓存把前者移出关键路径、并在同 schema 的请求间摊薄；这是『异步编译门』的定量解释。",
        "字符语法 ↔ token 词表的适配：语法定义在字符/字节流上，采样发生在离散 token 上——一个 token 可覆盖多个字符、不同 token 可以共享前缀（重叠适配问题）。这正是 GrammarCompiler 必须吃 TokenizerInfo（backend_xgrammar.py:L60-L70）的原因：编译产物不是字符 DFA，而是直接在词表上回答『下一步哪些 token 合法』的状态机（fill_next_token_bitmask 的输出即其答案）。xgrammar 的具体适配算法在库内，本章以契约为界不深入库实现。",
        "validate_tokens 的『试走再回退』等价性：xgrammar 的实现对前缀逐个 accept，成功计数 k，然后 rollback(k)。当且仅当状态机的 rollback 是精确逆操作时，这才与『不推进』语义等价——而 GrammarMatcher 的回滚深度上限被设为 num_speculative_tokens，所以这条等价性在投机 token 数以内成立，超出则不保证。",
        "阻塞态门控的正确性：调度器只在 grammar property 返回非 None 时才把请求晋级到 WAITING。_grammar 一旦从 Future 换成成品/Exception 就不再变回，属性读是幂等的——『就绪』是单调的（false→true 后不回退），因此不会出现『晋级后又发现没编译完』的竞态；grammar_init 只在输入处理线程调用，写 _grammar 的也只有它与属性 getter 所在的调度线程（后者只做 Future→成品/Exception 替换），源码注释明确断言无竞态（core.py:L985-L990）。v0.27.1 的 Exception 分支同样单调：一旦写入 Exception，每拍晋级检查都会把它转进 grammar_compile_error_reqs，同拍 FINISHED_ERROR 出队。",
        "ReDoS 的指数爆炸与超时防线：嵌套量词类 pattern（如 (a+)+b）会使 NFA→DFA 确定化的状态空间指数爆炸，编译一个恶意 regex 可能永久挂死线程。compile_regex_with_timeout（utils.py:L48-L83）用单线程池 + future.result(timeout=VLLM_REGEX_COMPILATION_TIMEOUT_S，默认 5s) 把『编译不终止』转成 ValueError——与异步编译门同一哲学：任何不可预期的慢，都不许碰引擎的关键路径。三处消费：xgrammar 编译 REGEX（backend_xgrammar.py:L91-L95）与校验（L282-L291）、outlines 编 Index（backend_outlines.py:L65-L68）、LMFE 编 RegexParser（L110-L114）。",
    ],
    "mechanisms": [
        {"id": "m01-mask-not-sample", "name": "约束解码 = 采样前的合法性掩码（不改采样算法）", "kind": "dataflow", "difficulty": "core", "needs_figure": True, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/backend_types.py:L72-L80", "vllm/sampling_params.py:L72-L83"]},
        {"id": "m02-async-grammar-compile-gate", "name": "异步语法编译 + WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 状态门（本章命门）", "kind": "protocol", "difficulty": "core", "needs_figure": True, "needs_worked_example": True,
         "source_anchors": ["vllm/v1/structured_output/__init__.py:L166-L175", "vllm/v1/request.py:L106-L114", "vllm/v1/core/sched/scheduler.py:L2050-L2062", "vllm/v1/core/sched/scheduler.py:L2678-L2703"]},
        {"id": "m03-future-poll-100us", "name": "Future.result(timeout=0.0001) 的『几乎非阻塞』就绪轮询与三态原地替换（Future→成品/Exception）", "kind": "algorithm", "difficulty": "core", "needs_figure": False, "needs_worked_example": True,
         "source_anchors": ["vllm/v1/structured_output/request.py:L50-L79"]},
        {"id": "m04-grammar-abc-six-methods", "name": "请求级契约六方法：为什么恰好是这六个", "kind": "protocol", "difficulty": "core", "needs_figure": True, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/backend_types.py:L31-L95"]},
        {"id": "m05-backend-abc-three-methods", "name": "引擎级契约三方法 + 三个 dataclass 字段（编译器/词表这类重资源的归属）", "kind": "protocol", "difficulty": "core", "needs_figure": False, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/backend_types.py:L98-L136"]},
        {"id": "m06-rollback-for-spec-decode", "name": "rollback 是为投机解码留的口子：max_rollback_tokens = num_speculative_tokens；LMFE 撑不住直接拒绝", "kind": "protocol", "difficulty": "core", "needs_figure": False, "needs_worked_example": True,
         "source_anchors": ["vllm/v1/structured_output/backend_xgrammar.py:L119-L126", "vllm/v1/structured_output/backend_xgrammar.py:L190-L193", "vllm/v1/structured_output/backend_lm_format_enforcer.py:L124-L133"]},
        {"id": "m07-accept-vs-validate", "name": "accept_tokens（推进）与 validate_tokens（试走再回退）的语义分离", "kind": "algorithm", "difficulty": "core", "needs_figure": False, "needs_worked_example": True,
         "source_anchors": ["vllm/v1/structured_output/backend_xgrammar.py:L152-L188", "vllm/v1/structured_output/backend_types.py:L34-L60"]},
        {"id": "m08-structured-output-key", "name": "structured_output_key：六种约束归一成 (枚举, 字符串) 的编译键", "kind": "dataflow", "difficulty": "core", "needs_figure": False, "needs_worked_example": True,
         "source_anchors": ["vllm/v1/structured_output/request.py:L77-L103", "vllm/v1/structured_output/backend_types.py:L19-L28"]},
        {"id": "m09-compile-cache-reuse", "name": "编译复用是「各后端各自为政」：xgrammar 库内缓存（默认 512MB 预算）/ outlines 自建 dict（可选 SQLite 磁盘）/ guidance 无 / LMFE 仅 tokenizer_data", "kind": "config", "difficulty": "core", "needs_figure": False, "needs_worked_example": True,
         "source_anchors": ["vllm/v1/structured_output/backend_xgrammar.py:L65-L70", "vllm/v1/structured_output/backend_outlines.py:L58-L71", "vllm/v1/structured_output/utils.py:L282-L301", "vllm/v1/structured_output/backend_lm_format_enforcer.py:L34-L40"],
         "note": "纠正常见误解：vLLM 侧不做跨请求编译去重（structured_output_key 只是每请求 cached_property）。worked example 建议：两个同 schema 请求在 xgrammar 下命中库内缓存 vs 在 guidance 下每次重编的对照。"},
        {"id": "m10-xgrammar-compile-dispatch", "name": "xgrammar 编译分派：五个分支（JSON/JSON_OBJECT/GRAMMAR/REGEX/STRUCTURAL_TAG）——无 CHOICE 分支", "kind": "dataflow", "difficulty": "core", "needs_figure": True, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/backend_xgrammar.py:L78-L126"],
         "note": "关键：CHOICE 走不到这里——校验期 validate_xgrammar_grammar 已把 choice 原地改写成 EBNF grammar（L293-L303），到引擎侧 structured_output_key 已是 (GRAMMAR, ebnf串)。"},
        {"id": "m11-backend-selection-auto", "name": "后端选择与 auto 回退阶梯（发生在前端校验期，落到 params._backend；引擎侧只认 _backend）", "kind": "config", "difficulty": "core", "needs_figure": True, "needs_worked_example": False,
         "source_anchors": ["vllm/sampling_params.py:L1000-L1082", "vllm/v1/structured_output/__init__.py:L124-L164"]},
        {"id": "m12-single-backend-per-engine", "name": "全引擎单后端：惰性构造 + 请求级指定被拒（_backend / _backend_was_auto 记账）", "kind": "config", "difficulty": "supporting", "needs_figure": False, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/__init__.py:L124-L164", "vllm/sampling_params.py:L949-L966"]},
        {"id": "m13-four-backend-contract-matrix", "name": "四后端同契约对照：支持的约束形态、回滚能力、终态语义各自的取舍", "kind": "protocol", "difficulty": "core", "needs_figure": True, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/backend_xgrammar.py:L135-L203", "vllm/v1/structured_output/backend_guidance.py:L142-L221", "vllm/v1/structured_output/backend_outlines.py:L111-L168", "vllm/v1/structured_output/backend_lm_format_enforcer.py:L43-L91"]},
        {"id": "m14-terminated-semantics-divergence", "name": "is_terminated 的语义分歧：xgrammar 缓存标志位 / guidance 用 EOS+rollback_lag / outlines 故意延迟一步 / LMFE 看前缀末位是否 EOS", "kind": "protocol", "difficulty": "supporting", "needs_figure": False, "needs_worked_example": True,
         "source_anchors": ["vllm/v1/structured_output/backend_xgrammar.py:L198-L199", "vllm/v1/structured_output/backend_guidance.py:L165-L171", "vllm/v1/structured_output/backend_guidance.py:L216-L217", "vllm/v1/structured_output/backend_outlines.py:L119-L121", "vllm/v1/structured_output/backend_outlines.py:L159-L163", "vllm/v1/structured_output/backend_lm_format_enforcer.py:L82-L88"]},
        {"id": "m15-bitmask-layout", "name": "位掩码的内存布局：[max_num_seqs×(1+num_spec), ceil(vocab/32)] 的 int32 行，-1 表全允许；每行比 logits 小 32 倍", "kind": "layout", "difficulty": "supporting", "needs_figure": True, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/backend_outlines.py:L99-L105", "vllm/v1/structured_output/backend_lm_format_enforcer.py:L141-L147", "vllm/v1/structured_output/backend_xgrammar.py:L128-L129", "vllm/v1/structured_output/__init__.py:L225-L234"]},
        {"id": "m16-external-launcher-sync-fallback", "name": "external_launcher 模式退回同步编译（多 rank 状态跃迁必须同步，否则死锁）", "kind": "config", "difficulty": "supporting", "needs_figure": False, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/__init__.py:L46-L55"]},
        {"id": "m17-thread-safety-invariant", "name": "线程安全不变式：grammar_init 只在输入处理线程调用；就绪是单调的，调度器只做 Future→成品/Exception 替换", "kind": "protocol", "difficulty": "supporting", "needs_figure": False, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/engine/core.py:L969-L991", "vllm/v1/structured_output/request.py:L50-L59"]},
        {"id": "m18-request-entry-and-handoff", "name": "端到端接缝：校验 → Request 阻塞态 → grammar_init → 晋级 → 每拍 fill/advance → GrammarOutput 交棒（ch31 起点）", "kind": "dataflow", "difficulty": "core", "needs_figure": True, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/request.py:L87-L114", "vllm/v1/engine/core.py:L969-L991", "vllm/v1/core/sched/scheduler.py:L1646-L1668"]},
        {"id": "m19-capability-matrix-in-dispatch", "name": "能力矩阵藏在编译分派里：outlines 只认 JSON/REGEX/CHOICE（其余当场 ValueError），LMFE 认 JSON/JSON_OBJECT/REGEX/CHOICE", "kind": "config", "difficulty": "supporting", "needs_figure": False, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/backend_outlines.py:L73-L97", "vllm/v1/structured_output/backend_lm_format_enforcer.py:L101-L123"]},
        {"id": "m20-validation-rewrites-request", "name": "校验期不只是选后端，还会原地改写请求：choice → EBNF grammar", "kind": "dataflow", "difficulty": "core", "needs_figure": False, "needs_worked_example": True,
         "source_anchors": ["vllm/v1/structured_output/backend_xgrammar.py:L293-L303", "vllm/v1/structured_output/utils.py:L553-L561"],
         "note": "validate_xgrammar_grammar 里：choice_as_grammar(so_params.choice) → Grammar.from_ebnf 试编 → so_params.choice=None; so_params.grammar=choice_grammar 就地改写后 return。这解释了 xgrammar 的 compile_grammar 为什么没有 CHOICE 分支、也是『六种入口枚举 → 引擎侧只见五种』的原因。"},
        {"id": "m21-who-advances-the-grammar", "name": "谁在推进语法状态机：accept_tokens 的真实调用点在调度器 update_from_output；validate_tokens 只喂 spec draft 做『不推进的试走』", "kind": "dataflow", "difficulty": "core", "needs_figure": False, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/core/sched/scheduler.py:L1817-L1843", "vllm/v1/core/sched/scheduler.py:L2163-L2166", "vllm/v1/core/sched/scheduler.py:L2191-L2194"]},
        {"id": "m22-compile-failure-path", "name": "编译失败经 Future 闭环：Exception 存进 _grammar → grammar_compile_error_reqs → FINISHED_ERROR 只杀一个请求（v0.27.1 新闭环：v0.21.0 此处还是 TODO）", "kind": "protocol", "difficulty": "core", "needs_figure": False, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/request.py:L57-L58", "vllm/v1/structured_output/__init__.py:L169-L174", "vllm/v1/structured_output/__init__.py:L184-L192", "vllm/v1/core/sched/scheduler.py:L2699-L2701", "vllm/v1/core/sched/scheduler.py:L1954-L1962"]},
        {"id": "m23-regex-redos-timeout", "name": "ReDoS 防护：compile_regex_with_timeout 单线程池+超时（VLLM_REGEX_COMPILATION_TIMEOUT_S 默认 5s），xgrammar/outlines/LMFE 三处消费", "kind": "algorithm", "difficulty": "supporting", "needs_figure": False, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/utils.py:L48-L83", "vllm/v1/structured_output/backend_xgrammar.py:L91-L95", "vllm/envs.py:L1566-L1568"]},
        {"id": "m24-outlines-disk-cache", "name": "outlines SQLite 磁盘索引缓存：serde 二进制替代 pickle 消除反序列化任意代码执行风险；默认关（LRUCache(128) 内存兜底）", "kind": "config", "difficulty": "supporting", "needs_figure": False, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/utils.py:L218-L301", "vllm/envs.py:L1486-L1487"]},
        {"id": "m25-thinking-gate-pointer", "name": "思考段门控指针：should_fill_bitmask/should_advance/trim_reasoning_for_advance 判『思考段结束后语法才生效』（v0.27 bug 高发区 #44006/#42452/#43388）——本章只认调用点，展开归 ch31", "kind": "protocol", "difficulty": "supporting", "needs_figure": False, "needs_worked_example": False,
         "source_anchors": ["vllm/v1/structured_output/__init__.py:L361-L439", "vllm/v1/structured_output/__init__.py:L462-L486", "vllm/v1/core/sched/scheduler.py:L1817-L1831"]},
    ],
    "subtraction_plan": {
        "delete": [
            {"what": "前端校验阶梯不入引擎侧最小集：sampling_params.py 的 _validate_structured_outputs（L923-L1086）与 validate_guidance_grammar（backend_guidance.py:L293-L303）、validate_structured_output_request_outlines（backend_outlines.py:L171-L204）、validate_structured_output_request_lm_format_enforcer（backend_lm_format_enforcer.py:L153-L191）", "why_safe": "校验发生在 API 进程（InputProcessor.verify 链），引擎侧最小集从 params._backend 已定开始；正文以 embed 讲。**validate_xgrammar_grammar（backend_xgrammar.py:L272-L363）必须保留**——choice→EBNF 改写的唯一发生地（m20），CHOICE 请求若不先过它、引擎侧 compile_grammar 无分支可走（同款只删被调函数留调用点的坑）。"},
            {"what": "所有 Mistral/tekken 分词器兼容分支（backend_xgrammar.py:L42-L59 的手工 TokenizerInfo；backend_guidance.py:L97-L102 的 mistral 分支）", "why_safe": "只影响 TokenizerInfo/LLTokenizer 的构造方式，不改变『规格字符串 → 编译 → matcher → 掩码行』这条主控制流；精简版固定走 from_huggingface/from_tokenizer 主路径。"},
            {"what": "structural_tag 的 deprecated 编译路径（backend_xgrammar.py:L98-L108 的 structures/triggers 拆解；validate_xgrammar_grammar 内 L349-L359 同款）", "why_safe": "同一入口的旧格式兼容分支，新格式路径 compile_structural_tag(grammar_spec) 已覆盖语义；删掉不影响其余四种约束形态。"},
            {"what": "thinking/reasoning 全部：manager 的 _get_reasoner/should_fill_bitmask/should_advance/_find_reasoning_end_index/trim_reasoning_for_advance（__init__.py:L99-L112, L361-L486）；StructuredOutputRequest 的 reasoning_ended/reasoning_end_token_index/reasoning_parser_kwargs/reasoner 字段（request.py:L27-L37）；Request 构造的注入（vllm/v1/request.py:L90-L94）；manager 的 reasoner 装配（__init__.py:L81-L97）", "why_safe": "ch31 主题；trim_reasoning_for_advance 在 reasoning_end_token_index is None 时原样返回 new_token_ids（L479-L481 早退）——删包装后 accept_tokens 直接吃 new_token_ids，与无 reasoner 的运行路径逐 token 等价。"},
            {"what": "grammar_bitmask 的批装配中段与并行填充：executor_for_fillmask（L60-L68）、_async_submit_fill_bitmask（L207-L210）、并行分支（L244-L271）、串行分支的 spec 窗口预推进+rollback 与 thinking 探测（L283-L350）", "why_safe": "本章站 7 只到『每请求 fill 自己那一行』；批装配/并行/spec 窗口归 ch31。精简版保留 _fill_bitmasks（L194-L205）、惰性预算分配（L225-L234）、串行骨架（逐请求 _fill_bitmasks + bonus 行 + cumulative_index 递增）与 numpy 转换（L352-L359），足以演示一请求一行。"},
            {"what": "outlines 与 lm-format-enforcer 两个后端的实现（backend_outlines.py 的 OutlinesBackend/OutlinesGrammar 与 backend_lm_format_enforcer.py 全文；utils.py 的 outlines 词汇表/缓存 helpers L178-L301；grammar_init 里两个 elif 分支 L145-L162）", "why_safe": "正文以 xgrammar 深挖 + guidance 对照已证明『同一契约、不同实现』；这两家以对照表呈现。删 grammar_init 两分支后 outlines/lmfe 后端请求落 else ValueError——如实降级：精简版只接受 xgrammar/guidance 两后端（与测试夹具约束一致），拒绝即报错而非静默。"},
            {"what": "utils.py 的 apply_grammar_bitmask（L86-L175）与 worker 侧一切", "why_safe": "worker 侧重排+pinned H2D+apply_token_bitmask_inplace 与两段式 GPU 窗口全部归 ch31，本章终点是 GrammarOutput 交棒。"},
            {"what": "logger 调用与 TYPE_CHECKING/LazyLoader 样板、# type: ignore 注释", "why_safe": "纯工程装饰，无控制流影响。注意：grammar_init/_create_grammar/preprocess_add_request 的**注释与 docstring 必须保留**（线程安全断言、Future 携错语义是叙事证据），只删 logger.xxx 调用行。"},
        ],
        "must_keep": [
            {"symbol": "StructuredOutputOptions", "why": "六种约束形态的枚举，structured_output_key 的第一元与编译分派的开关"},
            {"symbol": "StructuredOutputKey", "why": "(枚举, 规格字符串) 键类型——编译键的载体，读者要看懂它为什么可哈希"},
            {"symbol": "StructuredOutputGrammar", "why": "请求级契约 ABC，本章骨架"},
            {"symbol": "accept_tokens", "why": "推进状态机——契约六方法之一，且是投机解码回滚的对偶"},
            {"symbol": "validate_tokens", "why": "『不推进的试走』——与 accept_tokens 的语义区分是本章要点"},
            {"symbol": "rollback", "why": "专为投机解码留的口子，本章明确讲清、ch31/ch33 展开"},
            {"symbol": "fill_bitmask", "why": "契约与采样的唯一接口——语法在这里变成掩码（F6 回收的落点）"},
            {"symbol": "is_terminated", "why": "语法走到终态的判据，决定还要不要继续掩码"},
            {"symbol": "reset", "why": "六方法之一，缺则 ABC 不完整。注：v0.27.1 全仓仍无 in-tree 调用者（grep 只命中定义与各后端内部 matcher.reset/guide.reset/ll_matcher.reset）——writer 勿编造使用场景"},
            {"symbol": "StructuredOutputBackend", "why": "引擎级契约 ABC（vllm_config/tokenizer/vocab_size 三字段 + 三方法）"},
            {"symbol": "compile_grammar", "why": "『一段语法怎么变成对象』的落点，本章标题的动词"},
            {"symbol": "allocate_token_bitmask", "why": "掩码张量的形状与所有权在这里定，是与 ch31 的接缝"},
            {"symbol": "destroy", "why": "后端级清理，契约完整性"},
            {"symbol": "StructuredOutputManager", "why": "引擎级编排者，异步编译的发起方"},
            {"symbol": "grammar_init", "why": "命门：惰性建后端 + 把编译提交线程池"},
            {"symbol": "_create_grammar", "why": "线程池里真正执行的工作函数，key→compile_grammar 的一跳 + Future 携错"},
            {"symbol": "_use_async_grammar_compilation", "why": "external_launcher 同步回退开关，解释异步为何不是无条件的"},
            {"symbol": "executor", "why": "ThreadPoolExecutor 实例——异步编译的物理载体（max_workers=半数 CPU）"},
            {"symbol": "StructuredOutputRequest", "why": "请求级容器，Future/成品/Exception 三态共存点"},
            {"symbol": "_check_grammar_completion", "why": "timeout=0.0001 非阻塞轮询，门控的核心实现"},
            {"symbol": "is_grammar_ready", "why": "属性之一。注：v0.27.1 全仓仍零 in-tree 调用者（grep 只命中定义）——门控实际走 grammar property，勿写成『对外就绪查询接口』"},
            {"symbol": "grammar", "why": "StructuredOutputRequest 的 property/setter（request.py:L65-L75）——调度器实际读的就是它，门控真正的落点"},
            {"symbol": "from_sampling_params", "why": "『无约束就返回 None』——决定 use_structured_output 真假的源头"},
            {"symbol": "structured_output_key", "why": "cached_property，编译键的缓存点"},
            {"symbol": "get_structured_output_key", "why": "六种约束→二元组的归一函数，含 json.dumps 归一细节"},
            {"symbol": "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR", "why": "本章的『门』——真实枚举名，正文与精简版都必须用它（不是 WAITING_FOR_FSM）"},
            {"symbol": "XgrammarBackend", "why": "默认后端，深挖主线"},
            {"symbol": "XgrammarGrammar", "why": "六方法的参考实现"},
            {"symbol": "GrammarCompiler", "why": "cache_enabled=True 的编译器——编译复用的真实所在"},
            {"symbol": "GrammarMatcher", "why": "逐 token 状态机本体，max_rollback_tokens 与投机 token 数挂钩"},
            {"symbol": "max_rollback_tokens", "why": "回滚深度=投机 token 数，rollback 存在理由的关键证据"},
            {"symbol": "num_processed_tokens", "why": "状态推进计数，rollback 时同步回退——状态是成对维护的"},
            {"symbol": "GuidanceBackend", "why": "第二实现，证明契约的可替换性"},
            {"symbol": "GuidanceGrammar", "why": "同上；rollback_lag 是其独有的语义差异"},
            {"symbol": "rollback_lag", "why": "guidance 在 EOS 后回滚要少退一格——契约相同、实现细节不同的最好例证"},
            {"symbol": "StructuredOutputsParams", "why": "用户面入口，六选一互斥"},
            {"symbol": "_backend", "why": "后端选择的落点字段，连接前端校验与引擎侧构造"},
            {"symbol": "_backend_was_auto", "why": "记账位，解释为什么复用 params 不会误报后端冲突"},
            {"symbol": "__post_init__", "why": "StructuredOutputsParams 的六选一互斥校验（L90-L111，count>1 与 count<1 双向报错）——writer 要断言『六选一互斥』就得有这段源码"},
            {"symbol": "all_constraints_none", "why": "from_sampling_params 判『无约束』的依据（L113-L127），入口机制的证据"},
            {"symbol": "_is_blocked_waiting_status", "why": "m02 门控的调度侧一半（scheduler.py:L2050）——阻塞态分流 skipped_waiting；dossier 已逐字内嵌且标 core，不进 must_keep 则 lint_fidelity 无从校验、implementer 极可能整块删掉"},
            {"symbol": "_try_promote_blocked_waiting_request", "why": "门控另一半（scheduler.py:L2678）：语法就绪晋级回 WAITING、Exception 转 grammar_compile_error_reqs"},
            {"symbol": "grammar_compile_error_reqs", "why": "v0.27.1 编译失败闭环的账本（scheduler.py:L211 初始化、L1954-L1962 收割 FINISHED_ERROR）——m22 的锚，删则失败路径断链"},
            {"symbol": "use_structured_output", "why": "vllm/v1/request.py:L267-L269，定义即 structured_output_request is not None；preprocess_add_request 的分支条件、data_flow 第 3 步都用它"},
            {"symbol": "validate_xgrammar_grammar", "why": "choice→EBNF 原地改写的唯一发生地（backend_xgrammar.py:L293-L303），m20 的锚；删则 CHOICE 路径崩"},
            {"symbol": "choice_as_grammar", "why": "utils.py:L553-L561，choice→EBNF 改写链条的一环，与 validate_xgrammar_grammar 同进退"},
            {"symbol": "compile_regex_with_timeout", "why": "ReDoS 防线（utils.py:L48-L83，m23）：xgrammar REGEX 编译与校验的共用被调函数，删则 REGEX 路径崩"},
            {"symbol": "_fill_bitmasks", "why": "站 7 的活体（__init__.py:L194-L205）：一请求一行 + is_terminated/非语法行填 -1 的全允许兜底——F6 回收的可跑对照"},
            {"symbol": "grammar_bitmask", "why": "每拍③入口与预算分配（__init__.py:L212-L234）：[max_batch×(1+max_num_spec)] 行掩码的惰性分配——ch31 首对象，删则站 7 无宿主"},
            {"symbol": "GrammarOutput", "why": "交棒载体（vllm/v1/core/sched/output.py:L287-L291：(request_ids, ndarray) 二元组）——ch31 的第一对象，本章终点"},
        ],
    },
    "foreshadow_due": {
        "source": "v3 伏笔真相源 = book/cartography/pedagogy-plan.json foreshadows（bible.py due 是 v2 旧章号账本，不跑——任务指令明确）",
        "should_plant": [],
        "should_payoff": [
            {"id": "F6", "text": "grammar bitmask 窗口", "planted": 9, "paid": 30,
             "how": "ch9 站 4（③ get_grammar_bitmask）只讲掩码在这拍的位置——『表怎么算出来，是本书约束解码两章的整章主场』。本章第 7 站正式回收前半：语法编译成 FSM 后，fill_next_token_bitmask 用 FSM 当前状态在预分配 [批×(1+spec), ceil(V/32)] int32 掩码上逐请求写行（开关表语义）；每拍⑤ accept_tokens 推进 FSM、下一拍③ 算出新的合法集。后半（批装配/并行填充/spec 窗口预推进+rollback/跨进程 ndarray/worker 侧 H2D+apply/两段式 GPU 窗口）按大纲归 ch31『bitmask 落地』——正文须显式说清这个对半切分，勿把 ch31 的内容提前展开"},
        ],
        "note": "pedagogy-plan.foreshadows 中 F6（planted=9, paid=30）是本章唯一回收项；本章无埋设任务。邻章衔接（非伏笔账本、正文点名即可）：站 1 的前端校验链回指 ch4/ch6（API 进程校验期、params 过线）；每拍③/⑤ 的拍位与重叠窗口回指 ch9 五拍；采样交接点（掩码先行、logits 已被改过）回指 ch29 站 1-2；spec 窗口 rollback 前指 ch31/ch33；thinking 门控前指 ch31。",
    },
    "l2_spec": "book/cartography/l2-specs/ch30.json（stations 与本档案 stations 同一份账本）",
}

OUT.write_text(json.dumps(DOSSIER, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("written:", OUT)
print("excerpts:", len(EXCERPTS))
for e in EXCERPTS:
    n_lines = e["code"].count("\n") + 1
    print(f"  {e['lines']:>14}  {n_lines:>3}行  {e['path']}")
