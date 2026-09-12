# impl-notes — v3 ch30《约束解码 I：语法编译》(Part VII)

标准代码章（非 primer）：对 vLLM **v0.27.1（6e448d0ea）** 的只做减法精简版——
同名、同结构、同控制流；只删 dossier `subtraction_plan.delete` 批准项，
`must_keep` 全保留；每 def/class 标 `# SOURCE: vllm/...:Lxxx`（行号全部对
v0.27.1 现核，v2 资产旧行号未沿用），删除处标 `# SUBTRACTED:`。
测试 host `python -m pytest` 全量可跑（**159 passed**），跑的是**真实状态机**
——host 已装 xgrammar==0.2.6 / llguidance==1.7.6（均为 vLLM
`requirements/common.txt` 钉版区间 `xgrammar>=0.2.1,<1.0.0` /
`llguidance>=1.7.0,<1.8.0` 的 Windows wheel），非 Fake 注入。
`lint_fidelity` 全绿（无 BLOCKING）。

## 文件清单（implementation/ 下与真实仓库同构镜像）

**本章脊柱（subtract-only 主角）**

- `vllm/sampling_params.py` —— StructuredOutputsParams 六选一互斥（L70-L142
  逐字）+ `_is_non_tekken_mistral`/`_get_llg_tokenizer`（L191-L196 逐字）+
  SamplingParams HOST SEAM 切面（verify L755-L770 / _validate_structured_
  outputs L923-L1086 主体）。
- `vllm/v1/structured_output/__init__.py` —— StructuredOutputManager：
  异步编译门（grammar_init L114-L175 / _create_grammar L177-L192）+ 推进侧
  思考门四方法（_get_reasoner / should_advance / _find_reasoning_end_index /
  trim_reasoning_for_advance，L99-L486 全逐字）。
- `vllm/v1/structured_output/backend_types.py` —— 整文件逐字（两层契约 ABC
  m5 + 六形态枚举 + 键类型）。
- `vllm/v1/structured_output/request.py` —— 整文件逐字（三态容器 m12 +
  structured_output_key 归一 m2）。
- `vllm/v1/structured_output/backend_xgrammar.py` —— 默认后端主线：五分派
  compile_grammar（m6）+ XgrammarGrammar 六方法（m7/m8/m9）+ 能力预检 +
  validate_xgrammar_grammar（m3/m4）。
- `vllm/v1/structured_output/backend_guidance.py` —— 第二实现（m18）：
  rollback_lag / serialize 六形统一（choice 原生）/ validate_guidance_grammar。
- `vllm/v1/structured_output/utils.py` —— ReDoS 超时护栏（m19）+ 校验期改写
  工具箱（choice_as_grammar / grammar_is_likely_lark / convert_lark_to_ebnf，
  m4）。
- `vllm/v1/request.py` —— Request 语法门切面（出生即阻塞 L109-L114 逐字）+
  RequestStatus 全枚举 + from_engine_core_request + append/use_structured_
  output/is_finished 面。
- `vllm/v1/engine/core.py` —— preprocess_add_request L969-L991 逐字（站 3
  入口、grammar_init 唯一调用点、线程安全注释原文）。
- `vllm/v1/core/sched/scheduler.py` —— 调度器语法门切面：侧队分流
  （L2050-L2062 逐字）/ schedule() WAITING 相位窥队头晋级（L700-L711 逐字）/
  _try_promote_blocked_waiting_request 语法分支（L2696-L2705 逐字）/
  update_from_output 推进块（L1817-L1843 逐字）与编译失败收账（L1954-L1971
  逐字）+ finish_requests 两遍式。
- `vllm/v1/core/sched/request_queue.py` —— SchedulingPolicy + RequestQueue
  ABC + FCFSRequestQueue 逐字（窥队/prepend 回侧队的物理载体）。
- `vllm/v1/core/sched/output.py` —— SchedulerOutput 字段面（行序契约载体；
  GrammarOutput 归 ch31 不镜像）。
- `vllm/config/structured_outputs.py` —— 六字段+校验器近逐字（m20 对照基准；
  pydantic @config → 显式 `__init__`+尾调用承载同一校验体）。

**HOST SEAM 支撑面（消费面最小承载，逐字或注明退化）**

- `vllm/exceptions.py`（VLLMValidationError 逐字）
- `vllm/logger.py` / `vllm/envs.py`（两个结构化输出环境变量的真实 lambda
  逐字；机制=字典+`__getattr__` 同构，去 functools.cache 以便测试切换超时）
- `vllm/utils/import_utils.py`（LazyLoader 逐字）/ `vllm/utils/mistral.py`
  （is_mistral_tokenizer 逐字）
- `vllm/tokenizers/__init__.py`（cached_tokenizer_from_config 退化
  AutoTokenizer 直载；语义=HF fast tokenizer）
- `vllm/config/__init__.py`（VllmConfig/ModelConfig/ParallelConfig/
  SchedulerConfig/SpeculativeConfig 字段面）
- `vllm/pooling_params.py`（类型占位）/ `vllm/v1/utils.py`（ConstantList
  逐字）/ `vllm/v1/engine/__init__.py`（FinishReason/EngineCoreEventType/
  EngineCoreEvent/EngineCoreRequest/EngineCoreOutput 字段逐字，
  msgspec.Struct→@dataclass）
- `vllm/v1/core/sched/utils.py`（remove_all 逐字）

## 减法执行账（subtraction_plan.delete 七项逐一）

| delete | 内容 | 落点 |
|---|---|---|
| [0] Mistral/tekken 兼容分支 | backend_xgrammar L42-L59 手工 TokenizerInfo + backend_guidance L97-L102 mistral ll_tokenizer 三分支（else 体提升直线）；连坐删两文件仅服务该分支的 import（is_mistral_tokenizer/MistralCommonBackend）。**sampling_params 侧 _is_non_tekken_mistral + skip_guidance 计算按计划明令保留不动** | backend_xgrammar / backend_guidance |
| [1] structural_tag deprecated 路径 | compile_grammar L97-L109 与 validate_xgrammar_grammar L347-L359 的 structures/triggers 拆解（StructuralTagItem）；精简版只走 compile_structural_tag(str) / from_structural_tag(str) 新路径 | backend_xgrammar |
| [2] 批掩码装配整块 | grammar_bitmask(L212-L359)/_fill_bitmasks/_async_submit_fill_bitmask/executor_for_fillmask 装配(L60-L68)/_grammar_bitmask/_full_mask 字段(L57-L58)/should_fill_bitmask(L361-L379)；**scheduler.get_grammar_bitmask(L1646-L1668) 同边界不进精简版**（计划原文：交棒只在 dossier 内嵌展示） | structured_output/__init__ / scheduler |
| [3] outlines+LMFE 两后端 | 两文件不创建；utils.py outlines 专属全家（OutlinesVocabulary/get_outlines_cache_path/OutlinesDiskCache/get_outlines_cache/_reduced_vocabulary 及其正则/get_outlines_vocabulary）；sampling_params 配套删：两段 import（L1004-L1009）、两显式分支（L1031-L1042）、auto 阶梯 outlines 降级支（L1069-L1073，else 体提升直线）；grammar_init 两 elif（L145-L162）。**auto 阶梯骨架按计划保留**：try xgrammar→except→skip_guidance 两判据计算（L1058-L1067）→validate_guidance_grammar 兜底→_backend_was_auto=True | 全线 |
| [4] reasoning parser 装配段 | __init__.py L81-L93（plugin import+查表构造）；reasoner_cls 停留 L43 None 初值=源码真实路径；**四方法本体原样保留**（must_keep） | structured_output/__init__ |
| [5] serialize 的 STRUCTURAL_TAG 分支 | backend_guidance L258-L282（StructTag 拆解/to_grammar）；落到 else 报非法类型 | backend_guidance |
| [6] logger 调用/TYPE_CHECKING 断言块/# type: ignore/类头 NOTE | 全部 logger.* 调用行（含 accept_tokens『Please file an issue』/check_error/_create_grammar 的 logger.exception/serialize else 分支）；各 `if TYPE_CHECKING: assert(...)` 块；XgrammarGrammar 类头 jump-forward NOTE（L137-L142）；backend_xgrammar L50 的 `# type: ignore`（随 [0] 分支删） | 全线 |

**章边界删除（impl-notes 记账，非 delete 项而是镜像范围裁定）**：

- `utils.py:apply_grammar_bitmask`（L86-L176）——worker 侧掩码落地（重排/
  pinned H2D/apply_token_bitmask_inplace），与 get_grammar_bitmask 同为 ch31
  交棒件（dossier scope_note 边界原文），不进本章精简版。
- scheduler `_try_promote_blocked_waiting_request` 的 REMOTE_KVS（L2683-L2694）
  与 STREAMING_REQ（L2705-L2708）两分支——P/D KV 传输（ch16）/流式会话
  （ch38）域，依赖 _update_waiting_for_remote_kv/kv_cache_manager/
  streaming_queue 等本章不镜像的状态；**语法分支三出口逐字保留**、
  _is_blocked_waiting_status 三阻塞态枚举逐字保留（『机制共用』的证据面）。
- scheduler schedule()/update_from_output 的非语法域块（RUNNING 相位 KV 预算/
  抢占/前缀命中/encoder/LoRA/stale/spec 回扣）——ch10-ch12/ch13/ch15/ch33 域，
  逐块 SUBTRACTED 注明；schedule()/update_from_output/add_request/
  finish_requests/`_update_after_schedule` 均保留真实方法名与控制流骨架。
- sampling_params `verify` 的六项非结构化校验调用（L762-L767）——本体归 ch29
  的 SamplingParams 完整面（HOST SEAM 未载其字段面）。

## 已知行为差异（delete[3] 既定口径，writer 勿掩盖；m18 对照表给全量真相）

1. **非 tekken Mistral 分词器或 guidance 不支持的 schema（patternProperties）**：
   真实源码 auto 阶梯降 outlines；精简版无 outlines，经 validate_guidance_
   grammar 报错拒单。multipleOf 等 xgrammar 不支持而 guidance 支持的 schema
   正常降 guidance（测试 `test_auto_ladder_falls_back_to_guidance_on_multiple_of`）。
2. **显式配 outlines/lm-format-enforcer 的引擎**：精简版两显式分支已删，会
   落进 auto 阶梯 else；真实源码各自走专门校验。
3. 其余六选一/auto 阶梯/冲突检查/改写路径与真实源码逐字一致（159 测试覆盖）。

## HOST SEAM 偏差清单（writer/explainer 须知）

1. **msgspec→dataclass/普通类**：SamplingParams（真实 msgspec.Struct 数百
   字段）、EngineCoreRequest/EngineCoreOutput/EngineCoreEvent（真实 msgspec
   序列化件）。语法门主控制流不涉序列化面。
2. **pydantic→普通类**：StructuredOutputsConfig（校验体逐字，装饰器退化）；
   VllmConfig 对象图=字段面（真实 ch03 域）。
3. **tokenizers**：cached_tokenizer_from_config 退化 AutoTokenizer 直载
   （真实 registry 有 tokenizer_mode 路由与各家专版）。
4. **xgrammar 0.2.6 版本事实**（正文引用需知，源码口径以 vLLM pin 为准）：
   ① `GrammarMatcher(max_rollback_tokens=…)` 仍按源码传参（m9 证据），但库侧
   已弃用其限制语义（构造时发 DeprecationWarning，内部恒无限回滚）——
   测试以该 Warning 为『参数传入』的可观测证据；② GrammarMatcher.is_
   terminated 需吃过 EOS/stop token 才为 True（完整 choice 串后仍 False）；
   ③ XgrammarGrammar.reset 按源码只清 num_processed_tokens 与 matcher，
   **不清 `_is_terminated`**——终态后 accept 恒走头部短路（源码即此，测试
   `test_is_terminated_and_reset` 按真实行为断言）。
5. **llguidance 1.7.6 版本事实**：LLMatcher 吃完合法 choice 即 is_stopped
   （与 xgrammar 的『需 EOS』形成 m18 语义分歧的又一格）；GRAMMAR 形的坏语法
   在 serialize 段的 GBNF→Lark 转换处报错（"Failed to convert the grammar
   from GBNF to Lark"），不进 validate_grammar 的 "Grammar error:" 分支。
6. **scheduler 返回型**：update_from_output 返回 `dict[int, list[EngineCore
   Output]]`（真实 `dict[int, EngineCoreOutputs]` 的 outputs 字段同内容面——
   EngineCoreOutputs 的 DP wave/统计包装面不进精简版）。
7. **lint 工具口径**：lint_fidelity 跳过 `__init__.py`（包标记豁免），而本章
   主角 StructuredOutputManager 恰在 `vllm/v1/structured_output/__init__.py`
   （与真实仓库同构）。为使 must_keep 可检测，scheduler.py 的 import 处注明
   grammar_init/_create_grammar/_use_async_grammar_compilation/_get_reasoner/
   _find_reasoning_end_index 编排面所在（内容为真实架构事实，非虚构）。

## 测试与运行

- host：`python -m pytest instances/vllm/artifacts-v3/ch30-grammar-compilation/tests -q`
  → **159 passed**（6 文件：params/key、rewrite utils、xgrammar backend、
  validation rewrite、async gate、reasoning gate）。
- 依赖：`pip install xgrammar==0.2.6 llguidance==1.7.6`（host 曾缺，本次已装；
  gpt2 词表 50257 走 HF 本地缓存，无网络依赖）。
- 行为基准全部对真实源码 v0.27.1 现核 + 真库实测：六选一互斥双向报错
  （L90-L111）、json.dumps 归一（L82-L103）、auto 阶梯降级（multipleOf→guidance）、
  choice→EBNF 原地改写（L293-L303）、lark 判定无 ::= 即 Lark（L417）、
  五分派无 CHOICE 分支、GrammarMatcher 逐 token 语义（拒收 False 不前进/
  validate 试走+rollback 幂等/rollback 与计数成对）、半 CPU 线程池
  （max_workers=(cpu+1)//2）、100µs 探测三态（Future→成品/Exception 原地替换、
  timeout=0.0001）、external_launcher 同步回退且异常包 Future、阻塞态入
  skipped_waiting、每拍窥队未就绪 prepend 回侧队、就绪当拍入批、编译失败
  同拍收账只杀单请求（FINISHED_ERROR+空 token 回执）、采样后 accept 拒收=
  FINISHED_ERROR+resumable=False、思考门 reasoner=None 常量路径/#44006 混步
  剔除/#43388 delta 窗口。
- 结果台账：`tests/test-report.json`。
