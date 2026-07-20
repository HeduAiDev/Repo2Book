# ch31 精简版实现说明（只做减法 / pin ad7125a4 / v0.21.0）

精简版忠实镜像 vLLM v1 结构化输出（约束解码）子系统的**语法编译与后端契约**切面：
六选一互斥参数 → 异步语法编译门 → 请求级契约六方法 / 引擎级契约三方法 → 默认
（xgrammar）与第二实现（guidance）两个后端 → 调度器的阻塞态判定/晋级与
accept_tokens/validate_tokens 调用点。与真实 vLLM **同名、同结构、同控制流**，只删
不增。所有删除点都带 `# SUBTRACTED:` 注释并标注原 `vllm/...:Lxxx`。

## 可运行性

纯 Python（不 import vllm，不依赖真实 xgrammar/llguidance——本轮无容器，host 也没有
装这两个第三方库，见 dossier.analyst_notes_on_plan）。`backend_xgrammar.py` /
`backend_guidance.py` 用 `try: import xgrammar as xgr / except ImportError: xgr =
None` 顶层导入（真实源码是 LazyLoader，效果等价）；测试里用轻量 Fake 对象替身
monkeypatch 模块级 `xgr`/`llguidance` 名字——被替身的只是外部库对象，vLLM 自身的
分派逻辑、状态推进、掩码调用一字不改地被真实执行。

## 验收判据

把真实 vLLM 删掉所有 `# SUBTRACTED:` 标注的分支，应 ≈ 得到本精简版。删除项严格限于
`dossier.subtraction_plan.delete` 批准范围。`must_keep` 的全部符号原样保留（用
`python3 scripts/lint_fidelity.py` 校验）。

## 文件与真实源码对应

| 精简版文件 | 真实源码 | 说明 |
|---|---|---|
| `backend_types.py` | `vllm/v1/structured_output/backend_types.py` | 六种约束枚举 + 请求级/引擎级两个 ABC，原样保留 |
| `so_request.py` | `vllm/v1/structured_output/request.py` | 改名避免与 `request.py` 撞名；`StructuredOutputRequest` + `get_structured_output_key` |
| `request.py` | `vllm/v1/request.py` | 只保留 `RequestStatus` 枚举与 `Request.__init__` 里挂载结构化请求、置初始阻塞态的切面 |
| `sampling_params.py` | `vllm/sampling_params.py` | `StructuredOutputsParams`（六选一互斥）+ 精简版 `SamplingParams`（只留 `structured_outputs`/`max_tokens`）+ `_validate_structured_outputs`（后端选择 + auto 阶梯） |
| `backend_xgrammar.py` | `vllm/v1/structured_output/backend_xgrammar.py` | 默认后端，深挖主线：`XgrammarBackend`/`XgrammarGrammar`/`validate_xgrammar_grammar`（choice→EBNF 改写）/`has_xgrammar_unsupported_json_features` |
| `backend_guidance.py` | `vllm/v1/structured_output/backend_guidance.py` | 第二实现：`GuidanceBackend`/`GuidanceGrammar`（`rollback_lag` 语义差异）/`serialize_guidance_grammar`/`has_guidance_unsupported_json_features` |
| `utils.py` | `vllm/v1/structured_output/utils.py` | 只留 `choice_as_grammar` |
| `structured_output_manager.py` | `vllm/v1/structured_output/__init__.py` | `StructuredOutputManager`：`grammar_init`/`_create_grammar`（改名避免撞 `__init__.py`——本工厂约定 `__init__.py` 只作包标记，不参与保真度扫描） |
| `scheduler.py` | `vllm/v1/core/sched/scheduler.py` | 抽取四个切面：`_is_blocked_waiting_status`、`_try_promote_blocked_waiting_request`（仅结构化输出分支）、accept_tokens/validate_tokens 两个调用点（抽成独立方法）、`get_grammar_bitmask` 的 id 过滤逻辑 |

## 1:1 Source Map（关键符号）

| 精简版符号 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| `StructuredOutputsParams.__post_init__` | `sampling_params.py:L59-80` | 原样保留 | 六选一互斥，count>1/count<1 双向校验 |
| `StructuredOutputRequest.grammar`/`is_grammar_ready` | `structured_output/request.py:L42-64` | 原样保留（含 `self.status = RequestStatus.WAITING` 残留代码） | 门控真正落点是 `grammar` property，`is_grammar_ready` v0.21.0 零调用者 |
| `get_structured_output_key` | `structured_output/request.py:L77-98` | 原样保留 | 六种约束归一成 `(枚举, 字符串)` |
| `XgrammarBackend.compile_grammar` | `backend_xgrammar.py:L77-122` | 删 Mistral 分支、deprecated structural_tag 拆解 | 只留 5 个分支，**无 CHOICE**——choice 已在校验期改写 |
| `validate_xgrammar_grammar` | `backend_xgrammar.py:L268-354` | 删 Lark 转换调用、deprecated structural_tag 解析 | choice 分支原地把 `so_params.choice=None; so_params.grammar=EBNF` |
| `has_xgrammar_unsupported_json_features` | `backend_xgrammar.py:L221-259` | 删嵌套递归下钻，只留顶层三类检测 | 批准项6 |
| `XgrammarGrammar` 六方法 | `backend_xgrammar.py:L148-199` | 原样保留 | 请求级契约参考实现 |
| `GuidanceGrammar.accept_tokens`/`rollback` | `backend_guidance.py:L153-216` | 原样保留 | `rollback_lag`：EOS 后回滚少退一格，guidance 独有 |
| `GuidanceBackend.__post_init__` | `backend_guidance.py:L88-101` | 删 Mistral 分支 | 批准项1 |
| `choice_as_grammar` | `utils.py:L451-459` | 原样保留（`regex`→标准库 `re`） | choice→EBNF 改写链条的落点 |
| `StructuredOutputManager.grammar_init`/`_create_grammar` | `structured_output/__init__.py:L114-183` | 删 outlines/lm-format-enforcer 两个 `elif` | 命门：编译扔线程池，绝不阻塞调度循环 |
| `Scheduler._is_blocked_waiting_status` | `core/sched/scheduler.py:L1515-1521` | 原样保留 | 阻塞态判定 |
| `Scheduler._try_promote_blocked_waiting_request` | `core/sched/scheduler.py:L1998-2023` | 只留结构化输出分支，删 REMOTE_KVS/STREAMING_REQ | 二者依赖的状态不在本精简版 Scheduler 范围 |
| `Scheduler._advance_grammar_on_sampled_tokens` | `core/sched/scheduler.py:L1360-1369` | `should_advance` 门控换成 `use_structured_output` | reasoning 门控整体不在本章范围（批准项3） |
| `Scheduler._validate_spec_tokens_against_grammar` | `core/sched/scheduler.py:L1617-1621` | 同上 | `validate_tokens`「不推进的试走」调用点 |

## 测试

`tests/test_structured_output.py`（73 个，纯 Python，host 直跑）覆盖：六选一互斥
（count>1/count<1）、`get_structured_output_key` 六种归一、`choice_as_grammar` 转义、
`StructuredOutputRequest.grammar` 的 Future 门控（未就绪→None、就绪→原地替换、
幂等）、`Request` 初始阻塞态、xgrammar 五分支编译派发（CHOICE 不在其中）、
`validate_xgrammar_grammar` 的 choice→EBNF 原地改写、`XgrammarGrammar` 六方法
（含推进失败短路、`validate_tokens` 试走不留痕迹、rollback 计数对称）、
`GuidanceGrammar` 的 EOS+`rollback_lag` 语义、auto 阶梯（xgrammar 成功/失败降级到
guidance/skip_guidance 时 outlines 分支明确 raise 标出边界）、请求级后端选择冲突
校验、`StructuredOutputManager.grammar_init` 的惰性建后端 + 真实线程池异步编译 +
external_launcher 同步回退、调度器阻塞态判定与晋级、accept_tokens/validate_tokens
两个真实调用点的行为、`get_grammar_bitmask` 交棒点的请求 id 过滤。
