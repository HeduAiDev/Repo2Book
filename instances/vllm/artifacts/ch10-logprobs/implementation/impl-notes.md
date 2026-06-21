# ch10 Logprobs 精简版 — 实现笔记

只做减法的忠实精简版，source pin `f3fef123`。与 vLLM 同名、同结构、同控制流；只删不增。
所有装配/字节回退/累计/flat-vs-nested 控制流逐字保留，仅删去与 logprobs 无关的编排与
未在本章路径上调用的工具方法（均带 `# SUBTRACTED:` 注释）。

## 文件布局

| 精简版文件 | 镜像的真实 vLLM 文件 |
| --- | --- |
| `logprobs.py` | `vllm/logprobs.py` |
| `logprobs_processor.py` | `vllm/v1/engine/logprobs.py` |
| `detokenizer_utils.py` | `vllm/tokenizers/detokenizer_utils.py`（仅 `convert_ids_list_to_tokens`） |
| `outputs.py` | `vllm/v1/outputs.py`（仅 `LogprobsLists`/`LogprobsTensors` 字段） |
| `output_processor.py` | `vllm/v1/engine/output_processor.py`（仅 `_new_completion_output` 下游接口） |

## 1:1 Source Map

| 精简版符号 | vllm/...:Lxxx | 改动 | 原因 |
| --- | --- | --- | --- |
| `LogprobsProcessor`（dataclass + 字段） | `vllm/v1/engine/logprobs.py:L29-L40` | 无改动（去日志器 import） | 主角类，字段语义全保留 |
| `LogprobsProcessor.from_new_request` | `vllm/v1/engine/logprobs.py:L42-L67` | 字面照搬 | 三个 None/0.0 分支（哪些 logprobs 被开启 + cumulative 初值） |
| `LogprobsProcessor._update_sample_logprobs` | `vllm/v1/engine/logprobs.py:L69-L119` | 字面照搬 | sample 主装配：逐 step tolist、非增量去 token、`cumulative += logprobs[0]`、append |
| `LogprobsProcessor._update_prompt_logprobs` | `vllm/v1/engine/logprobs.py:L121-L187` | 字面照搬 | prompt 装配：自行 Pythonize（`.shape`/`.flatten().tolist()`/`.tolist()`）、扁平化 offset 切片、无 cumulative |
| `LogprobsProcessor._get_sampled_context_ids` | `vllm/v1/engine/logprobs.py:L208-L247` | 字面照搬 | 字节修正上下文来源；flat（`start_indices`）vs nested（`next(iter)`）双路径；空区间跳过；max_context=4 |
| `LogprobsProcessor._correct_decoded_token` | `vllm/v1/engine/logprobs.py:L249-L310` | 字面照搬 | 上下文感知 UTF-8 多字节重建（本章技术核心），逐字保留 |
| `LogprobsProcessor._verify_tokens` | `vllm/v1/engine/logprobs.py:L312-L346` | 字面照搬 | 批量检测以 U+FFFD 结尾的候选并修正；横向候选 vs 纵向上下文 |
| `LogprobsProcessor.pop_prompt_logprobs` | `vllm/v1/engine/logprobs.py:L189-L206` | 字面照搬 | DELTA 语义：一次性返回并清空 |
| `LogprobsProcessor.update_from_output` | `vllm/v1/engine/logprobs.py:L348-L352` | import 路径改本地 | 唯一对外入口与 sample/prompt 分派（两 if 互不排斥） |
| `Logprob` | `vllm/logprobs.py:L12-L24` | 字面照搬 | nested 叶子记录（logprob/rank/decoded_token） |
| `FlatLogprobs`（+ `append`/`append_fast`/`__getitem__`/`__len__`/...） | `vllm/logprobs.py:L30-L152` | 字面照搬 | 扁平存储降 GC；`__getitem__(slice)` 保留以支撑 DELTA 切尾 |
| `create_prompt_logprobs` / `create_sample_logprobs` | `vllm/logprobs.py:L162-L172` | 字面照搬 | 容器构造；prompt 首位 `append(None)` 占位 |
| `append_logprobs_for_next_position` | `vllm/logprobs.py:L175-L206` | 字面照搬 | flat vs nested 写入分叉 + rank 链 `chain((rank,), 1..K)`（"第一个=被选中 token"的源头） |
| `convert_ids_list_to_tokens` | `vllm/tokenizers/detokenizer_utils.py:L83-L104` | 字面照搬 | 非增量逐 token 去 token（U+FFFD 的来源） |
| `LogprobsLists` / `LogprobsTensors` | `vllm/v1/outputs.py:L26-L37, L51-L59` | 仅留字段定义 | 本章只解包字段；构造/搬运/切片工具方法属其它章节 |
| `RequestState._new_completion_output` | `vllm/v1/engine/output_processor.py:L376-L407` | 仅留 logprobs 相关字段 | 下游接口：sample logprobs/cumulative 进 CompletionOutput，DELTA 切尾 `logprobs[-len(token_ids):]`，cumulative 不切 |

## 主要减法（均带 `# SUBTRACTED:`）

- `output_processor.py`：RequestOutputCollector 队列合并、abort、tracing/stats、streaming、parallel sampling 合并 —— 与 logprobs 装配无关。
- `outputs.py`：`slice_request`/`tolists`/`to_cpu_nonblocking`/`filter`/`empty_cpu` —— 发生在 sampler/EngineCore 侧。
- `detokenizer_utils.py`：`detokenize_incrementally` 等增量去 token —— logprobs 走非增量路径。
- 各文件 SPDX 头与 `init_logger`。

## 关于桩张量（不破坏保真度的说明）

`_update_prompt_logprobs` 在真实 vLLM 里吃 torch 张量（`.shape`/`.flatten().tolist()`/`.tolist()`）。
精简版**方法体逐字未改**；测试用一个最小桩 `FakeArray` 提供同名接口（numpy 与 torch 均支持
这三个调用），使其能在 host（无 CUDA/torch）直接跑，验证 Pythonize + 切片逻辑。这是测试侧的
输入替换，不是对被测代码的改动。

## 验证

- `python3 -m pytest tests/ -q` → 34 passed（纯单元，不 import vllm）。
- `python3 scripts/lint_fidelity.py instances/vllm/artifacts/ch10-logprobs` → 全部通过，21 个 `must_keep` 符号均在。
