# ch06 精简版实现笔记 — Parallel Sampling 扇出 (n>1)

只做减法的忠实子集：与 vLLM v1 同名/同结构/同控制流，纯单元（不 import vllm/torch/msgspec），
读者可直接跑通 n>1 扇出 → 独立下发 → 归并 → 级联 abort 全链路。源码 pin `f3fef123`。

## 模块构成

- `types.py` — 支撑类型的减法子集（SamplingParams / CompletionOutput / EngineCoreRequest /
  RequestOutput / RequestOutputCollector / random_uuid），只保留本章触及字段。
- `parallel_sampling.py` — **本章核心** ParentRequest（扇出派生 + 归并）。
- `input_processor.py` — assign_request_id（child id 派生前提）。
- `output_processor.py` — RequestState 归并 + OutputProcessor 登记/级联 abort。
- `async_llm.py` — AsyncLLM.add_request 的 n>1 扇出主控制流 + EngineCore 接收端。

## 1:1 Source Map

| 精简版符号 | 真实 vLLM 源 | 改动 | 原因 |
|---|---|---|---|
| `ParentRequest.__init__` | `vllm/v1/engine/parallel_sampling.py:L36` | 删 docstring；其余 1:1 | output_aggregator 预置 / child_requests / cached 初始化全保留（must_keep） |
| `ParentRequest._get_child_sampling_params` | `parallel_sampling.py:L51` | 删 docstring；逻辑 1:1 | 保留 seed+index 递进 vs 无 seed 缓存复用的关键不对称 |
| `ParentRequest.get_child_info` | `parallel_sampling.py:L83` | 删 docstring | `f"{index}_{request_id}"` 唯一 id + 登记 child_requests |
| `ParentRequest.get_outputs` | `parallel_sampling.py:L100` | 1:1（含注释） | 流式逐条转发 / FINAL_ONLY 按 index 聚合 / 去重已返还 / finished 判定 |
| `ParentRequest.observe_num_generation_tokens` | `parallel_sampling.py:L128` | 1:1 | 指标观测保留（observe_finished_request 静态方法 SUBTRACTED） |
| `InputProcessor.assign_request_id` | `vllm/v1/engine/input_processor.py:L214` | 删 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION 分支 | external_req_id 内部化 + `-{random_uuid():.8}` 8hex 后缀 |
| `AsyncLLM.add_request` | `vllm/v1/engine/async_llm.py:L368` | 删流式输入/deprecated 直传/reasoning 注入/校验上文 | n>1 扇出循环主控制流（兑现 ch04 延迟分支） |
| `AsyncLLM._add_request` | `async_llm.py:L400` | 删 log_requests | output_processor + engine_core 双挂 |
| `OutputProcessor.add_request` | `vllm/v1/engine/output_processor.py:L508` | 删流式复用分支 | 建独立 RequestState + parent_requests + external_req_ids 反查表 |
| `OutputProcessor.abort_requests` | `output_processor.py:L466` | 删 abort 终态 output 产出 | 父→级联未完成 child 的 id 归集 |
| `RequestState.make_request_output` | `output_processor.py:L321-L331` | 删 detokenize/DELTA偏移/pooling/_new_request_output 组装 | parent_req.get_outputs 归并 + request_id 改回 external_req_id |

## 验收判据

把真实 vLLM 删掉所有 `# SUBTRACTED:` 分支，应当 ≈ 得到本精简版。
must_keep 全部 14 个符号保留（lint_fidelity 校验通过）。

## 已 SUBTRACTED（dossier.subtraction_plan.delete 批准项）

1. async_llm AsyncGenerator 流式输入分支（与 n>1 互斥）。
2. async_llm EngineCoreRequest 直传 deprecated + reasoning_ended/reasoning_parser_kwargs 注入。
3. async_llm kv_sharing_fast_prefill + prompt_logprobs 前置校验。
4. ParentRequest.observe_finished_request 指标静态方法。
5. output_processor pooling_output / delta token 偏移 / logprobs / _new_request_output 组装。
6. input_processor 完整 process_inputs（只保留 assign_request_id）。

## 运行

```
python3 -m pytest tests/ -q          # 14 passed，纯单元，无需 Docker
python3 scripts/lint_fidelity.py instances/vllm/artifacts/ch06-input-processor
```
