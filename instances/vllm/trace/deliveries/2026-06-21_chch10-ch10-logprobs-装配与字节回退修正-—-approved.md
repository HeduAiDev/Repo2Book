# ch10 Logprobs 装配与字节回退修正 — APPROVED

- **Type**: delivery
- **Chapter**: ch10
- **Date**: 2026-06-21
- **Timestamp**: 2026-06-21T14:30:41Z
- **Agents involved**: archivist, analyst, implementer, tester, writer, reviewer
- **User present**: False
- **Tags**: ch10, logprobs, byte-fallback, utf8, flat-vs-nested, APPROVED, delivery

## What happened

精简版忠实移植 vllm/v1/engine/logprobs.py + vllm/logprobs.py + vllm/tokenizers/detokenizer_utils.py(convert_ids_list_to_tokens) + vllm/v1/outputs.py(LogprobsLists/Tensors 字段) + vllm/v1/engine/output_processor.py(_new_completion_output) @ f3fef123，只做减法。覆盖：from_new_request 三分支决定哪些 logprobs 开启 + cumulative 初值；_update_sample_logprobs(cumulative+=logprobs[0]) / _update_prompt_logprobs(自行 Pythonize+扁平 offset 切片, 无 cumulative)；_get_sampled_context_ids(flat start_indices vs nested next(iter), 上限4, 跳空区间)；_correct_decoded_token 上下文感知 UTF-8 多字节重建(本章核心, '中'=e4 b8 ad 跨位置重解 byte-fallback U+FFFD)；_verify_tokens 横向候选 vs 纵向上下文；FlatLogprobs(扁平降GC, 不可改, slice 支撑 DELTA 切尾) vs nested；append_logprobs_for_next_position rank 链 chain((rank,),1..K)；pop_prompt_logprobs DELTA 清空；update_from_output 双分派。四 linter 全 PASS(formulas 0 提示)，host 34/34 passed。

## Why it matters

兑现 Stage 3 输出处理在 logprobs 维度的展开，承接 ch08 OutputProcessor 主循环与 ch09 去 token 化；为读者厘清 sample vs prompt logprobs 装配差异、flat/nested 双格式、以及 byte-fallback 多字节字符在 logprobs 路径上的上下文感知修正机制。

## What to remember

ch10 bible due 无应埋/应回收伏笔，伏笔账目无悬挂；bible 新增 10 条 ch10 接口(LogprobsProcessor/from_new_request/update_from_output/_correct_decoded_token/_get_sampled_context_ids/pop_prompt_logprobs/FlatLogprobs/append_logprobs_for_next_position/create_*/convert_ids_list_to_tokens)；测试 host 跑(纯单元不 import vllm, 桩 ByteFallbackTokenizer/FakeArray 复现真实行为)合理无需容器。
