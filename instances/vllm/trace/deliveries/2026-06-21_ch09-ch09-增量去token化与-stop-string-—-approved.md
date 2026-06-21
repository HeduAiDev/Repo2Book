# ch09 增量去token化与 stop string — APPROVED

- **Type**: delivery
- **Chapter**: 09
- **Date**: 2026-06-21
- **Timestamp**: 2026-06-21T13:18:43Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch09, detokenizer, stop-string, utf8, min_tokens, delivery

## What happened

ch09《增量去token化与stop string》交付并归档。精简版只做减法忠实移植 vllm/v1/engine/detokenizer.py + tokenizers/detokenizer_utils.py + v1/engine/output_processor.py(去token化切片) @ f3fef123。四 linter 无 BLOCKING(fidelity/structure/grounding PASS, formulas 6 条非阻塞 inline 密度提示 WONTFIX), host 测试 25 passed + 1 skipped(Fast 路径 DecodeStream 容器门控, importorskip)。reviewer overall_verdict=APPROVED。bible 新增 6 条 ch09 接口(from_new_request 三路工厂/update/get_next_output_text holdback/detokenize_incrementally UTF-8 双窗口/check_stop_strings/process_outputs 去token化切片)。

## Why it matters

ch09 落地 Stage 3 输出处理的去token化与停止判定细节(holdback 防 stop-string 跨块漏判、min_tokens 守卫、UTF-8 多字节边界扣留再释放), 是 ch08 OutputProcessor 主循环的下游展开。bible due 显示无应埋/应回收伏笔, 伏笔账目无悬挂。

## What to remember

ch09《增量去token化与stop string》交付并归档。精简版只做减法忠实移植 vllm/v1/engine/detokenizer.py + tokenizers/detokenizer_utils.py + v1/engine/output_processor.py(去token化切片) @ f3fef123。四 linter 无 BLOCKING(fidelity/structur...
