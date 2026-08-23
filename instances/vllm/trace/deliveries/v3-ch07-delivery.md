# v3 ch07《上行：从 token 到文字》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第七章、Part II 第四章）
- **Chapter**: v3 ch07 · Part II 分而治之：进程边界与消息 · kind=code（L0 缩放：API 进程上行泳道）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-22 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，13 条 issue 全 negotiable/non-blocking（零 blocking），全文见 `artifacts-v3/ch07-uplink-token-to-text/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 1 轮（72 用例 host 全绿 ~3.5s，真 Rust DecodeStream/真 msgpack/真 asyncio，无平台分支不进容器）；write↔review 3 轮；L2/图 1 轮；盲审 1 轮零失败（10 张图全 PASS）。注：figure-manifest selfcheck_note 里的 R1-R3/盲审②长史是 Illustrate 站内返修与 agent 限流重试记录，与 ledger 正式轮次不同源，读账时注意。
- **归档时抽查（issue 兑现状态）**：13 条全 negotiable，抽查代表 9 处——PR #12287 负载画像（「6000 并发请求」应为「6000 条请求/并发上限 400」）、裸文件名 serving.py:L602（应为 chat_completion/serving.py）、省略标注「两行」实为 5 行、站号三处口径互斥（章首图注 5-12 解包流水/信箱节标题 12-13/信箱图注 11-12）、外部基准 +6.4% 与「不增产」论断缺一句机制性调和、ABORT 终态收条在断连路径无人取未点破、「下一节门口的扣留」指错节、「后厨」一词两指（L452 三岔口 vs L1333 引擎）、update-pipeline 图注「守卫三轮」与图面轮 3「期满」标签口径不一——**均未在稿**，以 APPROVED 归档；全部为词级/半句级，writer 定点小修一轮可全消，清单已留 review-report.json。
- **bible 登记（v3 侧车）**：glossary-v3 +18（detokenize/BPE/byte-level BPE/SentencePiece/▁/cleanup 算法/DecodeStream/TokenizersBackend/byte-fallback/U+FFFD/冻结/双 offset 滑窗/扣留 holdback/stream_interval/min_tokens/一单多杯 n>1/ParentRequest/终态收条）；concepts-v3 +14（两本账/三路工厂/Fast DecodeStream/慢线双 offset/byte-fallback 边界/尾部扣留/停止串仲裁/min_tokens 双生效/stream_interval 节流/n>1 扇出父聚合/停止串反向 abort/错误广播/abort 双轨入参与终态收条/BPE 词表地基）；interfaces-v3 +20 条（RequestOutputCollector/RequestOutput.add/RequestOutputKind/BaseIncrementalDetokenizer update+get_next_output_text/Fast/Slow/check_stop_strings/ParentRequest/RequestState 三道闸/OutputProcessor 唯一单循环/AsyncMPClient 到港面/AsyncLLM 上行五件套等）；figures.json 追加 10 张（L2-ch7=l2 章图 + m2/m4/m5/m7/m9/m13/m15/m16/m18 九机制图，book:v3）。
- **伏笔对账**：F5 断连反向 abort planted done:true（正文「客人离席」节完整三层接力+两跳实测，L1438 节尾钩子「引擎那一侧收到 ABORT 之后怎么接…到 Part VIII 服务面一章展开」+ L1450 总结 + L2 站 14 拍片/页脚 ch38 回收标；第一跳 with_cancellation 只嵌锚点不展开——ch38 域）。收款按 pedagogy-plan paid=38 未到期。本章无应收（无 paid=7 条目，与 run-ledger 一致）。ch2/ch4/ch6 已立概念（单槽信箱/三态契约/断连反向 abort/分片理货/双登记/双轨 id）按回指处理，未重复登记。
- **图登记门禁**：`REPO2BOOK_INSTANCE=vllm PYTHONIOENCODING=utf-8 python scripts/lint_figures_registered.py <章目录>` 显式传参 exit 0（active_instance=triton-ascend，无参模式照不到 vllm——ch06 同款第三次记录）。

## Why it matters

Part II 上行章：把 ch1「文本不过线、token 过线」的回程半边与 ch2 十六站走读只给结论的上行段（变换 5-6）展开成整条泳道的机制账——增量 detokenize 两本账与三路工厂（v0.27.1 判据 TokenizersBackend 非 v0.21.0 旧讲法）、byte-fallback UTF-8 边界的位置判定、停止串窗口仲裁与尾部扣留的精确账（max(len)−1）、三道闸与 stream_interval 节流、n>1 扇出父聚合（兑现 ch6 结尾留的念想）、单槽信箱状态机（Exception 无条件抢槽）、断连反向 abort 两跳与 ABORT 终态收条（兑现 ch2「判停两地协作」的前端反向出径）。此后 ch8（logprobs 回填 RequestState.logprobs_processor 恒 None 占位）、ch38（F5 回收：HTTP 层取消传导全貌+FINISHED_ABORTED 落账）可按「前章已立」直接引用；interfaces-v3 自 ch04 开张后登记上行泳道接口 20 条。

## What to remember

1. **【writer 定点小修清单待用】** 13 条 negotiable 全部未兑现即归档（APPROVED 不阻断是合规的，ch02 有先例）——但其中 4 条是可核性/导航硬伤类（PR 负载画像失真、裸 serving.py、站号三处互斥、「下一节」指错），建议下轮 writer 小修优先吃掉；「后厨」一词两指与「守卫三轮」图注口径两条是错配诱因，也值得修。
2. **【F5 收条交代】** ch38 回收时按 dossier foreshadow_due.how 展开：StreamingResponse 取消传导、非流式 Client disconnected、abort 终态输出的时序竞争——本章正文已如实留「帧在路上、引擎当前 step 还会做完的有界浪费」半句钩子。
3. **【跨章契约】** ch8 回填点已在 impl-notes 机械表登记（CompletionOutput.cumulative_logprob/logprobs 加默认 None、RequestState.logprobs_processor 字段名保留恒 None）——ch8 写作时按此恢复传参即可，不必重考古。
4. **【工具卫生延续】** active_instance=triton-ascend 期间，vllm 实例一切 bible 门禁继续须 REPO2BOOK_INSTANCE=vllm 显式前缀（连续三章同况）；lint_dossier/lint_fidelity 对 artifacts-v3 的静默跳过缺陷（ch04/ch05/ch06 已记）仍未修，本章由 impl-notes 收工审计（15 verbatim 段相似度全对齐批准删除项）兜底。
