# ch15《一次 execute_model 的真实数据流——昇腾 forward context 与 DP 跨卡同步》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 15
- **Date**: 2026-06-29
- **Timestamp**: 2026-06-29T18:46:26Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: execute_model, forward-context, dp-sync, moe-comm, cudagraph-mode, part-IV

## What happened

ch15 多维评审 APPROVED 交付。主线串起一拍前向：NPUModelRunner.execute_model → _prepare_inputs → _build_attention_metadata → set_ascend_forward_context（昇腾在基座 set_forward_context 之上注入 moe_comm_type/method、flashcomm v1/v2、mmrs_fusion、mc2_mask，经 select_moe_comm_method 选定）→ _model_forward → _sample；DP 同步 _sync_metadata_across_dp 把 num_tokens+cudagraph_mode 打包 [2,dp] 零张量各填己列、一次 sum-allreduce → tokens 取 max / mode 取 min（NONE 为吸收元），昇腾比基座多同步 2 个标志、可走 NPU device group。精简版只验可读控制流（派发骨架/forward context 注入/select_moe_comm_method 决策/DP 打包，纯 Python 可跑；真实算子与 all_reduce 不真跑）。

## Why it matters

Part IV 执行主干第三章，把 ch13(NPUWorker)/ch14(NPUModelRunner 猴补) 搭好的台子真正跑一拍，承上启下到 Part V 注意力后端。

## What to remember

Reviewer APPROVED，0 blocking。22 条 non-blocking 微调（含 reader-comprehension 维度术语注解、§15.3 两拍追踪表 eager 子情形数值需修正、dp-sync.png『量』字 tofu 缺字、§15.5 省略标记/自指节号）已存 reviews/review-report.json，留 writer 定点小修，不退章。三处前向引用伏笔已登记：select_moe_comm_method→ch26(f10)、attn_metadata 后端实体→ch18(f9)、_sample 采样器→ch28(f11)。bible.py due ch15 为空，本章无应回收伏笔。
