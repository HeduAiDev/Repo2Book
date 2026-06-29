# ch12 《KV 卸载：host/CPU 分层与 OffloadingHandler 对位》交付（APPROVED）

- **Type**: delivery
- **Chapter**: 12
- **Date**: 2026-06-29
- **Timestamp**: 2026-06-29T14:50:50Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: kv-offload, cpu-npu, device-host, block-view-rebuild, dma, review-approved

## What happened

reviewer 判定 APPROVED（12 条 issue 全为 negotiable/non-blocking：5 条 reader-comprehension 加分项+3 条图文示例对齐/重叠可视化+1 条 must_keep BatchMemcpyParams 仅散文未内嵌+1 条五步标号一致+1 条长复合句拆分）。两条路径成稿：标准路径 NPUOffloadingSpec/get_manager->CPUOffloadingManager + CpuNpuOffloadingHandler.transfer_async（torch.npu.Event 异步分层搬运节拍，对位 vLLM offloading 框架 offloading_connector.py / v1/kv_offload/cpu/manager.py）；极简路径 SimpleCPUOffloadNPUWorker.register_kv_caches 重建 block view + NPUDmaCopyBackend DMA 直拷线程 + npu_mem_ops 指针算术。host 精简版 22 passed，lint_fidelity 全过（must_keep 28 符号全在）。review-report.json 已落盘 reviews/。

## Why it matters

ch12 是昇腾 KV 三件事的第三件（卸载=device<->host 分层搬运省显存，与 ch10 PD 分离跨节点直传、ch11 KV 池化外存复用并列）。交付后 ch12 接口入 bible（7 条精简版签名），无 ch12 伏笔需埋/回收。issue 均非阻塞，可作 writer 定点小修候选，不退章。

## What to remember

ch12 已交付 APPROVED。bible 已登记 ch12 七条接口（NPUOffloadingSpec/transfer_async/get_finished/expand_block_ids/register_kv_caches/NPUDmaCopyBackend/build_params+copy_blocks）。12 条 review issue 全 negotiable+non-blocking，留作 writer 可选小修。无 ch12 foreshadow。下一章 ch13。
