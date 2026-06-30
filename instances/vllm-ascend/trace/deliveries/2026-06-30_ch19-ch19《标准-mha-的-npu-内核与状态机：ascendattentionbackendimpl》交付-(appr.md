# ch19《标准 MHA 的 NPU 内核与状态机：AscendAttentionBackendImpl》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 19
- **Date**: 2026-06-30
- **Timestamp**: 2026-06-30T00:32:50Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch19, attention, mha, state-machine, torch_npu, paged-attention, fused-infer, f7-payoff, part-v

## What happened

reviewer APPROVED（14 项 issue 全 non-blocking/negotiable：1 处省略标注、1 处比喻措辞、句长软线、长代码块导航、argmax 单调性论证、forward-flow 图 workspace 预取版位、paged 量化对比、state-machine 图 sparse_mode 列澄清，及 5 项 reader-comprehension 名词/数据结构补释）。本章讲透 AscendAttentionState 五态机分流、AscendAttentionMetadataBuilder.build 装配(split_decodes_and_prefills 拆批/slot_mapping/block_table)、forward_impl 按状态选 paged vs fused-infer 算子路径、reshape_and_cache 写回分页 KV、workspace 预取节拍；AscendC8AttentionBackendImpl 作 subtract-only 候选选讲。承接 ch18 选定的 AscendAttentionBackend，本章是其 impl_cls。

## Why it matters

把 FlashAttention 的 CUDA 内核逐一换成 torch_npu 的 paged/fused-infer 算子，状态机决定走哪条算子路径——standard MHA 后端在 NPU 上的主路径范例，姊妹篇对照 vllm/v1/attention/backends/flash_attn.py。

## What to remember

f7（CP 组排布，埋于 ch08）在本章收口：enable_cp()(utils.py，prefill/decode_context_parallel_size>1) 运行期分流，AscendAttentionBackend.get_impl_cls/get_builder_cls 据此切 CP 版 AscendAttentionCPImpl/Builder；CP 组排布本身回指 ch08，深入 CP attention 算子在 context_parallel/ 子模块非本章主线。reviewer 强调 sparse_mode(0/3/4)由 causal/sliding_window 决定、与五态正交（写者已在 19.6 厘清，仅图注待澄清）。
