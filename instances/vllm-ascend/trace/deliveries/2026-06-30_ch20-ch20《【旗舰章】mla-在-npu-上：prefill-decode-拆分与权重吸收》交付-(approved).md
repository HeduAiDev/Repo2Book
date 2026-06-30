# ch20《【旗舰章】MLA 在 NPU 上：prefill/decode 拆分与权重吸收》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 20
- **Date**: 2026-06-30
- **Timestamp**: 2026-06-30T01:47:45Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch20, mla, attention, weight-absorption, torch_npu, prefill-decode, fused-operators, part-v, flagship

## What happened

reviewer APPROVED（15 项 issue 全 non-blocking/negotiable，blocking 计数=0）。本章讲透 MLA 五件套：(1) 权重吸收 _q_proj_and_k_up_proj——npu_format_cast 转 FRACTAL 格式、split W_UK/W_UV、permute W_UK_T、torch.bmm 把 q_nope 吸收进 latent 空间；process_weights_after_loading 做准备；§20.4 给结合律代数证明+形状级 runnable 验证。(2) 三段 metadata——AscendMLAMetadataBuilder.build 派生 build_chunked/prefill/decode_metadata，build_for_graph_capture 走图捕获。(3) prefill 路 _forward_prefill(npu_fused_infer_attention_score)+_compute_prefill_context(chunked context + npu_attention_update LSE 在线 softmax 合并,回指 ch19)。(4) decode 路 _forward_decode(MQA 吸收,npu_fused_infer_attention_score_v2+get_max_workspace 预取);exec_kv_decode/prefill 用 npu_kv_rmsnorm_rope_cache 融合算子。(5) forward 按 num_decodes/num_prefills 派发。非阻断 issue 集中在术语首现即注(MQA/LSE/PA/workspace/speculative-decode)、ql_nope→q_nope 命名切换桥接、KV 压缩比 1/50 vs 1/57 正文/图不一致、吸收恒等式可补数值演示、impl-notes 源码落点<3(归档侧)。

## Why it matters

MLA(多头潜在注意力)=低秩压缩砍 KV cache + 权重吸收省 decode 解压；昇腾把通用 MLA 的 prefill/decode 拆分与权重吸收换成最密集的 torch_npu 融合算子(format_cast/kv_rmsnorm_rope_cache/attention_update/fused_infer_score_v2)。承接 ch18 选定的 AscendMLABackend，本章是其 impl_cls(旗舰)。姊妹篇对照基座 vLLM v0.21.0 vllm/model_executor/layers/attention/mla_attention.py(MLACommonMetadataBuilder/MLACommonImpl 基类,注意基类在此文件非 v1/attention/backends/mla/common.py)。

## What to remember

host 无 NPU/CANN，torch_npu 融合算子不真跑——精简版只验形状级可读控制流(absorb 的 split/permute/bmm 形状代数、三段 metadata 装配、prefill chunked-context 的 LSE 合并循环、forward 按 decode-prefill 派发,纯 Python 可跑)。ch20 无伏笔应埋/应回收(bible due 空)。决策遗留：reviewer 建议 §20.4 补 2 路数值对照(朴素解压 vs 吸收)allclose 演示作加分项；absorb.png 副标题 1/50 与正文推导 1/57 待统一(gen_absorb.py 改一处字符串重渲染)。已登记 12 个 ch20 精简版接口到 bible interfaces.json。
