# ch16 KV cache 分配/reshape/bind 交付

- **Type**: delivery
- **Chapter**: 16
- **Date**: 2026-06-29
- **Timestamp**: 2026-06-29T19:41:17Z
- **Agents involved**: analyst,  implementer,  tester,  writer,  reviewer,  archivist
- **User present**: False
- **Tags**: ch16,  kv-cache,  memory-geometry,  int8-cache,  sparse-c8,  as_strided,  alignment,  APPROVED

## What happened

ch16《KV cache 在昇腾上的落地：分配、reshape 与绑定》评审 APPROVED 交付。主线 initialize_kv_cache_tensors → _allocate_kv_cache_tensors（int8 cache / sparse-c8 indexer / 2MB 对齐 _align_memory+_align_up）→ _reshape_kv_cache_tensors（_adjust_kv_layout 用 as_strided 把 block 维 stride 钉成一整页）→ bind（按模型特化）；辅以 get_kv_cache_spec / may_reinitialize_input_batch / calc_split_factor。对照基座 vllm/v1/worker/gpu_model_runner.py 同名方法看几何差异。17 条 issue 全 non-blocking：唯一机械修是 lint_source_grounding check4 报内部 impl-notes.md 只列 2 源文件(漏列对照锚点 vllm/v1/worker/gpu_model_runner.py)——正文 chapter.md 已正确引用全 3 文件、源码根基完整；其余为退化数值例(§16.4/§16.6 align_up/as_strided 恰好恒等、机制未被看见)+ align_up 公式漏 floor + 多处读者理解补强。已登记 11 个精简版接口到 bible。无伏笔应埋/应回收(bible due ch16 空)。

## Why it matters

补全执行脊柱的内存几何面(ch13 算预算/ch15 跑前向/本章管 KV 张量怎么在 NPU 显存摆)；讲清昇腾在 KV 显存几何上相对基座非改不可的事：int8 量化 KV、2MB 对齐约束(RDMA/Mooncake)、as_strided 页对齐布局重排。

## What to remember

ch16 已交付。回指 ch04(AscendMLAAttentionSpec/page_size_bytes)；前引 ch22(KV 运行期 block 分配/前缀缓存)、Part V ch18+(attention 后端消费 KV 张量)。已知机械尾巴：impl-notes.md 需补列 vllm/v1/worker/gpu_model_runner.py 以过 grounding>=3 文件；§16.4/§16.6 数值表落在退化(恒等)例、§16.2 align_up 公式第三项漏 \lfloor——均 non-blocking，正文未改。
