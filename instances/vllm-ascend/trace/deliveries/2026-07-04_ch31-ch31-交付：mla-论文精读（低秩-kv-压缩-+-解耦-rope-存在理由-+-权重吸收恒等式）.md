# ch31 交付：MLA 论文精读（低秩 KV 压缩 + 解耦 RoPE 存在理由 + 权重吸收恒等式）

- **Type**: delivery
- **Chapter**: 31
- **Date**: 2026-07-04
- **Timestamp**: 2026-07-04T11:47:58Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: ch31, delivery, APPROVED, primer, mla, paper-fidelity

## What happened

reviewer 判定 APPROVED（14 条 issue 全 negotiable/non-blocking，无阻断）。四段式：动机（KV cache 是长上下文显存瓶颈，MHA 每 token 2·n_h·d_h·l 元素）→数学推导（低秩 KV 联合压缩为何可行 + 完整推导解耦 RoPE 的存在理由：对压缩后 key 加 RoPE 会在 W^Q 与 W_UK 之间夹入依赖相对位置 δ 的旋转矩阵 M(δ)，矩阵乘不满足交换律故不可吸收进静态权重，这是 ch20 读者最大的认知悬崖 + 权重吸收恒等式 W~=W_UK^T W^Q + q 侧低秩 q_lora）→小参数数值推演（参考实现 mla_reference.py 复现论文 Table 1 四种注意力 KV cache 对比 + decoupled_rope.py 数值验证 M(δ) 依赖相对位置）→落地（vllm_ascend/attention/mla_v1.py AscendMLAImpl 真实代码锚点：process_weights_after_loading 权重拆分/_q_proj_and_k_up_proj q 侧吸收/_v_up_proj o 侧吸收/decode 吸收路径 vs prefill 物化路径，回指第 20 章 MLA on NPU）。论文包在 book/papers/ch31-primer-mla/。已登记 4 条精简版接口签名（MLAReference/precompute_absorbed_query_weights/effective_middle_matrix/compare_kv_cache）到 bible interfaces.json；本章无待埋/待回收伏笔（bible.py due ch31 为空）。

## Why it matters

全书'原理篇'论文精读支线之一（与 ch33 投机采样 primer 同属），专门补足 ch20 昇腾 MLA 实现章读者的最大认知悬崖——为什么解耦 RoPE 必须存在、权重吸收恒等式为何成立。把 mla_v1.py 的吸收矩阵变换（W_UK_T/W_UV）与 DeepSeek-V2 论文 §2.1 的公式推导对应起来，读者读完能看懂'为什么代码要这样拆权重'而不止'代码在做什么'。

## What to remember

reviewer APPROVED，14 条 issue 全非阻断（4 条 paper-fidelity/排版一致性打磨 + 10 条 reader-comprehension 术语/记号定义建议）。四段式：动机（KV cache 瓶颈）→解耦 RoPE 存在理由完整推导（M(δ) 依赖相对位置故不可吸收）+ 权重吸收恒等式→数值推演（Table 1 复现 + M(δ) 数值验证）→mla_v1.py 落地锚点，回指 ch20。4 接口已登记 bible，无伏笔缺口。
