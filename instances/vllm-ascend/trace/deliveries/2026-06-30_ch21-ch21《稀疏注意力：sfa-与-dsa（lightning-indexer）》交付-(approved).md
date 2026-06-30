# ch21《稀疏注意力：SFA 与 DSA（Lightning Indexer）》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 21
- **Date**: 2026-06-30
- **Timestamp**: 2026-06-30T05:11:11Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: attention, sparse, sfa, dsa, lightning-indexer, mla, additive-extension, part-v

## What happened

ch21 多维评审 APPROVED 交付。本章是 OOT 插件「加法式扩展」在注意力子系统的现身：vLLM 主干无对位后端，昇腾在 MLA 之上叠加稀疏注意力。AscendSFAImpl 继承 ch20 MLAAttentionImpl(复用低秩 KV 压缩/权重吸收)+INT8 索引器+q_hadamard，走 top-2048 稀疏 flash；AscendDSAImpl 继承昇腾自有 DSAAttentionImpl，核心是 Lightning Indexer——build_prefill/decode_metadata 各调 npu_quant_lightning_indexer_metadata(sparse_count=index_topk=512,sparse_mode=3) 先选每 query top-512 KV 再只算这部分；DeviceOperator(device_op.py) 设备算子门面注意力各章共用。精简版仅验可读控制流(MLA 复用+稀疏选择+元数据装配+门面派发,形状级)，私有 NPU 算子 host 不真跑。

## Why it matters

稀疏注意力把长上下文注意力从 O(L) 降到 O(top-k)，是插件在 vLLM 之外新增(而非替换)的能力，印证 OOT 插件不止顶替内核还能增量扩展。完成 Part V 注意力后端线 ch18→19→20→21 的稀疏延伸。

## What to remember

ch21 接口已登记 interfaces.json(AscendSFAImpl/AscendSFAMetadataBuilder/AscendDSAImpl/build_prefill_metadata/build_decode_metadata/DeviceOperator)。无 ch21 应埋/应回收伏笔(bible due 为空)。评审 APPROVED 0 阻断；全部 issue 为 negotiable=true/blocking=false 的定点小修(writer 侧)：①§21.4 indexer 字段行号 L403→L468 ②术语漂移「增量扩展」vs glossary「加法式扩展」(ch08 canonical)——建议加桥接句并补登同义词 ③§21.5 形状表 [T,N·L_kv] 记号未定义/L_kv 撞名 ④§21.2 key 二元/三元同名困惑 ⑤§21.3「便宜一个量级」归因应补 indexer 省 softmax+compressor cmp_ratio=4 ⑥lint_formulas L90 单段 3 inline 软提示 ⑦多条 reader-comprehension 补全(L²推导/对位定义/model runner v2/索引器具体数字/MLA prolog 衔接/inline 重写/内核签名)。术语漂移项需 Archivist 后续决定是否补登「增量扩展」为「加法式扩展」同义词。
