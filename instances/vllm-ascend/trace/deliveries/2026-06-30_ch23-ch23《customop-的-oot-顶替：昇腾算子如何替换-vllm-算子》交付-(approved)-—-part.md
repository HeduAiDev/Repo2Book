# ch23《CustomOp 的 OOT 顶替：昇腾算子如何替换 vLLM 算子》交付 (APPROVED) — Part VI 开篇

- **Type**: delivery
- **Chapter**: 23
- **Date**: 2026-06-30
- **Timestamp**: 2026-06-30T07:14:35Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: delivery, part-vi, customop, oot, forward_oot, register_ascend_customop, enable_custom_op, subtract-only, approved

## What happened

reviewer 判 APPROVED。本章是 Part VI（算子与编译层）开篇，讲整个插件「换头不换身」的总开关：(1) 注册表总开关——utils.py register_ascend_customop 构造 REGISTERED_ASCEND_OPS（~27 项：RMSNorm→AscendRMSNorm/SiluAndMul→AscendSiluAndMul/QuickGELU→AscendQuickGELU/FusedMoE→AscendFusedMoE…，310P 另覆盖一批 *310），遍历把每个 vLLM CustomOp 子类的注册项 register_oot 顶替为昇腾子类，一处调用全模型算子被批量换实现；(2) 分发链——基座 custom_op.py CustomOp.dispatch_forward 按 enabled/compile 选 forward_oot vs forward_native，昇腾子类只覆写 forward_oot；(3) 标本两则——AscendSiluAndMul（只覆写 forward_oot）、AscendRMSNorm（forward_oot 里 enable_custom_op() 真二分：真→torch.ops._C_ascend.npu_add_rms_norm_bias 融合算子，否→回退 torch_npu.npu_add_rms_norm/npu_rms_norm 原子算子）。精简版只删 dossier 批准项；host 无 NPU/CANN，融合/回退二分纯 Python 可跑、真实 AscendC 算子不真跑。7 个新接口已登记 bible。

## Why it matters

收束 OOT 三段式（ch02 发现/ch03 monkey-patch/ch04-05 配置）到算子层的落地：示范「不改 vLLM 模型定义一行，靠一张注册表 + forward_oot 覆写把每个算子的身体从 CUDA 换成昇腾」。是 Part VI（ch24 算子注册 / ch25 编译 ACLGraph / ch26 FusedMoE）的总入口，enable_custom_op() 的融合 vs 回退二分为 ch25 编译层埋线。

## What to remember

review-report.json：APPROVED，16 个 issue 全 negotiable=true/blocking=false。非阻断点集中三类：(a) 个别行号锚点 ±1 偏移（layernorm.py L27→L28、§23.8 L644→L645/L334→L335，其余锚点核对均准；lint_source_grounding 已过）；(b) 算法/数值：§23.6 HBM 往返表的回退「2 颗 kernel」隐含 self.bias 非 None，非量化场景 bias=None 时回退退化为单颗 npu_add_rms_norm，建议表注限定带 bias 场景；等价性建议补算子级一一对应论证（host 无 NPU 无法端到端数值追踪，已诚实交代）；(c) fig-overview「其余约 24 项」(4+24≈28) 与正文钉死 27 不一致，建议图改「其余约 23 项」对齐。其余为 reader-comprehension 术语/动机补注（kernel/访存受限/张量并行/meta 实现/vllm_ascend_C/self.bias 总初始化为 None 与「只在量化场景」措辞矛盾）。无伏笔到期（埋/回收均空）。
