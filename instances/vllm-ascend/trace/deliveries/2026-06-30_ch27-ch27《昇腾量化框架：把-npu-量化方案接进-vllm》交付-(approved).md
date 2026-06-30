# ch27《昇腾量化框架：把 NPU 量化方案接进 vLLM》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 27
- **Date**: 2026-06-30
- **Timestamp**: 2026-06-30T17:11:57Z
- **Agents involved**: archivist, reviewer, writer, implementer, analyst, tester
- **User present**: False
- **Tags**: delivery, ch27, quantization, part-vii, oot-registry-adapter

## What happened

昇腾量化框架章交付，reviewer APPROVED。三入口 @register_quantization_config 注 ascend/compressed-tensors/fp8（后两者先删 vLLM 原版再同名顶替）；AscendModelSlimConfig.get_quant_method 按 layer 类型+quant_type 分发；methods/registry.py 的 _SCHEME_REGISTRY[(quant_type,layer_type)] + register_scheme 装饰器选 scheme；method_adapters.py 三 wrapper（AscendLinearMethod/AscendKVCacheMethod/AscendFusedMoEMethod）适配 vLLM 基类、全转交内部 scheme；以 W8A8_DYNAMIC 走通 create_weights→apply 全链（npu_dynamic_quant + npu_quant_matmul）；逐层 scheme 决策在 modelslim_config.get_linear_quant_type/quant_description，quant_parser.py 管 MXFP dtype 映射。三图全绿、四 linter 通过。

## Why it matters

Part VII 开篇章，是 OOT「注册表+适配器」范式最干净样本（~6k LOC 量化全栈）；承接 ch23 算子顶替 + ch26 FusedMoE（量化 MoE 经 AscendFusedMoEMethod）；为 ch28 采样/ch29 投机/ch30 加载铺路。

## What to remember

review APPROVED，24 条 issue 全 negotiable/non-blocking（含 21 条 reader-comprehension 增解释类 + 3 条措辞/图示微调）；无 must-fix。bible 已登记 6 个精简版接口。本章无伏笔应埋/应回收。
