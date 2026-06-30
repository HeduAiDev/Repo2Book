# ch25《AscendCompiler 与 ACLGraph：torch.compile + cudagraph 栈整体顶替》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 25
- **Date**: 2026-06-30
- **Timestamp**: 2026-06-30T12:52:25Z
- **Agents involved**: archivist, reviewer, writer
- **User present**: False
- **Tags**: ch25, compilation, aclgraph, torch.compile, f12, f4b, delivery

## What happened

本章完工，reviewer APPROVED（17 issues 全 negotiable/non-blocking：6 个评审维度建议 + 11 个 reader-comprehension 名词补释；含 1 个 fusion_pass_stack 图缺席的可商榷项）。核心讲两个平台钩子(get_compile_backend→AscendCompiler / get_static_graph_wrapper_cls→ACLGraphWrapper，外加 get_pass_manager_cls→GraphFusionPassManager)整体换掉 vLLM 编译+图捕获栈；AscendCompiler.compile() 二分(npugraph_ex/torchair vs compile_fx+aot_autograd+融合 pass)；ACLGraphWrapper 的 NPUGraph capture/replay+BatchDescriptor 分桶+207008 兜底；GraphFusionPassManager 串 9 个融合 pass + register_dummy_fusion_op 占位锚点+双注册。

## Why it matters

编译层(Part VI)关键章：torch.compile/cudagraph 两大性能支柱的 OOT 整体顶替范式，承接 ch24 算子 meta、收口本章图捕获。回收伏笔 f12(ch24 埋，meta 备齐→本章图编/捕得起)与 f4b(ch06 埋/ch07 点，ca_comm custom all-reduce 归口图捕获框架，诚实说明框架在此、接入待来)，两者均已在 arc-map 标 resolved。

## What to remember

ch25 delivered/APPROVED。5 个新接口已入 bible interfaces.json(三钩子/AscendCompiler.compile 二分/ACLGraphWrapper/GraphFusionPassManager/register_dummy_fusion_op)。f12+f4b 已回收(narrative L5/L131/L661 兑现 meta→图、L643-651 诚实交代 ca_comm 框架在此接入待来)。reviewer 17 issues 全非阻断，未要求退章。
