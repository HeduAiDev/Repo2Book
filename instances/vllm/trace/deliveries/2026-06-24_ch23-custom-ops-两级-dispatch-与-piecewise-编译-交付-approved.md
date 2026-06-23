# ch23《模型层如何变成融合、图捕获的 kernel：CustomOp 两级 dispatch 与 piecewise 编译》交付 APPROVED

- **Type**: delivery
- **Chapter**: ch23
- **Date**: 2026-06-24
- **Timestamp**: 2026-06-24T00:00:00Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: custom-op, dispatch, torch-compile, piecewise, cuda-graph, rms-norm, split-graph, f17-payoff, APPROVED

## What happened

Part VI 模型层第二章，承接 ch22 模型契约，讲模型层算子如何变成融合、图捕获的 kernel——两级 dispatch。第 1 级（构造期、单算子粒度）：`CustomOp` 基类在 `__init__` 调 `dispatch_forward()` 一次性选定 `self._forward_method`（`enabled` 且 CUDA→`forward_cuda` 手写融合 kernel；否则 `maybe_compile(forward_native)` 纯 torch），之后 `forward` 只做零开销转发、无任何运行期平台/配置判断。开关 `enabled() = (default_on() or enabled) and not disabled`，`default_on()` 由 `custom_ops` 列表 all/none 计数定、`-name` 优先级最高；纯 eager 默认 all→走 forward_cuda，开 Inductor 默认 none→走 forward_native 给编译器留融合余地。以 `@CustomOp.register("rms_norm") RMSNorm` 为实例，forward_native(纯 torch online-add) vs forward_cuda(融合 `fused_add_rms_norm`) 数值对拍一致。第 2 级（首次前向、整图粒度）：`@support_torch_compile` 把 nn.Module 包成可编译，`VllmBackend.__call__` 经 `split_graph` 在 `splitting_ops`（attention）处 piecewise 切图——规整段 `PiecewiseBackend` 编译 + CUDA graph 捕获，attention 段保持 eager；切图正确性靠 split id 单调不减、相对顺序不变（切完仍是同一函数）。最后 `direct_register_custom_op` + `unified_attention_with_output`/`*_fake`：self.attn 注册为不透明 torch op 进 torch.compile 图避免 graph break，`use_direct_call = not opaque_attention_op()`（CUDA/ROCm/CPU 返 True→False→走 torch.ops 分支）。回收 f17。reviewer verdict=APPROVED，6 条 issue 全 non-blocking + negotiable（§23.6 不变量长句拆段、§23.8 双重否定结论前置、『定死』关键词变奏、default_on `not` 优先级点破、enabled 真值表逐步求值、ch24 前向链接对齐）。

## Why it matters

ch23 解答了「写在模型层的算子（RMSNorm、attention）如何最终变成融合的、被 CUDA graph 捕获的 kernel」这一 vLLM v1 性能关键。两级 dispatch 的分工——构造期选实现、首次前向切整图——是理解 vLLM 为何同时拥有手写融合 kernel 与 torch.compile/Inductor 两条路径、且二者不冲突的核心心智模型。piecewise 切图在 attention 处断开、其余段走 CUDA graph 的设计，正是 ch22 留下的 f17（Attention 吞掉的 self.attn 自定义算子进 torch.compile 图）的落地，也为 ch24 走进 self.attn 内部铺路。

## What to remember

两级 dispatch：第 1 级 CustomOp.__init__ 一次性选 forward_cuda vs forward_native（enabled()=(default_on() or enabled) and not disabled，-name 优先级最高，纯 eager→cuda/开 Inductor→native），forward 零开销转发；第 2 级 @support_torch_compile + VllmBackend 经 split_graph 在 splitting_ops(attention) piecewise 切图，规整段编译+CUDA graph 捕获、attention 段 eager，切图正确性靠 id 单调+顺序不变。self.attn 经 direct_register_custom_op 注册为不透明 torch op 避免 graph break（use_direct_call = not opaque_attention_op()）。RMSNorm 为 CustomOp 实例（forward_native/forward_cuda 数值对拍）。回收 f17(resolved_in=ch23)。bible 注册 8 条 ch23 精简版接口。verdict=APPROVED，6 条全 non-blocking+negotiable。归档已知坑：archivist.py record 对 ch-前缀 slug 在 _update_state_chapter 抛 ValueError，delivery 文件由 archivist 手工落地、state.json 手工回写（同 ch04/ch10/ch11/ch22）。
