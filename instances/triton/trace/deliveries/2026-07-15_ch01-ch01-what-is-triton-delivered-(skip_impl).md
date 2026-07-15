# ch01 What is Triton delivered (skip_impl)

- **Type**: delivery
- **Chapter**: 01
- **Date**: 2026-07-15
- **Timestamp**: 2026-07-15T15:16:37Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch01, skip_impl, primer-framing, backend-seam, constexpr, five-stage-lowering

## What happened

全书首章《Triton 是什么，以及本书怎么读》交付。kind=skip_impl(取景框章、无精简版)。中心机制 visit_Call 三岔分发(①JITFunction 抄 tt.func+tt.call/②@builtin 注入 _builder 建 op/③普通 Python 编译期执行)，配 tl 两层结构(@builtin 55 : @jit 30)、cdiv 追踪期 vs make_ttir 内联两形态、五级降级+现编 C launcher 发射、FOR_EACH_P 后端接缝。真 pin v3.2.0 实机编译取追踪期 make_ir IR 与阶段二 IR 对照。6 张图全 blind PASS；全部门禁绿；GitHub 渲染真值 0 未渲染。dossier 5 轮对抗性自核 + 2 图外科修(fig-m10 后端数=2/宏上限=4 澄清) + provenance 脱敏。

## Why it matters

开篇立起全书心智模型「Python 不是在跑而是在被追踪」，并把三条主线(constexpr 边界/看对 IR 的层/TRITON_KERNEL_DUMP 逐层读)埋成后续每章性能归因的地基；§8 后端接缝是姊妹篇 Triton-Ascend 逐章对位的锚。

## What to remember

ch01 done(skip_impl)。foreshadows f1..f5 已登记 arc-map(后端接缝→ch36+姊妹篇/五级降级→ch32/constexpr 特化→ch25/三岔与 tl 两层→ch16/TRITON_INTERPRET→ch13)。6 图入 figures.json，17 术语入 glossary，23 概念入 concepts。
