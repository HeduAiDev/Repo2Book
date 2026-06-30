# ch28 交付：采样的 NPU 对位——Gumbel-max 避同步 + Triton 优雅回退 + 薄壳继承

- **Type**: delivery
- **Chapter**: 28
- **Date**: 2026-06-30
- **Timestamp**: 2026-06-30T18:12:02Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch28, sampling, gumbel-max, rejection-sampler, triton-fallback, thin-subclass, f11-payoff, APPROVED

## What happened

ch28《采样的 NPU 对位》评审 APPROVED 并归档。核心：(1) random_sample 用 Gumbel-max 等价式 probs.div_(q).argmax(dim=-1)（q=exponential_）替代 torch.multinomial，避开 CPU-NPU 同步，并用 npu_stream_switch(global_stream()) 把指数随机放独立 stream 异步做；(2) AscendRejectionSampler 从 ops/triton/reject_sample 引 Triton kernel 做投机解码拒绝采样，HAS_TRITON 不可用时优雅回退基类 pytorch 实现；(3) 两个采样器都是薄壳子类，只覆写碰 NPU 同步/能上 Triton 的几处。回收伏笔 f11（ch15 _sample 二选一派发的采样器内部对位）。review 19 条 issue 全 non-blocking。已登记 4 个精简版接口到 bible。

## Why it matters

兑现 ch15 留给采样章的伏笔 f11；与 ch26 batch-invariant 同源（都用 ops/triton/）；确立『子类化只覆写热点 + 数学等价式绕硬件短板 + 加速器可选回退』的薄壳典范，是后续 NPU 对位章节的参照。

## What to remember

ch28《采样的 NPU 对位》评审 APPROVED 并归档。核心：(1) random_sample 用 Gumbel-max 等价式 probs.div_(q).argmax(dim=-1)（q=exponential_）替代 torch.multinomial，避开 CPU-NPU 同步，并用 npu_stream_switch(global_stream()) 把指数随机放独立 stre...
