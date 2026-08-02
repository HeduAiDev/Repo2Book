# ch12 retrofit: build-block-views figures + argument gap fix

- **Type**: delivery
- **Chapter**: 12
- **Date**: 2026-07-03
- **Timestamp**: 2026-07-03T13:54:49Z
- **Agents involved**: archivist, reviewer, illustrator, writer
- **User present**: False
- **Tags**: retrofit, ch12, figures, review

## What happened

外科回修 ch12(kv-offloading-host-cpu):对 retrofit-plan.json 中 flagged 机制 build-block-views(depth=shallow/figure=missing)产出素材(explainer.json+traces/build_block_views.json)并补 2 张图(byte-layout-single-segment, layout-reduction-single-vs-multi),盲审 PASS;§12.8 多段分支不变量论证的隐含前提(seg_stride_bytes≥seg_data_bytes 的来源)在评审中被指出但不 blocking,已登记为非阻塞待办。评审 verdict=APPROVED。

## Why it matters

close ch12 体检发现的 shallow/missing-figure 缺口,同时把评审发现的论证缺口显式记录以便后续任一次回修顺带补上。

## What to remember

外科回修 ch12(kv-offloading-host-cpu):对 retrofit-plan.json 中 flagged 机制 build-block-views(depth=shallow/figure=missing)产出素材(explainer.json+traces/build_block_views.json)并补 2 张图(byte-layout-single-segment,...
