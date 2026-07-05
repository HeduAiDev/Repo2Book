# ch10 PD 分离(Mooncake) 存量回修交付

- **Type**: delivery
- **Chapter**: 10
- **Date**: 2026-07-03
- **Timestamp**: 2026-07-03T13:19:02Z
- **Agents involved**: archivist, implementer, illustrator, writer, reviewer
- **User present**: False
- **Tags**: retrofit, ch10, layerwise-push, figures

## What happened

ch10-pd-disaggregation-mooncake 完成存量外科回修：新增 layerwise-pipeline-overlap 插图(逐层 KV 传输与下一层计算重叠)，chapter.md 补充数值追踪表(4层顺序传输64 vs 逐层流水线46，隐藏比(L-1)/L，L=80时约99%)，reviewer 盲审 APPROVED(1条非阻塞图注冗余建议)。

## Why it matters

layerwise 连接器是本章'省下跨节点KV传输延迟'核心论点的关键机制，此前缺图/缺数值追踪，回修补齐素材真相源与图文一致性。

## What to remember

新图+新机制 layerwise-push 已登记进 bible/figures.json(figure_id=layerwise-pipeline-overlap)；review 结论为 APPROVED，剩余1条建议(图注与表格重复)为 negotiable/non-blocking，可留给下次微调，不影响归档。
