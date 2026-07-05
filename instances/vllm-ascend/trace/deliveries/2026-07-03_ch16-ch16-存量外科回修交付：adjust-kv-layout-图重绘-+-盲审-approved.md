# ch16 存量外科回修交付：adjust-kv-layout 图重绘 + 盲审 APPROVED

- **Type**: delivery
- **Chapter**: 16
- **Date**: 2026-07-03
- **Timestamp**: 2026-07-03T14:24:47Z
- **Agents involved**: archivist, reviewer, illustrator
- **User present**: False
- **Tags**: retrofit, ch16, kv-cache, figure-fix

## What happened

对 ch16(KV-cache allocation/reshape/bind)跑 chapter-retrofit：体检 8 个机制，仅 adjust-kv-layout 判 figure=wrong(旧图只画退化情形一 gap=0，与图注承诺的情形二 gap=8 矛盾)；复用 gen_diagrams.py 重绘为情形一/情形二两带对比图，Read PNG 亲眼核对 block 落点 0/24/48 与 gap=8 后盲审 PASS。reviewer 对全章 7 个机制逐一核对三层递进(直觉/数值推演/源码)后判 APPROVED，另记 3 条 non-blocking 协作性观察(reshape-view 缺独立数值表、align-primitives 不变量未展开、adjust-kv-layout.png gap 方块未直接标数字)供未来体检参考。

## Why it matters

外科回修纪律要求逐机制体检+免修早退，只对体检标记的 figure=wrong 项动手，不碰未登记机制，避免过度改写；reviewer 的 non-blocking 观察为未来体检轮次留下具体线索而不阻塞本轮交付。

## What to remember

ch16 retrofit 完成：adjust-kv-layout.png 已换成情形一/二对比图；figures.json 已补登 ch16 五个机制→三张图(alloc-geometry/adjust-kv-layout/bind-dispatch)映射；reshape-view 数值表薄弱是未来观察项，非本轮 blocking。
