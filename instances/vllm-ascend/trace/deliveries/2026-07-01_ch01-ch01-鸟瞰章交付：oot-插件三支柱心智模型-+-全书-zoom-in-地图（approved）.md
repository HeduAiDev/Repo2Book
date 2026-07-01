# ch01 鸟瞰章交付：OOT 插件三支柱心智模型 + 全书 zoom-in 地图（APPROVED）

- **Type**: delivery
- **Chapter**: 01
- **Date**: 2026-07-01
- **Timestamp**: 2026-07-01T16:03:10Z
- **Agents involved**: reviewer,  writer,  archivist
- **User present**: False
- **Tags**: meta,  birdseye,  skip_impl,  three-pillars,  entry-points,  NPUPlatform,  monkey-patch

## What happened

ch01 meta 鸟瞰章（全书开篇、最后写）经多维评审 APPROVED（0 阻断，11 条 negotiable 小修：均为内嵌源码删减补省略标记/『都不做实际计算』措辞软化/一处抽象含糊落地/§1.3 六代码块加节奏路标/图 1-2 第四对钩子图注对齐/§1.4 两段 patch 补量级 25·22/reader-comprehension 四条术语补注 envs·backend_map 元组键·engine-core·conftest）。skip_impl：无精简版/无 companion，只内嵌三支柱最小真源码锚点（setup.py entry_points / __init__.py register+adapt_patch / platform.py NPUPlatform 类头 + 三个 get_*_cls），对照基座 vLLM v0.21.0。review-report.json 已落 reviews/。

## Why it matters

鸟瞰章是读者进全书的心智模型总闸：『装上就被发现(entry points)+一个平台类接管分发(NPUPlatform)+改不动处两段式打补丁(monkey-patch)+往每个扩展点登记昇腾实现』。它把后续 29 个代码章的 zoom-in 挂到一张全书地图上。全部 issue 非阻断，机械/措辞级小修，可交付。

## What to remember

ch01 APPROVED 交付。skip_impl 无新接口入 bible（三支柱锚点接口已归 ch02/ch03 owner）。due ch01 显示无伏笔应埋/应回收——鸟瞰章不 own 伏笔，只做全书地图。11 条 negotiable 小修待 writer 定点消化，无阻断。
