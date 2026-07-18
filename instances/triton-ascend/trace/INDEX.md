# triton-ascend Trace INDEX

《triton-ascend 源码解读》长期记忆索引（Archivist 维护）。基座对照《Triton 源码解读》(上游 Triton v3.2.0)。只保留最近 10 条；更早条目见 deliveries/、decisions/ 目录。

## Recent Activity

| Date | Type | Chapter | Summary | File |
|---|---|---|---|---|
| 2026-07-18 | delivery | ch04 | Part 2 语言层首章·deep：前端接缝——CodeGenerator 并挂双 builder(self.builder 标准 IR/self.ascend_builder 昇腾方言)+visit_Call 第四岔(extension.is_builtin 选路)+@al.builtin 双标记+WITH_DISPATCH 表+插入点/loc 接力；瞬时 review-agents-failed 逃逸→Lead resume 清除→APPROVED；writer 补 6 处；埋 scope/region/SSA→ch08；全 green | [2026-07-18_ch04-dual-builder-ascend-dispatch.md](deliveries/2026-07-18_ch04-dual-builder-ascend-dispatch.md) |
| 2026-07-18 | delivery | ch01 | 全书首章·flagship 鸟瞰：三支柱(fork 非插件/三段结构化下降链/达芬奇 cube-vector 双核)+ book-map；dossier-verify 抓 Lead 发车 focus 事实错→修 dossier 复跑；writer 补 7 处 reader-comp；APPROVED 全 green | [2026-07-18_ch01-birdseye-ascend-backend.md](deliveries/2026-07-18_ch01-birdseye-ascend-backend.md) |
