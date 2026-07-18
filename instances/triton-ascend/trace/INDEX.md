# triton-ascend Trace INDEX

《triton-ascend 源码解读》长期记忆索引（Archivist 维护）。基座对照《Triton 源码解读》(上游 Triton v3.2.0)。只保留最近 10 条；更早条目见 deliveries/、decisions/ 目录。

## Recent Activity

| Date | Type | Chapter | Summary | File |
|---|---|---|---|---|
| 2026-07-18 | delivery | ch03 | Part 1 收尾·on-ramp·skip_impl：vector-add 的 GPU→NPU 最小改写——拿 tutorials/01 活体标本逐行看它相对基座 triton 同名核改了什么。7 机制(移植相关改动只有两处 import torch_npu+device='npu'/核体逐字节不变、torch_npu 设备注册、block-mask 满块-尾块算法、逻辑 grid 跑通 vs 物理核跑快呼应 ch02、i1/i8 mask 布尔存储、测试即真相源、compile_hint preview 留 ch08)；linter bug 修正(fidelity/source_grounding 路径正则不含连字符→昇腾教程文件名假 narrative_grounding BLOCKING→Lead 修正则+TDD f7a5dec2)；writer 补 4 处+ill 修 fig-m1 行号徽标；埋 compile_hint→ch08(前向线索)；APPROVED 全 green·3 图盲审 PASS | [2026-07-18_ch03-first-kernel-vector-add.md](deliveries/2026-07-18_ch03-first-kernel-vector-add.md) |
| 2026-07-18 | delivery | ch02 | Part 1 flagship primer·达芬奇 NPU 硬件模型：三支柱(为什么非 GPU/AI Core cube+2vector+scalar 1:2/片上 GM-UB 192KB-L0A-B-C 显式层级+搬运+double-buffer 减半)+四约束(tiling 硬件必然三级 ncore-xblock-xblock_sub/32B-512B 对齐/mix_mode 双向判据/grid 强绑物理核)；诚实边界三层一致(source-cited 写死/paper-attributed 软化/图面 16×16×16 降视觉确定性)；dossier-verify 抓 triton_better_kernel 引错文件名(实在 architecture_difference.md:97-124)→Lead 修复跑；blind 1→2 修图；论文包 2 篇 web-verified+5 source-docs；无正式伏笔(收束表作前向线索)；全 green | [2026-07-18_ch02-davinci-npu-hardware-model.md](deliveries/2026-07-18_ch02-davinci-npu-hardware-model.md) |
| 2026-07-18 | delivery | ch04 | Part 2 语言层首章·deep：前端接缝——CodeGenerator 并挂双 builder(self.builder 标准 IR/self.ascend_builder 昇腾方言)+visit_Call 第四岔(extension.is_builtin 选路)+@al.builtin 双标记+WITH_DISPATCH 表+插入点/loc 接力；瞬时 review-agents-failed 逃逸→Lead resume 清除→APPROVED；writer 补 6 处；埋 scope/region/SSA→ch08；全 green | [2026-07-18_ch04-dual-builder-ascend-dispatch.md](deliveries/2026-07-18_ch04-dual-builder-ascend-dispatch.md) |
| 2026-07-18 | delivery | ch01 | 全书首章·flagship 鸟瞰：三支柱(fork 非插件/三段结构化下降链/达芬奇 cube-vector 双核)+ book-map；dossier-verify 抓 Lead 发车 focus 事实错→修 dossier 复跑；writer 补 7 处 reader-comp；APPROVED 全 green | [2026-07-18_ch01-birdseye-ascend-backend.md](deliveries/2026-07-18_ch01-birdseye-ascend-backend.md) |
