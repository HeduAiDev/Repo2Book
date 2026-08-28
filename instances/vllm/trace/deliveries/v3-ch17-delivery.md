# v3 ch17《执行三层》交付归档（APPROVED，补归档）

- **Type**: delivery（v3 Archive 站，chapter-pipeline 第十七章、Part V「GPU 不等 Python：执行管线」开篇——**2026-08-29 archivist 补归档**：定稿 commit c9136b23 于 2026-08-27 落库，本记录与 bible 回写均为归档缺口补账）
- **Chapter**: v3 ch17 · Part V · kind=code（L0 缩放：GPU 执行臂列上层——三层各答一问 Executor 在哪跑/Worker 设备归谁管/ModelRunner 这一拍怎么算）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 定稿 2026-08-27 · 补归档 2026-08-29 · **Agents**: pipeline 各站（analyst→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED（定稿 commit 自述「评审 pass 零 blocking」；reviews/review-report.json 为 algorithm-pedagogy 单维度工件：verdict=pass、20/20 机制勾选表、1 条 negotiable 且**已修**——explainer m4.quantified 锚点 L597→L585 素材侧过期、正文本就正确，补归档时 grep 复核 explainer.json 已无 L597、L585×2 在位）

## What happened

- **评审与轮次**（如实记：原无 run-ledger.json，2026-08-29 补归档时按 Lead 指示回填最小台账 `reviews/run-ledger.json`——provenance 注明无原生台账、事实出自 manifest/commit e426eea4·fd224da9；轮次口径摘自 review-report.json、figure-manifest selfcheck/blind_review 与 git 史）：dossier 对抗自核两轮（e426eea4：EXECUTOR_FAILED 锚点四处+史实+车道）；图系两轮返修后均重盲审 PASS——mp-bringup-star figure-integration 阻断（图面/manifest 锚点 L597 误，真锚 L585=MessageQueue(1,1) 每 worker 一条；连带 READY 发出点 L870-L877→L886-L893）定点修+revise 后再验证 PASS；futurewrapper-fifo REVISE r1（writer 侧改字：耗时对比/FIFO 配对不变式/core.py:L655-L673）后两轮独立重盲审 PASS；writer 评审 1 条 negotiable（同 explainer 锚点）已修。
- **测试**：impl 84 passed（~28s host 三连跑稳定：真 torch/pyzmq/cloudpickle/真 mp spawn 子进程——WorkerProc 经真实 make_worker_process/worker_main/READY 握手出生、真 ZMQ 广播 MQ/逐 worker 应答 MQ、真 busy loop/监控线程/三级关停；无 vllm 包）；lint_fidelity 全过（must_keep 66 符号 over_subtraction 空账）。
- **bible 登记（v3 侧车，2026-08-29 补归档回写）**：glossary-v3 +18（ch17 正主 11 条：执行三层三问切分/控制面·数据面分离/FutureWrapper FIFO 配对/output_rank 单点收割/延迟初始化 WorkerWrapperBase/cloudpickle/MessageQueue（SHM 广播）/AsyncIntermediateTensors 懒同步/RayExecutorV2/CuMem 池 tag；首现章如实记先现章 7 条：collective_rpc→ch09、SPMD·NCCL·external_launcher·qualname·PCP→ch03、RLHF→ch15、EXECUTOR_FAILED→ch05，各注明正主 ch17——同 LoRA/fork 勘正先例）；concepts-v3 +16（对齐 pedagogy-plan introduces 三项 Executor·Worker·ModelRunner/collective_rpc/控制面数据面分离+拆细：mp 星形装配 READY 握手/延迟初始化三步硬约束/NCCL 先于显存快照/显存三锚点两池 tag/compile_or_warm_up 前移编排/一次 enqueue 全员可见/FIFO 配对不变式/output_rank/两段式墙内三面/失败两路关停三级/同一抽象两种拓扑/PP 接力懒同步/async worker 半边/worker_extension_cls 注入）；interfaces-v3 ch17 +39（Executor 抽象面五件+uni 全类+run_method+mp 全链十七件+worker_base 两类+gpu_worker 十一件+runner 契约面+平台/工具收尾，出自 impl-notes 1:1 Source Map）；figures.json 追加 5 张（L1-partV(l1-partv-guide) + L2-ch17(l2-executor-worker-model-runner) + three-layers(m1)/mp-bringup-star(m4)/futurewrapper-fifo(m12)，book:v3；needs_figure 3/3、其余机制无独立图由正文表承载）。
- **伏笔对账**：本章不埋不收（pedagogy-plan foreshadows 无 planted=17/paid=17 条目，与 dossier foreshadow_due should_plant/payoff 均空一致），foreshadow-v3.json 零改动。前向指针（ch18 差量调和/固定地址、ch19 编译捕获、ch33 spec drafter、ch34 PP 与 ray 全貌）与回指（ch9 两段式外壳、ch12 异步调度 worker 半边、ch14 显存账本）均为信息性指路。**注意 F7（block_table 间接寻址 planted=13→paid=22）本章站 10 只到「return None 之前是黑盒」为止——正中 ch13 埋点与 ch22 收款的中间，无踩线。**
- **图登记门禁**：`REPO2BOOK_INSTANCE=vllm python scripts/lint_figures_registered.py <章目录>` 显式传参 exit 0；manifest 5 图与 bible v3-ch17 条目逐 id 集合相等。

## Why it matters

Part V 开篇章落位归档：L0 图 GPU 执行臂列上层点亮，ch9 立的五拍外壳（②④ 两拍）至此有了墙内实现的三层骨架——后续 ch18（持久批次/固定地址）、ch19（编译捕获）都挂在这根骨架上。方法论一笔：**同一抽象两种拓扑**（uni run_method 三分支与 mp busy_loop 前两支逐字同构）与 **FIFO 配对不靠 id 靠顺序**（三条队列天然同序免掉整套请求编号协议）是本章最值得后章复用的两个设计句型；EXECUTOR_FAILED 与 ENGINE_CORE_DEAD 的「同形状对偶」把 ch05 的死讯纪律接进了执行臂。

## What to remember

1. **【补归档性质】** 本章定稿（2026-08-27）早于 archivist 归档（2026-08-29）两天——期间 bible 侧车与 state.chapters 均缺 ch17 段，本次补齐；ch15 delivery「账本缺口」条所列 ch13/ch17 两项至此清账（ch14 由另一条线补归档中、ch16 未发车）。
2. **【评审工件单薄、如实记】** reviews/ 仅 algorithm-pedagogy 单维度 review-report（pass+1 negotiable 已修），run-ledger.json 为 2026-08-29 补归档回填（非原生台账，provenance 在档）——与 ch15/ch18 的多维度并行评审工件不同，系当时流程记录不齐，非评审未跑（图系 REVISE 轮与 dossier 对抗自核两轮均有 manifest/commit 可考）；后续如需完整轮次口径以 git 史为准。
3. **【impl 取证环境】** host 无 GPU/无 vllm：广播/应答 MQ 用 ZMQ 回环 tcp 等价替代（控制面契约一致），凡毫秒数只取结构与量级（1823.8ms 拉起大头是 Windows spawn 冷启动）；runner 前向深水为注释占位（delete 6 批准、ch18/19 域），两段式协议面全真可观察；determine_available_memory HOST SEAM 返回 0（账本归 ch14）。
4. **【卫生】** 定稿 commit 曾误入库 16 张 .chk-* 临时裁片，次日 fd224da9 已剔除（磁盘保留惯例之外的一次入库事故，已清）。
