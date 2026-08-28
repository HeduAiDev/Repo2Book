# v3 ch15《前缀缓存》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第十五章、Part IV「显存是主角：分页 KV」第三块拼图）
- **Chapter**: v3 ch15 · Part IV · kind=code（L0 缩放：KV 账本列缓存区——缓存面点亮，KV 半区自上而下全通）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-28 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，11 条 issue（0 blocking、11 negotiable，其中 3 条 reader-comprehension 维、1 条为逐机制勾选表存档非缺陷），全文见 `artifacts-v3/ch15-prefix-caching/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 2 轮（r1：45/45 pytest host 绿 + lint_fidelity 0 BLOCKING，但**容器差分电池**（impl vs 钉版 v0.27.1，47 CHECK 场景）抓出 2 个测试断言 impl-fallback 数字而非 pin 真实行为——都在 block_size==hash_block_size 的 mamba-align 配置：真实 MambaManager null 填充后只登记最后一个状态块（hash[4]@80），max_cache_hit_length=num_tokens−1 永远探不到链尾 → mamba 组恒 miss、不动点把整笔命中拖到 0（钉版实测 hit==0/boundary==64）；impl 基类回退（dossier.delete #6 批准）使其像 full 一样逐块登记——断言数字是减法副作用。r2：两测改 partial-hit 配置（full(64)+mamba(64,'align')、hash_bs=16）重驱，impl 与 pin **双侧逐字节一致**（hit==0/boundary==48），impl-notes 登记该 impl≠pin 边界；其余 43 测差分/锚点核过、46/47 电池行字节相同）；write↔review 3 轮；L2 1 轮；盲审 2 轮（r1 失败 1 处：phase1-miss-stop B 面板链位 0..3 与 spec 池块号 1..4 两套编号未对齐 → 统一池块号重渲；r2 零失败 13 图全 PASS）。foreshadow_due=[F2 payoff]（已收）、escalated=null。
- **bible 登记（v3 侧车）**：glossary-v3 +23（链式哈希/radix 树/APC/NONE_HASH/extra_keys/cache salt/prefix_caching_hash_algo/BlockHashToBlockMap/BlockHashListWithBlockSize/CoW/块内 CoW 部分命中/拷贝对/过户/逆序 free/劈分/惰性驱逐/混合不动点/junction（shared_prefix_boundary）/Marconi 钉住/稀疏驻留/retention_interval/replay 边界 + **LRU 补账**——首现章如实记 ch11（ch11 一词带过、ch15 展开为主角并补系统代价谱），同 ch02 补账先例）；concepts-v3 +13（对齐 pedagogy-plan introduces 四项+拆细：链式哈希/平面哈希表/extra_keys 语义隔离/粒度视图/命中窗口只进门开一次/touch 救回共享/满块写回/LRU 双不变量/惰性驱逐/F2 收口/块内 CoW 三件套/混合不动点/Marconi junction+稀疏驻留）；interfaces-v3 +36（哈希链族六件 + BlockPool 前缀面七方法 + single_type 六切面 + coordinator 三件 + manager 三件 + Request 三面 + scheduler 五件 + 装配/worker/stats 收尾，出自 impl-notes 1:1 Source Map）；figures.json 追加 13 张（L2-ch15(l2) + chained-hash(m01)/flat-hash-map(m02)/phase1-miss-stop(m04)/touch-refcount(m05)/writeback-mask(m06)/reverse-free(m07)/split-free(m08)/f2-preempt-rehit(m11)/hash-granularity-view(m12)/partial-cow(m13)/hybrid-fixed-point(m15)/marconi-junction(m16)，book:v3；mechanism_id 沿 ch18 的 chNN-mNN 零填充体例——dossier 账本 id 为裸 mN，映射一致；m03/09/10/14/17/18/19/20 无独立机制图，m09/m14 并入相邻图+正文表承载）。
- **伏笔对账**：本章应埋 0、应收 **F2 抢占恢复撞前缀缓存**（pedagogy-plan planted=11/paid=15，与 run-ledger foreshadow_due 一致）——正文实际收到：L888 显式「把第 11 章埋的伏笔收掉」接走 ch11 按下的第四问、L932 显式点名 F2 并定论期望收益取决于生存窗口、图 ch15-fig-f2-preempt-rehit+图注「第 11 章埋、本章收」、四步链路（free 不清哈希→逆序+劈分→重排重查→touch 救回）+重算量上下界 [1,P] 全论证——foreshadow-v3.json F2 paid→**done:true** 带 payoff_evidence。**遗留 writer 一行债**（ch11 evidence 已记、archivist 无权改 narrative）：ch11 chapter.md L11「（F2，第 15 章）」与 L530/L534「Part IV」仍为裸章号——ch15 slug 已定，链接化 `../../ch15-prefix-caching/narrative/chapter.md` 现在可做，留 Lead 排 writer 定点。
- **图登记门禁**：`REPO2BOOK_INSTANCE=vllm python scripts/lint_figures_registered.py <章目录>` 显式传参 exit 0（active_instance=triton-ascend 下必须带环境变量——ch01 已记的坑）；manifest 13 图与 bible v3-ch15 条目逐 id 集合相等（本记录内程序核对）。

## Why it matters

Part IV 第三块拼图落位：L0「调度 · 显存账本」列 KV 半区的**缓存面**点亮，ch13 的池与块表、ch14 的账本与门、ch15 的缓存自上而下全通——ch10 整章当黑盒用的 `get_computed_blocks` 与 ch11 按下的「free 不清哈希」全部有了完整机制。**F2 是本书首个跨 Part 长伏笔的正式收款**（ch11 埋 → ch15 收，「重算」变「重载元数据+补算」的期望收益由逆序+劈分两条纪律续命）。方法论两笔：① **差分电池**（impl vs 钉版同场景双跑）抓出「测试断言减法副作用而非 pin 行为」——host 全绿+lint 全绿都照不到这一层，值得进经验候选（与 ch18 的 impl_test_ledger 空数组对照：凡有容器差分电池的章 impl_test_ledger 必须非空）；② 本章对「绝对论断」的纪律（grep radix 零命中、ref_cnt+= 逐点枚举、版本锚逐 PR 核证）把 review 的可核性推到全书最高水位之一——issue-1 连版本边界方向都逐 git tag 核出来了。

## What to remember

1. **【writer 定点小修清单待用】** 11 条全部 negotiable 即归档（APPROVED 不阻断合规，ch02/ch07/ch08/ch09/ch10/ch12/ch18 同先例）。最优先 **issue-1 版本锚反向**（事实错向，非措辞问题）：L740「v0.21 之前的 free…」应为 v0.24（劈分 PR #42656 首个 release，git tag --contains 核证；v0.21-v0.23 都带着「全量 append 挂队尾」行为）；其余按 review 顺序：m5「唯一的例外」收窄一个定语（第 5 处 ref_cnt+= 在 single_type:L1624 connector/P-D 面归 ch16）、3/8=37.5% 分子口径（场景实为 2 无哈希块=2/8）、f2 图红虚线标签 map 4→0 两场景数字拼接、Merkle「对账」越 §8 账本系词边界、LRU 三层嵌套句拆两句、m16 表后三行补块大小/对齐参数、CoW 首现 L5 加一次性括注、move_block_hashes 补调用时机一句。
2. **【impl≠pin 边界登记】** block_size==hash_block_size 的 mamba-align 配置：真实 align 模式 mamba 组恒 miss（只登记最后状态块且 cap 探不到）；impl 基类回退是 dossier.delete #6 批准的减法、但其逐块命中的数字**不是 pin 行为**——写作与后续引用勿把该配置下的 mamba 命中讲成真实行为（impl-notes 与 run-ledger 双登记，m15/m16 场景一律用 partial-hit 粒度配置驱动）。
3. **【账本缺口如实上报（更新）】** bible 侧车与 state.v3.chapters 现覆盖 ch01-ch12 + ch15 + ch18：**ch13/ch14/ch17 仍未经 archivist 归档**（ch13 有 delivery 文件、ch14/ch17 连 delivery 也缺），ch16 未发车。本章按纪律**未抢注** ch13/ch14 已立概念（touch/自由队列/块表 append-only/hash_block_size/KV cache group 等，留给其补归档）——ch13-17 补归档前，gap 审计对「前章已立」的判定仍会失真，需 Lead 排期。
4. **【archivist 脚本坑（本日踩、已修）】** python dict 更新 glossary 时新条目值误用 tuple——json.dump 把 tuple 序列化成 JSON array（应为 {中文译名,一句释义,首现章} 对象），写完必须结构性核验（本次 23 条全中、当场转正）；另 python3 WindowsApps 坏桩照旧，全部走 Miniconda python。
