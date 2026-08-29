# v3 ch16《KVConnector——KV 账本+边界》定稿交付（2026-08-29 归档）

## 状态
- **全章定稿**：narrative 1448 行 + L2-ch16（12 站开篇章图）+ 8 张机制图（全部独立盲审 PASS）
- Review **APPROVED**：9 条 issue（2 blocking、7 negotiable）**全部修毕在稿/在档**——两条 blocking（logits 概念归章 ch7→ch8 两处跨章链接、ch14「第三道预算门」→「第一道」序号事实矛盾）已定点修在稿；档案侧两条（dossier embed「Worker-side 六原语」→七原语、explainer 补 m6 最小素材卡）也已修毕；行号锚取宽 4 处中仅 ①（bind_gpu_block_pool L289-L294→L291-L294）需对齐、已修，②③④为正文块有意加宽的设计常态、保留。全文存 reviews/review-report.json。
- run-ledger：impl_test 1 轮（60/60 host 纯控制流）、write_review 3 轮、L2 2 轮、盲审 1 轮零失败；无升级。
- **F8「KVConnector 双面契约」已埋**（L7 立术语「本章立的是新的东西：双面契约」/ L15 图注 F8 埋点注 / L254 钩子「两章各回收本章契约的一半」/ ch16-fig-role-split 承载契约全貌）——收款 ch36 P/D 分离 + ch37 KV 池化，均未到期。本章应收：无。

## 核心内容
双面契约（一个类按 KVConnectorRole 分居两进程、分开构建零共享、跨线只有 KVConnectorMetadata/KVConnectorOutput 两封信；We build separately to enforce strict separation）/ 外部缓存当第二个前缀缓存（None ≠ 0 三态、skipped 退避不堵队头）/ 子块尾仲裁（砍尾免 CoW、有远端写则本地截到块对齐）/ 预约护轨（async load 凑齐 Coffman 前三条、门口破持有并等待 fits in free−Σ预约−水位）/ 已分配未缓存窗口（账实分离三端点）/ worker 一拍（@contextmanager、finally 是正确性落点、no_forward 空拍）/ 逐层重叠 sum→max（wait_for_save 栅栏=正确性优先于重叠极限）/ slot 寻址确定性双射（block_id×block_size+offset 直写池张量）/ 全命中退一 token（远端缓存不豁免采样必要性）/ 第一个坏块截断+共享去重+补登记清零 / producer 终局接管（已交接未送达、has_finished_requests 保活）/ 边界三例外（deferred_frees 步序栅栏、drop_stale_output 抢占护栏、partial-tail 钉住交接）/ 16 注册后端四层地图。三个块生涯挂起态（已分配未缓存、已交接未送达、已释放未归还）+ Part IV 收官总结。

## 归档注意（archivist 2026-08-29）
- **bible v3 侧车回写完成**：glossary +12（双面契约 / 调度器侧-worker 侧 / None≠0 / 子块尾仲裁 / 预约护轨 / 已分配未缓存 / 已交接未送达 / 已释放未归还 / 逐层重叠 / 第一个坏块截断 / 两封信 / offload——KVConnector 本尊已在册（首现 ch09、正主 ch16），未重注）；concepts +14（对齐 pedagogy-plan introduces 双面契约+调度器侧/worker 侧并拆细到 14 机制概念）；interfaces +38（impl-notes 1:1 Source Map 全量：契约 base.py 面十签名+调度器落点+池侧开口+ExampleConnector+两封信）；figures +9（L2-ch16=l2-kv-connector、机制图沿 ch15/ch18 的 chNN-mNN 零填充体例映射 m01/m02/m03/m04/m07/m08/m09/m11——m5 护轨、m10 失败回滚、m12/m13/m15/m16/m17 无独立机制图，由 L2 站位+正文实测表承载）；foreshadow +F8（planted ch16 done:true 带 evidence，paid 36/37 done:false 未到期）。
- lint_figures_registered 显式章目录 + REPO2BOOK_INSTANCE=vllm **exit 0**（manifest 9 图与 bible 条目逐 id 相等）。
- explainer 素材真相源覆盖 12/17 机制（m1-m11+m14，含评审要求补登的 m6 不透明搬运单最小素材卡）；m12/m13/m15/m16/m17 五条 supporting 无素材卡——review 已登记知情，后续回修/审计勿以「explainer 无条目」为由误退正文。
- state.json v3.chapters.ch16 落段、v3.status 更新（Part IV ch13-16 收官）；不动 v2 字段。
