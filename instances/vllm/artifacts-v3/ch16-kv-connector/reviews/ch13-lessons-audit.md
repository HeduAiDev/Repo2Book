# ch16《KVConnector》ch13 四经验回扫审计

- 审计对象：`instances/vllm/artifacts-v3/ch16-kv-connector/`（narrative/chapter.md 1448 行 + diagrams/ 9 图 + figure-manifest.json）
- 审计依据：ch13 读者反馈修复沉淀的四条经验（L1 图注是图的影子 / L2 账目逐笔讲「此刻在哪」 / L3 单例讲不全角色清单 / L4 清单≠快照），另加 ch13 前向承诺逐条兑现与数字口径跨章一致。
- 方法：全部 9 图 Read PNG 亲眼看；3 张方位存疑图（L2 章图、role-split、layer-overlap）另以 SVG `<text x/y>` 坐标判定（非目测）；pin 源（v0.27.1, 6e448d0ea9）逐一核锚点；`lint_trace_consistency` 与 `lint_diagrams` 均跑（exit 0）。
- 结论：**BLOCKING 2 / ADVISORY 4**。L1（逐图）与 L4（不变量）全部过；失分集中在**前向承诺**（ch13 埋给 ch16 的两笔账没接）与 **L2**（窗口期块账的一个时序卡点、同步路径一笔带过）。

---

## 一、逐图核 L1（任务 1）——9/9 过

| 图 id | 方位词核对 | 数字/覆盖核对 | 判定 |
|---|---|---|---|
| L2-ch16 | 北排=装配+配置门（SVG y≈183-343）、中排①-⑦=站 3-9（y≈403-615，①双查…⑦完成回收逐枚对上）、南排=失败回滚/producer 终局/边界三例外+两 why 注+F8 注（y≈741-810）——坐标判定与图注逐字相符；左轨站号条「1-2 装配·3-6 查等待·7-9 收发回收·10-12 边界」是同一 12 站的更细分组，与图注无冲突 | 站号 1-12、①-⑦ 枚举、F8 注指向 Part V 首站（=ch16 末段 L1446）一致 | 过 |
| ch16-fig-role-split | 左泳道=调度器进程（label x=70）、右=worker（x=902）、中缝进程边界 x=750、下行信 Metadata（y=314）在上/上行信 Output（y=454）在下——坐标判定 | 5+7 原语逐名、SCHEDULER=0/WORKER=1、bind_gpu_block_pool/register_kv_caches 与正文 L211 图注一致 | 过 |
| ch16-fig-none-means-later | 三出边布局、底部两拍实测条在底 | 步 1 零调度零占块、步 2 算 64−32=32 转 RUNNING，与正文 m2 表一致 | 过 |
| ch16-fig-subtail-arbitration | 仲裁前条在上、A/B 并列在下，与图注「仲裁前/A/B」序一致 | 40=32+8、呈 connector 32、A 采用 48 本拍 8、B 外部 0 回退 32 本拍 24，与正文 m3 表一致（盲审已记档的「B 支未画远端 offer=8」为 cosmetic，维持非缺陷） | 过 |
| ch16-fig-allocated-not-cached | 纵向动线（五段条→双账→状态条→三端点）无方位词风险 | ext_comp 32★、2 块 [1,2]、缓存账 0、free 63→61、先行 32 零前向，与图注/正文/trace 三方一致 | 过 |
| ch16-fig-worker-tick | 主线在左、右栏=no_forward 空拍变体（坐标+目测一致） | 7 事件序、get_finished 收 {dead}、no_forward 仅 2 事件 wait_for_save 缺席，一致 | 过 |
| ch16-fig-layer-overlap | 两面板上下叠放；每面板内**传输泳道（y=159）在计算泳道（y=219）之上**（契约面板 375/435 同序）——坐标判定，图注「上泳道传输/下泳道计算」成立 | 就绪 2/4/6/8、朴素 20=Σ8+Σ12、契约 14=max(8,12)+2、t=14 红虚线、11 事件、教学模型页脚，全对 | 过 |
| ch16-fig-recv-promote | 状态链→展开→双案例卡结构，无方位词风险 | 部分命中 48/块账 4=3+1/续算 16；全命中 64→63/补算 1，一致 | 过 |
| ch16-fig-delayed-free | 交接条→三快照（63→61→61→63 演变自明）→底部注记 | free 61 不动→回 63、2 块、has_finished_requests、对照组 offload False，一致 | 过 |

机械面：`lint_diagrams` exit 0；manifest 9 图全登记、全部被正文引用。

## 二、账目段扫 L2（任务 2）

**过项**：m3 仲裁三支（呈值/砍尾/采用/免 CoW/回退，L464-L477）逐笔有账；m9 结算三步序（补缓存→退一→分流，L1068）时序显式；m10 四景（先行→截断→补登记→重算量，L1242-L1249）逐笔可答「此刻在哪」；m14 寻址账（slot 公式、40-token→2 块、尾 8 token 不进交接，L986-L995）讲透；prefix-cache-stats 的记账时机点明（准入时记、未调度不计数，L1316）；护轨无环的归纳论证（L554）与活性收尾有据。

**卡点两条 → ADVISORY A1、A2（见 findings）**：async 准入拍「只发 ext 段块」的账（`num_new_tokens=0`，scheduler.py:L866-L869，注释原话 *loading remote KV, do not allocate for new work*）未引未述，m5 表「护轨门本拍 4 vs full-ISL 门 8」之差、m4「64-token 只挂 2 块」都隐性踩在它上面；同步命中的块/缓存时序一笔带过。

## 三、例子覆盖 L3（任务 3）

**过项**：三态 0/None/数字各有落点（0 在 ExampleConnector 代码 `return 0, False` + m3 B 外部 0；None 专节；数字 m2）；handle_preemptions 默认空实现明说（L810）；kv_role 三态各有实例（consumer 主线 / producer m11+例外二 / both=offload 磁盘版存取一来一回）；失败双策 async/sync 都有 trace（m10）；MLA 分支、encoder-decoder 拒绝、mamba 组内部形态均「明说不展开」。

**缺口两条 → ADVISORY A3、A4**：take_events 全章仅 docstring 清单一处出场；例外一/例外三无实测例且取证口径段（L19）未交代。

## 四、不变量 L4（任务 4）——过

- 「有远端写则本地截到块对齐」：点破 + `truncate_computed_blocks` 断言引文（L483）✓
- 站号/读法：「正文按讲解需要编排、不必照站号读」（L15）——清单≠阅读顺序，明说 ✓
- 三挂起态同类递进枚举（L684「后面还有两个同族的」→ L1440 总结收拢）✓
- async/sync 的块缓存状态互斥（「async 的块还没进缓存、evict 开关关；sync 的可能已缓存」，L1124）✓
- 完成是入场券不是结算（L1030）✓；None≠0 两种「没有」专节 ✓
- ch13 立的「comp 与 new_comp/ext_comp 不同时非零」在本章全部例子（m4 窗口、m9 提升）无违反，无需重述 ✓

## 五、ch13 前向承诺逐条兑现（任务 5）

| ch13 出处 | 承诺 | ch16 兑现处 | 判定 |
|---|---|---|---|
| L612（另 L643 映射行、L646 病根的预测器半边、图上病根注） | E3 分配侧对账：真实挂账 [null,null,null] 占位 3 格、外部段另发 1 块、新块 3 块、实体块 1+3=4；allocate_external_computed_blocks 语义（cdiv(total,bs)−len(req_blocks)）；「ext_comp 抬高 total_computed 却不进 local_computed」 | **无**（grep E3/占位（块表义）/跳段分配/allocate_external_computed_blocks 全零命中；L682「先占位」、L466「跳段恢复」均另一义） | **失约 → B1** |
| L646 主体 | ext_comp 病根：token 已算 ≠ 块不用分配，两本账分道扬镳 | L598（ext_comp 段点亮「not cached by vLLM, but cached by the connector」）、L641（块照常挂表、缓存账为零）、L684（账实分离）、m4 可观测面 | 兑现（预测器半边归 B1） |
| L781 | delay_cache_blocks：块先分下去、写缓存账推迟 | L624/L641（L551 早退跳 cache_blocks）+ 窗口节整节 | 兑现 ✓ |
| L961 | async KV load 覆写区跳过集合 | L665-L674（_skip_zero_block_ids）+ L1218 反面补登记 | 兑现 ✓ |
| L1151 | 配 connector 时 output_rank 置 None、全 rank 各回一份、执行器收齐后聚合（直链 ch16） | **无**（机制本体实际写在 ch17 L801「KV 输出聚合」） | **失约 → B2** |
| ch14 L951 | reserved_blocks 那条线「Part IV 末章 KVConnector 的戏」 | L520「在这里兑现」+ m5 表 | 兑现 ✓ |
| ch15 L1656 | 远端 KV 怎么发现/搬回/与本地命中怎么仲裁 | 站 3（双查）/站 4（仲裁）接走，L7 明说 | 兑现 ✓ |

## 六、数字口径跨章一致 + 锚点现核（任务 6）

- **block_size=16**：全章一致（m2/m3/m4/m5/m9/m10/m14）；m3 的「块 16、哈希粒度 8」与 ch15 的部分命中形态同类（ch15 教学例是 64/16，此处 16/8，属同类配置非数字引用，无冲突）。**过**
- **E3 例数字**（W=16/外部 32/total 64/再算 36/null×3+外部 1+新 3=实体 4）：本章无引用亦无冲突——缺席本身即 B1。
- **64 块池基线 free 63**：m4（63→61）、m9（61 基础上 +1 新算）、m11（61→61→63）三图一致。**过**
- **v0.27.1 锚点抽查 11 处全过**：factory.py:L74（NOTE 原话逐字）、scheduler.py L137（encoder-decoder assert）、L773/L788（双查标记）、L866-L869（async 零新算三行——正文未引但位置核实，供 B1 修法用）、L1026/L1041（等待态标记）、L2820（截断行）、base.py L3（docstring）、L465（签名）、kv_cache_manager.py L535/L551、mixin L76、kv_transfer_utils L37。ch16 自身引文全部是重构后口径（`allocate_new_computed_blocks` 为 v0.27.1 协调器包装名，pin 中它内部仍调 `allocate_external_computed_blocks`——正是 B1 要接的线头，非 ch16 锚点错误）。
- **「据 v0.27.1 工厂注册表共 16 项」（L254）**：实数 16（factory.py L152-L243 恰 16 个 `KVConnectorFactory.register_connector`）。**过**

---

## Findings 汇总

### [BLOCKING] B1 —— ch13 前向承诺 a+c（含 b 的预测器半边）未兑现：E3 分配侧对账与 allocate_external_computed_blocks 语义整章缺席
- **位置**：全章缺席；最近正文 L601-L613（allocate_slots 引文止于 coordinator 包装层）、L641、m5 表 L547-L553。
- **问题**：ch13 L612 明文「真实挂账会是 [null, null, null] 占位 3 格、外部段另发 1 块、新块 3 块，实体块 1+3 = 4。E3 分配侧的对账…账归 KVConnector 章」，L643 给了函数名与公式（cdiv(total, bs) − len(req_blocks)），图上病根注「ext_comp 抬高 total_computed 却不进 local_computed」。ch16 全零命中。pin 源核：机制俱在——kv_cache_coordinator.py:L192-L240 两段式（先逐组 touch 本地命中、再逐组调 `allocate_external_computed_blocks`，防先发的外部块逐出后组未 touch 的命中块）、single_type_kv_cache_manager.py:L291-L326（`get_num_skipped_tokens` 钳位 + `cdiv(local+ext, bs) − len(req_blocks)` 另发）。且 ch16 自家数字正踩在这笔没讲的账上：m5 护轨门「本拍 4」对 full-ISL 门「8」之差、m4「64-token 请求块表只挂 2 块」。
- **违反**：前向承诺（ch13 L612/L643/L646 预测器半边）。
- **修法**（≤5 句，落点 L641 段后或 m5 表后）：引 scheduler.py:L866-L869 三行（`if load_kv_async: num_new_tokens = 0`，注释原话 *loading remote KV, do not allocate for new work*）——async 准入拍零新算 token，本拍只发外部段块；一句打开 L535 包装：coordinator 先逐组 touch 本地命中，再调 `allocate_external_computed_blocks` 按 cdiv(本地+外部, block_size) − 已持 另发（m5「占 4 块 ext」、m4 的 2 块正是这一笔）；末句对回 ch13 E3：滑窗部署窗外段先以 null 占位占表长、外部段另发 1 块、提升拍续算发新块，实体块 1+3=4，ch13 预留的账在此结清。

### [BLOCKING] B2 —— ch13 L1151 的 output_rank 聚合承诺落空（机制实际写在 ch17）
- **位置**：全章零命中（「聚合」仅 L779 collective_rpc 另一义）；机制本体在 ch17 L801（「KV 输出聚合（output_rank 置 None、应答改经聚合器汇总…」）。
- **问题**：ch13 L1151「配了 KV connector 时 output_rank 置 None、全 rank 各回一份、执行器收齐后聚合（账归 KVConnector 章（第 16 章））」带直链跳 ch16，读者落地无着。
- **违反**：前向承诺（ch13 L1151）。
- **修法**：ch16 L729 段末（两封信处）补两句——上行信在 tp>1 时不再只有 output_rank 一家回：配了 connector 时 output_rank 置 None、全 rank 各回一份、执行器收齐聚合，聚合内景归第 17 章。（或改 ch13 L1151 链接指向 ch17，二选一；前者不动 ch13。）

### [ADVISORY] A1 —— [L2] 窗口期「另一半块在哪」的时序卡点未讲
- **位置**：L684（64-token 请求块表只挂 2 块）、m5 表 L549-L550（本拍 4 vs 全程 8）。
- **问题**：读者会问「64 token 不是要 4 块吗，另外 2 块呢」「r1 全程 8 块、护轨门怎么只查 4」——答案（async 准入拍零新算、新算段块推迟到提升拍，m9 case1 的「+1 新算」即它）要从三张表反推。与 B1 同根，同一处修即可。
- **违反**：L2（一句话标签盖不住时序卡点）。
- **修法**：并入 B1；若 B1 落在 m5 表后，L684 处加一句回指即可。

### [ADVISORY] A2 —— [L2] 同步命中的块/缓存时序一笔带过
- **位置**：L364（m2 步 2 行「本地 0 + 外部 32 = 已算 32…转 RUNNING」）。
- **问题**：同步路径（load_kv_async=False）不走等待态：块当拍分配、当拍照常登记缓存（不跳 cache_blocks）、KV 数据在前向首层 wait 就位。正文只深讲了异步窗口，同步侧「此刻在哪」缺一句，读者可能把「已分配未缓存」误推广到同步命中。
- **违反**：L2。
- **修法**：m2 表后（L367 附近）补两句：同步命中没有窗口——块这一拍就分配并照常入缓存账，KV 数据在前向首层前就位；「已分配未缓存」只对 load_kv_async=True 开。

### [ADVISORY] A3 —— [L3] take_events / build_connector_worker_meta 裸列无戏
- **位置**：L80（12 原语清单）；take_events 全章仅此一处，build_connector_worker_meta 仅 L774 省略注露名。
- **问题**：契约清单立了 12 原语、10 个有正戏，这两个只挂名，「谁调它、何时」无答——正是 L3 说的「缺席角色要么对照例、要么明说」。
- **违反**：L3。
- **修法**：L774 省略注处补半句：这两个是观测面原语（KV 事件上报、worker 回信装配），本章主线不经过。

### [ADVISORY] A4 —— [L3] 例外一/例外三无实测例，取证口径段未交代
- **位置**：L1322-L1362（deferred_frees 步序栅栏）、L1385-L1432（partial-tail 钉住）；对照 L19 取证环境段（列了四处口径差别，不含这两节）。
- **问题**：「三个挂起态」在总结作为一类带走（L1440），两个有 trace（m4、m11）、「已释放未归还」没有；例外三同样纯代码走读。L19 不交代，细心读者会疑心漏跑。
- **违反**：L3。
- **修法**：L19 补半句（例外一/三是纯步序时序控制流，本章以代码走读呈现、未驱动实测），或两节内就地一句。

---

## 修复纪律备注

- B1+A1 一处落点、B2 两句、A2 两句、A3/A4 各半句——全部是**定点小修**，无整章重写式建议；每处落点都是读者卡住的那一步，承重句各 ≤5 句。
- 本审计只读，未改 narrative/ 与 diagrams/ 任何文件；本报告为唯一落盘产物。
- 环境注：本机 shell `python3` 被沙箱拒（exit 49 无输出），linter 以 `python` 跑通，结论不受影响。
