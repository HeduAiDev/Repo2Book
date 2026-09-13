# ch15《前缀缓存》ch13 四经验回扫审计报告

- 审计对象：`instances/vllm/artifacts-v3/ch15-prefix-caching/`（narrative/chapter.md 1658 行 + diagrams/ 全部 16 张图与 figure-manifest.json）
- 审计方法：16 张 PNG 逐张 Read 亲眼看；全部方位存疑处解析对应 SVG `<text x= y=>` 坐标定量判定（不目测定案）；涉及口径处亲核 pin 源码 v0.27.1（single_type_kv_cache_manager.py / kv_cache_manager.py / kv_cache_coordinator.py / block_pool.py）
- 经验基准：ch13 读者反馈修复沉淀的 L1（图注/alt 是图的影子）、L2（账目公式逐笔讲「此刻在哪」）、L3（单例讲不全角色清单）、L4（清单≠快照类不变量要点破）+ ch13→ch15 前向承诺 + 数字口径
- 判级：[BLOCKING]=方位错/数字漂移/承诺失约；[ADVISORY]=卡点可再讲透/覆盖可补
- 结论：**BLOCKING 2 / ADVISORY 6**。两条 BLOCKING 都是图注方位词与图面实际布局不符（L1 同款病，ch13 的教训原样复发）；承诺 a/b/c 全部兑现且口径一致，数字口径全过。

---

## 一、发现清单

### F1 [BLOCKING] [L1·方位错] call-panorama 图注「泳道自上而下」——泳道实为自左向右，自上而下的是时间轴

- 位置：chapter.md L39（ch15-fig-call-panorama 图注第一句）
- 问题：图注写「泳道自上而下：Request、Scheduler、KVCacheManager、Coordinator、管家、BlockPool」。SVG 实测六条泳道头在同一水平行（y=157，x=310→530→745→950→1160→1394），是**纵向生命线按左→右排布**；图面右上角自己的注释写的是「时间轴 = 调用序（自上而下，非墙钟）」——自上而下的是时间，不是泳道。读者按图注去找「最上面一条 Request 泳道」会落空。
- 违反经验：L1（图注方位词与图面不一致；ch13 实例同款——图升级后图注仍按旧布局说话）
- 定点修法（一词）：L39「泳道自上而下」→「泳道自左向右」。其余图注内容（事件 0 拍外 / ①-⑦ 查 / ⑧-⑰ allocate_slots 三段 / 拍尾虚线回 Request）逐项与图面核对全部一致，不动。

### F2 [BLOCKING] [L1·方位错] why-groups 图注把右上面板与底部横带都标错了方位

- 位置：chapter.md L341（ch15-fig-why-groups 图注）
- 问题：图注写「左：Hybrid 协调器两个管家…各持块表，**同一请求 decode 拍后两组块表分叉**（full 组 [1,2,3,4,9]…SWA 组…[NULL,6,7,8,10]）；**键构成条**：…。右：**判据**…加六张**案例卡**」。SVG 面板几何（rect 实测）：左面板 x=60-720 只含协调器/两管家/块池结构；**两组块表分叉实拍与键构成条画在右面板**（x=748-1440，y=96-566）；**判据带与六张案例卡是底部两条全宽横带**（y=584-736 与 y=782-940），不在「右」。manifest 自己的 selfcheck_note 记的也是三区（「右面板 request+两表+键条 / 左面板池+回收线+构造注 / 判据+案例条」），图注压成两区时把两块内容挂错了方位。
- 违反经验：L1（图注方位词失真；与 ch13「下半块条带」实例同构）
- 定点修法（两个方位词）：L341 改为「左：Hybrid 协调器两个管家…各持块表；**右上**：同一请求 decode 拍后两组块表分叉…键构成条…。**下**：判据…加六张案例卡…」。内容与数字（[1,2,3,4,9]/[NULL,6,7,8,10]/键尾 00 00 00 00 与 00 00 00 01/六卡）全部与图面一致，只动方位词。

### F3 [ADVISORY] [L2] 写回 min 公式里的 total_computed_tokens 全章未落地一句数值

- 位置：chapter.md L648-651（allocate_slots 尾部 embed）与 L655 邻近 prose；配套图 ch15-fig-call-panorama ⑮ 帧的「num_computed_tokens=80」
- 问题：`num_tokens_to_cache = min(total_computed_tokens + num_new_tokens, request.num_tokens)` 中 total_computed_tokens 只在代码里出现（本章未像 ch13 L616-627 那样给出定义），正文只讲了 min 的右侧。读者对着全景图 ⑮ 帧「cache_blocks 收到 80」追问「调度器不是只把已算数设为 32 吗，80 哪来」时，缺一句「total_computed_tokens＝准入定下的已算数（本例命中 32）；＋本拍新算 48 恰等于 num_tokens 80，min 不截」。
- 违反经验：L2（组合公式里的一笔没讲「此刻在哪、值是多少」）
- 定点修法：L655 段内补一个括注句（约 20 字），不重排段落。

### F4 [ADVISORY] [L3] kv_cache_manager.free embed 里 `_partial_tail_pins` 三行裸奔——不触发的分支没有明说挂哪

- 位置：chapter.md L930-932（free embed 开头三行）
- 问题：`pins = self._partial_tail_pins.pop(...)` 展示了但全章零解释。亲核 pin 源码：该表由 `take_partial_tail_offloads`（kv_cache_manager.py:L848-L874）填充——**connector 侧 partial-tail offload 的钉住块，无 connector 恒空**。本章刚在进阶一教过 `_partial_hit_reqs`，读者极易把 `_partial_tail_pins` 误认成 CoW 尾账，卡在「free 什么时候会见到 pins」。
- 违反经验：L3（缺席/不触发的角色要明说挂在哪）
- 定点修法：把这三行改为省略注释（「… 省略：connector 分支三行（外部 KV 传输 partial-tail offload 钉住的块，归 Part IV 末章；无 connector 恒空）」），与本章 connector 分支的一贯标注法（L427-428）对齐；或 prose 一句明说。

### F5 [ADVISORY] [L2·跨章口径] add_local_computed_blocks 的省略注释只盖了 skip 段的一半——「跳段块先切出、不 touch」被静默省掉

- 位置：chapter.md L565-566（省略注释）与 L569 `touch(new_computed_blocks)`
- 问题：pin 源码 L262-265 在 touch **之前**先把跳段块从 new_computed_blocks 切掉（`new_computed_blocks = new_computed_blocks[num_skipped_blocks:]`）；ch13 六行账明确立过「命中块里落在跳段前缀内的那部分——它们不会被 touch（不离开自由队列）」（ch13 L575）。ch15 的省略注释只说了「SWA 窗外段以 null 块占位」，切片这半句没了，留下的 `touch(new_computed_blocks)` 读起来像「全部命中块都 touch」，与 ch13 口径表面打架。
- 违反经验：L2/L3（一笔的「此刻在哪」被省略注释盖住；跨章口径风险）
- 定点修法：省略注释补半句：「（跳段前缀内的命中块先从 new_computed_blocks 切出——不 touch、只以 null 占位，ch13 六行账的 skipped 支）」。

### F6 [ADVISORY] [L1·图内压缩歧义] f2-preempt-rehit 图收据行「主线 16（25%）· 最坏 48/64」易读成分数

- 位置：图 ch15-fig-f2-preempt-rehit 的 [1,P] 注记盒（SVG x=1120 y=457）；正文 L1144 与图注 L1148 本身是谨慎的
- 问题：48 来自支线 48-token 场景（补 48＝该 prompt 的 100%）、64 是主线 P（最坏应 64/64）；两个不同分母的例并置成「48/64」，读作「最坏 48 个/共 64」就是错的。同图最坏分支框自己写对了（「补 48 token（本例 prompt 48 时即 100%）」），仅此收据行有歧义。
- 违反经验：L1（图面数字表达与正文口径的可误读距离）
- 定点修法：illustrator 定点改该行为「最坏＝全量：支线场景 48、主线 P=64」（一行文字，布局不动）；图注/正文无需改。

### F7 [ADVISORY] [L1·图角标窄于图注] 两张图的面板角标比图注/manifest 少写一拍

- 位置：ch15-fig-phase1-miss-stop 角标「命中主循环「查」」vs 图注 L517/manifest 的「查 → 链上走」两拍；ch15-fig-partial-cow 角标「命中主循环「CoW 换尾」」vs 图注 L1332 的「CoW 换尾」与「拷贝过线」两拍
- 问题：两图内容都完整覆盖两拍（phase1 图含准入+链查；partial-cow 图含过线盒），只是角标少写一拍，属图面自标签与图注的轻微不同步（欠声明，非指错）。
- 违反经验：L1（图注与图面标签同步纪律的边角）
- 定点修法：两张图角标补全为「查 → 链上走」「CoW 换尾 · 拷贝过线」（或反过来把图注/manifest 的 l0_anchor 收窄），低优先。

### F8 [ADVISORY] [L3·小] touch 的 `and not block.is_null` 条件无注

- 位置：chapter.md L536（touch embed 内条件）；prose L542「两件事」句未提 null 豁免
- 问题：null 块 ref_cnt 不维护、永不在自由队列（ch13 池的出生图注立过「处处特判」），这正是这里的特判之一；条件裸露易引出「touch 会不会去摘 null」的小疑问。
- 违反经验：L3（缺席角色/null 特判回指）
- 定点修法：L542「两件事」句后补半句回指（「null 块被豁免——它从不在自由队列、ref_cnt 不记账，第 13 章立过的特判」）。

---

## 二、检查过、没问题的项（「过」清单）

### 逐图 L1（16/16 张全部亲眼 Read + SVG 坐标核方位；14 张全过）

- **L2-ch15 章图**：图注（L15）北行/中排/南行三段与 SVG 实测一致（北行 y≈165-326＝请求侧增量哈希/装配开关/平面哈希表/哈希粒度；中排 y≈398-590＝①-⑦ 命中主循环；南行 y≈724-1000＝抢占哈希保留/逆序+劈分/惰性驱逐 + 3 条 why 注 + 邻章分界）；「站号 1-2 算 · 3-5 查 · 6-9 挂/写/拷 · 10-12 留与逐」与图面读图行逐字一致。第 4 站徽标出现两次（中排 ② 链上走 + 北行平面哈希表框）系跨排交叉引用，非错误。
- **chained-hash**：左侧指纹链三行 + NONE_HASH 种子 + 尾段 2 token 虚框不入账 + 右栏两请求对照（✓✓✗）与断链实验盒（h1 从 3c3c50f324ae 变 c6bd6310bdcd）全对；「50 token→3 满块」「共享前 32」与 m1 表一致。
- **flat-hash-map**：左字节条（32 青 + 4 橙 big-endian）/中 dict 三段（①②③ 标在 dict 行上）/右 radix 澄清 + NOTE #1/#2，与图注左/中/右映射一致；块 3/5/7/9 图例注明例示。
- **phase1-miss-stop**：B 四行（块 1 ✓/块 2 ✓/块 3 ✗ break/块 4 连查不查）、C 63→48（红竖线「连带 15 + 要 logits 的 1」）、D 17→命中 16/重算 1，与 m4 表及正文一致；正文 L501 的表/图编号差 1 有明注（L501 末）。角标问题见 F7。
- **touch-refcount**：左六时点表（1→0→1→2→1→0）、右上 B/C 交叉共引快照（ref_cnt=2）、右中 O(1) 解剖、右下队列尾段 …3、4、2、1；「← 驱逐先来」箭头指向队头（左）与 reverse-free 图同款约定、底部图例「自由队列头=先驱逐端」兜底，无歧义到须修的程度；数字与 m5 表一致。
- **writeback-mask**：左登记账（块 0/1 入表 @16/@32、块 2 尾 8 不入表、条目数=满块数 2）、右 mask=[True,False]（0→1 只 +1）+ 幂等闸注，与 m6 表一致。
- **reverse-free**：左真实（…3、2、1，3 最靠驱逐端，取 1 块后命中 32/67%）/右反事实（…1、2、3，命中 0），与 m7 表一致；「outside of this class」引文在图。
- **split-free**：左劈分（1,3,4,5,6,7,2）/右关缓存全 append（4,5,6,7,3,2,1），#42656/#48017 落页脚，与 m8 表一致。
- **f2-preempt-rehit**：主线三状态（64/4→PREEMPTED 哈希保留 4→重命中 48/补 16/25%）+ 最坏分支（map 0、补 48）+ [1,P] 两头，与 m11 表一致（除 F6 一行）。
- **hash-granularity-view**：顶部标尺 @16/@32/@48/@64、三行视图（原始 h0-h3 / 32 取索引 1、3 / 64 取索引 3、重算 0），图注「第 2、4 枚」＝索引 1、3，与 m12 表一致。
- **partial-cow**：左两级探测（phase 1 @64 miss / phase 2 @48 命中、60%）、右换尾三段（共享 1、2 → cow3/cow5 各 ref_cnt=2、拷贝对拷贝自 1/2 → 原块留表）、右下过线盒（retained 4·copies 2）、左下对照（无 partial-hit 命中 0），与 m13 表一致；rev2 的 phase 2 起探点订正（自 @48 起 1 次即中）已落图。
- **hybrid-fixed-point**：左 7 行调用账本（95→80→48、重启行、第二轮 full 缺席）+ 右阶梯与收据（2 轮/5 次、80−48=32），与 m15 表一致；simple hybrid（场景 A 1 轮 2 次）在右栏注记补齐。
- **marconi-junction**：①产出（hit=0/uncached 48/boundary=48 + 对照归零）②写回（两路读者）③a 特赦（159→128、112→64）③b 停点（停 64、192/208），与 m16 表逐数一致。
- **call-panorama**：事件 0 拍外、①-⑦ 查（3 次查表、hit=32、预算 79）、⑧-⑰ allocate_slots 三段（touch [1,2]→新块 [5,6,7]→登记 [2,5)/map 4→7/块表 5 项）、拍尾采样后回 Request，与 m21 表逐项一致（除 F1 方位词）。
- **partial-to-full**：顶部两路流程带（super 幂等闸 / _cache_partial_tail_block 不受闸管）+ 四拍时间轴（48/50/63/64，闸 0>=0×3 与 0>=1 翻转，map 1→3 构成拆开数），与 m22 表逐格一致。
- 全部 alt 文本为结论式、无方位词，无 alt 失真。

### 账目段 L2（逐笔「此刻在哪」扫描——过，除 F3/F5）

- 一拍调度表（L26-33）：查/挂/写新块/写回四行的函数链与数字逐笔清楚；ch13 的卡点（命中块此刻未挂表、touch 在 allocate_slots 才发生）在本章由结构事实 + 全景图顺序（查①-⑦→挂⑧-⑪）+ `assert len(req_blocks)==0` 注释三重钉住。过。
- 指纹增量账（m1）、extra_keys 隔离（m3）、phase 1 查表账（m4，查表次数=命中块数+1 的论证在场）、touch 六时点（m5，冷 0→1 出队/热 1→2 不出队两态都在）、写回登记账（m6）、逆序（m7）、劈分（m8，25%/37.5% 标注累积口径）、惰性驱逐（m9，map 1→1→0）、F2 上下界（m11，[1,P] 两头原因给全）、CoW 四步（m13，每组 1 块×2 组、retained 4=copies 2 两端、ref_cnt=2=请求+保留拆开）、move_block_hashes（m10，活重重指 map 不变）、不动点（m15，逐 finder 入/出 + full 缺席原因）、junction 三件套（m16，各停点行输入/输出齐全）、retention 三态（m17，87.5%/50% 算术自洽）。全过。
- CoW 预算 +1 与「只换不增/又换又增」分岔：ch15 embed（L1267-1290）与 ch13 拍 1/拍 4 同源同账。过（锚差一行见下）。

### 例子覆盖 L3（过，除 F4/F8）

全冷/全热（m5 C 进场=热态）、缓存开关（m8 右面板 + 装配开关码）、retention 三态（m17 全点亮）、SWA/Mamba 角色（m23 分叉实拍 + [NULL,…] 形态回指 ch14）、extra_keys 三源（m3 salt/LoRA/mm 全例）、uniform vs hybrid 对照（m23 行 1）、junction=0 无副作用对照（m16 行 1）。缺席角色均有明说：eagle/DCP「可省略项」（L483）、connector 三处「归 Part IV 末章」（L427、L1417、L1656）、prompt logprobs/pooling「命中恒零」（L267）、reset_prefix_cache（L1096）。

### 不变量 L4（全过）

- 命中与运行中互斥：点破（L41 结构事实一 + L438「为什么中途不查」+ 源码注释原话 "running requests are short-circuited there"）。
- 链式哈希失效传播：断链实验（m1 + 图）+「miss 后必 miss 是数学推论不是启发式」论证（L501）+ 图面不变量带。
- partial 命中的块边界：进阶一开篇三刀 + 触发判定 + 对照组（无 partial-hit 命中 0）。
- ref_cnt>0 绝不在自由队列 / +1−1 两对镜像：m5 不变量带 + L604。
- 无哈希块永不可能命中：L1002 论证 + 图。
- 幂等闸与进度账的配对、partial→full 晋升「是常态不是特例」：partial-to-full 节整节点破。
- junction=0 无副作用：m16 对照行 + 图不变量带。

### ch13 前向承诺兑现（3/3 过）

- **承诺 a**（ch13 L443「前缀命中块 new_computed_blocks…来源是 get_cached_block 按内容哈希查池（block_pool.py:L198），哈希什么时候算、怎么查，ch15 开门见山」）：兑现。`new_computed_blocks` 名字贯穿（L431 调度器解包、L624-629 allocate_new_computed_blocks 实参）；get_cached_block 在全景表锚 block_pool.py:L198-L217（L29）、phase 1 embed 调用行（L493）与 phase1 图逐字锚；哈希两时机（构造首算 L758-765 + 每拍续算 L769-783）与查法（「查」节整节）齐。
- **承诺 b**（「命中的由 touch 复用别人的、ref_cnt 0→1」）：兑现且同口径。ch15 L523 显式回指 ch13 伏笔原话；touch 码（L536-538）+1 与 ref_cnt==0 出队；m5 冷态 0→1 出队、热态 1→2（=ch13 全冷/全热两态）；L604「touch 挂命中块、get_new_blocks 挂新块」与 ch13「命中的不走 get_new_blocks」同账。
- **承诺 c**（CoW 尾巴真实时机，两章口径一致）：兑现且无冲突。触发条件（partial=命中长度非块整数倍；单组 hash_bs==bs 永不触发→ch13 L979 / ch15 进阶一）、预算 +1（ch13 拍 1 / ch15 L1317）、_apply_cow 三动作与保留引用（同码同账）、步序栅栏 scheduler.py:L1181-L1190（两章 embed 逐字同）、worker 先清零后拷贝（ch13 拍 6 / ch15 L1373-1384）——逐一对照一致。ch15 另讲的 move_block_hashes 过户时序（L1409「allocate_new_blocks 里 L1622/L1624 同步做完」）亲核 pin 源码属实（MambaManager 分配路径 L1616-1624，注释即「running 请求块表 append-only，条目反向重指」）；enable_partial_hash_hits 适用面（L1417）亲核属实（kv_cache_coordinator.py:L581-L588 + SlidingWindowManager assert L914-916）。小注：预算 +1 的锚 ch13 写 L226-L229、ch15 写 L226-L230（多含 return 行），口径无冲突，顺手可对齐为 L226-L229。

### 数字口径跨章一致（过）

- block_size=16：ch15 单组主线场景全程 16（进阶场景的 64/32 块均就地声明配置）。
- touch ref_cnt 0→1：两章同口径。
- ch13 E2 数字（100/cdiv7/命中48=3块/return7）：ch15 未复用该例；同型推导自洽——B 80/cdiv5/命中32=2块/+3新块=5 项块表（m21）、F2 64/重命中48=3块/补16（m11），算术全对。
- 其余共享数字（O(1) 摘除、LRU 尾、#42656/#48017、free 不清哈希表述）两章一致。

---

## 三、修法汇总（全部定点小修，无整章重写类建议）

| # | 级别 | 落点 | 动作量 |
|---|---|---|---|
| F1 | BLOCKING | chapter.md L39 | 改 1 个方位词 |
| F2 | BLOCKING | chapter.md L341 | 改 2 个方位锚（左→左/右上/下 三区） |
| F3 | ADVISORY | chapter.md L655 附近 | 加 1 个括注句 |
| F4 | ADVISORY | chapter.md L930-932 | 3 行改省略注释或加 1 句 |
| F5 | ADVISORY | chapter.md L565-566 | 省略注释补半句 |
| F6 | ADVISORY | 图 ch15-fig-f2-preempt-rehit | illustrator 改 1 行文字（经 workflow，非 writer 改图） |
| F7 | ADVISORY | 图 phase1-miss-stop / partial-cow 角标 | illustrator 补角标（低优先） |
| F8 | ADVISORY | chapter.md L542 | 补半句回指 |
