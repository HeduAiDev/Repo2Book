# ch14「怎么切」领地第三轮重写——定向评审报告（r3）

- **评审对象**：`narrative/chapter.md` L513「## 怎么切」至 L1476（「## 门多紧」之前），共 964 行；对照前两轮已 APPROVED 的 r2 稿（commit 507ee7b8，同领地 622 行）。
- **一级镜头**：叙事结构审（零复用残留 / 冷开场铺垫 / 理论覆盖率 / 类比恰当性）；验收标准 = 用户四条批评（①逻辑通顺零兼容旧结构 ②理论罩住后半 ③架构图体现新模型适配面 ④数值讲清）。
- **日期**：2026-09-22 ｜ **评审者**：reviewer ｜ r3 重写 commit：c0871424（+未提交收尾改动）。

## 一、客观检查（本轮独立复跑，解释器 `python`；注意本机 `python3` 是 Windows Store 哑桩、exit 49 无输出，勿用）

| 检查 | 结果 |
|---|---|
| lint_fidelity / lint_source_grounding / lint_chapter_structure / lint_formulas / lint_trace_consistency | 全 PASS |
| lint_chapter_map --require / lint_anchors --all / lint_punct --all / lint_diagram_geometry --all / lint_diagram_scaffolding --all | 全 PASS（裸章号 232 处为全书既有机械告警，按契约全书只记此一条） |
| lint_dossier / lint_explainer | PASS（warn：paper_origin 缺——机械告警非正文问题，r2 已登记；warn：adaptation-ladder manual trace 未经运行验证——该表为 pin 锚定的策展表，性质上不可跑，如实） |
| lint_diagrams | **1 BLOCKING**：`ch14-fig-decl-vs-physical` figure-spec（explainer mechanisms[26]）有 spec、manifest 未登记（spec 自注「Lead 已选 D 方案：留 spec 不出图」）。两张新图盲审 verdict 由本轮开场时的 PENDING 在评审期间变为 **PASS**（2026-09-22 独立盲审，只看 PNG+spec，含全分辨率裁片；中途的 `.blind_*.png` 裁片已清理，孤儿告警消失） |
| 逐字保真抽查 | 四路分发块（kv_cache_utils.py:L1794-L1819）与压缩器形状块（compressor.py:L173-L189）对 pin 逐行比对：逐字一致（仅多 `# Lnnn` 行号注）。机械面 lint_fidelity 绿 + r2 全量 43 块基线 |
| 绝对论断核验 | 「V4 三种小页又都不是 stride 索引层」：pin 实查 `indexes_kv_by_block_stride` 默认 False（kv_cache_interface.py:190），全书仅 inkling sconv 两处覆写，V4 族零覆写——论断成立；r2 的 B2 修复措辞（「分路本身不读任何模型名」）与内嵌代码不再打架，仍完好 |
| 数值复算 | 全部通过：576 对齐四笔（37376→37440=65×576、1168→1728、8448→8640、32768→32832=57×576）；形状因子 2×512×2×4=8192 / 2×512×1×4=4096、4×2=8×1 抵消；MLA 组页和 30×37440+30×8640+31×1728=1435968；10737418240//1435968=7477（余 685504）；243=167×1+31×2+3×3+1×5（202 桶、35 桶跨组相撞）；G0..G4 新开 offset 91+27+0+57+27=202 自洽；近似 GCD 玩具 pad(2)=pad(3)=1 平局取大 d=3 与 pin tie-break 一致；census 85=11×5+10×3、页宽计数 32/22/21/10；失败演示 33=cdiv(511,16)+1、223=3568//16、14,614,528 B；m18 收益合成回指 257/513/1.9961 与 m15、33 与失败演示一致 |
| 图（10 张全部亲读 PNG） | 两张新图（hybrid-theory-model-v2、adaptation-map）+ 被引用旧图 8 张（kvcache-init-chain、packed-slicing、v4-layer-anatomy、v4-census、v4-compress-accounting、hybrid-groups、tensor-sharing、grouping-lineage）：图面↔图注↔正文数字三方一致。含 r2 N4 修复核验（init-chain 琥珀闸门线图注已改「收集段之后」，线在收集段容器底，与图面一致）；packed-slicing 角注 1435968/202/7477 在图上；census 右条 32/22/21/10 在图上 |
| 跨章缝合 | ch15 对 ch14 的全部提法（「怎么切」一节、hash_block_size=GCD/prefix_match_unit、站 4 立 spec、12+13 实跑、四种页宽 37440/32832/8640/1728、unify 走读过、null 形态、假设第六条）与新节名/新内容逐一吻合；MLA 全书首现确在 ch14（ch10/12/13 零命中，见 R3-2） |
| 章内指针解析 | 全章「」引用节名的 promise 全量扫描：全部解析到现存节名；唯一内容漂移 = R3-1（L405）。门多紧侧「回收感知准入上限」L1659、「SWA 的还账方式」L1562 均在，L469「收益合成」在 L1734 兑现且回指「怎么切」新名 |

## 二、一级镜头：叙事结构四查

### 1. 零复用残留 —— PASS（领地内），接缝处 1 条定点缺陷（R3-1）

结构重组是真的，不是旧稿换标题：r2 顺序（类型→前史→两难→分桶→页统一→V4 三节→第三路→张量共享→尺子）→ r3 顺序（类型→**先立模型**→**spec 申报面+全链路导览**→迭代史→失败演示→**分发面（四路前置 + 分桶 + 页统一 + 接入面）**→**压力测试（四小节伞）**→落地张量→尺子）。量化：领地 622→964 行，实质行 45% 新写；最有结构意义的一步是**四路分发代码从 r2 的 V4 第三路段落尾部前移到分发面开头**——r2 里读者先学 分桶/页统一、直到后半才第一次见到它们所实现的分路代码，r3 修正了这个倒挂。逐节核对承接：每节开头/结尾都有新建桥（或显式回指哪条推论落地，或承接上节末句），无概念跳跃、无重复讲解（复现的 MambaSpec 页性质、UniformTypeKVCacheSpecs 等均是以新用途为框架的有意回收）。四节点名检查：迭代史（新增理论标签×4、V0 缺公理二本身的新框架）、失败演示（重写为 P1 式两遍实验，实验设置回指公理规则名，数字与下一节 m5 共用）、分发面（开头回指 init-chain 导览图第⑤步——与图面站⑤标注一致）、落地张量账（新增四段式开合 + 202 offset 逐组构成 + 三例跨组相撞）——全部为本轮重新组织。

唯一残留：**L405（非重构区、站 1-7 旧文）的前向指针**「真分配发生在后文『worker 侧落地』一节」——r3 把真分配（`_allocate_kv_cache_tensors`，Allocate once）搬进了「落地为物理张量」（L1316-1338），「worker 侧落地」（L1808）只剩块表/kernel 块细分，指针指错一站。见 R3-1。

### 2. 冷开场铺垫 —— PASS（1 条首现缺口 R3-2）

全领地函数/机制/概念首现逐一向前扫描：spec 收集（站 4 前铺 ✓）、UniformTypeKVCacheSpecs（迭代史先于分发面 ✓）、KVCacheGroupSpec/docstring（就地立 ✓）、SWA 双向交接（L459「下一节怎么切完整展开」↔ L521「站 7 只借了它有封顶这一个事实，这里正式介绍」✓✓）、EAGLE/MTP/HMA/c4/c128/c1/元组/四种页宽裸数字（r2 B3 三处修复在位 ✓）、管家/get_num_skipped_tokens（就地定义+前指装配节 ✓）、watermark/dcp/num_computed_tokens/prefix_match_unit/max_in_flight_tokens（就地 gloss ✓）、CSA/HCA/indexer/mHC/MLA/压缩三股流（L911-913 集中立 ✓）。分发面开头回指 init-chain 导览图站号：**在**（「第⑤步写着四路分发定账」）。缺口：MLA 首现 L607 无展开（定义在 306 行后的 L913，且全书前章零铺垫）；census 一词 L575/L905 首用时无 gloss（payoff 在 L1015-1025）。两条，低于判级闸 ≥3 阈值，整体不升 blocking。

### 3. 理论覆盖率 —— PASS

两公理四推论真罩住了后半每一节：失败演示→①（开头即「推论①说……这不是设计品味，可以当场检验」）、分桶等量→「第四路先兑现推论①，再兑现推论②」、页统一→「这就是推论②在第四路里的调度实现，三条出路正好对应理论里的三种情形」、接入面→「推论④在这里兑现」、装包→「推论③在分组侧的落地现场（布局侧归落地为物理张量）」、落地张量→「Packed 重叠就是推论③的物理形态」+收尾「从理论模型里的一句话长成了完整的代码形态」、两把尺子→「推论②那句块在组间自由流动在调度侧的余波」、V4→「四条推论 V4 全撞」。用户点名的四件全部在理论层预答：**对齐为什么**=推论②（自由队列不认组⇒面积必须全池统一，整除调 t/垫/死三条路+96/80 玩具数）、**pack 为什么**=推论③（跨型不整除⇒唯一无损解=重叠时间复用）、**等宽约束**=推论②同一论证（分桶节再推导一遍「组里 N 层，一个块号兑现 N 页」）、**适配程度**=推论④（「申报的离谱程度决定适配深度」，四档+边界判据）。L533 四道自测题与四条推论一一对应、各有兑现节。未发现任何节在高码环境裸奔：压缩记账的「管理面/物理面」之问由公理一的「（以及压缩）」+紧邻第三环换算尺承接，且整个 V4 节在「四条全撞」伞下。轻微松口一条（R3-3）：L597「（『压力测试』逐档点验）」的逐档清单实际散在适配面表+图注+纪元四，压力测试节内无成串点验。

### 4. 类比恰当性 —— PASS

水表已删（全章 grep 零命中）。余下比喻逐个验映射面：**货币**（块号↔货币、面积↔购买力，公理二→「撕成几份」→「对它根本不流通」全程一致）、**申报表**（谁交表/交白卷/账记邻居名下，一致）、**管家**（manager，一致）、**钥匙/房间面积**（块表条目↔物理字节，与 ch13 血统和 tensor-sharing 图注一致）、**压力测试**（V4，与「真实在产的旗舰就是这种量级」一致）。全部映射闭合、无一承载超出映射面的论证负担——核心论证确为失败演示+数字（P1/P2 兑现）。

## 三、用户四条验收标准

| 标准 | 判定 | 依据 |
|---|---|---|
| ① 逻辑通顺（零兼容旧结构） | **过**（领地内） | 二.1；唯一旧结构残留 = L405 接缝指针（R3-1，一行修） |
| ② 理论罩住后半 | **过** | 二.3：四推论逐节显式落地标签，四件预答全在理论层 |
| ③ 架构图体现新模型适配面 | **过** | adaptation-map 亲读：生命周期八站横轴 × 上行模型侧（四档色深：零档全绿/RSWA 13 行/SlidingWindowManager/自研后端）× 下行框架侧，右侧适配深度阶梯 + DSV4 徽标（四个申报面+复用 SlidingWindowManager+三个自研后端）——真回答「新模型要适配什么」，且与理论图推论④互指；theory-model-v2 亲读：两公理卡/四推论卡各带落地标签/底带判定链/t₁→A、t₂→C 时间互斥示意，与图注、正文三方一致。P7（社区无人画过、自行合成）兑现 |
| ④ 数值讲清 | **过** | [4,2*512*2*4] 五因子逐个有 pin 锚有含义（槽数/两份子状态/头宽 448+64/重叠系数/fp32 字节）+每槽/块/对齐三行乘法；202→1 四段式有开（L1207 定义）有表（243:202:1 直方图+逐组新开 offset+三例相撞偏移）有合（L1372「243 条层账→202 声明桶→1 次物理分配→243 视图」）；全章常数来历逐个在位（584=448+128+8、576 UE8M0、132=128+4、256=后端偏好、243=30×5+31×3、1435968 组页和、7477=整除取整、2.6%=37440/1435968），本轮全部手工复算通过。P5/P6 兑现且超出社区最深（A4） |

## 四、机制对账

lint_trace_consistency 绿（正文数值表↔explainer trace 逐格一致）。r2 的 12 机制勾选表在 r3 领地重排后仍全数在场（本轮逐一确认其正文载体未丢失：m5/m6/m7/m8/spec-collection/init-call-tree/packed-layout-semantics/v4-cache-spec-census/v4-cache-compress-write-path/v4-cache-uniform-groups-verify/packed-view-slicing/v4-61layer-config + 新增 tensor-declaration-account、adaptation-ladder）。本轮不再重印全表，抽验记录：tensor-declaration-account 的 202 逐组构成与三例相撞偏移为 r3 新增、trace 可溯（tensor_decl_account.json）；adaptation-ladder 为 pin 锚定策展表（explainer warn 已如实标注）。

## 五、Issues

### R3-1（blocking，writer，一行修）——旧结构接缝残留：前向指针指错节
- **dimension**: 连贯/叙事结构（用户标准①）
- **problem**: L405「缩的对象 `kv_cache_tensors` 在这首次露面……真分配发生在后文**『worker 侧落地』**一节」——r3 已把真分配（`_allocate_kv_cache_tensors`、Allocate once、packed backing）整段搬进「落地为物理张量：四段式张量账」（L1316-1338），「worker 侧落地」（L1808）现在只讲每组一张块表与 kernel 块细分。读者按指针翻到 L1808 找不到承诺的内容（那里的真分配只有一句带过），实际内容在 L1205+。这是站 1-7 保留旧文指向被重构领地的唯一未更新处——正是「零兼容旧结构」要抓的接缝。
- **suggested_fix**: L405 改「真分配发生在后文『落地为物理张量：四段式张量账』一节」（如愿意可再补半句「worker 侧的块表换算见『worker 侧落地』」）。
- **rationale**: promise-payoff 是本轮叙事重建设计的承重结构；错一站即断一环。
- **evidence**: chapter.md L405 vs L1316-1338（分配循环在落地张量节内嵌）vs L1808-1830（worker 侧落地开头自述「最后一块拼图……每组一张块表、大块拆小块」）；git diff 507ee7b8→c0871424 显示分配代码为本轮移入。
- **negotiable**: false ｜ **blocking**: true

### R3-2（negotiable，writer，两句级）——MLA/census 首现早于定义
- **dimension**: reader-comprehension（未达 ≥3 聚集阈值，不升 blocking）
- **problem**: (a) 「主 MLA 的 656 B/token」首现 L607（纪元二），展开定义在 L913，间隔 306 行；ch10/12/13 全零铺垫，全书首现即在此。(b) census 首现 L575/L905，词义（逐层户口清点）到 L1015-1025 才兑现。
- **suggested_fix**: (a) L607 就地加「主 MLA（multi-head latent attention，多头潜在注意力，数学归第 24 章）」；(b) L575 或 L905 加半句「census（逐层户口清点：数每层报几本账）」。
- **rationale**: r2 B3 同型问题（c4/c128/元组/页宽裸数字）已修，MLA 是同类漏网；两句级修复。
- **evidence**: grep 首现序 MLA→L607 vs 定义 L913；census→L575/L905 vs L1015；ch10/12/13 grep MLA 零命中
- **negotiable**: true ｜ **blocking**: false

### R3-3（negotiable，writer，一句级）——「压力测试逐档点验」承诺略超交付
- **dimension**: promise-payoff 一致
- **problem**: L597「V4 四档全占（『压力测试』逐档点验）」——逐档素材齐（适配面表 pin 实例列、adaptation-map 图注「四个申报面+复用 SlidingWindowManager+三个自研后端」、纪元四「走到第四档」）但压力测试节内无成串的 ①→④ 点验，读者按括注去找会拿到散件。
- **suggested_fix**: 二选一：括注改「（『协议的分发面』末的适配面小节逐档列出；V4 的兑现散见四小节）」；或在压力测试「边界三句话收束」处补一句四档归位。
- **rationale**: 本轮叙事的卖点就是 promise 精确兑现；这条是仅存的松口。
- **evidence**: chapter.md L597 vs L903-1203（无逐档清单）；L845-850 表、L899 图注
- **negotiable**: true ｜ **blocking**: false

### R3-4（blocking@图门禁，指派 illustrator/explainer，不让 writer 改）——decl-vs-physical spec 未登记
- **dimension**: figure-integration（lint_diagrams 1 BLOCKING）
- **problem**: explainer.json mechanisms[26].figure_specs[0] 有 `ch14-fig-decl-vs-physical` 完整 spec（spec 自注：Lead 已选 D 方案——留 spec、dossier 不升平 figure=false），但 manifest 无任何登记，lint_diagrams 阻断。正文从不引用该图（领地图文自洽），纯账面缺口。
- **suggested_fix**: manifest 补一条 skip/deferred 登记（或若 linter 不支持跳过语义，由 explainer 移除该 spec 并在 note 保留决策记录）。
- **rationale**: 图门禁红灯不因「图不存在」而豁免；登记语义要能表达「有意不出图」。
- **evidence**: lint_diagrams 输出；explainer.json mechanisms[26].figure_specs[0].note
- **negotiable**: true（修法二选一） ｜ **blocking**: true（作为图门禁项；归属 illustrator/explainer）

### R3-5（negotiable，archivist/收尾）——run-ledger 未记三轮重写
- **dimension**: 经验回流台账
- **problem**: reviews/run-ledger.json 止于 2026-08-29 原始流水线（write_review_rounds:2、blind_rounds:1）；此后三轮领地重写（含本轮 r3 评审与两张新图的独立盲审）均未入账。
- **suggested_fix**: 定稿收尾时补记 r1/r2/r3 轮次、盲审史（两张新图 2026-09-22 独立盲审 PASS）与用户裁决节点。
- **evidence**: run-ledger.json review_completed_at=2026-08-29 vs c0871424/507ee7b8 两个重写 commit + 本轮
- **negotiable**: true ｜ **blocking**: false

### R3-6（negotiable，Lead/archivist 裁定）——伏笔 f12 的 plant 在正文缺席而台账记 resolved
- **dimension**: bible 伏笔对账
- **problem**: bible.py due ch14 列 f11/f12 两条应埋。f11 在位（L1766「被抢者全部块释放、num_computed_tokens 归零、回队头」+ch15 多处钩子）；f12（skipped_waiting/WAITING_FOR_REMOTE_KVS → ch35）全章 grep 零命中，而 arc-map 标 `status:"resolved", resolved_in:"ch35"`——ch35 尚未写，resolved 状态系预填，plant 疑在历轮重写中丢失。
- **suggested_fix**: Lead 裁定：或回补一句 plant（自然落点在门多紧「请求留在 WAITING」附近，一句「等远程 KV 的请求另有隔离队列，PD 篇拆」级即可），或修台账状态。
- **evidence**: bible.py due ch14 输出；arc-map.json f12 条目；chapter.md grep skipped_waiting/WAITING_FOR_REMOTE/远程 KV 零命中（仅 get_num_skipped_tokens 同形词）
- **negotiable**: true ｜ **blocking**: false

## 六、判定

**REVISE（定点小修，不退整章）**：

- 一级镜头四查全过、用户四条验收标准全过——叙事结构本体放行。r2 两轮 APPROVED 却被用户判「不通顺」的教训在本轮没有复发：领地是真实重构（四路前置、理论伞、四段式开合、每节显式推论钩），不是旧稿打补丁。
- 阻断项两条、各一行级：**R3-1**（writer，L405 指针改节名）+ **R3-4**（illustrator/explainer，manifest 补登记）。两条落地后本章图门禁与叙事门禁均绿，即达 APPROVED。
- negotiable 四条（R3-2/R3-3/R3-5/R3-6）建议随本轮一并处理，不单独立门。

**去重说明**：与 r2 报告（review-report.json）无重复 issue；r2 的 B1/B2/B3/N1-N9 修复经本轮逐一复核仍完好。lint_dossier paper_origin 与 lint_anchors 裸章号两条机械告警按契约全书各只记一条（沿用 r2 登记）。reader-comprehension 本轮 2 条（R3-2 内两子项），未达 exp-2026-07-18-01 的 ≥3 聚集线，不触发强制回环。

**升级说明**：无 >3 轮复发项。R3-1 属本轮首报。
