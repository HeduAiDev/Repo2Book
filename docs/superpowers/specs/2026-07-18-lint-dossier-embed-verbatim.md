# SDD 简报：lint_dossier 新增 embed_verbatim 确定性检查

> 来源：exp-2026-07-18-02（book-retro 2026-07-18，triton 主书收官批次）。
> 性质：任务简报——**不直接改代码**，由 Lead 派 TDD 小任务落地（先写测试再实现）。
> 落点：`scripts/lint_dossier.py`（新增检查项）+ `scripts/tests/test_lint_dossier_embed.py`（新建）。

## 1. 问题（root cause）

analyst 给打印/格式化类等**易随版本漂移**的函数写 `embed_excerpts` 时，凭训练记忆**默写旧版本
代码**，而非逐字对照当前 pin commit 的 blob。已知 4 次样本（ch13/ch27/ch35/ch41，ch41
run-ledger 显式列举），全部靠 dossier 对抗性自核（LLM）救场——概率性兜底，无确定性闸门。
典型：ch41 m5 线程号编码方式抄了旧版 triton（per-warp vs 全局）。

## 2. 规则描述

对 `dossier.json` 的每个 `embed_excerpts` 条目（结构 `{path, lines, code, elide}`，实测全书
557 条均此四键）：按声明的 `path` 与行号区间 `lines`（格式 `Lnnn[-Lnnn]`）从**当前 pin 的
实际 blob** 取内容，与 `code` 字段做**空白归一后的逐字比对**——不一致即 **blocking**
（检查项名 `embed_verbatim`，输出归入现有 `res` 字典新键，报告风格与既有 `anchor` 类一致：
`{mid/序号}: {path}:{lines} 第 N 行与 pin 不符：dossier=…  pin=…`）。

**定级：blocking**。理由：embed_excerpts 是「正文要内嵌的真实源码」的唯一真相源（三支柱 A），
默写旧版直接违反 HARD RULE 2/3 的根基；且该检查是纯机械比对，误报可控（见 §4 上线纪律）。

## 3. 实现要点

### 3.1 pin blob 的定位方式

- 源根复用现有 `_source_root(chapter_dir)`（向上找 `artifacts` 的父目录 + `/source`）。
- **优先 `git -C <source> show HEAD:<path>` 取内容**，工作区读文件作 fallback：pin 仓是
  blobless clone、tag 检出（triton 实例 HEAD=v3.2.0），`git show` 对「工作区被临时改脏」免疫；
  git 不可用/路径不在 git（如实验性快照）时回退 `(src/<path>).read_text()`。
  两者都取不到 → 报 `embed_verbatim: 文件不存在/无法读取`（blocking——现状 embed_excerpts
  的 path 根本没有存在性检查，此项顺带补上）。
- **external-source 前缀（前瞻 primer，exp-0711-2 同源问题）**：path 带
  `../book/external-source/` 类前缀的条目，先按 `lint_chapter_structure._norm_anchor_path`
  同款逻辑归一/解析；解析到 external-source 快照目录则从该快照读，**不**从 pin 仓读。
  解析不到的一律按普通 path 处理。

### 3.2 归一化（空白归一的确切定义）

逐行比对，每行做：`expandtabs()` → `rstrip()` → 行内连续空白折叠为单空格。
**不做** lstrip/缩进折叠——Python/MLIR 缩进承载语义，缩进错=默写错，必须抓。
（行内折叠是为容忍对齐性空格差；如实测产生误报可收紧为仅 rstrip，以 oracle 对表为准。）

### 3.3 省略/注记的处理（关键：34% 条目非全量内嵌）

全书实测：557 条中 **190 条 `code` 行数 ≠ 声明区间行数**——analyst 在区间内做了抽行省略
（如 ch02 声明 57 行只嵌 15 行），省略说明写在 `elide` 字段（prose），`code` 内**目前零例**
内联省略标注。因此比对分两种模式：

- **全量模式**（code 行数 == 区间行数）：逐行严格比对（归一后），任一行不符 → blocking。
- **子集模式**（行数不符）：`code` 的每一行须能在 pin 区间内**按序**匹配到
  （有序子序列匹配，双指针即可）；匹配不到的第一行即报 blocking（附 dossier 行内容 +
  pin 区间内最相近行，便于定位是默写还是行号错）。乱序匹配（全部行都在但顺序颠倒）同报。
- **注记行豁免**：`code` 内若出现 `# … 省略 …`/`// … 省略`/`# SOURCE:`/`# PAPER:` 打头的
  标注行（当前 0 例，但正文惯例存在、analyst 未来可能内联），跳过不参与匹配。
- **空行**：归一后为空的行不作为子集模式的匹配锚（避免空行到处能配上造成假阴）。

### 3.4 适用范围与豁免

- 仅对 `path` 匹配现有 `ANCHOR` 文件模式且 `lines` 合法的条目做比对；primer 章 embed_excerpts
  可含**论文公式条目**（§/Eq 锚、无文件 path）——不匹配文件模式的条目跳过（不警告，属合法形态）。
- `src is None`（实例 source/ 不在）沿用现状：降级 warn「跳过 embed_verbatim」。

### 3.5 与既有检查的关系

- 现有 `source_anchors` 检查只核 **mechanisms 的锚点**（格式/文件存在/行号越界），完全不碰
  embed_excerpts；本检查补的是 embed_excerpts 的**内容逐字性**，两者互补不重叠。
- `lint_fidelity` 的 elision_gap/non_adjacent_splice（exp-0705-14）核的是**正文** vs dossier
  的省略标注一致性——它信任 dossier 为真相源；本检查补的正是「真相源自身 vs pin」这一环。
  三层闭环：pin ↔ dossier（本检查，新）→ dossier ↔ 正文（lint_fidelity，已有）。

## 4. 上线纪律（exp-0713-3：自造检测器必须与 oracle 对表）

落地顺序：实现 → 对 triton 全 43 章 + vllm/vllm-ascend 存量跑一遍 → **漏报=0、假阳逐条解释**
后才设 blocking；若存量假阳集中在某形态（如行内空白折叠不够），先修归一化再升级。
预期效果指标（台账）：analyst embed 非 pin 逐字被 dossier 自核救场次数 → 0（现 4 次）。

## 5. 测试用例草案（`scripts/tests/test_lint_dossier_embed.py`，pytest，fixture 用 tmp_path 造迷你 source+dossier）

1. **正例·全量**：code 与 pin 区间逐字一致（含 tab/行尾空格差异）→ 0 blocking。
2. **负例·默写旧版**：code 中一行与 pin 同位行差一个标识符（模拟旧版函数体）→ blocking，
   报文含该行 dossier/pin 两侧内容。
3. **正例·省略子集**：声明 L1-L20、code 只含其中 6 行（按序）→ 0 blocking。
4. **负例·子集含杜撰行**：省略子集中混入一行 pin 区间内不存在的代码 → blocking。
5. **负例·乱序**：code 行全在区间内但顺序颠倒 → blocking。
6. **豁免·注记行**：code 含 `# … 省略 …` 行 → 该行跳过、其余照常比对。
7. **豁免·primer 论文条目**：无文件 path 的 §/Eq 条目 → 跳过、无告警。
8. **边界**：path 不存在 → blocking「文件不存在」；source/ 缺失 → warn 跳过。
9. **git show 路径**：工作区文件被改脏而 HEAD blob 正确 → 仍 0 blocking（验证优先走 git show）。

## 6. 配套契约（已由 curator 另行落笔？——**否，留给 TDD 任务一并做**）

approved patch 还含 analyst.md 自检清单补一句硬检查：「本片段是否逐字取自 pin blob/diff，
而非训练记忆中的旧版本」。本条**未**随本批契约落笔（Lead 批准清单未列 analyst.md），
建议 TDD 落地 linter 时一并加（linter 上线后该句可直接引用检查项名）。

---

## 7. 落地记录(2026-07-18,phase-1 已上线)

TDD 完成:`scripts/tests/test_lint_dossier_embed.py` 11 用例 + 既有 17 回归全绿。
oracle 对表(118 章全corpus)修正了 §3 的三个假设偏差:

1. **「code 内零例内联省略标注」不成立**——裸 `...`/`# ...`/`// ...` 整行、行尾截断
   `…text ...`、行中压缩 `raise NameError(...)` 三形态大量存在 → 实现为**省略感知匹配**
   (纯省略行跳过、截断行按省略号前缀匹配)。
2. **统一 dedent 普遍存在**(ch27 .td 8 例:analyst 把嵌套代码整体去缩进后内嵌) →
   两侧各自去公共缩进再比,相对缩进仍严格。
3. **`\uXXXX` ASCII 转写**(ch23:`⊕` 抄成 `⊕`) → 归一化还原为字符。

**分级落点(与 §2 原案的差异)**:
- blocking:全量模式行不符、行号越界(oracle 后假阳=0,残余 10 例全为真实区间错;
  triton 3 例已修,vllm ch03×3 / ascend ch10·ch13·ch21·ch31 各 1 例遗留待修)。
- warn:path 不在 pin(31 例全为前瞻 primer 上游码/vllm-ascend 跨仓引 vllm/*/论文包路径——
  多真相源合法形态,原案「不存在即 blocking」不成立);子集模式不匹配(117 例混杂
  改行宽重拼/重复行贪心错配/列表重排等转录形态,假阳未清零,按 §4 纪律暂不升级)。
- 升级条件:下一本书新章的子集 warn 若能稳定人核清零,再评估升 blocking。
