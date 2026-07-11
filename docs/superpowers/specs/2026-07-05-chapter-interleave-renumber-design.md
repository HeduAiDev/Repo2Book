# 章节插入体系:交错重编号 + 补章 SOP + 首创期预留

日期:2026-07-05
状态:待用户评审
动机:Part VIII 六章原理篇内容是前置知识却坐在书尾像附录;指路框只是补丁。用户三条裁决:① 物理交错重编号、导语按真实阅读流重写;② **补章流程规范化为工厂能力**(插入是常规操作,不是手术);③ **预期前置到首次创作**——新书从第一天起让"日后插章"接近零迁移成本。
交付两层:**通用能力**(重编号引擎/补章 SOP/引用规范+lint)+ **首次应用**(ascend 书本次交错)。

---

## 1. 新章序(唯一真相:下表;36 章,P8 取消、原理章并入所属 Part)

| 新号 | 原号 | 内容 | Part |
|---|---|---|---|
| ch01-ch08 | 不变 | 接入/设备/并行前半 | P1-P3 |
| **ch09** | ch34 | **EPLB 算法本体(原理)** | P3 |
| ch10-ch20 | ch09-ch19 | 顺移(EPLB 工程→…→MHA) | P3-P5 |
| **ch21** | ch31 | **MLA 原理** | P5 |
| ch22 | ch20 | MLA 落地(NPU) | P5 |
| **ch23** | ch32 | **稀疏注意力谱系 NSA→DSA(原理)** | P5 |
| ch24 | ch21 | SFA/DSA 落地 | P5 |
| ch25 | ch22 | KV manager | P5 |
| **ch26** | ch36 | **V4 CSA/HCA(原理,Part V 收束)** | P5 |
| ch27-ch30 | ch23-ch26 | 顺移(算子/编译) | P6 |
| **ch31** | ch35 | **量化数学(原理)** | P7 |
| ch32-ch33 | ch27-ch28 | 顺移(量化框架/采样) | P7 |
| **ch34** | ch33 | **投机采样原理** | P7 |
| ch35-ch36 | ch29-ch30 | 顺移(投机管线/模型注册) | P7 |

目录名规则:仅换 `chNN` 前缀,slug 主体不变(`ch34-primer-eplb → ch09-primer-eplb`)。

## 2. 迁移引擎(通用工具 `scripts/renumber_chapters.py`,TDD)

**不做一次性脚本**——做通用重映射引擎,本次交错只是它的第一份输入:
- CLI:`python3 scripts/renumber_chapters.py --plan <plan.json> [--dry-run] [--instance <name>]`;plan.json = `{"moves": [{"old": "ch34-primer-eplb", "new_id": "ch09"}, …]}`(引擎自动级联计算受挤压章的新号,或 plan 显式全量给出——取显式全量,歧义最小);
- 另一入口 `--insert <slug>@before:<目标slug>` 供日后单章插入:引擎自动生成级联 plan 并打印供确认。
- 本次交错的 plan(§1 表)作为 `instances/vllm-ascend/book/cartography/renumber-2026-07-05.json` 落盘存档。

两阶段,幂等,`--dry-run` 先行:
1. **目录迁移**:`git mv` 28 个章目录(经临时名避免环形冲突)+ `book/papers/chNN-slug/` 同步改名。
2. **引用重写**(按旧号→新号映射,**整词边界** `chNN` 与路径 `chNN-slug`):
   - 全部 `artifacts/*/narrative/chapter.md`:相对链接 `../chNN-slug/`、文字「第 N 章」(仅当 N 属重映射集合且上下文为章引用——用「第 NN 章」全角模式匹配);
   - 全部章内 JSON 工件:dossier(prereq/pairs)、explainer、run-ledger/review-report/retrofit-plan 的 chapter_id;
   - `book/cartography/outline-final.json`(chapter_id/slug/deps/part;parts 数组删 P8、P3/P5/P7 的 intent 提及新增原理章)、`papers-map.json`(primer_chapter);
   - `book/bible/{arc-map,figures,concepts,glossary}.json` 中章号;
   - `book/assets/roadmap/roadmap.py` ALIASES(键与标签重排,原理章标签保留「原理篇:」前缀);
   - `trace/state.json` 键与内容;INSTANCE.md 章号引用。
   - **不改**:trace/deliveries 历史文件、docs/superpowers 历史 spec/plan、.superpowers 台账(历史记录保持原号,脚本在 trace 写一条 renumber 映射记录供考古)。
3. **校验器内置**:迁移后扫描全书 `chNN` 引用,凡指向不存在目录的链接报错;pytest 测试用 tmp fixture 验证目录迁移/链接重写/JSON 重写/幂等性。

## 3. 接缝导语重写(writer 定点,迁移后)

约 15 处接缝,每处只动开场 roadmap 图注/首段导语/收尾过渡:
- 6 章原理章:开场接真实前章(如新 ch21 MLA 原理开场接 ch20 MHA:「上一章看完昇腾 MHA;在进入下一章的 MLA 落地前,先把论文数学地基打好」),收尾预告落地章;
- 5 章被承接的码章(新 ch10/ch22/ch24/ch32/ch35):开场导语改为承接原理章;**昨日加的 5 处指路框措辞改写**(「见第 31 章」→「上一章刚建立的…」,不再是远程指路);
- 新 ch26(V4 收束)后的新 ch27(原 ch23 算子):开场「上一章」引用核对;
- ch01 全书地图章:book-map 图(gen 脚本)与正文 Part 描述按 36 章重生成(illustrator+writer 各一小任务)。
- 每章 roadmap.png:28 个重编号章以新 highlight 键重新生成(脚本循环 roadmap.py + rsvg,非 writer 任务)。

## 4. 复验门禁(全部须过)

- pytest 全绿(含迁移脚本新测试);
- 全书 linter:lint_anchors --all / lint_punct --all / lint_diagram_geometry --all / 逐章 lint_chapter_structure;
- 迁移校验器 0 悬空引用;roadmap ALIASES 每键可渲染(抽 3 章 Read PNG);
- **gap-audit 重跑**:cliffs 仍为 0(导语重写不得引入新裸奔);
- bible.py due 抽查 3 章正常。

## 5. 补章 SOP(RUNBOOK 新节,工厂能力)

日后任何补充章(原理章/新机制章/勘误扩章)的标准流程:
1. **定位**:先改 outline——新章条目插到目标位置(deps/part 定好),papers-map(若 primer)同步;**位置先于内容存在**。
2. **生产**:按最终章号+slug 直接发 chapter-pipeline(新书/尾部追加天然零迁移);若目标位置已被占(存量书中段插入)→ 先以临时尾号生产,APPROVED 后执行第 3 步。
3. **插入迁移**:`renumber_chapters.py --insert` 生成级联 plan → dry-run 审阅 → 执行 → 内置校验器 0 悬空。
4. **接缝导语**:writer 定点重写插入点前后章的开场/收尾(见 §3 纪律);roadmap 键与受影响章 roadmap.png 再生成。
5. **复验**:§4 门禁全跑(anchors/punct/structure/gap-audit 增量)。
RUNBOOK 记为「补章发车」节,与 primer 发车节互引。

## 6. 首创期预留(新书从第一天让插章便宜——落进规范与 lint)

插章的迁移成本 = 章号被写死的次数。三条规范把章号收敛到"链接与目录名"两处,其余全部解耦:
1. **跨章引用三规**(writer 契约 + lint_anchors 增强,成对落地):
   a. 跨章引用**必须**是 markdown 链接(`[第 N 章:标题](../chNN-slug/narrative/chapter.md)`),**禁止裸文字章号**(「详见第 21 章」无链接——lint 报 warn,存量豁免、新章 blocking);
   b. lint_anchors 增强:**链接文字中的章号须与链接目标目录号一致**——重编号后只需脚本改链接,linter 保证文字不漂移;
   c. 导语/图注衔接**按内容措辞**(「上一章的 MHA 后端」而非「第 19 章」),章号仅出现在链接里——号变时接缝措辞大多无需重写。
2. **位置开局规划**:cartography 定稿时,可预见的原理章/扩展章直接在 outline 占号排进物理序(papers-map.primer_chapter 与 outline 同步产生)——新书的 primer 章从第一天就在正确位置,零迁移。RUNBOOK §0 与 primer 发车节补此条款。
3. **工件章号最小化**:新章 JSON 工件(dossier/explainer/run-ledger)中 chapter_id 保持单一来源(目录名派生),禁止在 mechanism 描述、图注、trace 正文里冗余写死章号——lint_dossier 已核锚点,补一条 warn 级「非链接章号出现在 JSON 字符串中」检查(仅新章)。

## 7. 风险与对策

- **级联改名撞车**:两阶段临时名 + dry-run diff 审阅后再执行;git mv 保历史。
- **「第 N 章」文字误改**(如"第 3 章"指 vLLM 基座书):重写器只按全角「第 NN 章」+ NN∈重映射集合匹配,且对每处替换输出上下文行到迁移日志,Lead 抽查;姊妹书(vllm 实例)完全不动。
- **在飞工作**:执行窗口内不发任何 ascend 章 workflow;5h 循环若触发只做状态检查。
- **historical trace 断链**:trace 写 renumber 映射记录;台账/历史文档不改,考古凭映射表。
