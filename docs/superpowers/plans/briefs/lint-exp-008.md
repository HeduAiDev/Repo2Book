# SDD 任务简报 lint-exp-008 — core 机制源码层机检：source_anchor 行号区间须与正文代码块标注区间相交

> 来源：`instances/vllm-ascend/book/retro/retro-2026-07-05.json` candidate `exp-undated-008`（Lead 已批准，落点 `linter:lint_chapter_structure`，本简报为落地前的 SDD 任务定义，不改代码）。

## 背景 / 现象

`lint_chapter_structure.py` 当前的"内嵌真源码"检查是**全章级存在性检查**
（`MIN_SOURCE_BLOCKS = 2`：全章只要有 ≥2 个含规范路径标注的代码块即通过），不核对**逐个
`difficulty=core` 机制**是否各自都有对应的内嵌源码块。5 个已发生实例：

- ch02：`algorithm-pedagogy` 要求 2+ 轮数值追踪，唯独"懒加载单例"机制只有结论无过程仍整体通过；
- ch28：Gumbel-max 有三步严证，拒绝采样只给结论断言、缺推导；
- ch31：core 机制"decode/prefill 分派"把机制与源码揉成单一小标题，偏离全书统一的
  直觉/机制/源码三段式；
- ch32：两条 core 机制仅一句 prose 带过，没有内嵌对应源码，全章级检查未拦；
- ch33（`reviews/pending-issues.json` 第 3 条）：`expected-accepted-length` /
  `walltime-speedup` 两条 core 机制的 `dossier.code_spine` 锚点
  （`rejection_sampler.py:L1035-L1060`、`L289-L348`）从未被嵌入为代码块，仅第三节结尾一句
  prose 带过参数名，直到人工 Review 才发现——`lint_chapter_structure` 的全章级判据完全放行了
  这个缺口。

## 规则描述（拟）

对 `dossier/dossier.json` 里 `mechanisms[]` 中每一条 `difficulty == "core"` 的机制：

1. **锚点区间相交**：取该机制的 `source_anchors`（形如 `<repo>/x.py:Lnnn-Lnnn`），解析出
   `(path, La, Lb)`；在 `narrative/chapter.md` 里逐个内嵌源码块解析其标注的 `path:Lc-Ld`
   （复用 `lint_chapter_structure._SRC_PATH_RE` 的路径识别 + 行号后缀解析）；若同一 `path` 下
   存在至少一个代码块的 `[Lc, Ld]` 与该机制某个 `source_anchors` 条目的 `[La, Lb]` **区间相交**
   （`max(La,Lc) <= min(Lb,Ld)`），判该机制"源码层"达标；否则记为该机制缺失内嵌源码。
2. **三段式子标题**（可选加强项，若本次范围内一并做）：机制附近应能匹配到形如
   `#### 机制`/`#### 源码`（或语义等价）的独立子标题，而非揉进一个复合标题——此项优先级低于
   检查 1，可作为第二迭代。
3. 缺任一 `difficulty=core` 机制的源码层 → 报告里明确点名 `mechanism_id`，供 writer 定点回填，
   不笼统只报"全章源码块不足"。

## blocking / warn 定级建议

- 检查 1（core 机制锚点区间相交）→ **blocking**：这是全书统一的"必达物"契约字面要求
  （`writer.md`"每个 difficulty=core 的机制三层在场"），且判据是确定性的区间运算，无主观空间。
- 检查 2（三段式子标题）→ **warn（非阻断）先行**：标题措辞/结构判定的启发式误报风险更高（不
  同章节的小节命名习惯不完全统一），先作提示，观察一批章节后再考虑升级。

## 测试用例草案（参照 `scripts/tests/test_lint_chapter_structure.py` 风格）

```python
# 追加到 scripts/tests/test_lint_chapter_structure.py（或新文件 test_lint_chapter_structure_core.py）

def test_core_mechanism_missing_embedded_source_flagged(tmp_path):
    """dossier 声明一个 difficulty=core 机制，source_anchors 指向 L1035-L1060，
    但正文所有代码块标注的行号区间都不与之相交 → FAIL，且报告点名该 mechanism_id。"""
    ...

def test_core_mechanism_with_intersecting_anchor_passes(tmp_path):
    """正文某代码块标注 L1030-L1070（与 source_anchors 的 L1035-L1060 相交）→ 该机制判达标。"""
    ...

def test_core_mechanism_shared_anchor_across_two_mechanisms_ok(tmp_path):
    """两个 core 机制共享同一段代码块锚点（如 ch33 的 expected-accepted-length 与
    walltime-speedup 均落在同一 rejection_sample 函数签名块）→ 两者均应判达标，不要求各自
    独立代码块。"""
    ...

def test_supporting_mechanism_not_checked(tmp_path):
    """difficulty=supporting 的机制即便无内嵌源码块也不报（本检查只管 core）。"""
    ...

def test_report_names_the_failing_mechanism_id(tmp_path):
    """报告文案里必须包含缺失机制的 mechanism_id，不能只说"源码块数量不足"。"""
    ...
```

## 提醒

请 Lead 走 TDD 小任务落地。这条改动需要 `lint_chapter_structure.py` 读 `dossier/dossier.json`
（目前该脚本只吃 `chapter.md` 单文件路径参数，需要扩展调用方式或改签名为吃 `chapter_dir`）——
落地时注意向后兼容旧调用点（RUNBOOK §6 / writer 契约里目前是
`lint_chapter_structure {chapter}/narrative/chapter.md` 这种单文件调用）。
