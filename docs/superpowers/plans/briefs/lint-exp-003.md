# SDD 任务简报 lint-exp-003 — lint_source_grounding:读 chapter.md 而非 impl-notes.md；前缀按活动实例

> 来源：`instances/vllm-ascend/book/retro/retro-2026-07-05.json` candidate `exp-undated-003`（Lead 已批准，本简报为落地前的 SDD 任务定义，不改代码）。

## 背景 / 现象

`scripts/lint_source_grounding.py` 的 **Check 4（`vllm_files_listed`）** 只统计
`implementation/impl-notes.md`（内部脚手架记账文件）里出现的源码路径数，而不是发布正文
`narrative/chapter.md` 的真实引用。6 个已发生实例（ch03/ch07/ch08/ch16/ch20/ch22）里，正文已经
正确引用了足够的规范路径，但 `impl-notes.md` 漏记，导致误报 `Only N source files`；ch07/ch22
还出现同一缺陷被 **Check 3（`source_mapping_table`）与 Check 4 各报一遍**的重复噪音（这部分与
`exp-undated-004`/reviewer 去重职责有交叉，但根因在这里的计数源头不对）。

**代码现状核实**（写简报时已读源码，供任务定界）：Check 4 目前**已经**通过
`scripts/instance.py::canonical_prefix()` + `repo2book.json.depends_on` 动态取规范前缀
（`_SRC_PREFIXES`），并非硬编码 `vllm/`——即"路径前缀硬编码"这部分的原始症状在当前代码里已不
成立；**仍然成立、需要修的**是"计数源头是 `impl-notes.md` 而非 `chapter.md`"这一条主诉求。

## 规则描述（拟）

1. **源文件计数以正文为准**：Check 4 改为优先扫 `narrative/chapter.md` 里出现的规范路径引用
   （`_SRC_REF_RE` 同款正则，复用 `_SRC_PREFIXES`），达到 `>=3` 即通过；若 `chapter.md` 尚不存在
   （写作前跑），退回读 `impl-notes.md` 或跳过（不判 FAIL，只 warn）。
2. **impl-notes.md 完整性降级为提示**：`impl-notes.md` 里源码文件数不足，不再与正文计数共用
   同一硬阈值/同一 blocking 判据；改为独立的**非阻断**"提示"规则（`impl_notes_incomplete`），
   仅供 implementer 自查，不进 reviewer 的 blocking 汇总。
3. **path 前缀**：保持现状（已按活动实例动态读取），不需改动；简报仅确认其正确性，避免重复
   开发。

## blocking / warn 定级建议

- 正文源文件计数达标性检查（改造后的 Check 4）→ **blocking**（沿用现状定级，只是换了数据源）。
- `impl-notes.md` 完整性提示 → **warn（非阻断）**，避免对内部记账文件的疏漏拖累正文合规判定。

## 测试用例草案（参照 `scripts/tests/test_lint_source_grounding.py` 风格）

```python
# 追加到 scripts/tests/test_lint_source_grounding.py

def test_vllm_files_listed_counts_from_narrative_not_implnotes(tmp_path):
    """impl-notes.md 只登记 1 个路径，但正文 chapter.md 引用 3 个规范路径 → 不应 BLOCK。"""
    ...

def test_vllm_files_listed_still_blocks_when_narrative_insufficient(tmp_path):
    """正文本身引用不足 3 个规范路径（即便 impl-notes.md 凑够）→ 仍应 BLOCK（回归防漏判）。"""
    ...

def test_impl_notes_incomplete_is_warn_not_blocking(tmp_path):
    """impl-notes.md 缺路径但正文合规 → 只出现在非阻断的 impl_notes_incomplete，不计入 blocking 总数。"""
    ...

def test_prefix_still_dynamic_for_oot_instance(tmp_path):
    """vllm-ascend 实例：正文用 vllm_ascend/…py 路径应被正确计数（回归防前缀写死）。"""
    ...
```

## 提醒

请 Lead 走 TDD 小任务落地。改动范围小（Check 4 数据源切换 + 新增一条非阻断提示），建议与
`exp-undated-004`（reviewer 去重合并，已由本轮契约类落笔覆盖）配合验证：改后 ch07/ch22 一类的
"同一非阻断告警被两个维度各报一遍"现象应同时消失。
