# SDD 任务简报 lint-exp-002 — lint_fidelity:省略标记漏标 / 行号区间错位 / 悬空引用

> 来源：`instances/vllm-ascend/book/retro/retro-2026-07-05.json` candidate `exp-undated-002`（Lead 已批准，本简报为落地前的 SDD 任务定义，不改代码）。

## 背景 / 现象

`scripts/lint_fidelity.py` 目前的保真检查粒度停在"符号级 `# SOURCE` 覆盖"（`_spans_missing_source`
逐 def/class 核有无 `# SOURCE:`），**不核**：
1. 内嵌源码块声明的行号跨度（如 `L27-L50`）与实际展示的代码行数是否存在缺口；
2. 多个非相邻源码区间被拼接展示时，是否显式标了省略标记（如 `# … 省略 …`）；
3. 被保留下来的代码行里引用的自由变量，是否确实在同一保留区内有定义（悬空引用）。

6 个已发生实例（ch02/ch03/ch04/ch05/ch15/ch28）：三个非相邻方法拼接无省略标记；`import` 语句被
从 `if/else`/`try/except` 之后静默挪到条件块之前（相对位置被重排）；保留行引用了只在省略区定义
的变量；`L27-50` 声明连续区间实际静默抽掉中间两行；`forward_context.draft_attn_metadatas` 等
横切字段赋值被静默删除且无省略标记。这些目前只能靠人工逐段比对原文源码才能发现。

## 规则描述（拟）

对 `narrative/chapter.md` 里每个"内嵌真源码"代码块（即含规范源码路径标注 `path:Lxxx-Lyyy` 的
围栏代码块）：

1. **行号跨度 vs 展示行数比对**：解析代码块头部/紧邻文字里的 `path:La-Lb` 声明，与源码文件
   `instances/<active>/source/<path>` 里 `La..Lb` 的真实行数比较代码块内非空行数；若声明的是
   单一连续区间但实际展示行数明显少于 `b-a+1`（超出容忍的空行/注释折叠阈值），且块内无省略
   标记（如 `# … 省略 …` / `...`），判 **FAIL**。
2. **非相邻区间拼接必须显式标注**：若一个代码块同时引用了 dossier `embed_excerpts` 里两个不
   相邻的 `lines` 区间（通过与 dossier.json 的 `embed_excerpts[].lines` 比对判定"非相邻"），必须
   有省略标记分隔，否则判 **FAIL**。
3. **悬空引用检测**：对代码块内每一行使用的自由变量名（简单启发式：非内置、非当前块内赋值/
   参数定义的标识符），如果能在同一 `embed_excerpts` 条目的 `elide` 说明区间里找到其定义（即被
   省略区定义、保留区引用），判 **FAIL**（悬空引用）。此项允许一定误报率，先做保守启发式。

## blocking / warn 定级建议

- 检查 1（行号区间跨度缺口无省略标记）→ **blocking**：直接影响读者对照源码的可信度，且有
  确定性判据（行数差 + 无省略标记）。
- 检查 2（非相邻区间拼接无标注）→ **blocking**：与检查 1 同源，判据同样确定。
- 检查 3（悬空引用）→ **warn（非阻断）先行**：启发式误报风险较高（自由变量识别在复杂 Python
  语法下不稳，如闭包/装饰器参数），建议先作为非阻断提示跑一个批次观察误报率，稳定后再收紧为
  blocking。

## 测试用例草案（参照 `scripts/tests/test_lint_fidelity.py` 风格）

```python
# scripts/tests/test_lint_fidelity_elision.py（新文件，或并入 test_lint_fidelity.py）
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_fidelity import lint_fidelity

def test_gap_without_elision_marker_flagged(tmp_path):
    """声明 L1-L10 但代码块只展示 6 行、无省略标记 → FAIL。"""
    ...

def test_gap_with_elision_marker_passes(tmp_path):
    """同上跨度缺口，但代码块含 `# … 省略 …` → 不报。"""
    ...

def test_non_adjacent_blocks_concatenated_without_marker_flagged(tmp_path):
    """dossier.embed_excerpts 声明两个不相邻 lines 区间，chapter.md 代码块无缝拼接展示 → FAIL。"""
    ...

def test_dangling_reference_to_elided_region_flagged(tmp_path):
    """保留行引用一个只在 elide 说明区间定义的变量 → warn。"""
    ...

def test_contiguous_full_block_no_false_positive(tmp_path):
    """行号区间与展示行数完全吻合、无拼接 → 不报（回归防误报）。"""
    ...
```

## 提醒

请 Lead 走 TDD 小任务落地：先写以上测试用例（红），再实现 `lint_fidelity.py` 新增检查（绿），
悬空引用检查建议先以非阻断形式在若干章（ch02/ch03/ch04/ch05/ch15/ch28 的历史现象章）跑一轮
观察误报率，再决定是否收紧。
