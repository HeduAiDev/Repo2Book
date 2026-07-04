# SDD 任务简报 lint-exp-N2 — lint_paper_grounding：按 dossier sources 读全部论文包做小节核对

> 来源：`instances/vllm-ascend/artifacts/ch33-primer-speculative-sampling/reviews/pending-issues.json`
> 第 2 条 `suggested_fix`（对应经验台账 `exp-0705-2`，Lead 已批准，本简报为落地前的 SDD 任务定义，
> 不改代码）。

## 背景 / 现象（原始诉求）

`scripts/lint_paper_grounding.py` 在核对 `dossier.mechanisms[].paper_origin.sections` 能否在
论文包里查到时，被指出"只读本章目录下的 `paper.md`，没有考虑本章 dossier 里显式登记的第二篇
论文包 `paper-mtp.md`"，导致 `mtp-as-speculative-proposer` 机制的 section `'MTP in Inference'`
误报——人工核对 `paper-mtp.md` 后确认原文小节标题（`##### MTP in Inference.`）与章节引用逐字
匹配，判定为 linter 的多论文场景盲点而非章节缺陷。

原始 `suggested_fix`：给 `lint_paper_grounding.py` 增加对 `dossier.sources` 里除
`primary_paper` 外其余论文包（如 `supplementary_paper.pack`）的读取，按
`mechanism.paper_origin.paper`（arXiv id）匹配到对应论文包文件再做小节存在性检查，而不是硬编码
只读同目录 `paper.md`。

## ⚠️ 落地前请 Lead 先复核：代码现状可能已部分解决

写本简报时读取了当前 `scripts/lint_paper_grounding.py`（`git log` 显示最近一次相关改动是
`233d6359 fix(primer): 终审修复——source_grounding primer 分支/expect-primer 闸/ledger/类型护栏`）：

```python
# 3) dossier paper_origin.sections 可在论文包里找到(WARNING)
#    双论文 primer 章可能有 paper.md + paper-dsa.md 等多份,拼接全部 *.md 再 grep
inst_book = d.resolve().parent.parent / "book"
pack_dir = inst_book / "papers" / d.resolve().name
pack_files = sorted(pack_dir.glob("*.md")) if pack_dir.exists() else []
ptext = "\n".join(p.read_text(...) for p in pack_files) if pack_files else None
```

即：当前实现已经 `glob("*.md")` 拼接同目录下**全部** `.md` 文件（不止 `paper.md`），并且实测
`instances/vllm-ascend/book/papers/ch33-primer-speculative-sampling/` 目录下 `paper.md` 与
`paper-mtp.md` 均已存在——按当前代码路径，ch33 的这条误报**理论上应已消失**。

原始 `pending-issues.json` 的 issue 描述与当前代码行为不一致的可能原因（供 Lead 判断，不代由
Curator 下结论）：
1. issue 是在 `233d6359` 修复**之前**的评审轮次发现、写入 pending-issues 时未及时标记为已解决；
2. 或者当时 `paper-mtp.md` 尚未落盘到目录里，评审时目录里确实只有 `paper.md`（后来才补的第二
   份论文包），误报是当时的真实现象，现在补齐文件后已自然消失，不需要再改代码；
3. 或者还存在别的匹配盲点（如 `sections` 里的小节号在**其中一份**论文包里能找到，但代码逻辑
   要求"按 `paper_origin.paper`（arXiv id）精确匹配到唯一论文包"而非"任意一份论文包命中即可"
   ——当前实现是**任意一份命中即可**（拼接全部再 grep），比原始 suggested_fix 提议的"按 arXiv id
   精确路由"更宽松，一般情况下更不容易误报，但也可能在极端情况下（两份论文包碰巧都不含某小节
   号字面量、但用不同措辞表达同一小节）依然误报——这与"读哪个文件"无关，是小节号匹配本身的
   模糊性，原始 fix 不能解决。

**建议**：Lead 在派 TDD 任务前，先在 ch33 实际目录上重跑一次 `lint_paper_grounding.py`，确认
该具体误报是否已消失；若已消失，本简报的意义降级为"补一条回归测试防止未来重犯"，不必再改
逻辑；若仍然误报，则按下面的规则描述做真正的行为修复。

## 规则描述（拟，若确认仍需修复）

1. **按 `dossier.sources` 路由到具体论文包**（原始诉求）：若 `dossier.json` 顶层有 `sources`
   字段（如 `{"primary_paper": {...}, "supplementary_paper": {"pack": "paper-mtp.md", "paper": "arXiv:xxxx"}}`），
   核对 `mechanism.paper_origin.sections` 时优先按 `mechanism.paper_origin.paper`（arXiv id）
   匹配到 `dossier.sources` 里对应的 `pack` 文件名，只在该文件里 grep；找不到对应路由信息时，
   退回现状的"拼接目录下全部 `*.md`"作为兜底（不收紧、只加精确路由作为优先路径）。
2. 若确认现状（拼接全部 `*.md`）已能覆盖 ch33 这类场景，且 `dossier.sources` 字段本身尚未在
   多数章节的 dossier 里稳定登记，则本条可降级为：**不改代码，只补回归测试**固化当前行为，
   并在 `experience-ledger.md` 标注该 pattern 复发判定为"已由其他改动连带修复"。

## blocking / warn 定级建议

- `paper_ref`（小节号核对）目前已是 **warn（非阻断）**（`print_report` 的 `blocking` 只统计
  `impl`/`citation`/`expect`），本次改动不改变定级，只提升准确性、减少误报噪音。

## 测试用例草案（参照 `scripts/tests/test_lint_paper_grounding.py` 风格）

```python
# 追加到 scripts/tests/test_lint_paper_grounding.py

def test_section_found_in_secondary_paper_pack_no_false_positive(tmp_path):
    """dossier.mechanisms 里一条机制的 paper_origin.sections 只存在于第二份论文包
    （如 paper-mtp.md）而非 paper.md，且两份文件同在 papers/<chapter>/ 目录下
    → 不应报 paper_ref 误报（回归 ch33 现象，验证现状是否已修复）。"""
    ...

def test_section_missing_from_all_paper_packs_still_warns(tmp_path):
    """小节号在目录下所有论文包文件里都找不到 → 仍应 warn（防止改动后误伤真正的引用锚缺失）。"""
    ...

def test_sources_routing_prefers_matched_pack_when_present(tmp_path):
    """若 dossier.sources.supplementary_paper.pack 显式指向某文件且 paper_origin.paper
    匹配其 arXiv id → 优先只在该文件里核对（若采纳精确路由改动）。"""
    ...
```

## 提醒

请 Lead 先确认 ch33 现状是否已解决（见上文"落地前请复核"），再决定本任务范围是"补回归测试"
还是"改行为逻辑"；两种情况都建议走 TDD 小任务，并在 `experience-ledger.md` 的 `exp-0705-2` 行
补充复核结论。
