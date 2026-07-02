import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_trace_consistency import lint_trace_consistency

EXPLAINER = {"mechanisms": [{
    "mechanism_id": "m1",
    "worked_example": {"table": {"columns": ["轮次", "队列长"],
                                 "rows": [["1", "3"], ["2", "2"]]}},
}]}
DOSSIER = {"mechanisms": [{"id": "m1", "needs_worked_example": True}]}

GOOD_MD = """# 第 X 章

<!-- trace: m1 -->

| 轮次 | 队列长 |
|---|---|
| 1 | 3 |
| 2 | 2 |
"""


def _mk(tmp, md, explainer=EXPLAINER, dossier=DOSSIER):
    ch = tmp / "inst" / "artifacts" / "ch01"
    (ch / "narrative").mkdir(parents=True)
    (ch / "explainer").mkdir(parents=True)
    (ch / "dossier").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text(md, encoding="utf-8")
    if explainer is not None:
        (ch / "explainer" / "explainer.json").write_text(json.dumps(explainer), encoding="utf-8")
    (ch / "dossier" / "dossier.json").write_text(json.dumps(dossier), encoding="utf-8")
    return str(ch)


def test_matching_table_passes(tmp_path):
    r = lint_trace_consistency(_mk(tmp_path, GOOD_MD))
    assert not r["invalid"] and not r["drift"] and not r["coverage"]


def test_drifted_number_blocking(tmp_path):
    md = GOOD_MD.replace("| 2 | 2 |", "| 2 | 99 |")   # 99 不在素材里
    assert lint_trace_consistency(_mk(tmp_path, md))["drift"]


def test_unknown_mechanism_mark_blocking(tmp_path):
    md = GOOD_MD.replace("trace: m1", "trace: m9")
    assert lint_trace_consistency(_mk(tmp_path, md))["invalid"]


def test_mark_without_table_blocking(tmp_path):
    md = "<!-- trace: m1 -->\n\n这里没有表格。\n"
    assert lint_trace_consistency(_mk(tmp_path, md))["invalid"]


def test_missing_mark_is_coverage_gap(tmp_path):
    md = "# 第 X 章\n\n正文完全没有数值推演表标记。\n"
    assert lint_trace_consistency(_mk(tmp_path, md))["coverage"]


def test_no_explainer_old_chapter_warns_only(tmp_path):
    r = lint_trace_consistency(_mk(tmp_path, GOOD_MD, explainer=None))
    assert r["warn"] and not r["invalid"] and not r["drift"] and not r["coverage"]
