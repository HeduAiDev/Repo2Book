import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_explainer import lint_explainer

DOSSIER = {"mechanisms": [{
    "id": "m1", "name": "抢占回退循环", "kind": "algorithm",
    "source_anchors": ["pkg/sched.py:L1-L2"], "needs_figure": True,
    "needs_worked_example": True, "difficulty": "core"}]}

GOOD_ENTRY = {
    "mechanism_id": "m1",
    "intuition": "像叠盘子,永远从最上面拿",
    "worked_example": {
        "params": {"queue": 3},
        "trace_source": "run",
        "trace_ref": "traces/m1.json",
        "table": {"columns": ["轮次", "队列长", "返回"],
                  "rows": [["1", "3", "-"], ["2", "2", "req9"]]},
    },
    "invariant": {"claim": "队列长每轮严格减 1",
                  "argument": "每轮必 pop 一次,非负整数单调递减必有限步触底"},
    "quantified": "3 个请求 2 轮完成,O(len(running))",
    "figure_specs": [{
        "figure_id": "fig-m1", "claim": "抢占按 LIFO 弹出尾部,队列长每轮减 1",
        "template": "state-table",
        "numbers": [{"value": "3", "provenance": "traces/m1.json"}],
        "elements": ["逐轮状态表"], "caption_draft": "队列长 3→2→1:LIFO 抢占每轮恰弹出一个",
    }],
}
TRACE = '{"rounds": [{"round": 1, "qlen": 3}, {"round": 2, "qlen": 2, "victim": "req9"}]}'


def _mk(tmp, entry, trace=TRACE, dossier=DOSSIER):
    ch = tmp / "inst" / "artifacts" / "ch01"
    (ch / "dossier").mkdir(parents=True)
    (ch / "explainer" / "traces").mkdir(parents=True)
    (ch / "dossier" / "dossier.json").write_text(json.dumps(dossier), encoding="utf-8")
    (ch / "explainer" / "explainer.json").write_text(
        json.dumps({"mechanisms": [entry] if entry else []}), encoding="utf-8")
    if trace is not None:
        (ch / "explainer" / "traces" / "m1.json").write_text(trace, encoding="utf-8")
    return str(ch)


def test_good_entry_passes(tmp_path):
    r = lint_explainer(_mk(tmp_path, GOOD_ENTRY))
    assert not r["invalid"] and not r["mechanism"] and not r["trace"] and not r["figure"]


def test_missing_mechanism_entry_blocking(tmp_path):
    assert lint_explainer(_mk(tmp_path, None))["mechanism"]


def test_table_number_not_in_trace_blocking(tmp_path):
    e = json.loads(json.dumps(GOOD_ENTRY))
    e["worked_example"]["table"]["rows"][0][1] = "777"   # trace 里没有 777
    assert lint_explainer(_mk(tmp_path, e))["trace"]


def test_single_row_table_blocking(tmp_path):
    e = json.loads(json.dumps(GOOD_ENTRY))
    e["worked_example"]["table"]["rows"] = [["1", "3", "-"]]
    assert lint_explainer(_mk(tmp_path, e))["mechanism"]


def test_manual_without_reason_blocking(tmp_path):
    e = json.loads(json.dumps(GOOD_ENTRY))
    e["worked_example"]["trace_source"] = "manual"
    del e["worked_example"]["trace_ref"]
    assert lint_explainer(_mk(tmp_path, e, trace=None))["trace"]


def test_manual_with_reason_warns_only(tmp_path):
    e = json.loads(json.dumps(GOOD_ENTRY))
    e["worked_example"]["trace_source"] = "manual"
    e["worked_example"]["manual_reason"] = "本章 skip_impl,无精简版可跑"
    del e["worked_example"]["trace_ref"]
    r = lint_explainer(_mk(tmp_path, e, trace=None))
    assert not r["trace"] and r["warn"]


def test_needs_figure_without_spec_blocking(tmp_path):
    e = json.loads(json.dumps(GOOD_ENTRY))
    e["figure_specs"] = []
    assert lint_explainer(_mk(tmp_path, e))["figure"]


def test_figure_number_without_provenance_blocking(tmp_path):
    e = json.loads(json.dumps(GOOD_ENTRY))
    e["figure_specs"][0]["numbers"] = [{"value": "3"}]
    assert lint_explainer(_mk(tmp_path, e))["figure"]


FIG_ONLY_DOSSIER = {"mechanisms": [{
    "id": "m1", "name": "抢占回退循环", "kind": "layout",
    "source_anchors": ["pkg/sched.py:L1-L2"], "needs_figure": True,
    "needs_worked_example": False, "difficulty": "supporting"}]}

FIG_ONLY_ENTRY = {
    "mechanism_id": "m1",
    "figure_specs": [{
        "figure_id": "fig-m1", "claim": "抢占按 LIFO 弹出尾部",
        "template": "state-table",
        "numbers": [{"value": "3", "provenance": "traces/m1.json"}],
        "elements": ["逐轮状态表"], "caption_draft": "队列长 3→2→1",
    }],
}


def test_figure_only_mechanism_no_worked_example_required(tmp_path):
    r = lint_explainer(_mk(tmp_path, FIG_ONLY_ENTRY, trace=None, dossier=FIG_ONLY_DOSSIER))
    assert not r["mechanism"] and not r["trace"] and not r["figure"]


def test_needs_figure_missing_entry_blocking(tmp_path):
    r = lint_explainer(_mk(tmp_path, None, trace=None, dossier=FIG_ONLY_DOSSIER))
    assert r["figure"]
