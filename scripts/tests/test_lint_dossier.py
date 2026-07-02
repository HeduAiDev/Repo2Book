import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_dossier import lint_dossier

GOOD_MECH = {
    "id": "m1", "name": "抢占回退循环", "kind": "algorithm",
    "source_anchors": ["pkg/sched.py:L2-L4"], "needs_figure": True,
    "needs_worked_example": True, "difficulty": "core",
}


def _mk(tmp, mechanisms, with_source=True):
    """构造 instances 形状的树:<inst>/artifacts/ch01 + <inst>/source/pkg/sched.py(5 行)。"""
    inst = tmp / "inst"
    ch = inst / "artifacts" / "ch01"
    (ch / "dossier").mkdir(parents=True)
    if with_source:
        (inst / "source" / "pkg").mkdir(parents=True)
        (inst / "source" / "pkg" / "sched.py").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    (ch / "dossier" / "dossier.json").write_text(
        json.dumps({"mechanisms": mechanisms}), encoding="utf-8")
    return str(ch)


def test_valid_mechanisms_pass(tmp_path):
    r = lint_dossier(_mk(tmp_path, [GOOD_MECH]))
    assert not r["invalid"] and not r["mechanism"] and not r["anchor"]


def test_missing_mechanisms_blocking(tmp_path):
    r = lint_dossier(_mk(tmp_path, []))
    assert r["invalid"]


def test_algorithm_without_worked_example_blocking(tmp_path):
    m = dict(GOOD_MECH, needs_worked_example=False)
    assert lint_dossier(_mk(tmp_path, [m]))["mechanism"]


def test_anchor_line_out_of_range_blocking(tmp_path):
    m = dict(GOOD_MECH, source_anchors=["pkg/sched.py:L2-L99"])
    assert lint_dossier(_mk(tmp_path, [m]))["anchor"]


def test_anchor_bad_format_blocking(tmp_path):
    m = dict(GOOD_MECH, source_anchors=["sched.py 第2行"])
    assert lint_dossier(_mk(tmp_path, [m]))["anchor"]


def test_missing_source_dir_warns_only(tmp_path):
    r = lint_dossier(_mk(tmp_path, [GOOD_MECH], with_source=False))
    assert r["warn"] and not r["anchor"]


def test_duplicate_id_blocking(tmp_path):
    assert lint_dossier(_mk(tmp_path, [GOOD_MECH, dict(GOOD_MECH)]))["mechanism"]
