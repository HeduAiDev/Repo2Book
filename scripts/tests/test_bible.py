import sys
import pathlib
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import bible


def test_due_lists_plant_and_payoff(tmp_path):
    arc = tmp_path / "arc-map.json"
    arc.write_text(json.dumps([
        {"id": "f1", "what": "per-request 队列", "plant": "ch04", "payoff": "ch08", "status": "open"},
        {"id": "f2", "what": "DP wave", "plant": "ch21", "payoff": "ch21", "status": "open"},
    ]), encoding="utf-8")
    due = bible.due("ch04", arc_path=str(arc))
    assert any(x["id"] == "f1" for x in due["plant"])
    assert not due["payoff"]
    due8 = bible.due("ch08", arc_path=str(arc))
    assert any(x["id"] == "f1" for x in due8["payoff"])


def test_resolved_payoff_not_due(tmp_path):
    arc = tmp_path / "arc-map.json"
    arc.write_text(json.dumps([
        {"id": "f1", "what": "x", "plant": "ch04", "payoff": "ch08", "status": "resolved"},
    ]), encoding="utf-8")
    assert not bible.due("ch08", arc_path=str(arc))["payoff"]
