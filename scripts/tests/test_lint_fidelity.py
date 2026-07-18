import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_fidelity import lint_fidelity


def _mk(tmp, impl_files: dict, narrative: str = ""):
    d = tmp / "ch"
    (d / "implementation").mkdir(parents=True)
    for name, body in impl_files.items():
        (d / "implementation" / name).write_text(body, encoding="utf-8")
    (d / "narrative").mkdir(parents=True)
    (d / "narrative" / "chapter.md").write_text(narrative, encoding="utf-8")
    return str(d)


def test_missing_source_comment_is_blocking(tmp_path):
    impl = {"scheduler.py": "def schedule():\n    return 1\n"}  # no # SOURCE:
    res = lint_fidelity(_mk(tmp_path, impl))
    assert res["missing_source"], "function without # SOURCE: must be flagged"


def test_source_comment_passes(tmp_path):
    impl = {"scheduler.py": (
        "def schedule():\n"
        "    # SOURCE: vllm/v1/core/sched/scheduler.py:L352\n"
        "    # SUBTRACTED: no preemption — vllm L466\n"
        "    return 1\n")}
    res = lint_fidelity(_mk(tmp_path, impl))
    assert not res["missing_source"]
    assert not res["no_subtraction"]


def test_invention_marker_blocking(tmp_path):
    impl = {"x.py": "def f():\n    # SOURCE: vllm/a.py:L1\n    # TOY: fake loop\n    return 1\n"}
    res = lint_fidelity(_mk(tmp_path, impl))
    assert res["invention"]


def test_narrative_overexplains_companion(tmp_path):
    impl = {"x.py": "def f():\n    # SOURCE: vllm/a.py:L1\n    # SUBTRACTED: x\n    return 1\n"}
    nar = "see implementation/x.py and implementation/x.py and implementation/x.py\nvllm/a.py:L1\n"
    res = lint_fidelity(_mk(tmp_path, impl, nar))
    assert res["narrative_grounding"]


def test_over_subtraction_flagged(tmp_path):
    import json
    impl = {"x.py": "def f():\n    # SOURCE: vllm/a.py:L1\n    # SUBTRACTED: y\n    return 1\n"}
    d = _mk(tmp_path, impl)
    doss = pathlib.Path(d) / "dossier"
    doss.mkdir(parents=True)
    (doss / "dossier.json").write_text(json.dumps({
        "subtraction_plan": {"must_keep": [{"symbol": "RequestOutputCollector", "why": "核心队列"}]}
    }), encoding="utf-8")
    res = lint_fidelity(d)
    assert res["over_subtraction"], "must_keep 符号缺失应判过度删减"


def test_must_keep_present_passes(tmp_path):
    import json
    impl = {"x.py": (
        "class RequestOutputCollector:\n"
        "    # SOURCE: vllm/v1/engine/output_processor.py:L45\n"
        "    # SUBTRACTED: y\n"
        "    pass\n")}
    d = _mk(tmp_path, impl)
    doss = pathlib.Path(d) / "dossier"
    doss.mkdir(parents=True)
    (doss / "dossier.json").write_text(json.dumps({
        "subtraction_plan": {"must_keep": [{"symbol": "RequestOutputCollector", "why": "核心队列"}]}
    }), encoding="utf-8")
    res = lint_fidelity(d)
    assert not res["over_subtraction"]


def test_hyphenated_source_paths_match(tmp_path):
    """回归(triton-ascend):官方教程文件名含连字符(01-vector-add.py)+ 路径段连字符,
    _SRC_REF_RE 的字符类须含 '-',否则 hyphen 后整段路径匹配失败→假 narrative_grounding BLOCKING。
    (ch03 vector-add on-ramp 实证:引 ≥6 个连字符 .py 却被判 2 处。)"""
    import re
    from lint_fidelity import _SRC_REF_RE
    # 活动实例前缀下的连字符路径必须能整段匹配
    m = _SRC_REF_RE.search("vllm/tutorials/01-vector-add.py")
    assert m and m.group(0) == "vllm/tutorials/01-vector-add.py", \
        "连字符路径应整段匹配,不能只从连字符后截取"
    # 多段连字符 + 深路径
    assert _SRC_REF_RE.search("vllm/a-b/c-d-e/02-fused-softmax.py")
