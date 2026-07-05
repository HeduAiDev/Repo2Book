import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_chapter_structure import lint_structure, print_report

ROADMAP = "## Roadmap：你在这里\n地图\n\n"


def _mk(tmp, narrative_body, mechanisms):
    ch = tmp / "ch"
    (ch / "narrative").mkdir(parents=True)
    (ch / "dossier").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text(ROADMAP + narrative_body, encoding="utf-8")
    (ch / "dossier" / "dossier.json").write_text(
        json.dumps({"mechanisms": mechanisms}), encoding="utf-8"
    )
    return str(ch / "narrative" / "chapter.md")


def test_core_mechanism_missing_embedded_source_flagged(tmp_path):
    """dossier 声明一个 difficulty=core 机制，source_anchors 指向 L1035-L1060，
    但正文所有代码块标注的行号区间都不与之相交 → FAIL，且报告点名该 mechanism_id。"""
    mechanisms = [{
        "id": "expected-accepted-length", "difficulty": "core",
        "source_anchors": ["vllm/sample/rejection_sampler.py:L1035-L1060"],
    }]
    body = (
        "```python\n"
        "# vllm/sample/rejection_sampler.py:L200\n"
        "def unrelated():\n"
        "    return 1\n"
        "```\n"
        "```python\n"
        "# vllm/sample/rejection_sampler.py:L300\n"
        "def also_unrelated():\n"
        "    return 2\n"
        "```\n"
    )
    md = _mk(tmp_path, body, mechanisms)
    res = lint_structure(md)
    assert res["core_mechanism_missing_source"]
    assert "expected-accepted-length" in res["core_mechanism_missing_source"][0]


def test_core_mechanism_with_intersecting_anchor_passes(tmp_path):
    """正文某代码块标注 L1030-L1070（与 source_anchors 的 L1035-L1060 相交）→ 该机制判达标。"""
    mechanisms = [{
        "id": "expected-accepted-length", "difficulty": "core",
        "source_anchors": ["vllm/sample/rejection_sampler.py:L1035-L1060"],
    }]
    body = (
        "```python\n"
        "# vllm/sample/rejection_sampler.py:L1030-L1070\n"
        "def rejection_sample():\n"
        "    return 1\n"
        "```\n"
    )
    md = _mk(tmp_path, body, mechanisms)
    res = lint_structure(md)
    assert not res["core_mechanism_missing_source"]


def test_core_mechanism_single_line_marker_within_anchor_passes(tmp_path):
    """正文 marker 只给起始行号(真实语料常见写法)，只要落在 source_anchors 区间内
    (用展示行数近似出的 Ld 与 anchor 相交) 也应判达标。"""
    mechanisms = [{
        "id": "expected-accepted-length", "difficulty": "core",
        "source_anchors": ["vllm/sample/rejection_sampler.py:L1035-L1060"],
    }]
    body = (
        "```python\n"
        "# vllm/sample/rejection_sampler.py:L1035\n"
        "def rejection_sample(max_spec_len: int):\n"
        "    \"\"\"Maximum speculative length.\"\"\"\n"
        "    return 1\n"
        "```\n"
    )
    md = _mk(tmp_path, body, mechanisms)
    res = lint_structure(md)
    assert not res["core_mechanism_missing_source"]


def test_core_mechanism_shared_anchor_across_two_mechanisms_ok(tmp_path):
    """两个 core 机制共享同一段代码块锚点 → 两者均应判达标，不要求各自独立代码块。"""
    mechanisms = [
        {"id": "mech-a", "difficulty": "core",
         "source_anchors": ["vllm/a.py:L100-L120"]},
        {"id": "mech-b", "difficulty": "core",
         "source_anchors": ["vllm/a.py:L100-L120"]},
    ]
    body = (
        "```python\n"
        "# vllm/a.py:L100-L120\n"
        "def shared():\n"
        "    return 1\n"
        "```\n"
    )
    md = _mk(tmp_path, body, mechanisms)
    res = lint_structure(md)
    assert not res["core_mechanism_missing_source"]


def test_supporting_mechanism_not_checked(tmp_path):
    """difficulty=supporting 的机制即便无内嵌源码块也不报（本检查只管 core）。"""
    mechanisms = [{
        "id": "some-supporting-mech", "difficulty": "supporting",
        "source_anchors": ["vllm/a.py:L500-L520"],
    }]
    body = (
        "```python\n"
        "# vllm/a.py:L1-L10\n"
        "def unrelated():\n"
        "    return 1\n"
        "```\n"
    )
    md = _mk(tmp_path, body, mechanisms)
    res = lint_structure(md)
    assert not res["core_mechanism_missing_source"]


def test_report_names_the_failing_mechanism_id(tmp_path):
    """报告文案里必须包含缺失机制的 mechanism_id，不能只说"源码块数量不足"。"""
    mechanisms = [{
        "id": "walltime-speedup", "difficulty": "core",
        "source_anchors": ["vllm/a.py:L900-L950"],
    }]
    body = (
        "```python\n"
        "# vllm/a.py:L1-L10\n"
        "def unrelated():\n"
        "    return 1\n"
        "```\n"
    )
    md = _mk(tmp_path, body, mechanisms)
    res = lint_structure(md)
    assert any("walltime-speedup" in i for i in res["core_mechanism_missing_source"])


def test_no_dossier_skips_check_no_false_positive(tmp_path):
    """无 dossier/dossier.json（旧调用点/写作前跑）→ 静默跳过，不影响既有行为。"""
    ch = tmp_path / "ch"
    (ch / "narrative").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text(
        ROADMAP + "```python\n# vllm/a.py:L1\nx = 1\n```\n```python\n# vllm/a.py:L2\ny = 2\n```\n",
        encoding="utf-8",
    )
    res = lint_structure(str(ch / "narrative" / "chapter.md"))
    assert not res["core_mechanism_missing_source"]


def test_core_mechanism_missing_source_is_warn_not_blocking(tmp_path):
    """core_mechanism_missing_source 报告问题但降级为非阻断 warn——
    落地实测发现 ch31/ch32 等既有章节用逗号列表式 marker(如 `# path:L721,L729,L731`)
    或纯 prose 引用锚点，本检查的区间解析会漏认；按"既有章节不得新增 BLOCKING"的
    防回归硬规则不计入返回码（问题仍打印，供 writer/reviewer 自查）。"""
    import io
    import contextlib

    mechanisms = [{
        "id": "expected-accepted-length", "difficulty": "core",
        "source_anchors": ["vllm/sample/rejection_sampler.py:L1035-L1060"],
    }]
    body = (
        "```python\n"
        "# vllm/sample/rejection_sampler.py:L200\n"
        "def unrelated():\n"
        "    return 1\n"
        "```\n"
        "```python\n"
        "# vllm/sample/rejection_sampler.py:L300\n"
        "def also_unrelated():\n"
        "    return 2\n"
        "```\n"
    )
    md = _mk(tmp_path, body, mechanisms)
    res = lint_structure(md)
    assert res["core_mechanism_missing_source"], "问题本身仍应被报告出来"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = print_report(res, md)
    assert code == 0, "warn 级不应使 exit code 为 1"
    assert "⚠️ core_mechanism_missing_source" in buf.getvalue()
