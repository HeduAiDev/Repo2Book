import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_source_grounding import lint_source_grounding

NARRATIVE = """# 第 33 章 测试章

## 背景

这里没有任何源码引用，用于触发"无来源"检查。

## 实现细节

vllm/attention/backend.py:120 这里有真实引用。
"""


def _mk(tmp, kind=None, impl_comment="# hello world\n", dossier=True):
    ch = tmp / "inst" / "artifacts" / "ch33-test"
    (ch / "narrative").mkdir(parents=True)
    (ch / "implementation").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text(NARRATIVE, encoding="utf-8")
    (ch / "implementation" / "impl.py").write_text(impl_comment, encoding="utf-8")
    if dossier:
        (ch / "dossier").mkdir(parents=True)
        doc = {}
        if kind:
            doc["kind"] = kind
        (ch / "dossier" / "dossier.json").write_text(json.dumps(doc), encoding="utf-8")
    return str(ch)


def test_non_primer_minimal_fixture_still_blocks(tmp_path):
    """回归：非 primer 章，行为与修复前字节级一致——缺来源 section 与 <3 锚点均应 BLOCK。"""
    ch = _mk(tmp_path, kind=None, impl_comment="# just a comment\n")
    r = lint_source_grounding(ch)
    assert r["narrative_vllm_refs"], "非 primer 缺来源的 section 应报告"
    assert r["implementation_references"], "非 primer <3 SOURCE/REFERENCE 锚点应报告"


def test_non_primer_no_dossier_still_blocks(tmp_path):
    """没有 dossier.json 时（旧章节/边界情况），非 primer 行为不变。"""
    ch = _mk(tmp_path, kind=None, impl_comment="# just a comment\n", dossier=False)
    r = lint_source_grounding(ch)
    assert r["narrative_vllm_refs"]
    assert r["implementation_references"]


def test_primer_with_paper_anchors_passes(tmp_path):
    """primer 章：# PAPER: 锚点计入实现锚点计数，且跳过逐 section 来源检查。"""
    impl = (
        "# PAPER: §3.1 Eq.1\n"
        "def a(): pass\n\n"
        "# PAPER: §3.2 Eq.2\n"
        "def b(): pass\n\n"
        "# PAPER: §3.3 Eq.3\n"
        "def c(): pass\n"
    )
    ch = _mk(tmp_path, kind="primer", impl_comment=impl)
    r = lint_source_grounding(ch)
    assert not r["narrative_vllm_refs"], "primer 章应跳过缺来源 section 检查"
    assert not r["implementation_references"], "primer 章 >=3 个 # PAPER 锚点应视为通过"


def test_primer_with_too_few_paper_anchors_still_blocks(tmp_path):
    """primer 章即便跳过 section 检查，实现锚点数仍需 >= 3。"""
    impl = "# PAPER: §3.1 Eq.1\ndef a(): pass\n"
    ch = _mk(tmp_path, kind="primer", impl_comment=impl)
    r = lint_source_grounding(ch)
    assert r["implementation_references"], "primer 章 <3 个锚点仍应 BLOCK"


NARRATIVE_3_REFS = """# 第 33 章 测试章

## 背景

这里没有任何源码引用，用于触发"无来源"检查。

## 实现细节

vllm/attention/backend.py:120 这里有真实引用。
vllm/attention/layer.py:45 另一个引用。
vllm/worker/model_runner.py:200 第三个引用。
"""

NARRATIVE_INSUFFICIENT_REFS = """# 第 33 章 测试章

## 背景

这里没有任何源码引用，用于触发"无来源"检查。

## 实现细节

vllm/attention/backend.py:120 只有一个引用。
"""


def _mk_with_narrative(tmp, narrative, impl_notes=None, kind=None, impl_comment=None):
    ch = tmp / "inst" / "artifacts" / "ch33-test"
    (ch / "narrative").mkdir(parents=True)
    (ch / "implementation").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text(narrative, encoding="utf-8")
    if impl_notes is not None:
        (ch / "implementation" / "impl-notes.md").write_text(impl_notes, encoding="utf-8")
    if impl_comment is not None:
        (ch / "implementation" / "impl.py").write_text(impl_comment, encoding="utf-8")
    (ch / "dossier").mkdir(parents=True)
    doc = {"kind": kind} if kind else {}
    (ch / "dossier" / "dossier.json").write_text(json.dumps(doc), encoding="utf-8")
    return str(ch)


IMPL_3_ANCHORS = (
    "# SOURCE: vllm/attention/backend.py:120\n"
    "def a(): pass\n\n"
    "# SOURCE: vllm/attention/layer.py:45\n"
    "def b(): pass\n\n"
    "# SOURCE: vllm/worker/model_runner.py:200\n"
    "def c(): pass\n"
)


def test_vllm_files_listed_counts_from_narrative_not_implnotes(tmp_path):
    """impl-notes.md 只登记 1 个路径，但正文 chapter.md 引用 3 个规范路径 → 不应 BLOCK。"""
    impl_notes = "只提到 vllm/attention/backend.py 一个路径。\n"
    ch = _mk_with_narrative(tmp_path, NARRATIVE_3_REFS, impl_notes=impl_notes)
    r = lint_source_grounding(ch)
    assert not r["vllm_files_listed"], "正文已引用 3 个规范路径，不应因 impl-notes.md 漏记而 BLOCK"


def test_vllm_files_listed_still_blocks_when_narrative_insufficient(tmp_path):
    """正文本身引用不足 3 个规范路径（即便 impl-notes.md 凑够）→ 仍应报告（回归防漏判）。"""
    impl_notes = (
        "vllm/attention/backend.py\nvllm/attention/layer.py\nvllm/worker/model_runner.py\n"
    )
    ch = _mk_with_narrative(tmp_path, NARRATIVE_INSUFFICIENT_REFS, impl_notes=impl_notes)
    r = lint_source_grounding(ch)
    assert r["vllm_files_listed"], "正文引用不足 3 个规范路径时仍应报告，不能被 impl-notes.md 掩盖"


def test_impl_notes_incomplete_is_warn_not_blocking(tmp_path):
    """impl-notes.md 缺路径但正文合规 → 只出现在非阻断的 impl_notes_incomplete，不计入 blocking 总数。"""
    import io
    import contextlib
    from lint_source_grounding import print_report

    narrative_all_refs = """# 第33章 测试章

## 实现细节

vllm/attention/backend.py:120 引用一。
vllm/attention/layer.py:45 引用二。
vllm/worker/model_runner.py:200 引用三。
"""
    impl_notes = "只提到 vllm/attention/backend.py 一个路径。\n"
    ch = _mk_with_narrative(
        tmp_path, narrative_all_refs, impl_notes=impl_notes, impl_comment=IMPL_3_ANCHORS
    )
    r = lint_source_grounding(ch)
    assert not r["vllm_files_listed"]
    assert not r["narrative_vllm_refs"]
    assert not r["implementation_references"]
    assert r["impl_notes_incomplete"], "impl-notes.md 路径不足应记为非阻断提示"

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(r, ch)
    out = buf.getvalue()
    assert "BLOCKING" not in out, "impl_notes_incomplete 不应被计入 blocking 汇总"


def test_prefix_still_dynamic_for_oot_instance(tmp_path):
    """vllm-ascend 实例：正文用 vllm_ascend/…py 路径应被正确计数（回归防前缀写死）。"""
    from lint_source_grounding import _SRC_PREFIXES

    prefix = _SRC_PREFIXES[0]
    narrative = f"""# 第 33 章 测试章

## 实现细节

{prefix}/worker/worker.py:10 引用一。
{prefix}/attention/backend.py:20 引用二。
{prefix}/models/model.py:30 引用三。
"""
    ch = _mk_with_narrative(tmp_path, narrative)
    r = lint_source_grounding(ch)
    assert not r["vllm_files_listed"], f"{prefix}/ 前缀应被正确识别并计数"


def test_hyphenated_paths_counted(tmp_path):
    """回归(triton-ascend):src_ref_re 须匹配连字符文件名(tutorials/01-vector-add.py),
    否则姊妹书引官方教程文件全被漏计→假 vllm_files_listed 不足。"""
    import re
    import lint_source_grounding as m
    alt = "|".join(re.escape(p) for p in m._SRC_PREFIXES)
    rx = rf'(?:{alt})/[\w/-]+\.(?:py|pyi|td|cpp|cc|cu|cuh|h|hpp)'
    found = set(re.findall(rx, "见 vllm/tutorials/01-vector-add.py 与 vllm/x/02-fused-softmax.py"))
    assert "vllm/tutorials/01-vector-add.py" in found and "vllm/x/02-fused-softmax.py" in found
