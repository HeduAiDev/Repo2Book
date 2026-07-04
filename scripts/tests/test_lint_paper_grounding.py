import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_paper_grounding import lint_paper_grounding

IMPL_OK = '''
# PAPER: §3.1 Eq.4
def rejection_step(p, q, u):
    return u < min(1.0, p / q)


class Sampler:
    # PAPER: §3.2
    def draw(self):
        return 1
'''

IMPL_BAD = '''
def mystery(x):
    return x + 1
'''

MD_OK = """# 第 33 章

投机采样的保分布性由拒绝采样定理保证(arXiv:2211.17192 §3.1)。

$$
P(x) = \\min(1, p(x)/q(x))
$$

上式即论文 Eq.4 的接受准则。
"""

MD_NO_ARXIV = MD_OK.replace("arXiv:2211.17192 §3.1", "论文里")

PAPER_MD = "## 3.1 Speculative Sampling\nEq.4 acceptance...\n"


def _mk(tmp, kind="primer", impl=IMPL_OK, md=MD_OK, paper=PAPER_MD, sections=None, paper_origin=None):
    ch = tmp / "inst" / "artifacts" / "ch33-primer-speculative-sampling"
    (ch / "dossier").mkdir(parents=True)
    (ch / "implementation").mkdir(parents=True)
    (ch / "narrative").mkdir(parents=True)
    po = paper_origin if paper_origin is not None else {
        "paper": "arXiv:2211.17192", "sections": sections or ["§3.1"]}
    doc = {"mechanisms": [{"id": "m1", "paper_origin": po}]}
    if kind:
        doc["kind"] = kind
    (ch / "dossier" / "dossier.json").write_text(json.dumps(doc), encoding="utf-8")
    (ch / "implementation" / "ref_impl.py").write_text(impl, encoding="utf-8")
    (ch / "narrative" / "chapter.md").write_text(md, encoding="utf-8")
    if paper is not None:
        pd = tmp / "inst" / "book" / "papers" / "ch33-primer-speculative-sampling"
        pd.mkdir(parents=True)
        (pd / "paper.md").write_text(paper, encoding="utf-8")
    return str(ch)


def test_non_primer_chapter_all_empty(tmp_path):
    r = lint_paper_grounding(_mk(tmp_path, kind=None, impl=IMPL_BAD, md=MD_NO_ARXIV))
    assert not any(r[k] for k in ("impl", "citation", "formula", "paper_ref"))


def test_good_primer_passes(tmp_path):
    r = lint_paper_grounding(_mk(tmp_path))
    assert not r["impl"] and not r["citation"] and not r["formula"]


def test_missing_paper_anchor_blocking(tmp_path):
    assert lint_paper_grounding(_mk(tmp_path, impl=IMPL_BAD))["impl"]


def test_no_arxiv_in_narrative_blocking(tmp_path):
    assert lint_paper_grounding(_mk(tmp_path, md=MD_NO_ARXIV))["citation"]


def test_formula_without_nearby_anchor_warns(tmp_path):
    md = MD_OK.replace("上式即论文 Eq.4 的接受准则。", "就是这样。").replace(
        "(arXiv:2211.17192 §3.1)", "")
    md += "\n\narXiv:2211.17192\n" + "\n" * 30 + "$$\ny = x\n$$\n" + "\n" * 15
    r = lint_paper_grounding(_mk(tmp_path, md=md))
    assert r["formula"] and not r["citation"]


def test_section_not_in_paper_pack_warns(tmp_path):
    r = lint_paper_grounding(_mk(tmp_path, sections=["§9.9"]))
    assert r["paper_ref"]


def test_expect_primer_mismatch_blocks(tmp_path):
    r = lint_paper_grounding(
        _mk(tmp_path, kind=None, impl=IMPL_BAD, md=MD_NO_ARXIV), expect_primer=True
    )
    assert r["expect"]


def test_expect_primer_with_kind_primer_passes(tmp_path):
    r = lint_paper_grounding(_mk(tmp_path), expect_primer=True)
    assert not r["expect"]


def test_string_paper_origin_no_crash(tmp_path):
    # paper_origin 本应是 {paper, sections} 对象；这里给一个字符串，
    # 校验 lint_paper_grounding 不因 AttributeError 崩溃（跳过该机制的 paper_ref 检查即可）。
    r = lint_paper_grounding(_mk(tmp_path, paper_origin="arXiv:2211.17192 §3.1"))
    assert not r["paper_ref"]
