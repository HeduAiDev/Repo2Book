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


# ── lint-exp-N2 回归:多论文包场景(ch33 mtp-as-speculative-proposer 曾误报)──
# Lead 复核代码现状后确认:Check 3 早已 glob 目录下全部 *.md 拼接再 grep,并非只读
# paper.md——ch33 的这条误报现状已不成立。以下测试固化该行为,防止未来重犯。

def test_section_found_in_secondary_paper_pack_no_false_positive(tmp_path):
    """dossier.mechanisms 里一条机制的 paper_origin.sections 只存在于第二份论文包
    （如 paper-mtp.md）而非 paper.md，且两份文件同在 papers/<chapter>/ 目录下
    → 不应报 paper_ref 误报（回归 ch33 现象:拼接全部 *.md 再 grep 已能覆盖）。"""
    ch = _mk(tmp_path, sections=["MTP in Inference"])
    pd = tmp_path / "inst" / "book" / "papers" / "ch33-primer-speculative-sampling"
    (pd / "paper-mtp.md").write_text(
        "##### MTP in Inference.\nDeepSeek-V3 uses MTP during inference...\n",
        encoding="utf-8",
    )
    r = lint_paper_grounding(ch)
    assert not r["paper_ref"], "存在于第二份论文包的小节不应因只读 paper.md 而误报缺失"


def test_section_missing_from_all_paper_packs_still_warns(tmp_path):
    """小节号在目录下所有论文包文件里都找不到 → 仍应 warn（防止改动后误伤真正的引用锚缺失）。"""
    ch = _mk(tmp_path, sections=["§9.9"])
    pd = tmp_path / "inst" / "book" / "papers" / "ch33-primer-speculative-sampling"
    (pd / "paper-mtp.md").write_text("##### Something else entirely.\n", encoding="utf-8")
    r = lint_paper_grounding(ch)
    assert r["paper_ref"], "所有论文包都找不到的小节仍应报警,不能因多包拼接而漏判"


# ── symbol_context(WARN) + key_figure_missing(BLOCKING) ──
# 独立的最小 primer/code 章 fixture:只关心 narrative body 与 meta.json key_figures,
# 不复用上面 _mk() 的 IMPL/dossier.mechanisms 细节。

def _mk_primer(tmp_path, body, meta_key_figures=None, chapter_name="ch40-primer-symbol-fixture",
               create_paper_pack=True):
    ch = tmp_path / "inst" / "artifacts" / chapter_name
    (ch / "dossier").mkdir(parents=True)
    (ch / "narrative").mkdir(parents=True)
    (ch / "dossier" / "dossier.json").write_text(
        json.dumps({"kind": "primer", "mechanisms": []}), encoding="utf-8"
    )
    (ch / "narrative" / "chapter.md").write_text(body, encoding="utf-8")
    if create_paper_pack:
        pd = tmp_path / "inst" / "book" / "papers" / chapter_name
        pd.mkdir(parents=True)
        (pd / "paper.md").write_text(PAPER_MD, encoding="utf-8")
        meta = {}
        if meta_key_figures is not None:
            meta["key_figures"] = meta_key_figures
        (pd / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return str(ch)


def _mk_code_chapter(tmp_path, body, chapter_name="ch41-normal-chapter"):
    ch = tmp_path / "inst" / "artifacts" / chapter_name
    (ch / "dossier").mkdir(parents=True)
    (ch / "narrative").mkdir(parents=True)
    (ch / "dossier" / "dossier.json").write_text(json.dumps({"mechanisms": []}), encoding="utf-8")
    (ch / "narrative" / "chapter.md").write_text(body, encoding="utf-8")
    return str(ch)


def run_lint(chapter_dir):
    return lint_paper_grounding(chapter_dir)


def test_symbol_without_context_warns(tmp_path):
    # $$ 块引入 \delta,前后 3 行 prose 无 δ/delta 提及、无表格行 → warn 1 条
    ch = _mk_primer(tmp_path, body="推导如下。\n\n$$\n\\delta = j - t\n$$\n\n继续。\n")
    res = run_lint(ch)
    assert any("delta" in w or "δ" in w for w in res["symbol_context"])


def test_symbol_with_nearby_prose_ok(tmp_path):
    # 全式扫描下 RHS 的 j/t 也算候选,须在正文里一并交代清楚(不再只顾 LHS 的 δ)
    ch = _mk_primer(
        tmp_path,
        body=(
            "其中 $\\delta$ 是相对位置偏移,$j$ 遍历 $t$ 之前的历史 token。\n\n"
            "$$\n\\delta = j - t\n$$\n"
        ),
    )
    assert run_lint(ch)["symbol_context"] == []


def test_symbol_in_table_ok(tmp_path):
    # 表格同样须覆盖 RHS 的 j/t,而不只是 LHS 的 δ
    ch = _mk_primer(
        tmp_path,
        body=(
            "| 符号 | 含义 |\n|---|---|\n"
            "| $\\delta$ | 相对偏移 |\n"
            "| $j$ | 键位置 |\n"
            "| $t$ | 查询位置 |\n\n"
            "$$\n\\delta = j - t\n$$\n"
        ),
    )
    assert run_lint(ch)["symbol_context"] == []


def test_rhs_symbol_without_context_warns(tmp_path):
    # 本次修复的行为定义:等号右侧首现的符号(M)同样要查——LHS-only 会对此视而不见
    ch = _mk_primer(
        tmp_path,
        body="总时延如下。\n\n$$\nL = N^2 d / M\n$$\n\n继续讨论。\n",
    )
    res = run_lint(ch)
    assert any(" M " in w for w in res["symbol_context"])


def test_key_figure_registered_and_redrawn_ok(tmp_path):
    ch = _mk_primer(
        tmp_path,
        meta_key_figures=[{
            "fig": "Fig.2", "arxiv": "arXiv:2405.04434", "shows": "x",
            "why_essential": "y", "target_section": "z",
        }],
        body="![重绘自 arXiv:2405.04434 Fig.2:MLA 压缩路径](../diagrams/paper-fig-2.png)\n",
    )
    assert run_lint(ch)["key_figure_missing"] == []


def test_key_figure_not_redrawn_fail(tmp_path):
    ch = _mk_primer(
        tmp_path,
        meta_key_figures=[{
            "fig": "Fig.2", "arxiv": "a", "shows": "x", "why_essential": "y", "target_section": "z",
        }],
        body="没图。\n",
    )
    assert len(run_lint(ch)["key_figure_missing"]) == 1


def test_orphan_redraw_caption_fail(tmp_path):
    ch = _mk_primer(
        tmp_path, meta_key_figures=[], body="![重绘自 arXiv:x Fig.9:孤儿](../diagrams/paper-fig-9.png)\n"
    )
    assert len(run_lint(ch)["key_figure_missing"]) == 1


def test_degraded_redraw_caption_by_section_ok(tmp_path):
    # 「按 arXiv:xxxx §y 描述重绘」降级句式(无法精确对应具体 Fig 号时的措辞)同样应算已重绘
    ch = _mk_primer(
        tmp_path,
        meta_key_figures=[{
            "fig": "Fig.2", "arxiv": "arXiv:2405.04434", "shows": "x",
            "why_essential": "y", "target_section": "z",
        }],
        body="![按 arXiv:2405.04434 §4.2 描述重绘 Fig.2:MLA 压缩路径](../diagrams/paper-fig-2.png)\n",
    )
    assert run_lint(ch)["key_figure_missing"] == []


def test_redraw_caption_multi_fig_numbers_all_counted(tmp_path):
    # 图注「重绘自…Fig.2 与 Fig.3 合并」须 findall 全收两个 Fig 号,而非只取首个
    ch = _mk_primer(
        tmp_path,
        meta_key_figures=[
            {"fig": "Fig.2", "arxiv": "a", "shows": "x", "why_essential": "y", "target_section": "z"},
            {"fig": "Fig.3", "arxiv": "a", "shows": "x", "why_essential": "y", "target_section": "z"},
        ],
        body="![重绘自 arXiv:x Fig.2 与 Fig.3 合并](../diagrams/paper-fig-2-3.png)\n",
    )
    assert run_lint(ch)["key_figure_missing"] == []


def test_paper_pack_missing_no_duplicate_meta_warn(tmp_path):
    # 论文包目录整体缺失时只应记一条"论文包缺失"warn,不应叠加同根因的
    # "meta 缺 key_figures 字段"重复 warn
    ch = _mk_primer(tmp_path, body="没有公式。\n", create_paper_pack=False)
    res = run_lint(ch)
    assert sum("论文包缺失" in w for w in res["warn"]) == 1
    assert not any("缺 key_figures 字段" in w for w in res["warn"])


def test_code_chapter_skipped(tmp_path):
    ch = _mk_code_chapter(tmp_path, body="$$\n\\delta=1\n$$\n")
    res = run_lint(ch)
    assert res["symbol_context"] == [] and res["key_figure_missing"] == []
