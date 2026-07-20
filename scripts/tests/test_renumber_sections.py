"""正文 §N.M / `## N.M` 小节号重编号(exp-2026-07-20-02)。

来源:vLLM 约束解码双章插入(ch31-37 → ch33-39)后,7 章正文的小节标题与 §徽标全部滞留旧号——
`ch39-engine-core` 顶部写「第 39 章」、正文却全是 `## 37.x` 与 `§37.x`;而 diagrams/*.py 的
§徽标已被引擎改成 §39.x,于是 7 章 `lint_chapter_map --require` 全红(该 gate 不在 --all 里,
被 --all 全绿掩盖)。根因:_rewrite_text 规则 4 把 §N.M 重写限定在 diagrams/*.py,正文全豁免。

正文不能像 diagrams 那样无脑改——论文引用 `arXiv:2210.17323 §3.1` 必须免伤。因此按可判定性分三档:
  A 小节标题 `^## N.M`:绝不可能是论文引用 → 本章旧号即改。
  B 本章自引 `§N.M`(N == 本章旧号) → 改。
  C 跨章引 `§N.M`:仅当同行近处有指向该章的链接/「第 N 章」/「上一章|下一章」时才改;
    否则不改,只登记进 unresolved 交人核(宁漏勿伤)。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from renumber_chapters import Move, _rewrite_sections

# ch31→ch33, ch32→ch34, ch37→ch39(与真实修复计划同构)
MOVES = [Move("ch31-primer-eagle", "ch33"),
         Move("ch32-spec-decode", "ch34"),
         Move("ch37-engine-core", "ch39")]


def _rw(text, chapter_dir):
    return _rewrite_sections(text, MOVES, chapter_dir)


# ---------- A. 小节标题 ----------

def test_heading_self_renumbered():
    out, un = _rw("## 37.4 调度器怎么收口\n### 37.12 小结\n", "ch39-engine-core")
    assert "## 39.4 调度器怎么收口" in out
    assert "### 39.12 小结" in out
    assert un == []


def test_heading_of_unmoved_chapter_untouched():
    out, _ = _rw("## 13.4 标题\n", "ch13-scheduler")
    assert out == "## 13.4 标题\n"


def test_h1_chapter_line_not_touched_by_section_pass():
    # 「# 第 39 章…」由规则 3 负责,本函数不得插手
    src = "# 第 39 章　高级引擎运维\n"
    out, _ = _rw(src, "ch39-engine-core")
    assert out == src


# ---------- B. 本章自引 ----------

def test_self_section_ref_renumbered():
    out, un = _rw("细节见 §37.8,以及 [§37.11](#anchor)。\n", "ch39-engine-core")
    assert "§39.8" in out and "§39.11" in out
    assert "§37." not in out
    assert un == []


def test_paper_section_ref_immune():
    """论文引用 §3.1 与本章旧号无关 → 逐字不动(原引擎最担心的误伤)。"""
    src = "按 arXiv:2210.17323 §3.1 的推导,并见 §2.9。\n"
    out, un = _rw(src, "ch39-engine-core")
    assert out == src
    assert un == []


# ---------- C. 跨章引 ----------

def test_cross_ref_resolved_by_link():
    src = "证明在[下一章](../../ch34-spec-decode/narrative/chapter.md) §32.5,本章只用结论。\n"
    out, un = _rw(src, "ch33-primer-eagle")
    assert "§34.5" in out and un == []


def test_cross_ref_resolved_by_chapter_word():
    """引擎规则 3 已把「第 32 章」改成「第 34 章」,§ 须据新号回推旧号 32 才敢改。"""
    src = "见 [第 34 章 §32.5](../../ch34-spec-decode/narrative/chapter.md)。\n"
    out, un = _rw(src, "ch13-scheduler")
    assert "§34.5" in out and un == []


def test_cross_ref_resolved_by_relative_word():
    src = "保分布定理的完整证明留给下一章 §32.5,这里只借结论。\n"
    out, un = _rw(src, "ch33-primer-eagle")   # ch33 的下一章 = ch34(旧 32)
    assert "§34.5" in out and un == []


def test_unresolvable_cross_ref_reported_not_rewritten():
    """近处没有任何指向该章的线索 → 不动,登记 unresolved。"""
    src = "某处提到 §32.5 但没说是哪一章。\n"
    out, un = _rw(src, "ch20-distributed-parallelism")
    assert out == src
    assert len(un) == 1 and "32.5" in un[0]


# ---------- 幂等 ----------

def test_idempotent():
    src = "## 37.4 标题\n见 §37.8 与[下一章](../../ch34-spec-decode/narrative/chapter.md) §32.5。\n"
    once, _ = _rw(src, "ch39-engine-core")
    twice, un = _rw(once, "ch39-engine-core")
    assert once == twice
    assert un == []


def test_no_cascading_double_shift():
    """§31 → §33 后不得在同一趟里被当作『旧 33』再推成 §35。"""
    out, _ = _rw("§31.2\n", "ch33-primer-eagle")
    assert "§33.2" in out and "§35" not in out


# ---------- 标题派生锚点 ----------

def test_heading_derived_anchor_follows():
    """`## 36.2 顶层编排` 的 GitHub 锚点是 `#362-顶层编排`;标题改号后锚点链接须同步,否则章内链接断。"""
    src = ("## 36.2 顶层编排:一个 async with\n"
           "回看[顶层编排](#362-顶层编排一个-async-with)。\n")
    out, un = _rewrite_sections(src, [Move("ch36-entrypoints", "ch38")], "ch38-entrypoints")
    assert "## 38.2 顶层编排" in out
    assert "(#382-顶层编排一个-async-with)" in out
    assert "#362-" not in out and un == []


def test_multi_level_heading_anchor():
    src = ("### 36.5.1 流式\n"
           "见[流式](#3651-流式)。\n")
    out, _ = _rewrite_sections(src, [Move("ch36-entrypoints", "ch38")], "ch38-entrypoints")
    assert "### 38.5.1 流式" in out and "(#3851-流式)" in out


def test_anchor_of_unmoved_chapter_untouched():
    src = "## 13.2 标题\n见[标题](#132-标题)。\n"
    out, _ = _rewrite_sections(src, [Move("ch36-entrypoints", "ch38")], "ch13-scheduler")
    assert out == src


def test_new_chapter_filling_vacated_slot_self_ref_untouched():
    """新章补进被腾空的号位时,它的自引 §N.M 会与某条 move 的旧号撞号——纯属巧合,不得改也不得报。

    真实场景:约束解码新 ch31/ch32 补进 ch31-primer-eagle/ch32-spec-decode 腾出的号位,
    `ch31-structured-output` 正文里 19 处 §31.x 全是它自己的小节。
    """
    src = "## 31.4 掩码装配\n本节承 §31.2 的账,细节见 §31.9。\n"
    out, un = _rewrite_sections(src, MOVES, "ch31-structured-output")
    assert out == src
    assert un == []


def test_no_double_shift_when_chapter_word_equals_another_moves_old_id():
    """「第 36 章 §36.8」不得被再推成 §38.8——哪怕 36 恰好是另一条 move 的旧号。

    真实翻车:vLLM ch11 原文「第 36 章 §34.8」第一趟正确改成「第 36 章 §36.8」;第二趟里
    _refers_to 认了 tgt(ch36-entrypoints→ch38) 的旧号 36,把窗口里那个『新号 36』当成线索,
    于是 §36.8 → §38.8。规则 5 永远跑在规则 1/3 之后,窗口里只可能是新号,故只认新号。
    """
    moves = [Move("ch34-pd-disaggregation", "ch36"), Move("ch36-entrypoints", "ch38")]
    src = "见 [第 36 章 §36.8](../../ch36-pd-disaggregation/narrative/chapter.md)。\n"
    out, un = _rewrite_sections(src, moves, "ch11-engine-core")
    assert out == src
    assert un == []


def test_full_corpus_style_idempotence():
    """跨章引 + 自引 + 标题 + 锚点混排,连跑两趟结果一致(第二趟必须是彻底 no-op)。"""
    moves = [Move("ch31-primer-eagle", "ch33"), Move("ch32-spec-decode", "ch34"),
             Move("ch34-pd-disaggregation", "ch36"), Move("ch36-entrypoints", "ch38")]
    src = ("## 31.2 标题\n"
           "自引 §31.7;跨引见 [第 34 章 §32.5](../../ch34-spec-decode/narrative/chapter.md);\n"
           "锚点[标题](#312-标题);论文 §3.1。\n")
    one, un1 = _rewrite_sections(src, moves, "ch33-primer-eagle")
    two, un2 = _rewrite_sections(one, moves, "ch33-primer-eagle")
    assert one == two, "第二趟不是 no-op"
    assert un1 == [] and un2 == []
    assert "## 33.2 标题" in one and "§33.7" in one and "§34.5" in one
    assert "(#332-标题)" in one and "§3.1" in one


def test_moved_chapter_new_number_colliding_with_another_old_id_is_silent():
    """移动章改完号后,新号可能正好是另一条 move 的旧号(ch35 新号 35 vs 旧 ch35→ch37)。
    此时 §35.x 是它自己的小节,既不改也不报——否则第二趟会刷出上百条假 unresolved。"""
    moves = [Move("ch33-pd-disaggregation", "ch35"), Move("ch35-entrypoints", "ch37")]
    src = "## 35.4 契约\n本节承 §35.2,详见 §35.5。\n"
    out, un = _rewrite_sections(src, moves, "ch35-pd-disaggregation")
    assert out == src
    assert un == []


def test_relative_word_counts_as_consistency_signal():
    """「留给下一章 §34.5」写在 ch33 里本就自洽(下一章正是 ch34),不得报 unresolved。"""
    moves = [Move("ch34-pd-disaggregation", "ch36")]
    src = "保分布定理的完整证明留给下一章 §34.5,这里只借结论。\n"
    out, un = _rewrite_sections(src, moves, "ch33-primer-eagle")
    assert out == src and un == []
