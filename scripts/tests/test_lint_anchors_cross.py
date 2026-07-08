import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_anchors import check_cross, check_stale_section_prefix


def _mk(tmp, md):
    arts = tmp / "artifacts"
    (arts / "ch02-beta" / "narrative").mkdir(parents=True)
    (arts / "ch01-alpha" / "narrative").mkdir(parents=True)
    f = arts / "ch01-alpha" / "narrative" / "chapter.md"
    f.write_text(md, encoding="utf-8")
    return str(f)


def _mk_chapter(tmp, slug, body):
    """建一个独立章节目录 tmp/artifacts/<slug>/narrative/chapter.md，返回章目录路径(str)。"""
    d = tmp / "artifacts" / slug / "narrative"
    d.mkdir(parents=True, exist_ok=True)
    (d / "chapter.md").write_text(body, encoding="utf-8")
    return str(d.parent)


def _all_nums(tmp):
    """同 main() --all 的推导方式：实例现存章号集合从 artifacts/ 目录名推导。"""
    nums = set()
    for d in (tmp / "artifacts").glob("ch??-*"):
        nums.add(int(d.name[2:4]))
    return nums


def run_lint_collect_warns(tmp):
    """扫 tmp/artifacts 下全部章节目录，汇总 check_stale_section_prefix 的 warn 文案。"""
    nums = _all_nums(tmp)
    warns = []
    for d in sorted((tmp / "artifacts").glob("ch??-*")):
        warns.extend(check_stale_section_prefix(str(d), nums))
    return warns


def test_good_cross_link_passes(tmp_path):
    r = check_cross(_mk(tmp_path, "见[第 2 章：乙](../../ch02-beta/narrative/chapter.md)。\n"))
    assert not r["broken"] and not r["num_mismatch"] and not r["bad_depth"] and not r["bare"]


def test_broken_target_blocking(tmp_path):
    r = check_cross(_mk(tmp_path, "[第 9 章](../../ch09-nope/narrative/chapter.md)\n"))
    assert r["broken"]


def test_number_mismatch_blocking(tmp_path):
    r = check_cross(_mk(tmp_path, "[第 3 章：乙](../../ch02-beta/narrative/chapter.md)\n"))
    assert r["num_mismatch"]


def test_single_depth_legacy_blocking(tmp_path):
    r = check_cross(_mk(tmp_path, "[第 2 章](../ch02-beta/narrative/chapter.md)\n"))
    assert r["bad_depth"]


def test_bare_chapter_number_warns(tmp_path):
    r = check_cross(_mk(tmp_path, "详见第 2 章的讨论。\n"))
    assert r["bare"] and not r["broken"]


def test_bare_inside_link_text_not_flagged(tmp_path):
    r = check_cross(_mk(tmp_path, "[第 2 章：乙](../../ch02-beta/narrative/chapter.md)\n"))
    assert not r["bare"]


def test_stale_section_prefix_warns(tmp_path):
    """「见 19.5」：19 是实例现存章号(另有 ch19)、≠ 本章号 20、同行无 ch19- 链接、
    前有「见 」节引用语境 → warn(② 模式，前缀语境)。"""
    _mk_chapter(tmp_path, "ch19-bar", "# 第 19 章 Y\n")
    _mk_chapter(tmp_path, "ch20-foo", "# 第 20 章 X\n\n正文见 19.5 揭晓。\n")
    warns = run_lint_collect_warns(tmp_path)
    assert any("19.5" in w for w in warns)


def test_section_prefix_with_link_ok(tmp_path):
    """同行已有指向 ch19- 的跨章链接 → 19.5 视为正当引用，不 warn。"""
    _mk_chapter(tmp_path, "ch19-bar", "# 第 19 章 Y\n")
    body = "# 第 20 章 X\n\n见[第 19 章](../../ch19-bar/narrative/chapter.md) 19.5 节。\n"
    _mk_chapter(tmp_path, "ch20-foo", body)
    warns = run_lint_collect_warns(tmp_path)
    assert not any("19.5" in w for w in warns)


def test_own_section_and_versions_ok(tmp_path):
    """本章节号(20.3)、版本号(v0.21.0)、与实例中不存在的章号(3.5 倍，无 ch03)均不应 warn。"""
    body = "# 第 20 章 X\n\n本章 20.3 讲;v0.21.0 与 3.5 倍不受影响。\n"
    _mk_chapter(tmp_path, "ch20-foo", body)
    warns = run_lint_collect_warns(tmp_path)
    assert warns == []


def test_section_mark_warns_without_link(tmp_path):
    """真阳性①：「§14.4」无跨章链接、14 是现存章号且 ≠ 本章号 → warn(§ 前缀恒为节引用)。"""
    _mk_chapter(tmp_path, "ch14-bar", "# 第 14 章 Y\n")
    body = "# 第 20 章 X\n\n参考 §14.4 的实现。\n"
    _mk_chapter(tmp_path, "ch20-foo", body)
    warns = run_lint_collect_warns(tmp_path)
    assert any("§14.4" in w for w in warns)


def test_bare_with_section_suffix_warns(tmp_path):
    """真阳性②：「见 19.5 节」——裸 N.M 前有「见 」、后接「节」，双重命中节引用语境 → warn。"""
    _mk_chapter(tmp_path, "ch19-bar", "# 第 19 章 Y\n")
    body = "# 第 20 章 X\n\n见 19.5 节的讨论。\n"
    _mk_chapter(tmp_path, "ch20-foo", body)
    warns = run_lint_collect_warns(tmp_path)
    assert any("19.5" in w for w in warns)


def test_bare_numeric_ratios_not_flagged(tmp_path):
    """假阳性收紧：纯数值/比率/耗时——即便整数部分撞上现存章号(ch02/ch03)——
    没有 §、没有 见/详见/参见/接 前缀、没有 节/小节 后缀，一律不 warn。"""
    _mk_chapter(tmp_path, "ch02-bar", "# 第 2 章 B\n")
    _mk_chapter(tmp_path, "ch03-alpha", "# 第 3 章 A\n")
    body = "# 第 20 章 X\n\n加速比 3.5 倍，延迟 2.4 ms。\n"
    _mk_chapter(tmp_path, "ch20-foo", body)
    warns = run_lint_collect_warns(tmp_path)
    assert not any("3.5" in w for w in warns)
    assert not any("2.4" in w for w in warns)


def test_stale_section_primer_chapter_exempt(tmp_path):
    """primer 章的 §N.M 指论文章节,整章豁免 stale-section warn。"""
    _mk_chapter(tmp_path, "ch24-primer-flash-attention",
                "# 第 24 章 FA 原理\n\n论文 §3.1 给出 tiling。回看 §3.1。\n")
    _mk_chapter(tmp_path, "ch03-config", "# 第 3 章 配置\n")
    warns = run_lint_collect_warns(tmp_path)
    assert warns == []


def test_stale_section_uppercase_v_version_ok(tmp_path):
    """大写 V 版本号(DeepSeek-V3.2)不 warn,即使带节引用语境。"""
    _mk_chapter(tmp_path, "ch17-kv", "# 第 17 章 KV\n\n见 DeepSeek-V3.2 的稀疏化设计。\n")
    _mk_chapter(tmp_path, "ch03-config", "# 第 3 章 配置\n")
    assert run_lint_collect_warns(tmp_path) == []


def test_stale_section_compound_jie_word_ok(tmp_path):
    """『节』打头复合词(节拍/节点)不算节引用语境。"""
    _mk_chapter(tmp_path, "ch20-foo", "# 第 20 章 X\n\n每步 12.4 节拍打一个包。\n")
    _mk_chapter(tmp_path, "ch12-bar", "# 第 12 章 Y\n")
    assert run_lint_collect_warns(tmp_path) == []


def test_stale_section_standalone_jie_warns(tmp_path):
    """独立『节』后接标点仍应 warn。"""
    _mk_chapter(tmp_path, "ch20-foo", "# 第 20 章 X\n\n细节展开在 12.4 节。\n")
    _mk_chapter(tmp_path, "ch12-bar", "# 第 12 章 Y\n")
    warns = run_lint_collect_warns(tmp_path)
    assert any("12.4" in w for w in warns)
