import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_anchors import check_cross


def _mk(tmp, md):
    arts = tmp / "artifacts"
    (arts / "ch02-beta" / "narrative").mkdir(parents=True)
    (arts / "ch01-alpha" / "narrative").mkdir(parents=True)
    f = arts / "ch01-alpha" / "narrative" / "chapter.md"
    f.write_text(md, encoding="utf-8")
    return str(f)


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
