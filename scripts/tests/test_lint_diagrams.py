import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_diagrams import lint_diagrams

SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 100"><text {attrs}>多路复用解扇出</text></svg>'
SAFE = 'x="10" font-size="12" font-family="Droid Sans Fallback"'


def _mk(tmp, svgs: dict, pngs: list, narrative: str = ""):
    d = tmp / "ch"
    (d / "diagrams").mkdir(parents=True)
    (d / "narrative").mkdir(parents=True)
    for name, body in svgs.items():
        (d / "diagrams" / name).write_text(body, encoding="utf-8")
    for name in pngs:
        (d / "diagrams" / name).write_bytes(b"\x89PNG" + b"0" * 4000)
    (d / "narrative" / "chapter.md").write_text(narrative, encoding="utf-8")
    return str(d)


def test_cjk_unsafe_font_blocking(tmp_path):
    # CJK with sans-serif (the real bug) → flagged
    d = _mk(tmp_path, {"fig-x.svg": SVG.format(attrs='x="10" font-size="12" font-family="sans-serif"')},
            ["fig-x.png"], "![](../diagrams/fig-x.png)")
    assert lint_diagrams(d)["cjk_font"]


def test_cjk_no_font_blocking(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG.format(attrs='x="10" font-size="12"')},
            ["fig-x.png"], "![](../diagrams/fig-x.png)")
    assert lint_diagrams(d)["cjk_font"]


def test_cjk_comma_mixed_blocking(tmp_path):
    # listing a latin family alongside breaks CJK render → flagged
    d = _mk(tmp_path, {"fig-x.svg": SVG.format(attrs='x="10" font-size="12" font-family="Droid Sans Fallback, sans-serif"')},
            ["fig-x.png"], "![](../diagrams/fig-x.png)")
    assert lint_diagrams(d)["cjk_font"]


def test_cjk_safe_font_passes(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG.format(attrs=SAFE)},
            ["fig-x.png"], "![](../diagrams/fig-x.png)")
    r = lint_diagrams(d)
    assert not r["cjk_font"] and not r["orphan"] and not r["png_missing"] and not r["svg_invalid"]


def test_orphan_png_blocking(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG.format(attrs=SAFE)}, ["fig-x.png"], "正文没有引用这张图")
    assert lint_diagrams(d)["orphan"]


def test_missing_png_blocking(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG.format(attrs=SAFE)}, [], "![](../diagrams/fig-x.png)")
    assert lint_diagrams(d)["png_missing"]
