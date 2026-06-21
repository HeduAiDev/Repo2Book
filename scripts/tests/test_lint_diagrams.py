import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_diagrams import lint_diagrams

# 干净 SVG：sans-serif 混排，交给 rsvg-convert 逐字回退（不强制 CJK 字体）
SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 100">'
       '<text x="10" y="40" font-family="sans-serif" font-size="14">AsyncLLM 三段式 → 队列</text></svg>')


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


def test_valid_diagram_passes(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG}, ["fig-x.png"], "![](../diagrams/fig-x.png)")
    r = lint_diagrams(d)
    assert not r["svg_invalid"] and not r["png_missing"] and not r["orphan"]


def test_orphan_png_blocking(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG}, ["fig-x.png"], "正文没有引用这张图")
    assert lint_diagrams(d)["orphan"]


def test_missing_png_blocking(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG}, [], "![](../diagrams/fig-x.png)")
    assert lint_diagrams(d)["png_missing"]


def test_invalid_svg_blocking(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": "<svg><text>未闭合"}, ["fig-x.png"], "![](../diagrams/fig-x.png)")
    assert lint_diagrams(d)["svg_invalid"]
