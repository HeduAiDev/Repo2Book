import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_diagrams import lint_diagrams

# 干净 SVG：sans-serif 混排，交给 rsvg-convert 逐字回退（不强制 CJK 字体）
SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 100">'
       '<text x="10" y="40" font-family="sans-serif" font-size="14">AsyncLLM 三段式 → 队列</text></svg>')


def _mk(tmp, svgs: dict, pngs: list, narrative: str = "", explainer: str = None, manifest: str = None):
    d = tmp / "ch"
    (d / "diagrams").mkdir(parents=True)
    (d / "narrative").mkdir(parents=True)
    for name, body in svgs.items():
        (d / "diagrams" / name).write_text(body, encoding="utf-8")
    for name in pngs:
        (d / "diagrams" / name).write_bytes(b"\x89PNG" + b"0" * 4000)
    (d / "narrative" / "chapter.md").write_text(narrative, encoding="utf-8")
    if explainer is not None:
        (d / "explainer").mkdir(parents=True)
        (d / "explainer" / "explainer.json").write_text(explainer, encoding="utf-8")
    if manifest is not None:
        (d / "diagrams" / "figure-manifest.json").write_text(manifest, encoding="utf-8")
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


EXPL = ('{"mechanisms": [{"mechanism_id": "m1", "figure_specs": '
        '[{"figure_id": "fig-x", "claim": "c", "template": "flow"}]}]}')
SELF_OK = ('{"claim_readable_10s": true, "numbers_match_spec": true, "no_overlap": true, '
           '"arrows_attached": true, "cjk_rendered": true, "reading_order_clear": true}')


def _man(verdict="PASS", selfcheck=SELF_OK):
    return ('{"figures": [{"figure_id": "fig-x", "gen": "gen_fig-x.py", "svg": "fig-x.svg", '
            '"png": "fig-x.png", "selfcheck": ' + selfcheck +
            ', "blind_review": {"verdict": "' + verdict + '", "notes": ""}}]}')


def test_v3_manifest_ok_passes(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG, "gen_fig-x.py": "#"}, ["fig-x.png"],
            "![](../diagrams/fig-x.png)", explainer=EXPL, manifest=_man())
    assert not lint_diagrams(d)["manifest"]


def test_v3_missing_manifest_blocking(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG}, ["fig-x.png"],
            "![](../diagrams/fig-x.png)", explainer=EXPL)
    assert lint_diagrams(d)["manifest"]


def test_v3_blind_review_not_pass_blocking(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG, "gen_fig-x.py": "#"}, ["fig-x.png"],
            "![](../diagrams/fig-x.png)", explainer=EXPL, manifest=_man(verdict="PENDING"))
    assert lint_diagrams(d)["manifest"]


def test_v3_selfcheck_false_blocking(tmp_path):
    bad = SELF_OK.replace('"no_overlap": true', '"no_overlap": false')
    d = _mk(tmp_path, {"fig-x.svg": SVG, "gen_fig-x.py": "#"}, ["fig-x.png"],
            "![](../diagrams/fig-x.png)", explainer=EXPL, manifest=_man(selfcheck=bad))
    assert lint_diagrams(d)["manifest"]


def test_old_chapter_without_explainer_unaffected(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG}, ["fig-x.png"], "![](../diagrams/fig-x.png)")
    assert not lint_diagrams(d)["manifest"]
