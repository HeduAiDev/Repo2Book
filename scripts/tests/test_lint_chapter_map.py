import json, subprocess, sys
from pathlib import Path
LINT = Path(__file__).resolve().parents[1] / "lint_chapter_map.py"

def _mk(tmp_path, svg_texts, headings="## 20.1 入口\n## 20.2 分流\n", dossier=None, body_extra=""):
    ch = tmp_path / "ch20-foo"; (ch / "diagrams").mkdir(parents=True); (ch / "narrative").mkdir()
    tspans = "".join(f'<text x="0" y="0">{t}</text>' for t in svg_texts)
    (ch / "diagrams" / "chapter-map.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg">{tspans}</svg>', encoding="utf-8")
    (ch / "narrative" / "chapter.md").write_text(
        f"# 第 20 章 X\n\nhook。\n\n![本章地图](../diagrams/chapter-map.png)\n\n只想看结论,跳 §20.2。\n\n{headings}\n{body_extra}", encoding="utf-8")
    (ch / "dossier.json").write_text(json.dumps(dossier or
        {"mechanisms": [{"anchors": ["forward_impl", "_get_fia_params"]}]}), encoding="utf-8")
    return ch

def _run(ch, *flags):
    return subprocess.run([sys.executable, str(LINT), str(ch), *flags], capture_output=True, text=True)

def test_badge_matches_headings_pass(tmp_path):
    assert _run(_mk(tmp_path, ["§20.1", "forward_impl"])).returncode == 0

def test_badge_not_in_headings_fail(tmp_path):
    r = _run(_mk(tmp_path, ["§20.9"]))
    assert r.returncode == 1 and "20.9" in r.stdout

def test_fabricated_symbol_fail(tmp_path):
    r = _run(_mk(tmp_path, ["§20.1", "totally_fake_fn()"]))
    assert r.returncode == 1 and "totally_fake_fn" in r.stdout

def test_no_map_no_require_ok(tmp_path):
    ch = tmp_path / "ch21-bar"; (ch / "narrative").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text("# 第 21 章\n\n## 21.1 a\n", encoding="utf-8")
    assert _run(ch).returncode == 0

def test_no_map_with_require_fail(tmp_path):
    ch = tmp_path / "ch21-bar"; (ch / "narrative").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text("# 第 21 章\n\n## 21.1 a\n", encoding="utf-8")
    assert _run(ch, "--require").returncode == 1

def test_require_checks_position_and_guidance(tmp_path):
    ch = _mk(tmp_path, ["§20.1"])
    md = ch / "narrative" / "chapter.md"
    md.write_text(md.read_text(encoding="utf-8").replace("![本章地图](../diagrams/chapter-map.png)\n\n只想看结论,跳 §20.2。\n\n", "") +
                  "\n![本章地图](../diagrams/chapter-map.png)\n", encoding="utf-8")   # 图挪到第一个 ## 之后
    assert _run(ch, "--require").returncode == 1


# ── FIX-ROUND-2 ──────────────────────────────────────────────────────────

def test_bare_number_not_badge(tmp_path):
    """裸 N.M(如延迟/版本数值)不带 § 前缀，不当徽标核对——不应触发 badge_not_in_headings。"""
    r = _run(_mk(tmp_path, ["latency 20.5 ms"]))
    assert r.returncode == 0

def test_badge_natural_heading_chapter_gives_specific_guidance(tmp_path):
    """本章为自然标题(无 `## N.M` 编号标题，heading_set 为空)而图上仍有 §N.M 徽标——
    报错应明确指引:自然标题章节应改用标题词作站牌，禁用 §N.M 徽标。"""
    r = _run(_mk(tmp_path, ["§20.1"], headings="## 引言\n## 深入原理\n"))
    assert r.returncode == 1
    assert "自然标题" in r.stdout and "禁用" in r.stdout

def test_dotted_symbol_fabrication_fail(tmp_path):
    """纯点号限定名(如 `fake.method`)也要参与防杜撰核对——图上有、正文/dossier 都没有则 fail。"""
    r = _run(_mk(tmp_path, ["§20.1", "fake.method"]))
    assert r.returncode == 1 and "fake.method" in r.stdout


# ── FIX-ROUND-3 ──────────────────────────────────────────────────────────

def test_svg_pretty_tspan_parallel_symbols_no_false_glue(tmp_path):
    """同一 <text> 内两个并列的合法符号，各占一个 tspan、pretty-print 换行缩进
    (如 illustrator 生成器的输出风格)——不应被无分隔拼接粘成一个从未出现过的
    假 token(如 "forward_impl_get_fia_params")而误判杜撰；应保留词边界，
    两者各自都是 dossier 原文子串，exit 0。"""
    ch = tmp_path / "ch20-foo"
    (ch / "diagrams").mkdir(parents=True)
    (ch / "narrative").mkdir()
    (ch / "diagrams" / "chapter-map.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">\n'
        '  <text x="0" y="0">\n'
        '    <tspan>forward_impl</tspan>\n'
        '    <tspan>_get_fia_params</tspan>\n'
        '  </text>\n'
        '</svg>\n',
        encoding="utf-8")
    (ch / "narrative" / "chapter.md").write_text(
        "# 第 20 章 X\n\nhook。\n\n![本章地图](../diagrams/chapter-map.png)\n\n只想看结论,跳 §20.2。\n\n"
        "## 20.1 入口\n## 20.2 分流\n",
        encoding="utf-8")
    (ch / "dossier.json").write_text(
        json.dumps({"mechanisms": [{"anchors": ["forward_impl", "_get_fia_params"]}]}),
        encoding="utf-8")
    r = _run(ch)
    assert r.returncode == 0
    assert "forward_impl_get_fia_params" not in r.stdout

def test_svg_pretty_tspan_adjacent_fake_still_caught(tmp_path):
    """同一 <text> 内一个真符号紧挨一个杜撰符号(pretty-print 换行缩进)——空格拼接
    不能把这变成"放过杜撰"的漏洞：应仍精确报出干净的杜撰 token(`_evil_fake`)，
    而不是被粘连成不知所云的复合串。"""
    ch = tmp_path / "ch20-foo"
    (ch / "diagrams").mkdir(parents=True)
    (ch / "narrative").mkdir()
    (ch / "diagrams" / "chapter-map.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">\n'
        '  <text x="0" y="0">\n'
        '    <tspan>forward_impl</tspan>\n'
        '    <tspan>_evil_fake</tspan>\n'
        '  </text>\n'
        '</svg>\n',
        encoding="utf-8")
    (ch / "narrative" / "chapter.md").write_text(
        "# 第 20 章 X\n\nhook。\n\n![本章地图](../diagrams/chapter-map.png)\n\n只想看结论,跳 §20.2。\n\n"
        "## 20.1 入口\n## 20.2 分流\n",
        encoding="utf-8")
    (ch / "dossier.json").write_text(
        json.dumps({"mechanisms": [{"anchors": ["forward_impl"]}]}), encoding="utf-8")
    r = _run(ch)
    assert r.returncode == 1
    assert "_evil_fake" in r.stdout
    assert "forward_impl_evil_fake" not in r.stdout

def test_natural_language_trailing_punct_not_fabrication(tmp_path):
    """自然语言短语里带句点收尾的普通词(`decode.`)和缩写(`e.g.`)不应触发杜撰核对——
    剥离尾部句点后二者都不再含内部 `.`(或内部 `.` 后随字母的碎片过短)，不入核对。
    (`test_dotted_symbol_fabrication_fail` 已覆盖收窄后真杜撰 `fake.method` 仍被抓——
    此处不再重复，只补验证收窄后不误伤自然语言的这一半。)"""
    r = _run(_mk(tmp_path, ["§20.1", "fast path, e.g. decode."]))
    assert r.returncode == 0
