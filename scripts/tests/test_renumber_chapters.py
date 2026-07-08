import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import renumber_chapters as rc

MD_A = """# 第 1 章 开篇
见[第 2 章：乙](../ch02-beta/narrative/chapter.md)与[第 3 章：丙](../../ch03-gamma/narrative/chapter.md)。
正文提到第 2 章与第 3 章的内容。
"""


def _mk(tmp):
    """迷你实例:ch01-alpha / ch02-beta / ch03-gamma,含链接/裸章号/JSON 引用/roadmap 键。"""
    inst = tmp / "instances" / "mini"
    for d, md in [("ch01-alpha", MD_A), ("ch02-beta", "# 乙\n"), ("ch03-gamma", "# 丙\n")]:
        (inst / "artifacts" / d / "narrative").mkdir(parents=True)
        (inst / "artifacts" / d / "narrative" / "chapter.md").write_text(md, encoding="utf-8")
        (inst / "artifacts" / d / "reviews").mkdir(parents=True)
        (inst / "artifacts" / d / "reviews" / "run-ledger.json").write_text(
            json.dumps({"chapter_id": d[:4]}), encoding="utf-8")
    (inst / "book" / "cartography").mkdir(parents=True)
    (inst / "book" / "assets" / "roadmap").mkdir(parents=True)
    (inst / "book" / "bible").mkdir(parents=True)
    (inst / "trace").mkdir(parents=True)
    (inst / "book" / "assets" / "roadmap" / "roadmap.py").write_text(
        '"ch02": ("s", "乙"),\n"ch03": ("s", "丙"),\n', encoding="utf-8")
    (inst / "book" / "bible" / "concepts.json").write_text(
        json.dumps({"乙概念": "ch02"}), encoding="utf-8")
    (inst / "trace" / "state.json").write_text(json.dumps({"ch02": {"s": 1}}), encoding="utf-8")
    (inst / "INSTANCE.md").write_text("现状:第 2 章已交付\n", encoding="utf-8")
    return inst


SWAP = [{"old_dir": "ch03-gamma", "new_id": "ch02"}, {"old_dir": "ch02-beta", "new_id": "ch03"}]


def test_moves_and_simultaneous_swap(tmp_path):
    inst = _mk(tmp_path)
    rc.apply(inst, rc.parse_moves(SWAP), dry_run=False)
    assert (inst / "artifacts" / "ch02-gamma").is_dir()
    assert (inst / "artifacts" / "ch03-beta").is_dir()
    assert not (inst / "artifacts" / "ch02-beta").exists()
    md = (inst / "artifacts" / "ch01-alpha" / "narrative" / "chapter.md").read_text(encoding="utf-8")
    # 链接路径规范化为 ../../ 且目录名/文字章号同步交换
    assert "(../../ch03-beta/narrative/chapter.md)" in md
    assert "(../../ch02-gamma/narrative/chapter.md)" in md
    assert "[第 3 章：乙]" in md and "[第 2 章：丙]" in md
    assert "第 3 章与第 2 章的内容" in md            # 裸文字同步互换


def test_json_and_config_rewrites(tmp_path):
    inst = _mk(tmp_path)
    rc.apply(inst, rc.parse_moves(SWAP), dry_run=False)
    assert json.loads((inst / "book" / "bible" / "concepts.json").read_text(encoding="utf-8")) == {"乙概念": "ch03"}
    assert "ch02" in json.loads((inst / "trace" / "state.json").read_text(encoding="utf-8")) or \
           "ch03" in json.loads((inst / "trace" / "state.json").read_text(encoding="utf-8"))
    assert json.loads((inst / "trace" / "state.json").read_text(encoding="utf-8")) == {"ch03": {"s": 1}}
    rp = (inst / "book" / "assets" / "roadmap" / "roadmap.py").read_text(encoding="utf-8")
    assert '"ch03": ("s", "乙")' in rp and '"ch02": ("s", "丙")' in rp
    rl = json.loads((inst / "artifacts" / "ch03-beta" / "reviews" / "run-ledger.json").read_text(encoding="utf-8"))
    assert rl["chapter_id"] == "ch03"
    assert "第 3 章已交付" in (inst / "INSTANCE.md").read_text(encoding="utf-8")


def test_dry_run_touches_nothing(tmp_path):
    inst = _mk(tmp_path)
    rep = rc.apply(inst, rc.parse_moves(SWAP), dry_run=True)
    assert (inst / "artifacts" / "ch02-beta").is_dir()
    assert rep.planned_moves == 2 and rep.files_changed >= 3


def test_idempotent_second_apply(tmp_path):
    inst = _mk(tmp_path)
    rc.apply(inst, rc.parse_moves(SWAP), dry_run=False)
    rep2 = rc.apply(inst, rc.parse_moves(SWAP), dry_run=False)
    assert rep2.skipped_moves == 2 and rep2.files_changed == 0


def test_validator_catches_dangling(tmp_path):
    inst = _mk(tmp_path)
    bad = inst / "artifacts" / "ch01-alpha" / "narrative" / "chapter.md"
    bad.write_text(bad.read_text(encoding="utf-8") + "\n[坏](../../ch09-nope/narrative/chapter.md)\n", encoding="utf-8")
    probs = rc.validate(inst)
    assert any("ch09-nope" in p for p in probs)


def test_insert_generates_cascade_plan(tmp_path):
    inst = _mk(tmp_path)
    plan = rc.build_insert_plan(inst, new_slug="delta", before_dir="ch02-beta")
    m = {x["old_dir"]: x["new_id"] for x in plan["moves"]}
    assert m == {"ch02-beta": "ch03", "ch03-gamma": "ch04"}
    assert plan["new_chapter_dir"] == "ch02-delta"


def test_plan_archive_file_not_rewritten(tmp_path):
    inst = _mk(tmp_path)
    plan_file = inst / "book" / "cartography" / "renumber-x.json"
    plan_file.write_text(json.dumps({"moves": SWAP}), encoding="utf-8")
    rc.apply(inst, rc.parse_moves(SWAP), dry_run=False)
    assert json.loads(plan_file.read_text(encoding="utf-8")) == {"moves": SWAP}


def test_diagrams_gen_script_rewritten(tmp_path):
    """diagrams/gen_x.py 是章图的真相源(SVG 由它生成):§N.M 徽标与 chNN-slug 目录名须同批重写。"""
    inst = tmp_path / "instances" / "mini2"
    diagrams = inst / "artifacts" / "ch19-foo" / "diagrams"
    diagrams.mkdir(parents=True)
    (diagrams / "gen_x.py").write_text(
        'BADGE = "§19.2"\n# see ch19-foo/diagrams for source\n', encoding="utf-8")
    move = [{"old_dir": "ch19-foo", "new_id": "ch20"}]
    rc.apply(inst, rc.parse_moves(move), dry_run=False)
    moved = inst / "artifacts" / "ch20-foo" / "diagrams" / "gen_x.py"
    assert moved.is_file()
    content = moved.read_text(encoding="utf-8")
    assert '"§20.2"' in content
    assert "ch20-foo" in content
    assert "ch19-foo" not in content
    assert "§19.2" not in content


def test_papers_map_arxiv_ref_untouched(tmp_path):
    """papers-map.json 里的论文引用"arXiv:2210.17323 §3.1"不是章内小节徽标,§N.M 规则现仅限
    diagrams/*.py 自身章节徽标——即便 ch03 恰好在本次 moves 里被搬走,也不该被误伤。"""
    inst = tmp_path / "instances" / "mini-papersmap"
    (inst / "artifacts" / "ch03-quant" / "narrative").mkdir(parents=True)
    (inst / "artifacts" / "ch03-quant" / "narrative" / "chapter.md").write_text("# 量化\n", encoding="utf-8")
    (inst / "book" / "cartography").mkdir(parents=True)
    papers_map = inst / "book" / "cartography" / "papers-map.json"
    payload = {"quant": {"paper_note": "GPTQ 逐列 Hessian OBQ(arXiv:2210.17323 §3.1)"}}
    papers_map.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    move = [{"old_dir": "ch03-quant", "new_id": "ch05"}]
    rc.apply(inst, rc.parse_moves(move), dry_run=False)
    assert json.loads(papers_map.read_text(encoding="utf-8")) == payload


def test_chapter_self_reference_anchor_untouched(tmp_path):
    """正文自引用锚点"[§2.9](#anchor)"是章内小节链接,不是 diagrams 徽标;即便 ch02 在本次
    moves 里(SWAP: ch02→ch03),也不该被 §N.M 规则改写——否则会与未重写的 `## 2.9` 标题错位。"""
    inst = _mk(tmp_path)
    f = inst / "artifacts" / "ch02-beta" / "narrative" / "chapter.md"
    f.write_text(f.read_text(encoding="utf-8") + "\n参见[§2.9](#anchor)。\n", encoding="utf-8")
    rc.apply(inst, rc.parse_moves(SWAP), dry_run=False)
    moved = inst / "artifacts" / "ch03-beta" / "narrative" / "chapter.md"
    assert "[§2.9](#anchor)" in moved.read_text(encoding="utf-8")


def test_diagram_own_badge_scoped_and_zero_pad_normalized(tmp_path):
    """某章旧号 19 的 diagrams/gen_x.py 内同时含论文引用"§3.1"(N=3≠19,原样不变)与本章自身
    徽标"§19.2"(重写为"§20.2");附带验证零填充"§019.2"也能经 int() 归一命中同一条本章旧号。"""
    inst = tmp_path / "instances" / "mini-paperref"
    diagrams = inst / "artifacts" / "ch19-foo" / "diagrams"
    diagrams.mkdir(parents=True)
    (diagrams / "gen_x.py").write_text(
        'PAPER_NOTE = "cf. §3.1"\n'
        'BADGE = "§19.2"\n'
        'BADGE_PADDED = "§019.2"\n',
        encoding="utf-8")
    move = [{"old_dir": "ch19-foo", "new_id": "ch20"}]
    rc.apply(inst, rc.parse_moves(move), dry_run=False)
    content = (inst / "artifacts" / "ch20-foo" / "diagrams" / "gen_x.py").read_text(encoding="utf-8")
    assert '"cf. §3.1"' in content          # 论文引用(N=3≠19)原样不变
    assert content.count('"§20.2"') == 2    # BADGE 与零填充 BADGE_PADDED 均命中本章旧号 19
    assert "019" not in content
    assert "§19.2" not in content
