"""embed_verbatim 检查(exp-2026-07-18-02)——dossier embed_excerpts 与 pin blob 逐字比对。

SDD: docs/superpowers/specs/2026-07-18-lint-dossier-embed-verbatim.md
三层闭环里补『pin ↔ dossier』这一环(dossier ↔ 正文由 lint_fidelity 已管)。
"""
import json
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_dossier import lint_dossier

GOOD_MECH = {
    "id": "m1", "name": "示例机制", "kind": "dataflow",
    "source_anchors": ["pkg/mod.py:L1-L2"], "needs_figure": False,
    "needs_worked_example": False, "difficulty": "supporting",
}

# 9 行固定源文件(缩进承载语义:2/4 级缩进都有)
SRC = (
    "def f(x):\n"
    "    y = x + 1\n"
    "    # comment kept\n"
    "    z = y * 2\n"
    "    return z\n"
    "\n"
    "\n"
    "def g(a, b):\n"
    "    return a - b\n"
)


def _mk(tmp, excerpts, with_source=True, src_text=SRC):
    """instances 形状:<inst>/artifacts/ch01 + <inst>/source/pkg/mod.py。"""
    inst = tmp / "inst"
    ch = inst / "artifacts" / "ch01"
    (ch / "dossier").mkdir(parents=True)
    if with_source:
        (inst / "source" / "pkg").mkdir(parents=True)
        (inst / "source" / "pkg" / "mod.py").write_text(src_text, encoding="utf-8")
    doc = {"mechanisms": [dict(GOOD_MECH)], "embed_excerpts": excerpts}
    (ch / "dossier" / "dossier.json").write_text(
        json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return ch


def _blocking(res):
    return [i for k, v in res.items() if k != "warn" for i in v]


# ---------- 1. 正例·全量(容忍 tab/行尾空格差) ----------
def test_full_verbatim_ok(tmp_path):
    code = ("def f(x):\n"
            "\ty = x + 1   \n"          # tab 缩进 + 行尾空格:归一后应等价(expandtabs=8 恰与 pin 不同则应报——
            "    # comment kept\n"      # 这里 pin 是 4 空格,tab→8 空格会不等;故本行用回 4 空格版本
            "    z = y * 2\n"
            "    return z")
    # tab 行单独验证归一语义:pin 第 2 行是 4 空格缩进,tab 展开为 8 空格≠4 空格→缩进差必须抓。
    # 因此"正例"里不放 tab 缩进差,只放行尾空格差。
    code = ("def f(x):\n"
            "    y = x + 1   \n"
            "    # comment kept\n"
            "    z = y * 2\n"
            "    return z")
    ch = _mk(tmp_path, [{"path": "pkg/mod.py", "lines": "L1-L5", "code": code, "elide": []}])
    assert _blocking(lint_dossier(str(ch))) == []


# ---------- 2. 负例·默写旧版(同位行差一个标识符) ----------
def test_full_identifier_drift_blocks(tmp_path):
    code = ("def f(x):\n"
            "    y = x + 2\n"           # pin 是 x + 1
            "    # comment kept\n"
            "    z = y * 2\n"
            "    return z")
    ch = _mk(tmp_path, [{"path": "pkg/mod.py", "lines": "L1-L5", "code": code, "elide": []}])
    hits = [i for i in _blocking(lint_dossier(str(ch))) if "embed" in str(i) or "pin" in str(i)]
    assert hits, "默写漂移必须 blocking"
    assert any("x + 2" in i and "x + 1" in i for i in hits), "报文须含 dossier/pin 两侧内容"


# ---------- 3. 正例·省略子集(按序) ----------
def test_subset_in_order_ok(tmp_path):
    code = ("def f(x):\n"
            "    z = y * 2\n"
            "    return z")
    ch = _mk(tmp_path, [{"path": "pkg/mod.py", "lines": "L1-L9", "code": code,
                         "elide": ["中间两行省略"]}])
    assert _blocking(lint_dossier(str(ch))) == []


# ---------- 4. 负例·子集含杜撰行(阶段1:warn 人核,假阳清零后升 blocking) ----------
def test_subset_fabricated_line_warns(tmp_path):
    code = ("def f(x):\n"
            "    w = magic(y)\n"        # pin 区间内不存在
            "    return z")
    ch = _mk(tmp_path, [{"path": "pkg/mod.py", "lines": "L1-L9", "code": code, "elide": []}])
    res = lint_dossier(str(ch))
    assert _blocking(res) == []
    assert any("magic" in w for w in res["warn"])


# ---------- 5. 负例·乱序(阶段1:warn 人核) ----------
def test_subset_out_of_order_warns(tmp_path):
    code = ("        return z\n"
            "def f(x):")
    ch = _mk(tmp_path, [{"path": "pkg/mod.py", "lines": "L1-L9", "code": code, "elide": []}])
    res = lint_dossier(str(ch))
    assert _blocking(res) == []
    assert any("匹配不到" in w for w in res["warn"])


# ---------- 6. 豁免·注记行 ----------
def test_note_lines_skipped(tmp_path):
    code = ("def f(x):\n"
            "# … 省略:中间推导 …\n"
            "    return z\n"
            "# SOURCE: pkg/mod.py:L1-L9")
    ch = _mk(tmp_path, [{"path": "pkg/mod.py", "lines": "L1-L9", "code": code, "elide": []}])
    assert _blocking(lint_dossier(str(ch))) == []


# ---------- 7. 豁免·primer 论文条目(无文件 path) ----------
def test_paper_entry_skipped(tmp_path):
    ch = _mk(tmp_path, [
        {"anchor": "§2.1", "formula": "m_i = max(...)", "paper": "arXiv:2205.14135",
         "source": "论文", "note": "在线 softmax 递推"},
        {"path": "pkg/mod.py", "lines": "n/a:非仓库文件", "code": "x = 1", "elide": []},
    ])
    res = lint_dossier(str(ch))
    assert _blocking(res) == []
    assert not any("embed" in w for w in res["warn"]), "合法形态,不该告警"


# ---------- 8. 边界:path 不在 pin→warn 人核(前瞻/跨仓/论文包引用合法形态);source/ 缺→warn 跳过 ----------
def test_missing_file_warns(tmp_path):
    ch = _mk(tmp_path, [{"path": "pkg/nope.py", "lines": "L1-L2", "code": "x", "elide": []}])
    res = lint_dossier(str(ch))
    assert _blocking(res) == []
    assert any("不在 pin" in w for w in res["warn"])


def test_missing_source_warns_only(tmp_path):
    ch = _mk(tmp_path, [{"path": "pkg/mod.py", "lines": "L1-L2", "code": "x", "elide": []}],
             with_source=False)
    res = lint_dossier(str(ch))
    # source/ 不在:锚点检查同款降级,embed 不产 blocking
    embed_blocking = [i for i in _blocking(res) if "pin" in i or "embed" in i]
    assert embed_blocking == []


# ---------- 8b. 行号越界 ----------
def test_range_beyond_eof_blocks(tmp_path):
    ch = _mk(tmp_path, [{"path": "pkg/mod.py", "lines": "L1-L99", "code": "def f(x):", "elide": []}])
    assert any("越界" in i for i in _blocking(lint_dossier(str(ch))))


# ---------- 9. git show 优先(工作区脏、HEAD 干净→不报) ----------
def test_git_show_priority(tmp_path):
    ch = _mk(tmp_path, [{"path": "pkg/mod.py", "lines": "L1-L5", "code":
                         "def f(x):\n    y = x + 1\n    # comment kept\n    z = y * 2\n    return z",
                         "elide": []}])
    src = tmp_path / "inst" / "source"
    try:
        for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                    ["git", "-c", "user.email=t@t", "-c", "user.name=t",
                     "commit", "-qm", "pin"]):
            subprocess.run(cmd, cwd=src, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git unavailable")
    # 弄脏工作区(保持 9 行,免得连累既有 anchor 行号越界检查):HEAD 仍是正确内容
    (src / "pkg" / "mod.py").write_text("DIRTY\n" * 9, encoding="utf-8")
    assert _blocking(lint_dossier(str(ch))) == [], "须优先读 HEAD blob 而非脏工作区"
