"""lint_source_grounding 必须用退出码表达 BLOCKING(exp-2026-07-20-06)。

来源:该脚本 `print_report()` 没有返回值、`__main__` 也没有 `sys.exit(...)` ——
**无论有没有 BLOCKING,退出码恒为 0**。它在 CLAUDE.md 里是阻断式门禁,却对任何按退出码判定的
自动化(Lead 的批量扫、workflow 的门禁步)永远显示通过。ch06 的『小结』整节无源码引用,
报告里明明打印了 `🔴 1 BLOCKING`,退出码仍是 0,我的批量扫因此给了假绿。

对照 lint_fidelity.py 的收尾写法:`sys.exit(print_report(...))`,print_report 返回 0/1。
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from lint_source_grounding import lint_source_grounding, print_report

GOOD = "正文引用 vllm/v1/core/sched/scheduler.py:L100-L120。\n"


def _mk(tmp, narrative, impl_refs=3):
    ch = tmp / "instances" / "vllm" / "artifacts" / "ch01-x"
    (ch / "narrative").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text(narrative, encoding="utf-8")
    (ch / "implementation").mkdir()
    (ch / "implementation" / "m.py").write_text(
        "".join(f"# SOURCE: vllm/a{i}.py:L{i}\n" for i in range(impl_refs)), encoding="utf-8")
    return ch


def test_print_report_returns_1_when_blocking(tmp_path):
    ch = _mk(tmp_path, "# 第 1 章　标题\n\n钩子。\n\n## 1.1 空谈\n\n没有任何源码路径。\n")
    res = lint_source_grounding(str(ch))
    assert res["narrative_vllm_refs"], "前提:本该有 BLOCKING"
    assert print_report(res, str(ch)) == 1


def test_print_report_returns_0_when_clean(tmp_path):
    ch = _mk(tmp_path, "# 第 1 章　标题\n\n钩子。\n\n## 1.1 正文\n\n" + GOOD)
    res = lint_source_grounding(str(ch))
    assert print_report(res, str(ch)) == 0


def test_cli_exit_code_nonzero_on_blocking(tmp_path):
    """端到端:命令行跑一遍,退出码必须非 0——这是批量扫真正依赖的东西。"""
    ch = _mk(tmp_path, "# 第 1 章　标题\n\n钩子。\n\n## 1.1 空谈\n\n没有任何源码路径。\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "lint_source_grounding.py"), str(ch)],
                       capture_output=True, text=True)
    assert "BLOCKING" in r.stdout
    assert r.returncode != 0, "打印了 BLOCKING 却退出 0——门禁形同虚设"
