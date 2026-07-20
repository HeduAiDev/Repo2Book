"""按命令行里的章节路径定位实例(exp-2026-07-20-03)。

来源:active_instance=vllm 时对 triton-ascend 的章跑 lint_fidelity,前缀用的是 vllm 的,
正文里 `third_party/ascend/...`、`python/triton/...` 一个都匹配不上 → 报「真实源码引用仅 0 处
(需 >= 5)」的**假 BLOCKING**。三个消费前缀的 linter(fidelity/source_grounding/chapter_structure)
都在 import 期算 _SRC_PREFIXES,故只能在 instance.active_name() 这一层解决。

优先级:REPO2BOOK_INSTANCE(显式覆盖) > argv 里的 instances/<name>/ 路径 > active_instance > vllm。
--all 之类不带路径的调用不受影响,仍走 active_instance。
"""
import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _fresh(monkeypatch, argv, env=None):
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delenv("REPO2BOOK_INSTANCE", raising=False)
    if env:
        monkeypatch.setenv("REPO2BOOK_INSTANCE", env)
    import instance
    return importlib.reload(instance)


def test_argv_chapter_dir_wins_over_active(monkeypatch):
    m = _fresh(monkeypatch, ["lint_fidelity.py",
                             "instances/triton-ascend/artifacts/ch05-explicit-memory-hierarchy"])
    assert m.active_name() == "triton-ascend"


def test_argv_chapter_file_path(monkeypatch):
    m = _fresh(monkeypatch, ["lint_chapter_structure.py",
                             "instances/triton/artifacts/ch01-x/narrative/chapter.md"])
    assert m.active_name() == "triton"


def test_absolute_path_works(monkeypatch):
    m = _fresh(monkeypatch, ["x.py", "/mnt/e/Laboratory/Repo2Book/instances/triton-ascend/artifacts/ch05/"])
    assert m.active_name() == "triton-ascend"


def test_no_path_falls_back_to_active(monkeypatch):
    """--all 不带路径 → 仍用 active_instance,行为不变。"""
    m = _fresh(monkeypatch, ["lint_punct.py", "--all"])
    import json
    active = json.loads((pathlib.Path(m.ROOT) / "repo2book.json").read_text())["active_instance"]
    assert m.active_name() == active


def test_env_var_still_wins(monkeypatch):
    """显式环境变量是最高优先级,argv 不得夺权。"""
    m = _fresh(monkeypatch, ["lint_fidelity.py", "instances/triton-ascend/artifacts/ch05"], env="vllm")
    assert m.active_name() == "vllm"


def test_flag_that_merely_contains_instances_is_ignored(monkeypatch):
    """只有形如 instances/<name>/ 的真路径才算数,不能被裸词误触发。"""
    m = _fresh(monkeypatch, ["lint_punct.py", "--all", "instances"])
    import json
    active = json.loads((pathlib.Path(m.ROOT) / "repo2book.json").read_text())["active_instance"]
    assert m.active_name() == active
