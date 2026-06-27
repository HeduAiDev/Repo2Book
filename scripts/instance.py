#!/usr/bin/env python3
"""repo2book 实例解析器 —— 让脚本去 vLLM 化、按「当前活动实例」运转。

解析顺序：环境变量 REPO2BOOK_INSTANCE  >  顶层 repo2book.json 的 active_instance  >  "vllm"。
每个实例 = instances/<name>/，自带 repo2book.json（源仓信息/书配置）、artifacts/、book/、knowledge/、trace/。

CLI:  python3 scripts/instance.py [name|dir|artifacts|chapters|diagrams|source|config]
库:   from instance import active_name, artifacts_dir, chapters_glob, ...
（脚本以 `python3 scripts/<x>.py` 运行时，scripts/ 在 sys.path[0]，可直接 `import instance`。）
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _registry():
    p = ROOT / "repo2book.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (ValueError, OSError):
        return {}


def active_name():
    return os.environ.get("REPO2BOOK_INSTANCE") or _registry().get("active_instance") or "vllm"


def instance_dir(name=None):
    return ROOT / "instances" / (name or active_name())


def config(name=None):
    cfg = instance_dir(name) / "repo2book.json"
    try:
        return json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}
    except (ValueError, OSError):
        return {}


def artifacts_dir(name=None):
    return instance_dir(name) / "artifacts"


def source_dir(name=None):
    return instance_dir(name) / "source"


def book_dir(name=None):
    return instance_dir(name) / "book"


def chapters_glob(name=None):
    """活动实例的所有章节正文（相对仓库根，linter --all 用）。"""
    return os.path.relpath(artifacts_dir(name) / "ch*" / "narrative" / "chapter.md", ROOT)


def diagrams_glob(name=None):
    return os.path.relpath(artifacts_dir(name) / "ch*" / "diagrams" / "*.svg", ROOT)


def canonical_prefix(name=None):
    """正文里规范源码路径的前缀（如 vLLM 实例为 'vllm'）。"""
    return (config(name).get("source") or {}).get("canonical_prefix") or active_name()


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else "name"
    table = {
        "name": active_name(),
        "dir": str(instance_dir()),
        "artifacts": str(artifacts_dir()),
        "chapters": chapters_glob(),
        "diagrams": diagrams_glob(),
        "source": str(source_dir()),
        "config": json.dumps(config(), ensure_ascii=False),
    }
    print(table.get(key, active_name()))
