#!/usr/bin/env python3
"""repo2book 实例解析器 —— 让脚本去 vLLM 化、按「当前活动实例」运转。

解析顺序：环境变量 REPO2BOOK_INSTANCE  >  顶层 repo2book.json 的 active_instance  >  "vllm"。
每个实例 = instances/<name>/，自带 repo2book.json（源仓信息/书配置）、artifacts/、book/、trace/。

CLI:  python3 scripts/instance.py [name|dir|artifacts|chapters|diagrams|source|config]
库:   from instance import active_name, artifacts_dir, chapters_glob, ...
（脚本以 `python3 scripts/<x>.py` 运行时，scripts/ 在 sys.path[0]，可直接 `import instance`。）
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _registry():
    p = ROOT / "repo2book.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (ValueError, OSError):
        return {}


_ARGV_INSTANCE_RE = re.compile(r'(?:^|/)instances/([A-Za-z0-9._-]+)/')


def name_from_argv(argv=None):
    """从命令行参数里的 `instances/<name>/…` 路径推断实例名,推不出返回 None。

    动机(exp-2026-07-20-03):linter 常被显式喂一个章目录,而 canonical_prefixes 在 import 期
    就定死了。若此时 active_instance 指向另一本书,前缀完全对不上——lint_fidelity 会报
    「真实源码引用仅 0 处」这种**假 BLOCKING**(triton-ascend 的章 + vllm 的前缀实测如此)。
    命令行里那个路径才是调用者真正的意图,故让它压过全局 active_instance。
    要求形如 instances/<name>/ 的真路径(后面必须还有一层),裸词 "instances" 不算。
    """
    for a in (argv if argv is not None else sys.argv[1:]):
        if not isinstance(a, str) or a.startswith('-'):
            continue
        m = _ARGV_INSTANCE_RE.search(a.replace('\\', '/'))
        if m:
            return m.group(1)
    return None


def active_name():
    return (os.environ.get("REPO2BOOK_INSTANCE")
            or name_from_argv()
            or _registry().get("active_instance")
            or "vllm")


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
    """活动实例的所有章节正文 glob 模式（相对仓库根，linter --all 自行 glob）。

    v3 起章目录在 artifacts-v3/（artifacts/ 是 v2 存量）——`artifacts*` 通配
    两代都扫，否则 --all 对 v3 章节恒空转=假绿灯（exp-2026-09-05，ch27 评审
    发现，与 active_instance 扫错实例是两层叠加盲区）。返回值保持 pattern
    字符串契约（消费方 `glob.glob(instance.chapters_glob())`）。
    """
    return os.path.relpath(instance_dir(name) / "artifacts*" / "ch*" / "narrative" / "chapter.md", ROOT)


def diagrams_glob(name=None):
    """同 chapters_glob：`artifacts*` 通配两代章目录（v2 artifacts/ + v3 artifacts-v3/）。"""
    return os.path.relpath(instance_dir(name) / "artifacts*" / "ch*" / "diagrams" / "*.svg", ROOT)


def canonical_prefix(name=None):
    """正文里规范源码路径的前缀（如 vLLM 实例为 'vllm'）。"""
    return (config(name).get("source") or {}).get("canonical_prefix") or active_name()


def canonical_prefixes(name=None):
    """正文规范源码路径的**全部合法顶层根**（列表）。

    单前缀仓（vLLM: 'vllm'）返回 ['vllm']；多根仓（Triton fork：源码分布在
    python/ lib/ include/ third_party/ 等，无单一前缀）在 repo2book.json 的
    source.canonical_prefixes 里列出真实顶层根。向后兼容：无该字段时退回
    [canonical_prefix()]，故既有实例（vLLM）行为不变。
    """
    src = config(name).get("source") or {}
    lst = src.get("canonical_prefixes")
    if isinstance(lst, list) and lst:
        return list(lst)
    return [canonical_prefix(name)]


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
