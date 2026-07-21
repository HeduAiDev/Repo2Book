"""ch07 explainer 驱动脚本共用的环境装配。

复用 tests/conftest.py 的 `_build_env()`——它把 implementation/ 下的精简版按**规范
模块名**(triton.language.extra.cann.*/triton.language.*)装进 sys.modules，并用
FakeBuilder 站在真实 `ir.builder`(C++ 绑定，host 无昇腾 NPU/CANN 工具链)位置上。
所有 trace 都在 host 上纯 Python 跑出来，不涉及真机数值。
"""
import importlib.util
import json
import sys
from pathlib import Path

CH_DIR = Path(__file__).resolve().parents[2]
TESTS_DIR = CH_DIR / "tests"


def _conftest():
    spec = importlib.util.spec_from_file_location("ch07_conftest", TESTS_DIR / "conftest.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ch07_conftest"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_env(simt_enabled=False):
    return _conftest()._build_env(simt_enabled=simt_enabled)


def dump(obj, name):
    out = Path(__file__).resolve().parent / name
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    print(f"\n[trace written] {out}")
