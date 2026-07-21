"""ch09 explainer 驱动脚本公用脚手架。

把本章 `implementation/`（论文忠实的小型参考实现，纯 NumPy / CPU）加进 sys.path，
并提供统一的 trace 落盘函数。

trace_source: cpu-numpy-reference —— 宿主无昇腾 NPU/CANN，全部数值都是 CPU 上跑
参考实现得到的**结构性数值**（形状、区间、计数、逐点累加值），既不是也不冒充任何
真机性能数字。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_impl_dir() -> Path:
    env = os.environ.get("CH09_IMPL_DIR")
    if env and (Path(env) / "structured_op.py").exists():
        return Path(env)
    for base in [HERE] + list(HERE.parents):
        cand = base / "implementation"
        if (cand / "structured_op.py").exists():
            return cand
    raise RuntimeError("找不到 implementation/structured_op.py（可用环境变量 CH09_IMPL_DIR 指定）")


IMPL_DIR = _find_impl_dir()
if str(IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL_DIR))


def dump(name: str, payload: dict) -> Path:
    """把一次运行的原始输出写成 traces/<name>.json（同时 stdout 打一份）。"""
    out = HERE / f"{name}.json"
    payload = dict(payload)
    payload.setdefault("trace_source", "cpu-numpy-reference")
    payload.setdefault(
        "host_note",
        "host 无昇腾 NPU/CANN；本 trace 全部为 CPU/NumPy 参考实现的结构性数值，无真机数字",
    )
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    out.write_text(text, encoding="utf-8")
    print(text)
    return out
