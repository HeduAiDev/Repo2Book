#!/usr/bin/env python3
"""ch39 取证 — 实跑 pin v3.2.0 python/triton/testing.py:L20-L29 的真 _summarize_statistics。

testing.py 顶部 `from . import language/runtime` 需要已编译的 triton._C(源码树未编译),
无法整模块 import。但 _summarize_statistics 本体只在函数内 `import torch`、只吃一个 times 张量。
故从 pin 源文件按行抽出 L20-L29 这十行**逐字**源码 exec 进独立命名空间再调用——
执行的是 pin 源码本体(非重实现),只是把它与不可用的模块级 import 隔离开。

场景:构造一组含偶发尖峰的 times(ms)——4 次约 1.00ms 的正常轮 + 1 次 5.00ms 的调度尖峰。
对比 mean(被尖峰拖偏)vs median / 20-80 分位(抗离群),复现『全书性能数字为何取分位』。
"""
from pathlib import Path
import torch

SRC = Path("/mnt/e/Laboratory/Repo2Book/instances/triton/source")
lines = (SRC / "python/triton/testing.py").read_text().splitlines()
# L20-L29(1-indexed)= _summarize_statistics 完整函数体
src = "\n".join(lines[19:29])
print("=== pin 源码逐字(testing.py:L20-L29)===")
print(src)
ns = {}
exec(src, ns)
_summarize_statistics = ns["_summarize_statistics"]

# 含尖峰的一组实测(ms):4 稳态 + 1 尖峰,顺序打散
times = torch.tensor([1.02, 0.98, 5.00, 1.01, 0.99], dtype=torch.float)
print("\n=== 输入 times(ms)===")
print(list(times.tolist()))

print("\n=== 真源码输出 ===")
mean = _summarize_statistics(times, quantiles=None, return_mode="mean")
median = _summarize_statistics(times, quantiles=None, return_mode="median")
mn = _summarize_statistics(times, quantiles=None, return_mode="min")
mx = _summarize_statistics(times, quantiles=None, return_mode="max")
# do_bench 默认口径:quantiles=[0.5,0.2,0.8] → 中位 + 20/80 分位
quant = _summarize_statistics(times, quantiles=[0.5, 0.2, 0.8], return_mode="mean")
print(f"return_mode=mean   -> {mean:.4f} ms   (被 5.00 尖峰拖偏)")
print(f"return_mode=median -> {median:.4f} ms  (抗尖峰)")
print(f"return_mode=min    -> {mn:.4f} ms")
print(f"return_mode=max    -> {mx:.4f} ms  (=尖峰本身)")
print(f"quantiles=[0.5,0.2,0.8] -> {[round(x,4) for x in quant]} ms  (中位/20分位/80分位)")

import json
OUT = Path(__file__).with_name("summarize.json")
OUT.write_text(json.dumps({
    "source": "pin v3.2.0 python/triton/testing.py:L20-L29 _summarize_statistics (逐字 exec)",
    "input_times_ms_clean": [1.02, 0.98, 5.00, 1.01, 0.99],
    "input_times_ms_raw": times.tolist(),
    "mean": round(mean, 4), "median": round(median, 4),
    "min": round(mn, 4), "max": round(mx, 4),
    "quantiles_0.5_0.2_0.8": [round(x, 4) for x in quant],
}, indent=2))
print(f"\nWROTE {OUT}")
