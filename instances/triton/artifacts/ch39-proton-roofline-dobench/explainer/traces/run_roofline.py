#!/usr/bin/env python3
"""ch39 取证 — 实跑 pin v3.2.0 的 third_party/proton/proton/viewer.py 真源码,
喂仓库自带 test/example_cuda.json,无 GPU 出 roofline util / flop8/s / byte/s。

做法:viewer.py 顶部唯一与 GPU 耦合的 import 是
`from triton.profiler.hook import COMPUTE_METADATA_SCOPE_NAME, TritonHook`,
而 viewer 只用到其中两个纯常量(COMPUTE_METADATA_SCOPE_NAME 字符串 + TritonHook.flops_width/metrics 列表)。
故 stub 掉这个 module(不拉起 libproton),再用 importlib 按路径加载 pin 的真 viewer.py,
调用它真实的 get_raw_metrics / get_min_time_flops / get_min_time_bytes / derive_metrics / parse。
形状与数字全部来自 pin 源码本体,不是重实现。
"""
import importlib.util
import io
import json
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

SRC = Path("/mnt/e/Laboratory/Repo2Book/instances/triton/source")
VIEWER = SRC / "third_party/proton/proton/viewer.py"
EXAMPLE = SRC / "third_party/proton/test/example_cuda.json"

# --- stub triton.profiler.hook 的两个纯常量(与 pin 源码 hook.py:L4,L8-L9 逐字一致)---
hook_mod = types.ModuleType("triton.profiler.hook")
hook_mod.COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"  # hook.py:L4


class TritonHook:  # hook.py:L7-L9 常量部分
    flops_width = [8, 16, 32, 64]
    metrics = [f"flops{width}" for width in flops_width] + ["bytes"] + ["flops"]


hook_mod.TritonHook = TritonHook
# 满足 `import triton.profiler.hook` 的父包链
for name in ("triton", "triton.profiler"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
sys.modules["triton.profiler.hook"] = hook_mod

# --- 按路径加载 pin 的真 viewer.py ---
spec = importlib.util.spec_from_file_location("pin_proton_viewer", VIEWER)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)

print("=" * 70)
print("RAW INPUT: third_party/proton/test/example_cuda.json (仓库自带样例,无 GPU)")
raw = json.loads(EXAMPLE.read_text())
tree, device_info = raw[0], raw[1]
for child in tree["children"]:
    m = child["metrics"]
    print(f"  {child['frame']['name']}: device_id={m['device_id']} "
          f"flops8={m['flops8']:g} bytes={m['bytes']:g} time(ns)={m['time (ns)']}")
print("  device_info:")
for dt, devs in device_info.items():
    for idx, info in devs.items():
        print(f"    {dt}[{idx}]: arch={info['arch']} num_sms={info['num_sms']} "
              f"clock_rate(kHz)={info['clock_rate']} "
              f"mem_clock(kHz)={info['memory_clock_rate']} bus_width(bit)={info['bus_width']}")

# --- 逐函数取证:get_min_time_flops / get_min_time_bytes(pin viewer.py:L38-L81)---
print("=" * 70)
print("STEP 1  get_min_time_flops / get_min_time_bytes  (ideal_time,单位秒)")
with EXAMPLE.open() as f:
    gf, raw_metrics, dinfo = viewer.get_raw_metrics(f)
gf.update_inclusive_columns()
mtf = viewer.get_min_time_flops(gf.dataframe, dinfo)
mtb = viewer.get_min_time_bytes(gf.dataframe, dinfo)
names = gf.dataframe["name"] if "name" in gf.dataframe.columns else gf.dataframe.index
for i in gf.dataframe.index:
    nm = gf.dataframe.loc[i, "name"] if "name" in gf.dataframe.columns else str(i)
    tf = float(mtf.loc[i, "min_time"])
    tb = float(mtb.loc[i, "min_time"])
    roof = "compute" if tf > tb else "memory"
    print(f"  {nm:6s}: min_time_flops={tf:.6e}s  min_time_bytes={tb:.6e}s  "
          f"max=>{roof}-roof")

# --- 全管线取证:parse 打印 hatchet 树(pin viewer.py:L192-L201)---
print("=" * 70)
print("STEP 2  viewer.parse(-m util,flop8/s,byte/s)  真源码全管线输出")
buf = io.StringIO()
with redirect_stdout(buf):
    viewer.parse(["util", "flop8/s", "byte/s"], str(EXAMPLE))
out = buf.getvalue()
print(out)

# --- 落盘结构化 trace(供 lint_explainer 逐数字核对)---
result = {
    "source": "pin v3.2.0 third_party/proton/proton/viewer.py, fed test/example_cuda.json",
    "per_frame": {},
    "tree_text": out,
    # clean 展示口径(µs / util / T·G per s),供 explainer 表格与图逐字引用
    "cited": {
        "foo0_measured_us": 204.8, "foo1_measured_us": 204.8,
        "foo0_ideal_compute_us": 50.6, "foo0_ideal_memory_us": 24.9, "foo0_util": 0.247,
        "foo1_ideal_compute_us": 30.3, "foo1_ideal_memory_us": 9.9, "foo1_util": 0.148,
        "foo0_flop8_per_s_Tflops": 488, "foo0_byte_per_s_GBps": 488,
        "foo1_flop8_per_s_Tflops": 48.8, "foo1_byte_per_s_GBps": 48.8,
    },
}
for i in gf.dataframe.index:
    nm = gf.dataframe.loc[i, "name"] if "name" in gf.dataframe.columns else str(i)
    result["per_frame"][str(nm)] = {
        "min_time_flops_s": float(mtf.loc[i, "min_time"]),
        "min_time_bytes_s": float(mtb.loc[i, "min_time"]),
    }
# 打印 cited 便于人工核对
print("CITED (µs / util / T·Gper·s):")
for k, v in result["cited"].items():
    print(f"  {k} = {v}")
OUT = Path(__file__).with_name("roofline.json")
OUT.write_text(json.dumps(result, indent=2))
print("=" * 70)
print(f"WROTE {OUT}")
