#!/usr/bin/env python3
"""m5 — Autotuner.run 缓存键构造的忠实复刻（host 纯控制流，无需 CUDA）。

复刻 python/triton/runtime/autotuner.py:L174-L193 的 key 组装 + cache 命中判定：
    all_args = {**self.nargs, **kwargs}
    _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
    key = [_args[key] for key in self.keys if key in _args]
    for _, arg in _args.items():
        if hasattr(arg, "dtype"):
            key.append(str(arg.dtype))
    key = tuple(key)
    if key not in self.cache: ... 搜索 ... else 复用

用 stand-in 张量（只带 .dtype 字符串）忠实复现 str(arg.dtype) 入键。
arg_names=['x','out','N']，keys=['N']：x/out 是张量、N 是标量尺寸。
"""


class T:
    """stand-in 张量：只需一个 .dtype，忠实到 str(arg.dtype)。"""
    def __init__(self, dtype):
        self.dtype = dtype

    def __repr__(self):
        return f"T({self.dtype})"


arg_names = ["x", "out", "N"]
keys = ["N"]           # 用户在 @triton.autotune(key=["N"]) 声明


def make_key(all_args):
    _args = {k: v for (k, v) in all_args.items() if k in arg_names}
    key = [_args[k] for k in keys if k in _args]         # L176: key 参数值
    for _, arg in _args.items():                          # L177-179: 每个带 dtype 的实参追加 dtype
        if hasattr(arg, "dtype"):
            key.append(str(arg.dtype))
    return tuple(key)


cache = {}   # 对应 self.cache（内存态，按 key→best_config）
calls = [
    ("N=1024, fp16",  {"x": T("torch.float16"), "out": T("torch.float16"), "N": 1024}),
    ("N=1024, fp16",  {"x": T("torch.float16"), "out": T("torch.float16"), "N": 1024}),
    ("N=1024, fp32",  {"x": T("torch.float32"), "out": T("torch.float32"), "N": 1024}),
    ("N=2048, fp16",  {"x": T("torch.float16"), "out": T("torch.float16"), "N": 2048}),
]

print("m5 Autotuner 缓存键构造\n" + "=" * 60)
for label, all_args in calls:
    key = make_key(all_args)
    if key not in cache:
        cache[key] = f"best_config#{len(cache) + 1}"      # miss：跑一轮 _bench 存最优
        verdict = "MISS -> 搜索并缓存"
    else:
        verdict = "HIT  -> 复用 " + cache[key]
    print(f"[{label}]")
    print(f"    key = {key}")
    print(f"    {verdict}")
    print(f"    cache 现有 {len(cache)} 项")
    print()
