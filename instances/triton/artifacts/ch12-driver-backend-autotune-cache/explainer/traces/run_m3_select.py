#!/usr/bin/env python3
"""m3 — _create_driver 选择逻辑的忠实复刻（host 纯控制流，无需 CUDA）。

复刻 python/triton/runtime/driver.py:L5-L9 的 _create_driver：
    actives = [x.driver for x in backends.values() if x.driver.is_active()]
    if len(actives) != 1:
        raise RuntimeError(f"{len(actives)} active drivers ...")
    return actives[0]()

真实环境里 backends 由 _discover_backends() 建成 {name: Backend(compiler, driver)}，
driver 是各后端 driver.py 里唯一的 concrete 子类。这里用 stand-in 驱动（只带 is_active
+ 名字）忠实复现「筛唯一 is_active」的控制流；is_active 的真值来自各后端自报，
nvidia 为 torch.cuda.is_available() and (torch.version.hip is None)（driver.py:L468）。
"""


class FakeDriver:
    def __init__(self, name, active):
        self.name = name
        self._active = active

    def is_active(self):
        return self._active

    def __call__(self):
        return f"<{self.name}Driver instance>"

    def __repr__(self):
        return f"{self.name}(active={self._active})"


class Backend:
    def __init__(self, driver):
        self.driver = driver


def create_driver(backends):
    """L6-L9 忠实复刻。"""
    actives = [x.driver for x in backends.values() if x.driver.is_active()]
    n = len(actives)
    if n != 1:
        return ("RAISE", n, f"RuntimeError: {n} active drivers ({actives}). There should only be one.")
    return ("OK", n, actives[0]())


scenarios = [
    ("gpu-box  (nvidia 可用, amd 不可用)",
     {"nvidia": Backend(FakeDriver("Cuda", True)),
      "amd":    Backend(FakeDriver("HIP", False))}),
    ("cpu-only (nvidia 与 amd 均不可用)",
     {"nvidia": Backend(FakeDriver("Cuda", False)),
      "amd":    Backend(FakeDriver("HIP", False))}),
    ("双活歧义 (两个后端都自报 active)",
     {"nvidia": Backend(FakeDriver("Cuda", True)),
      "amd":    Backend(FakeDriver("HIP", True))}),
]

print("m3 _create_driver 选择逻辑\n" + "=" * 60)
for label, backends in scenarios:
    status, n_active, result = create_driver(backends)
    print(f"[{label}]")
    print(f"    active driver 数 = {n_active}")
    print(f"    结果: {result}")
    print()
