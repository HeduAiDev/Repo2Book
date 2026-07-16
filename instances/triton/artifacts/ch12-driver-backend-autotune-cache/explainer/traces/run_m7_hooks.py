#!/usr/bin/env python3
"""m7 — reset_to_zero / restore_value 的 pre/post hook 忠实复刻（host，无需 torch）。

复刻 python/triton/runtime/autotuner.py:L66-L83：
    def _pre_hook(kwargs, reset_only=False):
        for name in self.reset_to_zero:
            kwargs[name].zero_()
        if not reset_only:
            self.restore_copies = {name: kwargs[name].clone() for name in self.restore_value}
    def _post_hook(kwargs, exception):
        for name in self.restore_value:
            kwargs[name].copy_(self.restore_copies[name])
        self.restore_copies = {}

外加 _bench.kernel_call 的调用序（L147-163）：pre_hook -> fn.run -> post_hook。
用 stand-in 张量（list 承载值）忠实实现 zero_/clone/copy_，演示两个 config 连续 bench
时 acc 每轮从 0 起（不被上轮污染）、被 kernel 改写的 x 每轮还原到 5。
"""


class Tensor:
    """stand-in 张量：忠实 zero_/clone/copy_ 语义。"""
    def __init__(self, val):
        self.val = list(val)

    def zero_(self):
        self.val = [0 for _ in self.val]

    def clone(self):
        return Tensor(self.val)

    def copy_(self, other):
        self.val = list(other.val)

    def __repr__(self):
        return str(self.val)


reset_to_zero = ["acc"]
restore_value = ["x"]
restore_copies = {}


def pre_hook(kwargs, reset_only=False):
    for name in reset_to_zero:
        kwargs[name].zero_()
    if not reset_only:
        restore_copies.clear()
        for name in restore_value:
            restore_copies[name] = kwargs[name].clone()


def post_hook(kwargs, exception=None):
    for name in restore_value:
        kwargs[name].copy_(restore_copies[name])
    restore_copies.clear()


def kernel_run(kwargs):
    """模拟一个带副作用的 kernel：往 acc 累加、原地改写 x。"""
    kwargs["acc"].val = [v + 10 for v in kwargs["acc"].val]   # acc += 10
    kwargs["x"].val = [999 for _ in kwargs["x"].val]          # x 被覆盖


# 共享的 kwargs（autotune 对每个 config 复用同一组张量）
kwargs = {"acc": Tensor([7]), "x": Tensor([5])}   # acc 初值故意非 0，证明清零生效

print("m7 reset_to_zero / restore_value 钩子\n" + "=" * 60)
print(f"起始（上一 kernel 遗留）: acc={kwargs['acc']}  x={kwargs['x']}")
print()

for cfg in ("configA", "configB"):
    print(f"--- bench {cfg} ---")
    pre_hook(kwargs)
    print(f"    pre_hook 后 : acc={kwargs['acc']}  x={kwargs['x']}  restore_copies={{'x': {restore_copies['x']}}}")
    kernel_run(kwargs)
    print(f"    kernel 后   : acc={kwargs['acc']}  x={kwargs['x']}  (x 被 kernel 覆盖为 999)")
    post_hook(kwargs)
    print(f"    post_hook 后: acc={kwargs['acc']}  x={kwargs['x']}  (x 已还原)")
    print()

print("结论：两个 config 的 pre_hook 后 acc 恒为 [0]、x 恒为 [5]——面对完全相同的输入。")
