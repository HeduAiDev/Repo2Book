"""测试固定活动实例，与「当前在写哪本书」解耦。

多本书共存后，`active_instance` 会随生产切换（如切到 triton）。而 lint_* 脚本在
**import 时**就按活动实例绑定源码前缀（_SRC_PREFIXES），若测试固件用的是 vLLM 路径
（vllm/…），活动实例一变这些测试就假失败。此处在测试收集前把实例钉到 vllm，让用
vLLM 固件的测试对「谁是活动书」免疫。需要测别的实例的测试可自行 monkeypatch 覆盖。
"""
import os

os.environ.setdefault("REPO2BOOK_INSTANCE", "vllm")
