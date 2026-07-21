# 基座 get_ascend_devices 模块的最小子集——真实 is_compile_on_910_95 由
# get_ascend_devices()/check_npu_smi_device() 在导入时探测宿主 PCI 设备/npu-smi
# 得到，是纯硬件探测逻辑，与本章 mem_ops/vec_ops 的语言层语义无关。宿主无昇腾 NPU
# （见 INSTANCE.md），这里直接给出 False；测试里如需覆盖 910_95 分支，直接
# monkeypatch `vec_ops.is_compile_on_910_95`（与真实场景下"这台机器是不是 910_95"
# 是同一个模块级布尔开关，语义一致）。
#
# SOURCE: third_party/ascend/../python/triton/tools/get_ascend_devices.py:L55（节选）
is_compile_on_910_95 = False
