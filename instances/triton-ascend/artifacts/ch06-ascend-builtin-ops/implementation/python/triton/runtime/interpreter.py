# 基座 interpreter 模块的最小子集——sort()/cast() 只用 isinstance(_builder,
# InterpreterBuilder) 判断"是否在 interpreter 模式下运行"，本章不复现 interpreter
# 的完整 numpy 语义（那属于 python/triton/runtime/ascend_interpreter.py，宿主无
# 昇腾 NPU/CANN，见 INSTANCE.md），只留这个类型标记供 isinstance 分支判断为 False。
#
# SOURCE: python/triton/runtime/interpreter.py（类型标识节选）
class InterpreterBuilder:
    pass
