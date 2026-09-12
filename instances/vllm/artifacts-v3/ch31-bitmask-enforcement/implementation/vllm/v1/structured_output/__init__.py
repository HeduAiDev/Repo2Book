# SOURCE: vllm/v1/structured_output/__init__.py
# v3 ch31 脊柱①载体说明：真实 StructuredOutputManager 就住在
# vllm/v1/structured_output/__init__.py（L35 起）。本镜像将类体放在同包的
# manager.py、本文件退化为同款 re-export——纯布局适配（fidelity lint 不扫
# __init__.py；v2 ch31/ch32 的 structured_output_manager.py 同款处理），
# 类内全部 # SOURCE 锚点仍指向真实文件 vllm/v1/structured_output/__init__.py
# 的 v0.27.1 行号。减法说明随类体在 manager.py。
from vllm.v1.structured_output.manager import StructuredOutputManager

__all__ = ["StructuredOutputManager"]
