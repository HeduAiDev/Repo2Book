# SOURCE: vllm/model_executor/custom_op.py
# ch25 消费面：PluggableLayer（mla.py:L35 的 @PluggableLayer.register(
# "multi_head_latent_attention") 注册位——OOT 平台后门，must_keep 相关）
# —— 减法子集：register 装饰器 + registry 查表逐字。CustomOp 的派发面
#（dispatch_forward/enabled/default_on）本章无消费点——ch19/ch23 域。
from __future__ import annotations

import torch


# SOURCE: vllm/model_executor/custom_op.py op_registry —— 逐字位
op_registry: dict = {}


# SOURCE: vllm/model_executor/custom_op.py PluggableLayer —— 减法子集
#   （真实继承 torch.nn.Module——可替换层本体）
class PluggableLayer(torch.nn.Module):
    """A layer that can be replaced by a custom operation from an OOT plugin.

    Framework code instantiates these layers as placeholders. At import time,
    an OOT plugin can call `MyLayer.register("name")` to register a
    replacement class. The model loader checks the registry and uses the
    replacement class instead of the placeholder.
    """

    @staticmethod
    def register(name: str):
        # SOURCE: vllm/model_executor/custom_op.py PluggableLayer.register
        def decorator(cls):
            op_registry[name] = cls
            return cls

        return decorator
