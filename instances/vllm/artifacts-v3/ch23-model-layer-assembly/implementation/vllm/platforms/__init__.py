# SOURCE: vllm/platforms/__init__.py
# HOST SEAM：current_platform 的单进程退化承载。真实 platforms/__init__.py
# 按可见加速器实例化 Platform 子类（CUDA/ROCm/XPU/CPU…）；host 测试无任何
# 加速器——按 Platform 基类（platforms/interface.py）的默认行为回答：
#   is_rocm/is_xpu/is_cpu/is_cuda_alike/is_tpu → False
#   opaque_attention_op → False（interface.py:L1116-L1121 基类默认——
#     use_direct_call=True，Attention.forward 直调 Python 算子函数）
#   use_all_gather → False（CUDA/ROCm/CPU 同款——LogitsProcessor 走 gather
#     通道；基类 True 是 TPU 的 all-gather 语义，interface.py:L1102-L1106）
#   verify_model_arch → no-op（interface.py:L950 起的基类默认无操作版）
from __future__ import annotations


# SOURCE: vllm/platforms/interface.py Platform —— HOST SEAM：基类默认行为位
class _HostPlatform:
    # SOURCE: vllm/platforms/interface.py Platform.is_rocm —— HOST SEAM
    @classmethod
    def is_rocm(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_rocm —— HOST SEAM
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_xpu —— HOST SEAM
    @classmethod
    def is_xpu(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_xpu —— HOST SEAM
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_cpu —— HOST SEAM
    @classmethod
    def is_cpu(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_cpu —— HOST SEAM
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_cuda_alike —— HOST SEAM
    @classmethod
    def is_cuda_alike(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_cuda_alike —— HOST SEAM
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_tpu —— HOST SEAM
    @classmethod
    def is_tpu(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_tpu —— HOST SEAM
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_out_of_tree —— HOST SEAM
    @classmethod
    def is_out_of_tree(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_out_of_tree —— HOST SEAM
        return False

    # SOURCE: vllm/platforms/interface.py:L1116-L1121 opaque_attention_op
    #   —— 基类默认 False
    @classmethod
    def opaque_attention_op(cls) -> bool:
        """
        Returns True if we register attention as one giant opaque custom op
        on the current platform
        """
        # SOURCE: vllm/platforms/interface.py:L1116-L1121 opaque_attention_op
        return False

    # SOURCE: vllm/platforms/interface.py:L1102-L1106 use_all_gather
    #   —— HOST SEAM：取 CUDA/ROCm/CPU 的 False 覆盖值（gather 通道）
    @classmethod
    def use_all_gather(cls) -> bool:
        """
        Whether to use allgather in LogitsProcessor to gather the logits.
        """
        # SOURCE: vllm/platforms/interface.py:L1102-L1106 use_all_gather
        return False

    # SOURCE: vllm/platforms/interface.py:L950 verify_model_arch —— HOST SEAM
    #   基类默认无操作
    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        # SOURCE: vllm/platforms/interface.py:L950 verify_model_arch —— HOST SEAM
        return None


# SOURCE: vllm/platforms/__init__.py current_platform —— HOST SEAM 实例位
current_platform = _HostPlatform()
