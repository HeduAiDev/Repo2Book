# SOURCE: vllm/triton_utils.py
# HOST SEAM：`from vllm.triton_utils import tl, triton` 的 host 承载。
# 真实文件 re-export triton 语言 + 各类包装；本章精简版内所有 CUDA/Triton
# kernel 都以 HOST SEAM 精确数学镜像承载（见 impl-notes §Seam），@triton.jit
# 定义不再重复——但三处**直接以 kernel[grid](...) 形态被真实 Python 代码调
# 用**的核（_fill_short_context_topk_indices / _prepare_uniform_decode_kernel /
# _BUILD_PREFILL_CHUNK_METADATA_KERNEL）以「下标发射垫片」保住调用点逐字：
# 垫片 `kernel[(grid)](*args, **kw)` 在 host 转发到同文件登记的纯 torch 数学。
# triton.next_power_of_2/cdiv 为调用点消费的实用件（真实来自 triton 语言面）。
from __future__ import annotations


# SOURCE: vllm/triton_utils.py tl —— HOST SEAM 垫片（kernel 定义不重复；镜像
#   函数体内不使用 tl——真实 kernel 体由正文 embed_excerpts 呈现真实源码）
# SOURCE: vllm/triton_utils.py tl —— HOST SEAM 垫片（锚点双置）
class _TL:
    pass


tl = _TL()


# SOURCE: vllm/triton_utils.py triton —— HOST SEAM 垫片（next_power_of_2/cdiv
#   与真实 triton 语言面同式）
class _Triton:
    # SOURCE: triton.next_power_of_2 —— 同式
    @staticmethod
    def next_power_of_2(n: int) -> int:
        # SOURCE: triton language next_power_of_2
        return 1 if n <= 0 else 1 << (n - 1).bit_length()

    # SOURCE: triton.cdiv —— 同式
    @staticmethod
    def cdiv(a: int, b: int) -> int:
        # SOURCE: triton language cdiv
        return -(-a // b)

    # SOURCE: triton.jit —— HOST SEAM：直通装饰器（镜像函数不经编译）
    @staticmethod
    def jit(fn=None, **kwargs):
        # SOURCE: triton.jit —— HOST SEAM 直通
        def deco(f):
            return f

        return deco(fn) if fn is not None else deco


triton = _Triton()


# SOURCE: vllm/triton_utils.py —— HOST SEAM：下标发射垫片（kernel[grid](...)）
class TritonKernelShim:
    """`kernel[(grid)](*args, **kwargs)` 形态的 host 垫片。

    launch_fn 承载该 kernel 的精确数学（纯 torch），grid 与 num_warps 等
    launch 参数被忽略（host 串行等价）。每个使用点在其所在文件头登记
    对应真实 kernel 的源锚。
    """

    # SOURCE: vllm/triton_utils.py —— HOST SEAM 垫片位
    def __init__(self, launch_fn):
        self._launch_fn = launch_fn

    def __getitem__(self, grid):
        # SOURCE: vllm/triton_utils.py —— HOST SEAM 下标发射位
        def launch(*args, **kwargs):
            return self._launch_fn(*args, **kwargs)

        return launch
