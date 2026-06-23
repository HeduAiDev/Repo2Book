"""ch23 companion — 运行时替身（让本章精简版脱离 CUDA/vLLM 在 host 跑起来）。

本章正文聚焦三件事：CustomOp 两级 dispatch（custom_op.py）、@support_torch_compile +
piecewise 切图（compilation.py）、attention 自定义算子回收 f17（attention_op.py）。
本文件只提供它们赖以运行的最小上下文：CompilationConfig（只保留 custom_ops/mode/
splitting_ops 等本章用得到的字段）、CompilationMode 枚举、current_platform 平台探针、
以及 get_cached_compilation_config() 这个全局访问点。

这些都对应真实 vLLM 的同名实体（标 # SOURCE），把真实里依赖完整 VllmConfig 构造 /
多进程 / pydantic 校验的部分替换为可在单进程里设置的轻量替身（标 # SUBTRACTED）。
替换后控制流（如何读 custom_ops 决定 enabled、如何按 mode 决定 do_not_compile、在哪些
算子处切图）与真实 vLLM 完全一致，因此可数值/行为对照。正文不应喧宾夺主地讲它。
"""

from __future__ import annotations

import enum
from collections import Counter


# SOURCE: vllm/config/compilation.py:L37 (CompilationMode)
class CompilationMode(enum.IntEnum):
    NONE = 0
    STOCK_TORCH_COMPILE = 1
    DYNAMO_TRACE_ONCE = 2
    VLLM_COMPILE = 3


# SOURCE: vllm/config/compilation.py (CompilationConfig — 本章相关字段子集)
class CompilationConfig:
    # SUBTRACTED: 真实 CompilationConfig 是一个庞大的 pydantic dataclass（cudagraph_mode、
    # pass_config、inductor_compile_config、缓存目录等数十个字段，见 compilation.py:L100-L760）。
    # 本章只用到决定 dispatch 的少数字段，其余字段与本章主线无关，删之不损 dispatch/切图正确性。

    # SOURCE: vllm/config/compilation.py:L738 (_attention_ops 默认切图点)
    # SUBTRACTED: 真实 _attention_ops 含 mamba/short_conv/linear_attention/gdn 等十余项，
    # 都是同类「不可融合、需在此切图」的算子；正文以 unified_attention_with_output 为代表。
    _attention_ops = [
        "vllm::unified_attention_with_output",
        # "vllm::unified_mla_attention_with_output", "vllm::mamba_mixer2", ... (同类，省略)
    ]

    def __init__(
        self,
        custom_ops: list[str] | None = None,
        mode: CompilationMode = CompilationMode.NONE,
        backend: str = "inductor",
    ) -> None:
        # SOURCE: vllm/config/compilation.py:L469 (custom_ops: list[str])
        self.custom_ops = list(custom_ops) if custom_ops is not None else []
        # SOURCE: vllm/config/compilation.py (mode / backend)
        self.mode = mode
        self.backend = backend
        # SOURCE: vllm/config/compilation.py:L491 (splitting_ops: list[str] | None)
        self.splitting_ops: list[str] | None = None
        # SOURCE: vllm/config/compilation.py:L716,L718 (记账用 Counter)
        self.enabled_custom_ops: Counter[str] = Counter()
        self.disabled_custom_ops: Counter[str] = Counter()

    # SOURCE: vllm/config/compilation.py:L1082 (set_splitting_ops_for_v1 — 主路径)
    def set_splitting_ops_for_v1(self) -> None:
        # SUBTRACTED: 真实方法还处理 fuse_attn_quant / use_inductor_graph_partition /
        # unified_kv_cache_update 追加等分支（compilation.py:L1082-L1123）；本章只保留默认
        # 主路径：VLLM_COMPILE 下若 splitting_ops 未设，则取 _attention_ops（即在 attention 处切）。
        if self.mode != CompilationMode.VLLM_COMPILE:
            if self.splitting_ops is None:
                self.splitting_ops = []
            return
        if self.splitting_ops is None:
            self.splitting_ops = list(self._attention_ops)


# 模块级单例 + 访问点 ---------------------------------------------------------
# SUBTRACTED: 真实通过 set_current_vllm_config() 上下文与 get_cached_compilation_config()
# 的 functools.cache 取当前编译配置（vllm/config/__init__.py）。精简版用一个可 set 的模块
# 变量在单进程内切换「当前编译配置」视角，供测试切换 custom_ops/mode。
_CACHED_COMPILATION_CONFIG = CompilationConfig()


# SOURCE: vllm/config/__init__.py:get_cached_compilation_config
def get_cached_compilation_config() -> CompilationConfig:
    return _CACHED_COMPILATION_CONFIG


# SOURCE: vllm/config/__init__.py:set_current_vllm_config (测试侧切换替身)
def set_cached_compilation_config(cfg: CompilationConfig) -> None:
    # 测试钩子：切换当前编译配置视角（真实由 set_current_vllm_config 上下文驱动）。
    global _CACHED_COMPILATION_CONFIG
    _CACHED_COMPILATION_CONFIG = cfg


# 平台探针 -------------------------------------------------------------------
# SUBTRACTED: 真实 current_platform 是按构建后端选定的 Platform 子类（vllm/platforms），
# is_rocm/is_cpu/is_tpu/is_xpu/is_out_of_tree 等查询真实硬件。本章只需 CUDA 主路径与
# native 旁路两条线，故用一个可设置 kind 的轻量替身，默认 "cuda"。
class _Platform:
    # SOURCE: vllm/platforms/interface.py (Platform — 替身)
    def __init__(self) -> None:
        self.kind = "cuda"
        self.simple_compile_backend = "inductor"

    # SOURCE: vllm/platforms/interface.py:is_rocm
    def is_rocm(self) -> bool:
        return self.kind == "rocm"

    # SOURCE: vllm/platforms/interface.py:is_cpu
    def is_cpu(self) -> bool:
        return self.kind == "cpu"

    # SOURCE: vllm/platforms/interface.py:is_tpu
    def is_tpu(self) -> bool:
        return self.kind == "tpu"

    # SOURCE: vllm/platforms/interface.py:is_xpu
    def is_xpu(self) -> bool:
        return self.kind == "xpu"

    # SOURCE: vllm/platforms/interface.py:is_out_of_tree
    def is_out_of_tree(self) -> bool:
        return self.kind == "oot"


current_platform = _Platform()
