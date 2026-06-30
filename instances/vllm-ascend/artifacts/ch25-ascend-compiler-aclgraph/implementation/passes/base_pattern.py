# 精简版（只做减法）— 对照真实源码 vllm_ascend/compilation/passes/base_pattern.py
#
# 所有融合 pattern 的抽象基类。register() 把同一对 (pattern_fn, replacement_fn) 向两套引擎
# 各注册一份：torch inductor 的 pattern matcher（fusion_pass_compile 这条路用）+ npugraph_ex/
# torchair（npugraph_ex_compile 这条路用）。extra_stream_scope_check 拒绝跨 stream 的误融合。
# 本文件按真实源码原样保留（无删减）。
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.config import VllmConfig

try:
    import npugraph_ex as nge
except ImportError:
    import torchair as nge

from vllm_ascend.compilation.passes.utils.npugraph_ex_utils_check import extra_stream_scope_check

# Global set to track registered patterns and prevent duplicates
_registered_patterns: set[str] = set()


# SOURCE: vllm_ascend/compilation/passes/base_pattern.py:L20
class BasePattern(ABC):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        # SOURCE: vllm_ascend/compilation/passes/base_pattern.py:L21
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

    @abstractmethod
    def get_inputs(self) -> list[torch.Tensor]:
        # SOURCE: vllm_ascend/compilation/passes/base_pattern.py:L26
        pass

    @abstractmethod
    def get_pattern(self) -> Callable:
        # SOURCE: vllm_ascend/compilation/passes/base_pattern.py:L30
        pass

    @abstractmethod
    def get_replacement(self) -> Callable:
        # SOURCE: vllm_ascend/compilation/passes/base_pattern.py:L34
        pass

    def get_extra_stream_scope_check(self):
        # SOURCE: vllm_ascend/compilation/passes/base_pattern.py:L38
        return extra_stream_scope_check

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # SOURCE: vllm_ascend/compilation/passes/base_pattern.py:L41
        # Create a unique identifier for this pattern based on class name and eps
        pattern_id = f"{self.__class__.__name__}_{self.eps}"

        # Skip registration if this pattern has already been registered globally
        if pattern_id in _registered_patterns:
            return

        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        pm.register_replacement(pattern_fn, replacement_fn, example_inputs, pm.fwd_only, pm_pass)

        nge.register_replacement(
            search_fn=pattern_fn,
            replace_fn=replacement_fn,
            example_inputs=example_inputs,
            extra_check=self.get_extra_stream_scope_check(),
        )

        # Mark this pattern as registered
        _registered_patterns.add(pattern_id)
