# vllm_ascend/_310p/kv_block_zeroer.py —— subtract-only 精简版（ch17 主线之一：KV block 清零）
#
# 直接两层继承 vLLM 基类、跳过昇腾中间层：AscendKVBlockZeroer310(KVBlockZeroer)，
# KVBlockZeroer 直接 from vllm.v1.worker.utils。基类用 Triton kernel + 绝对字节地址表
# (seg_addrs/page_size_el) 清零；310 子类无 Triton，退化为：init_meta 收集每层 (k,v)
# 张量列表 + logical_page_ratio，zero_block_ids 用张量切片 .zero_() 直接写。
#
# 基类 kv_cache 是单 Tensor，310 是 (k,v) 二元组（配合 model_runner _allocate 的
# k_cache/v_cache 分开分配）。全段是纯 Python/torch 切片，host 可跑（用 CPU 张量即可，
# 见 ../tests）。
from collections.abc import Iterable
from typing import Any

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer


# SOURCE: vllm_ascend/_310p/kv_block_zeroer.py:L25
class AscendKVBlockZeroer310(KVBlockZeroer):
    """310P KV block zeroer without Triton.

    Atlas 300I DUO does not support Triton. For MTP >= 2 hybrid models, newly
    allocated attention KV blocks must be zeroed via direct tensor writes.
    """

    # SOURCE: vllm_ascend/_310p/kv_block_zeroer.py:L32-L35
    def __init__(self, device: torch.device, pin_memory: bool) -> None:
        super().__init__(device, pin_memory)
        self._kv_tensors: list[torch.Tensor] = []
        self._logical_page_ratio: int = 1

    # SOURCE: vllm_ascend/_310p/kv_block_zeroer.py:L37-L70
    def init_meta(
        self,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[list[int]],
        cache_dtype: str,
        runner_only_attn_layers: set[str],
        static_forward_context: dict[str, Any],
    ) -> None:
        seen_ptrs: set[int] = set()
        self._kv_tensors = []
        self._logical_page_ratio = 1

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id][0]
            # 逻辑块 → 物理页比例：调度器给的是逻辑 block id，清零时按 ratio 换算物理页区间。
            ratio = spec.block_size // kernel_bs
            if not self._kv_tensors:
                self._logical_page_ratio = ratio

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv_tuple = static_forward_context[layer_name].kv_cache
                assert len(kv_tuple) == 2, "K and V are not stored separately"
                for kv in kv_tuple:
                    dp = kv.data_ptr()
                    if dp in seen_ptrs:
                        continue
                    seen_ptrs.add(dp)
                    self._kv_tensors.append(kv)

    # SOURCE: vllm_ascend/_310p/kv_block_zeroer.py:L72-L81
    def zero_block_ids(self, block_ids: list[int]) -> None:
        if not block_ids or not self._kv_tensors:
            return

        ratio = self._logical_page_ratio
        for block_id in block_ids:
            start = block_id * ratio
            end = start + ratio
            for kv in self._kv_tensors:
                kv[start:end].zero_()
