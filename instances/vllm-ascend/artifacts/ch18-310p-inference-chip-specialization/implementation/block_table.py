# vllm_ascend/_310p/block_table.py —— subtract-only 精简版（ch17 主线之一：输入批/块表子类化）
#
# 立意：310P 无 Triton，基类（昇腾主栈 vllm_ascend/worker/block_table.py 的 `class
# BlockTable:` 独立类，只 import 复用 vLLM 的 @triton.jit _compute_slot_mapping_kernel）
# 在步末用 Triton kernel 写 slot_mapping.gpu。310 子类 BlockTable(AscendBlockTable)
# 覆写 compute_slot_mapping：改用纯 NumPy 在 **CPU** 上算同一公式
#   slot = block_number * block_size + (position % block_size)
# 再 copy_to_gpu。_to_numpy 的 "must be CPU" 守卫点明：310P 连 device 张量算术 / D2H
# 都规避。
#
# host 无 vllm_ascend/torch_npu，但 _compute_slot_mapping_numpy / _to_numpy /
# _normalize_slot_mapping_inputs 是纯 NumPy/torch.Tensor 控制流——测试用桩注入基类
# 与 slot_mapping 缓冲后可直接跑（见 ../tests）。
from typing import Any, cast

import numpy as np
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.worker.cp_utils import get_total_cp_world_size

from vllm_ascend.worker.block_table import BlockTable as AscendBlockTable
from vllm_ascend.worker.block_table import MultiGroupBlockTable as AscendMultiGroupBlockTable


# SOURCE: vllm_ascend/_310p/block_table.py:L14
class BlockTable(AscendBlockTable):
    # SOURCE: vllm_ascend/_310p/block_table.py:L15-L17
    def compute_slot_mapping(self, *args: Any) -> None:
        req_indices, positions = self._normalize_slot_mapping_inputs(*args)
        self._compute_slot_mapping_numpy(req_indices, positions)

    # SOURCE: vllm_ascend/_310p/block_table.py:L19-L51
    def _compute_slot_mapping_numpy(self, req_indices: np.ndarray, positions: np.ndarray) -> None:
        num_tokens = positions.shape[0]
        if num_tokens == 0:
            self.slot_mapping.copy_to_gpu(0)
            return

        # SUBTRACTED: dcp_world_size*pcp_world_size>1 的 CP/DCP 交错分支（block_table.py:L25-L43）。
        #   virtual_block_size / mask / PAD_SLOT_ID 的交错算法与昇腾基类 compute_slot_mapping_draft
        #   同构，CP（context parallel）是正交特性；单卡 dcp=pcp=1 主路径已足以演示
        #   "block_numbers*block_size+block_offsets 在 CPU 算" 的立意。
        logical_block_idx = positions // self.block_size
        block_table_indices = self._get_block_table_indices(req_indices, logical_block_idx)
        block_numbers = self.block_table.np.ravel()[block_table_indices]
        block_offsets = positions % self.block_size
        np.add(block_numbers * self.block_size, block_offsets, out=self.slot_mapping.np[:num_tokens])

        self.slot_mapping.copy_to_gpu(num_tokens)

    # SOURCE: vllm_ascend/_310p/block_table.py:L53-L55
    def _get_block_table_indices(self, req_indices, logical_block_idx):
        row_stride = self.max_num_blocks_per_req * self.blocks_per_phys_block
        return req_indices * row_stride + logical_block_idx

    # SOURCE: vllm_ascend/_310p/block_table.py:L57-L75
    def _normalize_slot_mapping_inputs(self, *args) -> tuple[np.ndarray, np.ndarray]:
        if len(args) == 2:
            req_indices, positions = args
            return self._to_numpy(req_indices), self._to_numpy(positions)

        if len(args) == 3:
            num_reqs, query_start_loc, positions = args
            query_start_loc_np = self._to_numpy(query_start_loc)[: num_reqs + 1]
            positions_np = self._to_numpy(positions)
            counts = np.diff(query_start_loc_np)
            req_indices_np = np.repeat(np.arange(num_reqs, dtype=np.int64), counts)
            if req_indices_np.shape[0] != positions_np.shape[0]:
                raise ValueError(
                    "query_start_loc and positions describe different token counts: "
                    f"{req_indices_np.shape[0]} != {positions_np.shape[0]}"
                )
            return req_indices_np, positions_np

        raise TypeError("compute_slot_mapping expects either 2 or 3 positional arguments")

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        # SOURCE: vllm_ascend/_310p/block_table.py:L77-L88
        if isinstance(value, np.ndarray):
            return value.astype(np.int64, copy=False)
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                # 310P 规避 device 张量算术/D2H：slot_mapping 必须全在 host 算。
                raise TypeError(
                    "310P slot mapping must be computed from CPU req_indices/positions; "
                    "device tensor inputs would require unsupported NPU arithmetic or D2H"
                )
            return value.detach().numpy().astype(np.int64, copy=False)
        return np.asarray(value, dtype=np.int64)


# SOURCE: vllm_ascend/_310p/block_table.py:L91
class MultiGroupBlockTable(AscendMultiGroupBlockTable):
    # SOURCE: vllm_ascend/_310p/block_table.py:L92-L158
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        num_speculative_tokens: int = 0,
        max_num_blocks: list[int] | None = None,
        kernel_sizes: list[list[int]] | None = None,
        cp_kv_cache_interleave_size: int = 1,
        kv_cache_groups: list[KVCacheGroupSpec] | None = None,
    ) -> None:
        if kernel_sizes is None:
            kernel_sizes = [[0]] * len(block_sizes)
        elif len(kernel_sizes) == 1 and len(block_sizes) > 1:
            kernel_sizes = kernel_sizes * len(block_sizes)
        elif len(kernel_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_sizes length ({len(kernel_sizes)}) must match block_sizes length ({len(block_sizes)})"
            )

        if max_num_blocks is None:
            total_cp_world_size = get_total_cp_world_size()
            max_num_blocks = [cdiv(max_model_len, block_size * total_cp_world_size) for block_size in block_sizes]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) must match block_sizes length ({len(block_sizes)})"
            )

        # 每个 KV-cache group 建一个 310 版 BlockTable（差异局部化：多 group 转发到 310 子类）。
        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max_num_blocks_per_req,
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_size_list,
                cp_kv_cache_interleave_size,
                num_speculative_tokens,
                kv_cache_group,
            )
            for block_size, kernel_size_list, max_num_blocks_per_req, kv_cache_group in zip(
                block_sizes, kernel_sizes, max_num_blocks, kv_cache_groups
            )
        ]
        # SUBTRACTED: kv_cache_groups is None 的备用构造分支（block_table.py:L142-L158）——
        #   与上面分支唯一差别是不透传 kv_cache_group；保留主分支已足以说明"多 group 转发"。

    # SOURCE: vllm_ascend/_310p/block_table.py:L160-L185
    def compute_slot_mapping(
        self,
        num_reqs_or_req_indices: int | np.ndarray | torch.Tensor,
        query_start_loc_or_positions: np.ndarray | torch.Tensor,
        positions: np.ndarray | torch.Tensor | None = None,
        positions_compressed_list: list[np.ndarray] | None = None,
        req_indices_compressed_list: list[np.ndarray] | None = None,
    ) -> None:
        for i, block_table_base in enumerate(self.block_tables):
            block_table = cast(BlockTable, block_table_base)
            if positions_compressed_list is not None and req_indices_compressed_list is not None:
                block_table.compute_slot_mapping(
                    req_indices_compressed_list[i],
                    positions_compressed_list[i],
                )
            elif positions is None:
                block_table.compute_slot_mapping(
                    num_reqs_or_req_indices,
                    query_start_loc_or_positions,
                )
            else:
                block_table.compute_slot_mapping(
                    num_reqs_or_req_indices,
                    query_start_loc_or_positions,
                    positions,
                )

    # SUBTRACTED: compute_slot_mapping_draft（block_table.py:L187-L202）—— draft 版与
    #   spec-decode 绑定（MTP/ngram 是 Part 另章主题），随 spec-decode 一并略去；转发逻辑
    #   与上面的 compute_slot_mapping 同构。
