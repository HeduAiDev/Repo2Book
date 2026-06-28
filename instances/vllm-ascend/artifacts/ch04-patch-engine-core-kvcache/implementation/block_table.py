# 案例5 — int32 slot_mapping：昇腾自有 worker 的子类覆盖（非 patch/ 目录）。
# 硬件算子约束：vllm-ascend 的 reshape_and_cache 要求 slot_mapping 为 int32（CUDA 用 int64），
# 故先 del 父类 int64 张量（立即 gc）再以 int32 重建。是「硬件算子约束驱动的数据类型改写」最小样本。
# subtract-only：与 vllm_ascend/worker/v2/block_table.py 同名/同结构/同控制流。
#
# SOURCE: vllm_ascend/worker/v2/block_table.py:L19-L22
import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables, _load_ptr


# SOURCE: vllm_ascend/worker/v2/block_table.py:L25-L105
class AscendBlockTables(BlockTables):
    """Block table for Ascend NPUs."""

    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        kernel_block_sizes: list[int] | None = None,
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
    ):
        # SOURCE: vllm_ascend/worker/v2/block_table.py:L28-L77
        from vllm_ascend.utils import vllm_version_is

        if vllm_version_is("0.21.0"):
            super().__init__(
                block_sizes,
                max_num_reqs,
                max_num_batched_tokens,
                max_num_blocks_per_group,
                device,
                cp_size,
                cp_rank,
                cp_interleave,
            )
        else:
            if kernel_block_sizes is None:
                kernel_block_sizes = block_sizes
            super().__init__(
                block_sizes,
                max_num_reqs,
                max_num_batched_tokens,
                max_num_blocks_per_group,
                device,
                kernel_block_sizes,
                cp_size,
                cp_rank,
                cp_interleave,
            )
        # because we will override these attribute, delete these attribute to
        # make sure it's collected by python gc immediately.
        del self.slot_mappings
        # vllm-ascend' reshape_and_cache function requires slot_mappings to be int32.
        # so we need to redefine slot_mappings to be int32.
        self.slot_mappings: torch.Tensor = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor:
        # SOURCE: vllm_ascend/worker/v2/block_table.py:L79-L105
        # SUBTRACTED: compute_slot_mappings 的 triton kernel 调用体
        # (vllm_ascend/worker/v2/block_table.py:L86-L105) 及 @triton.jit
        # _compute_slot_mappings_kernel（L108-L178）。案例5 只演示 slot_mappings 的 int32 dtype
        # 改写，kernel 启动 / CP slot 计算细节与 dtype 主旨无关，整体删除。
        raise NotImplementedError  # 占位：真实 triton kernel 见源码，本章不展开
