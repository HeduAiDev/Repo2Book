# SOURCE: vllm/v1/worker/block_table.py
# worker 侧多组落地（m14）：BlockTable——分配块大小 ≠ kernel 块大小时的
# 细分（32-token 内存块拆 2×16 kernel 块：use_hybrid_blocks/
# blocks_per_kv_block/map_to_kernel_blocks）；MultiGroupBlockTable——每 KV
# 组一张表；get_block_table_width——表宽 = 块数 × block_size // kernel_bs
#（token_alignment 对齐）。PAD_SLOT_ID 从 vllm/v1/attention/backends/
# utils.py:L45 折入（compute_slot_mapping 的 PAD 哨兵）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 8 条 DCP/PCP：get_dcp/get_pcp 组查询与 CP 交错位（L120-L135、
#     slot_mapping kernel 的 TOTAL_CP_WORLD_SIZE/is_local/interleave——单卡
#     恒 1 烘干）；
#   slot_mapping kernel 本体（_compute_slot_mapping_kernel L379-L442——槽位
#     恒等式归 ch13 精简版，PAD/CP 深讲归 ch22；compute_slot_mapping 派发
#     面随之删）；SlotMappingMode.NONE（mamba 状态槽寻址 → 邻章——默认
#     恒 TOKEN_TO_KV_SLOT）；swap_row/clear/move_row 的观测注释面按需保留。
import math
from enum import Enum

import numpy as np
import torch

from .math_utils import cdiv
from .v1_utils import CpuGpuBuffer

# SOURCE: vllm/v1/attention/backends/utils.py:L45 PAD_SLOT_ID（折入）
PAD_SLOT_ID = -1


# SOURCE: vllm/v1/worker/block_table.py:L20 get_block_table_width
def get_block_table_width(
    max_num_blocks: int,
    block_size: int,
    kernel_block_size: int | None = None,
    *,
    token_alignment: int | None = 128,
) -> int:
    """Return the width after optional alignment and virtual block splitting."""
    # SOURCE: vllm/v1/worker/block_table.py:L28-L34
    if kernel_block_size is None:
        kernel_block_size = block_size
    if block_size % kernel_block_size != 0:
        raise ValueError(
            f"kernel_block_size {kernel_block_size} must divide "
            f"block_size {block_size} evenly"
        )
    # SOURCE: vllm/v1/worker/block_table.py:L35-L40
    if token_alignment is not None:
        if token_alignment <= 0:
            raise ValueError("token_alignment must be positive")
        block_alignment = token_alignment // math.gcd(token_alignment, block_size)
        max_num_blocks = cdiv(max_num_blocks, block_alignment) * block_alignment
    return max_num_blocks * block_size // kernel_block_size


# SOURCE: vllm/v1/worker/block_table.py:L43 SlotMappingMode
class SlotMappingMode(Enum):
    # SOURCE: vllm/v1/worker/block_table.py:L44-L45
    TOKEN_TO_KV_SLOT = "token_to_kv_slot"
    NONE = "none"


# SOURCE: vllm/v1/worker/block_table.py:L48 BlockTable
class BlockTable:
    # SOURCE: vllm/v1/worker/block_table.py:L49 __init__
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
        slot_mapping_mode: SlotMappingMode = SlotMappingMode.TOKEN_TO_KV_SLOT,
    ):
        """
        Args:
            block_size: Block size used for KV cache memory allocation
            max_num_reqs: Maximum number of concurrent requests supported.
            max_num_blocks_per_req: Maximum number of blocks per request.
            max_num_batched_tokens: Maximum number of tokens in a batch.
            pin_memory: Whether to pin memory for faster GPU transfers.
            device: Target device for the block table.
            kernel_block_size: The block_size of underlying attention kernel.
                Will be the same as `block_size` if `block_size` is supported
                by the attention kernel.
            slot_mapping_mode: How this cache group maps scheduled tokens to
                cache slots. Mamba-like state caches do not use token slot
                mappings and should use SlotMappingMode.NONE.
        """
        # SOURCE: vllm/v1/worker/block_table.py:L76-L80
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device
        self.kv_cache_block_size = block_size

        # SOURCE: vllm/v1/worker/block_table.py:L82-L101 细分判定（标准 =
        #   同尺寸直映射；hybrid = 内存块拆 kernel 块，Example: 32-token
        #   memory blocks with 16-token kernel blocks → 2 kernel blocks each）
        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size
            # Memory blocks are subdivided to match kernel requirements
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        # SOURCE: vllm/v1/worker/block_table.py:L103
        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        # SOURCE: vllm/v1/worker/block_table.py:L105-L108（块表缓冲 + 行长账）
        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        # SUBTRACTED: slot_mapping 缓冲（L110-L112——槽位恒等式归 ch13）。

        # SOURCE: vllm/v1/worker/block_table.py:L114-L119 kernel 块 arange
        #   （细分的广播向量）
        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        # SUBTRACTED: CP 组查询（L120-L135——第 8 条单卡烘干）与
        #   slot_mapping_mode 存储（NONE 支 → 邻章；参数保留作装配面）。

    # SOURCE: vllm/v1/worker/block_table.py:L138 append_row
    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L143-L144
        if not block_ids:
            return

        # SOURCE: vllm/v1/worker/block_table.py:L146-L149（hybrid 细分：账本
        #   块号 → kernel 块号展开）
        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        # SOURCE: vllm/v1/worker/block_table.py:L151-L154
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    # SOURCE: vllm/v1/worker/block_table.py:L156 add_row
    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L157-L158
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    # SOURCE: vllm/v1/worker/block_table.py:L160 clear_row
    def clear_row(self, row_idx: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L161-L164
        num_blocks = self.num_blocks_per_row[row_idx]
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
        self.num_blocks_per_row[row_idx] = 0

    # SOURCE: vllm/v1/worker/block_table.py:L166 move_row
    def move_row(self, src: int, tgt: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L168-L175
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks
        # Clear the vacated source row: dummy-run batches dereference stale
        # rows as mamba state slots and write state in place there, possibly
        # after the blocks have been freed and reallocated.
        block_table_np[src, :num_blocks] = 0
        self.num_blocks_per_row[src] = 0

    # SOURCE: vllm/v1/worker/block_table.py:L177 swap_row
    def swap_row(self, src: int, tgt: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L178-L180
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    # SUBTRACTED: compute_slot_mapping（L182-L211——槽位恒等式 kernel 的
    #   派发面归 ch13；PAD/CP 深讲 → ch22）。

    # SOURCE: vllm/v1/worker/block_table.py:L213 commit_block_table
    def commit_block_table(self, num_reqs: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L214（每拍只拷活跃行）
        self.block_table.copy_to_gpu(num_reqs)

    # SUBTRACTED: clear（L216-L218——观测面）。

    # SOURCE: vllm/v1/worker/block_table.py:L220 map_to_kernel_blocks
    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """Convert kv_manager_block_id IDs to kernel block IDs.

        Example:
            # kv_manager_block_ids: 32 tokens,
            # Kernel block size: 16 tokens
            # blocks_per_kv_block = 2
            >>> kv_manager_block_ids = np.array([0, 1, 2])
            >>> Result: [0, 1, 2, 3, 4, 5]

            # Each kv_manager_block_id maps to 2 kernel block id:
            # kv_manager_block_id 0 → kernel block id [0, 1]
            # kv_manager_block_id 1 → kernel block id [2, 3]
            # kv_manager_block_id 2 → kernel block id [4, 5]
        """
        # SOURCE: vllm/v1/worker/block_table.py:L240-L241
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        # SOURCE: vllm/v1/worker/block_table.py:L243-L248（reshape(-1,1)×
        #   blocks_per_kv_block + arange 的纯 numpy 算术）
        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        return kernel_block_ids.reshape(-1)

    # SUBTRACTED: get_device_tensor / get_cpu_tensor / get_numpy_array
    #   （L250-L260——attention metadata 的读腿出口 → ch21）。

    # SOURCE: vllm/v1/worker/block_table.py:L262 _make_buffer
    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        # SOURCE: vllm/v1/worker/block_table.py:L265-L267
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


# SOURCE: vllm/v1/worker/block_table.py:L270 MultiGroupBlockTable
class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    # SOURCE: vllm/v1/worker/block_table.py:L273 __init__
    def __init__(
        self,
        max_num_reqs: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int],
        cp_kv_cache_interleave_size: int = 1,
        slot_mapping_modes: list[SlotMappingMode] | None = None,
    ) -> None:
        # SUBTRACTED: slot_mapping_modes 的 NONE 支与长度校验（L290-L296
        #   ——mamba 状态槽 → 邻章；本章全组 TOKEN_TO_KV_SLOT）。
        # SOURCE: vllm/v1/worker/block_table.py:L285-L289
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        # SOURCE: vllm/v1/worker/block_table.py:L298-L302
        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        # SOURCE: vllm/v1/worker/block_table.py:L304-L313（表宽 = 块数 ×
        #   block_size // kernel_bs，128-token 对齐）
        max_num_blocks = [
            get_block_table_width(n, block_size)
            for n, block_size in zip(max_num_blocks, block_sizes)
        ]

        # SOURCE: vllm/v1/worker/block_table.py:L315-L335 每组一张表
        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max_num_blocks_per_req,
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_block_size,
                cp_kv_cache_interleave_size,
            )
            for (
                block_size,
                kernel_block_size,
                max_num_blocks_per_req,
            ) in zip(
                block_sizes, kernel_block_sizes, max_num_blocks
            )
        ]

    # SOURCE: vllm/v1/worker/block_table.py:L337 append_row
    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L338-L339
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    # SOURCE: vllm/v1/worker/block_table.py:L341 add_row
    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L342-L343
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    # SOURCE: vllm/v1/worker/block_table.py:L345 clear_row
    def clear_row(self, row_idx: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L346-L347
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    # SOURCE: vllm/v1/worker/block_table.py:L349 move_row
    def move_row(self, src: int, tgt: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L350-L351
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    # SOURCE: vllm/v1/worker/block_table.py:L353 swap_row
    def swap_row(self, src: int, tgt: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L354-L355
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    # SUBTRACTED: compute_slot_mapping（L357-L364——ch13 槽位恒等式派发）。

    # SOURCE: vllm/v1/worker/block_table.py:L366 commit_block_table
    def commit_block_table(self, num_reqs: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L367-L368
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    # SUBTRACTED: clear（L370-L372——观测面）。

    # SOURCE: vllm/v1/worker/block_table.py:L374 __getitem__
    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        # SOURCE: vllm/v1/worker/block_table.py:L376
        return self.block_tables[idx]
