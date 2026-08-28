# SOURCE: vllm/v1/worker/block_table.py
# 本章切面（m08）：块表双镜像与先行拷贝——append_row CPU 增量 /
# commit_block_table 每拍只拷活跃行（_prepare_inputs 第一句，与 CPU 计算
# 重叠）。compute_slot_mapping 派发 Triton kernel（position→物理 slot 的
# 数学本体 → ch22，入口保留）。
# dossier.delete[8] 明示：CP 分片局部量与 hybrid kernel block 细分
#   （map_to_kernel_blocks 等）**不删**——kernel 本体是 ch22 主场不在本章
#   动刀；hybrid 由构造期 block_size 对比门控（kernel block == kv block 时
#   use_hybrid_blocks 恒 False）整段配置死码、原样保留零风险。
# HOST SEAM：CPU host 无 CUDA launch——compute_slot_mapping 在 CPU 设备走
#   kernel 本体的逐行镜像（同一恒等式 + 同一 PAD 尾 + 同一 CP 变量名）；
#   CUDA 设备逐字派发（容器内真跑，ch13 差分电池已验恒等式逐位一致）。
from __future__ import annotations

import math
from enum import Enum

import numpy as np
import torch
import triton
import triton.language as tl

from ._host_seams import get_dcp_group, get_pcp_group
from .math_utils import cdiv
from .utils import CpuGpuBuffer

# SOURCE: vllm/v1/attention/backends/utils.py:L45-L46 PAD/NULL 常量（本章折入
#   本文件——原文件是注意力后端工具族，其余成员 → ch21/22；ch13 同款折入）
PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0


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
    # SOURCE: vllm/v1/worker/block_table.py:L35-L39 token_alignment 对齐
    if token_alignment is not None:
        if token_alignment <= 0:
            raise ValueError("token_alignment must be positive")
        block_alignment = token_alignment // math.gcd(token_alignment, block_size)
        max_num_blocks = cdiv(max_num_blocks, block_alignment) * block_alignment
    # SOURCE: vllm/v1/worker/block_table.py:L40
    return max_num_blocks * block_size // kernel_block_size


# SOURCE: vllm/v1/worker/block_table.py:L43 SlotMappingMode
class SlotMappingMode(Enum):
    # SOURCE: vllm/v1/worker/block_table.py:L44
    TOKEN_TO_KV_SLOT = "token_to_kv_slot"
    # SOURCE: vllm/v1/worker/block_table.py:L45（Mamba/GDN 状态缓存组把块表当
    #   状态索引、跳过 per-token slot——ch14 域，成员与早退支保留）
    NONE = "none"


# SOURCE: vllm/v1/worker/block_table.py:L48 BlockTable —— 单 group 块表
class BlockTable:
    # SOURCE: vllm/v1/worker/block_table.py:L49-L75 __init__
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

        # SOURCE: vllm/v1/worker/block_table.py:L82-L101 标准支与 hybrid 支
        #   （hybrid 细分整段保留——delete[8]【不删】；构造期对比门控，
        #   kernel block == kv block 时走标准支）
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

        # SOURCE: vllm/v1/worker/block_table.py:L105-L112 双镜像缓冲
        #   （block_table CpuGpuBuffer + slot_mapping CpuGpuBuffer）
        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        # SOURCE: vllm/v1/worker/block_table.py:L114-L119 _kernel_block_arange
        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        # SOURCE: vllm/v1/worker/block_table.py:L121-L134 PCP/DCP 组探测
        #   （AssertionError → 未初始化时单卡退化 world_size=1/rank=0——
        #   HOST SEAM 的 get_pcp_group/get_dcp_group 抛 AssertionError 走同支）
        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        # SOURCE: vllm/v1/worker/block_table.py:L135-L136
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size
        self.slot_mapping_mode = slot_mapping_mode

    # SOURCE: vllm/v1/worker/block_table.py:L138 append_row —— 页表行写入口
    #   （老请求差量的落点）
    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L143-L144（空追加 no-op）
        if not block_ids:
            return

        # SOURCE: vllm/v1/worker/block_table.py:L146-L149 hybrid 细分改写
        #   （use_hybrid_blocks 门控——delete[8]【不删】）
        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        # SOURCE: vllm/v1/worker/block_table.py:L151-L154 差量追加（行内偏移
        #   由 num_blocks_per_row 记账）
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    # SOURCE: vllm/v1/worker/block_table.py:L156 add_row —— 重置并整行写
    #   （新增/恢复请求）
    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L157-L158
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    # SOURCE: vllm/v1/worker/block_table.py:L160 clear_row —— remove 打洞时清行
    def clear_row(self, row_idx: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L161-L164
        num_blocks = self.num_blocks_per_row[row_idx]
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
        self.num_blocks_per_row[row_idx] = 0

    # SOURCE: vllm/v1/worker/block_table.py:L166 move_row —— condense 搬行
    def move_row(self, src: int, tgt: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L167-L175
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks
        # Clear the vacated source row: dummy-run batches dereference stale
        # rows as mamba state slots and write state in place there, possibly
        # after the blocks have been freed and reallocated.
        block_table_np[src, :num_blocks] = 0
        self.num_blocks_per_row[src] = 0

    # SOURCE: vllm/v1/worker/block_table.py:L177 swap_row —— swap_states 的
    #   块表侧
    def swap_row(self, src: int, tgt: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L178-L180
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    # SOURCE: vllm/v1/worker/block_table.py:L182 compute_slot_mapping —— Triton
    #   派发入口（slot 数学 → ch22）
    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L188
        num_tokens = positions.shape[0]
        # SOURCE: vllm/v1/worker/block_table.py:L189-L192 Mamba/GDN 早退支
        if self.slot_mapping_mode == SlotMappingMode.NONE:
            # Mamba/GDN groups consume the block table as recurrent state
            # indices and do not use per-token slot mappings.
            return
        # SOURCE: vllm/v1/worker/block_table.py:L193
        assert self.slot_mapping_mode == SlotMappingMode.TOKEN_TO_KV_SLOT

        # HOST SEAM：CPU host 无 CUDA launch——kernel 本体的逐行镜像（同一
        # 恒等式、同一 PAD 尾、同一 CP 变量名）；CUDA 分支下方逐字保留，
        # 容器内真跑（ch13 差分电池在真 GPU 上对拍逐位一致）。
        if self.device.type == "cpu":
            self._compute_slot_mapping_host(
                num_reqs, num_tokens, query_start_loc, positions
            )
            return

        # SOURCE: vllm/v1/worker/block_table.py:L195-L211 kernel 派发（逐字）
        _compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_tokens,
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.block_table.gpu,
            self.block_table.gpu.stride(0),
            self.block_size,
            self.slot_mapping.gpu,
            KV_CACHE_BLOCK_SIZE=self.kv_cache_block_size,
            BLOCKS_PER_KV_BLOCK=self.blocks_per_kv_block,
            TOTAL_CP_WORLD_SIZE=self.dcp_world_size,
            TOTAL_CP_RANK=self.dcp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=self.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )

    # compute_slot_mapping 的 CPU 镜像（HOST SEAM——kernel L397-L442 的逐行
    # 对应：每 program 处理一请求的 token 区间；CP 变量名与单卡退化同款）
    # SOURCE: vllm/v1/worker/block_table.py:L397-L442（kernel 本体）
    def _compute_slot_mapping_host(
        self,
        num_reqs: int,
        num_tokens: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L399-L408 PAD 尾（CUDA graph
        #   兼容：[num_tokens, max_num_tokens) 每拍重填 PAD_ID）
        if num_tokens < self.max_num_batched_tokens:
            self.slot_mapping.np[num_tokens:] = PAD_SLOT_ID
        query_start_loc_np = query_start_loc.detach().cpu().numpy()
        positions_np = positions.detach().cpu().numpy()
        # kernel 视 block_table 为扁平缓冲（block_table_ptr + row_offset + idx）
        block_table_flat = self.block_table.np.reshape(-1)
        # 每 program 处理一请求的 token 区间（L410-L411）
        for req_idx in range(num_reqs):
            start_idx = int(query_start_loc_np[req_idx])
            end_idx = int(query_start_loc_np[req_idx + 1])
            if end_idx <= start_idx:
                continue
            # SOURCE: vllm/v1/worker/block_table.py:L413-L414
            virtual_block_size = self.kv_cache_block_size * self.dcp_world_size
            row_offset = req_idx * self.max_num_blocks_per_req
            # SOURCE: vllm/v1/worker/block_table.py:L415-L420（CP 局部量按
            #   self.dcp_* 取值——单卡 dcp_world_size=1 时 is_local 恒真、
            #   local_block_offsets 恒等，与 kernel 烘干值一致）
            pos = positions_np[start_idx:end_idx].astype(np.int64)
            virtual_block_indices = pos // virtual_block_size
            virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
            is_local = (
                virtual_block_offsets // self.cp_kv_cache_interleave_size
            ) % self.dcp_world_size == self.dcp_rank
            local_block_offsets = (
                virtual_block_offsets
                // (self.dcp_world_size * self.cp_kv_cache_interleave_size)
            ) * self.cp_kv_cache_interleave_size + (
                virtual_block_offsets % self.cp_kv_cache_interleave_size
            )

            # SOURCE: vllm/v1/worker/block_table.py:L430-L438
            block_indices = (
                virtual_block_indices * self.blocks_per_kv_block
                + local_block_offsets // self.block_size
            )
            block_numbers = block_table_flat[row_offset + block_indices].astype(
                np.int64
            )
            # SOURCE: vllm/v1/worker/block_table.py:L439-L442 恒等式本体
            slot_offsets = local_block_offsets % self.block_size
            slot_ids = block_numbers * self.block_size + slot_offsets
            slot_ids = np.where(is_local, slot_ids, PAD_SLOT_ID)
            self.slot_mapping.np[start_idx:end_idx] = slot_ids

    # SOURCE: vllm/v1/worker/block_table.py:L213 commit_block_table —— 每拍
    #   只拷活跃行（m08：_prepare_inputs 第一句先行拷贝 → 与 CPU 计算重叠）
    def commit_block_table(self, num_reqs: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L214
        self.block_table.copy_to_gpu(num_reqs)

    # SOURCE: vllm/v1/worker/block_table.py:L216 clear
    def clear(self) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L217-L218
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    # SOURCE: vllm/v1/worker/block_table.py:L220 map_to_kernel_blocks ——
    #   hybrid 细分（delete[8]【不删】，docstring 示例原文）
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

        # SOURCE: vllm/v1/worker/block_table.py:L243-L247
        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        return kernel_block_ids.reshape(-1)

    # SOURCE: vllm/v1/worker/block_table.py:L250 get_device_tensor —— 读侧出口
    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        # SOURCE: vllm/v1/worker/block_table.py:L252
        return self.block_table.gpu[:num_reqs]

    # SOURCE: vllm/v1/worker/block_table.py:L254 get_cpu_tensor
    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        # SOURCE: vllm/v1/worker/block_table.py:L256
        return self.block_table.cpu

    # SOURCE: vllm/v1/worker/block_table.py:L258 get_numpy_array
    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        # SOURCE: vllm/v1/worker/block_table.py:L260
        return self.block_table.np

    # SOURCE: vllm/v1/worker/block_table.py:L262 _make_buffer
    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        # SOURCE: vllm/v1/worker/block_table.py:L265-L267
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


# SOURCE: vllm/v1/worker/block_table.py:L270 MultiGroupBlockTable —— 按 KV
#   cache group 持多块表的扇出容器（InputBatch.block_table 的类型）
class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    # SOURCE: vllm/v1/worker/block_table.py:L273-L335 __init__
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
        # SOURCE: vllm/v1/worker/block_table.py:L285-L302 长度校验三连
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        if slot_mapping_modes is None:
            slot_mapping_modes = [SlotMappingMode.TOKEN_TO_KV_SLOT] * len(block_sizes)
        if len(slot_mapping_modes) != len(block_sizes):
            raise ValueError(
                f"slot_mapping_modes length ({len(slot_mapping_modes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        # SOURCE: vllm/v1/worker/block_table.py:L304-L313 宽度对齐（NONE 模式
        #   组不施加 token_alignment——Mamba 状态组）
        max_num_blocks = [
            (
                get_block_table_width(n, block_size, token_alignment=None)
                if slot_mapping_mode == SlotMappingMode.NONE
                else get_block_table_width(n, block_size)
            )
            for n, block_size, slot_mapping_mode in zip(
                max_num_blocks, block_sizes, slot_mapping_modes
            )
        ]

        # SOURCE: vllm/v1/worker/block_table.py:L315-L335 逐组构造
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
                slot_mapping_mode=slot_mapping_mode,
            )
            for (
                block_size,
                kernel_block_size,
                max_num_blocks_per_req,
                slot_mapping_mode,
            ) in zip(
                block_sizes, kernel_block_sizes, max_num_blocks, slot_mapping_modes
            )
        ]

    # SOURCE: vllm/v1/worker/block_table.py:L337-L339 append_row 扇出
    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L338-L339
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    # SOURCE: vllm/v1/worker/block_table.py:L341-L343 add_row 扇出
    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L342-L343
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    # SOURCE: vllm/v1/worker/block_table.py:L345-L347 clear_row 扇出
    def clear_row(self, row_idx: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L346-L347
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    # SOURCE: vllm/v1/worker/block_table.py:L349-L351 move_row 扇出
    def move_row(self, src: int, tgt: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L350-L351
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    # SOURCE: vllm/v1/worker/block_table.py:L353-L355 swap_row 扇出
    def swap_row(self, src: int, tgt: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L354-L355
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    # SOURCE: vllm/v1/worker/block_table.py:L357-L364 compute_slot_mapping 扇出
    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L363-L364
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    # SOURCE: vllm/v1/worker/block_table.py:L366-L368 commit_block_table 扇出
    #   （活跃行前缀拷贝）
    def commit_block_table(self, num_reqs: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L367-L368
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    # SOURCE: vllm/v1/worker/block_table.py:L370-L372 clear 扇出
    def clear(self) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L371-L372
        for block_table in self.block_tables:
            block_table.clear()

    # SOURCE: vllm/v1/worker/block_table.py:L374-L376 __getitem__
    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        # SOURCE: vllm/v1/worker/block_table.py:L376
        return self.block_tables[idx]


# SOURCE: vllm/v1/worker/block_table.py:L379 _compute_slot_mapping_kernel ——
#   槽位换算恒等式本体（slot = 块号 × block_size + 块内偏移；CP 分片局部量
#   L421-L428 整段保留——delete[8]【不删】，单卡 dcp=1 时退化恒等 → ch22）
@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
    block_table_stride,  # max_num_blocks_per_req
    block_size,
    slot_mapping_ptr,  # [max_num_tokens], int64
    KV_CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_KV_BLOCK: tl.constexpr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # SOURCE: vllm/v1/worker/block_table.py:L397
    req_idx = tl.program_id(0)

    # SOURCE: vllm/v1/worker/block_table.py:L399-L408 PAD 尾（CUDA graph 兼容）
    if req_idx == tl.num_programs(0) - 1:
        # Pad remaining slots for CUDA graph compatibility.
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    # SOURCE: vllm/v1/worker/block_table.py:L410-L411
    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    # SOURCE: vllm/v1/worker/block_table.py:L413-L414
    virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    # SOURCE: vllm/v1/worker/block_table.py:L415-L428
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        virtual_block_indices = pos // virtual_block_size
        virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        # SOURCE: vllm/v1/worker/block_table.py:L430-L438
        block_indices = (
            virtual_block_indices * BLOCKS_PER_KV_BLOCK
            + local_block_offsets // block_size
        )
        block_numbers = tl.load(
            block_table_ptr + row_offset + block_indices,
            mask=mask & is_local,
            other=0,
        ).to(tl.int64)
        # SOURCE: vllm/v1/worker/block_table.py:L439-L442 恒等式本体
        slot_offsets = local_block_offsets % block_size
        slot_ids = block_numbers * block_size + slot_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)
