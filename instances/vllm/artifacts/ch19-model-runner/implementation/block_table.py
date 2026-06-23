# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Subtract-only reduced companion for ch18.
# SOURCE: vllm/v1/worker/block_table.py (pin f3fef123)
# Reproduces BlockTable / MultiGroupBlockTable / _compute_slot_mapping_kernel
# faithfully; only the approved subtraction_plan.delete items are removed.

import numpy as np
import torch

# SUBTRACTED: vllm imports for distributed CP/DCP groups (get_dcp_group /
#   get_pcp_group) — CP/DCP sharding is an approved deletion; the reduced
#   companion fixes a single rank (TOTAL_CP_WORLD_SIZE == 1).
#   Orig: vllm/v1/worker/block_table.py:L7-L13
import triton
import triton.language as tl

# SOURCE: vllm/v1/attention/backends/utils.py PAD_SLOT_ID — padding slot id
#   used to mark token slots that must not be written (CUDA-graph padding).
PAD_SLOT_ID = -1


# SOURCE: vllm/v1/utils.py:L108  CpuGpuBuffer — the real vLLM helper that holds
#   a paired CPU tensor (+ numpy view) and GPU tensor. Kept verbatim because the
#   block table's "GPU 端镜像" is exactly this CPU/GPU pairing.
class CpuGpuBuffer:  # SOURCE: vllm/v1/utils.py:L108
    """Buffer to easily copy tensors between CPU and GPU."""

    def __init__(self, *size, dtype, device, pin_memory, with_numpy=True):
        # SOURCE: vllm/v1/utils.py:L111
        self.cpu = torch.zeros(*size, dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.gpu = torch.zeros_like(self.cpu, device=device)
        self.np: np.ndarray
        if with_numpy:
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        # SOURCE: vllm/v1/utils.py:L133
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)


# SOURCE: vllm/v1/worker/block_table.py:L18  BlockTable
class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int = 1,
    ):
        # SOURCE: vllm/v1/worker/block_table.py:L42
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        # SUBTRACTED: hybrid kernel-block subdivision branch (use_hybrid_blocks /
        #   blocks_per_kv_block / map_to_kernel_blocks). Approved deletion: when
        #   kernel_block_size == block_size the path is skipped (blocks_per_kv_block
        #   == 1). The reduced companion fixes standard equal block sizes.
        #   Orig: vllm/v1/worker/block_table.py:L47-L66
        if kernel_block_size != block_size:
            raise ValueError(
                "reduced companion only supports kernel_block_size == block_size"
            )
        self.block_size = block_size
        self.blocks_per_kv_block = 1
        self.use_hybrid_blocks = False

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        # SUBTRACTED: _kernel_block_arange (only used by hybrid path).
        #   Orig: vllm/v1/worker/block_table.py:L79-L84

        # SUBTRACTED: PCP/DCP world-size & rank discovery via distributed groups.
        #   Approved CP/DCP deletion — reduced companion fixes single rank.
        #   Orig: vllm/v1/worker/block_table.py:L86-L99
        self.pcp_world_size = 1
        self.pcp_rank = 0
        self.dcp_world_size = 1
        self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    # SOURCE: vllm/v1/worker/block_table.py:L102  append_row
    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return

        # SUBTRACTED: use_hybrid_blocks map_to_kernel_blocks call (approved hybrid
        #   deletion). Orig: vllm/v1/worker/block_table.py:L110-L113

        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    # SOURCE: vllm/v1/worker/block_table.py:L120  add_row
    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    # SOURCE: vllm/v1/worker/block_table.py:L124  clear_row
    def clear_row(self, row_idx: int) -> None:
        num_blocks = self.num_blocks_per_row[row_idx]
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
        self.num_blocks_per_row[row_idx] = 0

    # SOURCE: vllm/v1/worker/block_table.py:L130  move_row
    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    # SOURCE: vllm/v1/worker/block_table.py:L136  swap_row
    def swap_row(self, src: int, tgt: int) -> None:
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    # SOURCE: vllm/v1/worker/block_table.py:L141  compute_slot_mapping
    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        num_tokens = positions.shape[0]
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        _compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_tokens,
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.block_table.gpu,
            self.block_table.gpu.stride(0),
            self.block_size,
            self.slot_mapping.gpu,
            TOTAL_CP_WORLD_SIZE=total_cp_world_size,
            TOTAL_CP_RANK=total_cp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=self.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )

    # SOURCE: vllm/v1/worker/block_table.py:L166  commit_block_table
    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu(num_reqs)

    # SUBTRACTED: clear() / map_to_kernel_blocks() / get_cpu_tensor() /
    #   get_numpy_array() — clear() unused on the main path; map_to_kernel_blocks
    #   is the approved hybrid deletion; cpu/numpy accessors are not on the
    #   ch18 spine. Orig: vllm/v1/worker/block_table.py:L169-L213

    # SOURCE: vllm/v1/worker/block_table.py:L203  get_device_tensor
    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table.gpu[:num_reqs]

    # SOURCE: vllm/v1/worker/block_table.py:L215  _make_buffer
    def _make_buffer(self, *size, dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


# SOURCE: vllm/v1/worker/block_table.py:L223  MultiGroupBlockTable
class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int] | None = None,
        cp_kv_cache_interleave_size: int = 1,
    ) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L226
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError("kernel_block_sizes length must match block_sizes length")
        if max_num_blocks is None:
            # SUBTRACTED: get_total_cp_world_size() factor (CP/DCP). Approved
            #   deletion — single rank → total_cp_world_size == 1.
            #   Orig: vllm/v1/worker/block_table.py:L248
            max_num_blocks = [
                (max_model_len + block_size - 1) // block_size
                for block_size in block_sizes
            ]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError("max_num_blocks length must match block_sizes length")

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
            for block_size, kernel_block_size, max_num_blocks_per_req in zip(
                block_sizes, kernel_block_sizes, max_num_blocks
            )
        ]

    # SOURCE: vllm/v1/worker/block_table.py:L276  append_row
    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    # SOURCE: vllm/v1/worker/block_table.py:L280  add_row
    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    # SOURCE: vllm/v1/worker/block_table.py:L284  clear_row
    def clear_row(self, row_idx: int) -> None:
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    # SOURCE: vllm/v1/worker/block_table.py:L288  move_row
    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    # SOURCE: vllm/v1/worker/block_table.py:L292  swap_row
    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    # SOURCE: vllm/v1/worker/block_table.py:L296  compute_slot_mapping
    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    # SOURCE: vllm/v1/worker/block_table.py:L305  commit_block_table
    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    # SOURCE: vllm/v1/worker/block_table.py:L313  __getitem__
    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]


# SOURCE: vllm/v1/worker/block_table.py:L318  _compute_slot_mapping_kernel
@triton.jit
def _compute_slot_mapping_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
    block_table_stride,  # max_num_blocks_per_req
    block_size,
    slot_mapping_ptr,  # [max_num_tokens], int64
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # SOURCE: vllm/v1/worker/block_table.py:L318
    req_idx = tl.program_id(0)

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

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        block_indices = pos // virtual_block_size
        block_numbers = tl.load(block_table_ptr + row_offset + block_indices).to(
            tl.int64
        )

        virtual_block_offsets = pos - block_indices * virtual_block_size
        # SUBTRACTED: is_local / local_block_offsets CP-sharding math. Approved
        #   CP deletion — with TOTAL_CP_WORLD_SIZE == 1 and rank 0, is_local is
        #   always True and local_block_offsets degenerates to pos % block_size.
        #   Orig: vllm/v1/worker/block_table.py:L362-L369
        local_block_offsets = virtual_block_offsets

        slot_ids = block_numbers * block_size + local_block_offsets
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)
