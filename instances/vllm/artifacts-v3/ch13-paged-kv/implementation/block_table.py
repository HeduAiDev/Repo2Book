# SOURCE: vllm/v1/worker/block_table.py
# BlockTable——worker 侧页表镜像（m7/m9/m15 主角）：block_table
# [max_num_reqs, max_blocks_per_req] int32 的 CpuGpuBuffer（CPU 写行/commit 拷
# GPU）+ slot_mapping [max_num_batched_tokens] int64；append_row 差量追加、
# compute_slot_mapping 派发 Triton kernel（槽位恒等式本体）。
# SUBTRACTED（dossier.delete 批准项的落点）：
#   第 4 条 混合/多组：MultiGroupBlockTable（L270-L376）、map_to_kernel_
#     blocks（L220-L248）与 use_hybrid_blocks 细分分支（__init__ else 支与
#     append_row 的细分改写）、get_block_table_width 的 token_alignment 对齐
#     乘子、SlotMappingMode.NONE（Mamba/GDN 状态缓存模式）；
#   第 6 条 DCP/PCP：PCP/DCP 组探测 try/except（L121-L134）与
#     cp_kv_cache_interleave_size；kernel 的 CP 分片三处按常数 1 烘干
#     （TOTAL_CP_WORLD_SIZE=1 时 virtual_block_size 不放大、is_local 恒真、
#     local_block_offsets 恒等——控制流等价 → ch22）。
# HOST SEAM：CPU host 无 CUDA launch——compute_slot_mapping 在 CPU 设备走
#   kernel 本体的逐行镜像（同一恒等式 + 同一 PAD 尾）；CUDA 设备逐字派发。
from enum import Enum

import numpy as np
import torch
import triton
import triton.language as tl

from .v1_utils import CpuGpuBuffer

# SOURCE: vllm/v1/attention/backends/utils.py:L45-L46 PAD/NULL 常量（本章折入
#   本文件——原文件是注意力后端工具族，其余成员 → ch21/22）
PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0

# SUBTRACTED: vllm.triton_utils 的平台薄壳归一（import 直用 triton/tl）。


# SOURCE: vllm/v1/worker/block_table.py:L20 get_block_table_width
def get_block_table_width(
    max_num_blocks: int,
    block_size: int,
    kernel_block_size: int | None = None,
) -> int:
    """Return the width after optional alignment and virtual block splitting."""
    # SUBTRACTED: token_alignment 对齐乘子（L25-L26 关键字参数与 L35-L39——
    #   CUDA graph 对齐宽度 → ch14/22）。
    # SOURCE: vllm/v1/worker/block_table.py:L28-L34
    if kernel_block_size is None:
        kernel_block_size = block_size
    if block_size % kernel_block_size != 0:
        raise ValueError(
            f"kernel_block_size {kernel_block_size} must divide "
            f"block_size {block_size} evenly"
        )
    # SOURCE: vllm/v1/worker/block_table.py:L40
    return max_num_blocks * block_size // kernel_block_size


# SOURCE: vllm/v1/worker/block_table.py:L43 SlotMappingMode
class SlotMappingMode(Enum):
    # SOURCE: vllm/v1/worker/block_table.py:L44
    TOKEN_TO_KV_SLOT = "token_to_kv_slot"
    # SUBTRACTED: NONE 成员（L45——Mamba/GDN 状态缓存组把块表当状态索引、
    #   跳过 per-token slot，第 4 条 → ch14）


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
        """
        # SUBTRACTED: cp_kv_cache_interleave_size 参数（L58——第 6 条，单卡
        #   烘干为 1）；slot_mapping_mode 的 Mamba 说明段（L72-L74 docstring）。
        # SOURCE: vllm/v1/worker/block_table.py:L76-L80
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device
        self.kv_cache_block_size = block_size

        # SOURCE: vllm/v1/worker/block_table.py:L82-L87 标准支：分配块与 kernel
        #   块同尺寸、免细分直通
        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            self.block_size = block_size
            self.blocks_per_kv_block = 1
        else:
            # SUBTRACTED: hybrid 细分支（L88-L101——32 token 内存块拆两个
            #   16 token kernel 块等，第 4 条 → ch14/22）；本章只支持同尺寸。
            raise ValueError(
                "ch13 精简版只支持 kernel_block_size == block_size"
                "（kernel 块细分 → ch14/22）"
            )
        # SUBTRACTED: use_hybrid_blocks 标志与 _kernel_block_arange 预构
        #   （L87、L101、L114-L119——第 4 条）。

        # SOURCE: vllm/v1/worker/block_table.py:L103
        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        # SOURCE: vllm/v1/worker/block_table.py:L105-L112 双镜像缓冲
        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        # SOURCE: vllm/v1/worker/block_table.py:L108
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        # SOURCE: vllm/v1/worker/block_table.py:L110-L112
        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        # SUBTRACTED: PCP/DCP 组探测（L121-L134——第 6 条：单卡 pcp/dcp
        #   world_size=1/rank=0 烘干，不进 kernel 实参）。
        # SOURCE: vllm/v1/worker/block_table.py:L136
        self.slot_mapping_mode = slot_mapping_mode

    # SOURCE: vllm/v1/worker/block_table.py:L138 append_row —— 页表行写入口
    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L143-L144（空追加 no-op）
        if not block_ids:
            return

        # SUBTRACTED: use_hybrid_blocks 细分改写（L146-L149——map_to_kernel_
        #   blocks 拆块，第 4 条；blocks_per_kv_block=1 直通）。

        # SOURCE: vllm/v1/worker/block_table.py:L151-L154 差量追加（行内偏移
        #   由 num_blocks_per_row 记账）
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

    # SOURCE: vllm/v1/worker/block_table.py:L177 swap_row
    def swap_row(self, src: int, tgt: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L178-L180
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    # SOURCE: vllm/v1/worker/block_table.py:L182 compute_slot_mapping —— Triton
    #   派发入口
    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L188
        num_tokens = positions.shape[0]
        # SUBTRACTED: SlotMappingMode.NONE 早退（L189-L192——Mamba/GDN 状态
        #   缓存组，第 4 条）；断言保留。
        assert self.slot_mapping_mode == SlotMappingMode.TOKEN_TO_KV_SLOT

        # HOST SEAM：CPU host 无 CUDA——kernel 本体的逐行镜像（同一恒等式、
        #   同一 PAD 尾、同一变量名；CUDA 分支下方逐字保留，容器内真跑）。
        if self.device.type == "cpu":
            self._compute_slot_mapping_host(
                num_reqs, num_tokens, query_start_loc, positions
            )
            return

        # SOURCE: vllm/v1/worker/block_table.py:L195-L211（CP 三实参烘干删除；
        #   BLOCKS_PER_KV_BLOCK=1 直通）
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
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )

    # compute_slot_mapping 的 CPU 镜像（HOST SEAM——kernel L380-L442 的逐行对应）
    # SOURCE: vllm/v1/worker/block_table.py:L195-L211（派发面）/L380-L442（kernel 本体）
    def _compute_slot_mapping_host(
        self,
        num_reqs: int,
        num_tokens: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # 最后一个 program 的 PAD 尾（L399-L408：CUDA graph 捕获 max 形状，
        # [num_tokens, max_num_tokens) 每拍重填 PAD_ID）
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
            # 恒等式本体（L413-L418 + L430-L442 烘干 CP 后的主干）：
            # virtual_block_size = kv_cache_block_size（CP 乘子=1 烘干）
            virtual_block_size = self.kv_cache_block_size
            row_offset = req_idx * self.max_num_blocks_per_req
            pos = positions_np[start_idx:end_idx].astype(np.int64)
            virtual_block_indices = pos // virtual_block_size
            virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
            # CP 的 is_local/local_block_offsets 重排单卡退化恒等（→ ch22）
            block_indices = (
                virtual_block_indices * self.blocks_per_kv_block
                + virtual_block_offsets // self.block_size
            )
            block_numbers = block_table_flat[row_offset + block_indices].astype(
                np.int64
            )
            slot_offsets = virtual_block_offsets % self.block_size
            slot_ids = block_numbers * self.block_size + slot_offsets
            self.slot_mapping.np[start_idx:end_idx] = slot_ids

    # SOURCE: vllm/v1/worker/block_table.py:L213 commit_block_table —— 每拍
    #   只拷活跃行（_prepare_inputs 第一句先行拷贝 → m15）
    def commit_block_table(self, num_reqs: int) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L214
        self.block_table.copy_to_gpu(num_reqs)

    # SOURCE: vllm/v1/worker/block_table.py:L216 clear
    def clear(self) -> None:
        # SOURCE: vllm/v1/worker/block_table.py:L217-L218
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    # SUBTRACTED: map_to_kernel_blocks（L220-L248——分配块 id → kernel 块 id
    #   的细分展开，第 4 条 → ch14/22）。

    # SOURCE: vllm/v1/worker/block_table.py:L250 get_device_tensor —— 读侧
    #   出口（块表张量交给 attention metadata builder）
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


# SUBTRACTED: MultiGroupBlockTable（L270-L376——多组各一张块表的扇出容器，
#   第 4 条 → ch14）。单组全注意力下 InputBatch 持有的就是这一张 BlockTable。


# SOURCE: vllm/v1/worker/block_table.py:L379 _compute_slot_mapping_kernel ——
#   槽位换算恒等式本体（slot = 块号 × block_size + 块内偏移）
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
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # SUBTRACTED: TOTAL_CP_WORLD_SIZE / TOTAL_CP_RANK / CP_KV_CACHE_INTERLEAVE_
    #   SIZE 三 constexpr 参数（第 6 条——单卡 1/0/1 烘干：virtual_block_size
    #   不放大、is_local 恒真、local_block_offsets 恒等 → ch22）。
    # SOURCE: vllm/v1/worker/block_table.py:L397-L397
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

    # SUBTRACTED: TOTAL_CP_WORLD_SIZE 乘子（L413——单卡=1，烘干为常数）
    # SOURCE: vllm/v1/worker/block_table.py:L413-L414
    virtual_block_size = KV_CACHE_BLOCK_SIZE
    row_offset = req_idx * block_table_stride
    # SOURCE: vllm/v1/worker/block_table.py:L415-L420
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        virtual_block_indices = pos // virtual_block_size
        virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
        # SUBTRACTED: CP 分片（L421-L428——is_local 本秩判定与
        #   local_block_offsets 重排；单卡退化为恒等式 → ch22）

        # SOURCE: vllm/v1/worker/block_table.py:L430-L438（mask & is_local 的
        #   is_local 项单卡恒真、随 CP 烘干）
        block_indices = (
            virtual_block_indices * BLOCKS_PER_KV_BLOCK
            + virtual_block_offsets // block_size
        )
        block_numbers = tl.load(
            block_table_ptr + row_offset + block_indices,
            mask=mask,
            other=0,
        ).to(tl.int64)
        # SOURCE: vllm/v1/worker/block_table.py:L439-L442 恒等式本体
        slot_offsets = virtual_block_offsets % block_size
        slot_ids = block_numbers * block_size + slot_offsets
        # SUBTRACTED: 非本秩槽位打 PAD（L441 tl.where(is_local, ...)——单卡
        #   不触发 → ch22）
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)
