# ch22 explainer 驱动脚本 · m7 padded/unpadded 双口径裁决
# 同一实际 batch（120 真 token / 2 真 reqs），四种后端×图档组合下
# execute_model 交给 _get_slot_mappings 的口径（三元选择
# pad_attn or has_separate_kv_update → padded / else → unpadded）：
#   (FA 后端, NONE)        → has_separate_kv_update=True  → padded（120=unpadded）
#   (FA 后端, FULL 128×8)  → pad_attn=True                → padded 128
#   (默认后端, FULL 128×8) → pad_attn=True                → padded 128
#   (默认后端, NONE)       → 两者皆 False                 → unpadded 120
# FA 后端 forward_includes_kv_cache_update=False（KV 写独立成 op）——
# slot_mapping 必须用 padded 维度去匹配 key/value 张量。
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation._host_seams import (  # noqa: E402
    CUDAGraphMode,
    SchedulerOutputSeam,
    VllmConfigSeam,
    make_attn_group,
    make_kv_cache_config,
)
from implementation.flash_attn import FlashAttentionImpl  # noqa: E402
from implementation.forward_context import BatchDescriptor  # noqa: E402
from implementation.gpu_model_runner import GPUModelRunner  # noqa: E402

CPU = torch.device("cpu")
LAYER0 = "model.layers.0.self_attn.attn"
LAYER1 = "model.layers.1.self_attn.attn"


class _FALikeBackend:
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16]


class _DefaultBackend:
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1]


class _AttentionLayer:
    _k_scale = torch.tensor(1.0)
    _v_scale = torch.tensor(1.0)
    _q_scale = torch.tensor(1.0)

    def __init__(self, kv_cache):
        self.kv_cache = kv_cache
        self.impl = FlashAttentionImpl(
            num_heads=2, head_size=8, scale=1.0 / math.sqrt(8),
            num_kv_heads=1, alibi_slopes=None, sliding_window=None,
            kv_cache_dtype="auto")


class _ReqState:
    def __init__(self, req_id, block_ids):
        self.req_id = req_id
        self.block_ids = block_ids
        self.num_computed_tokens = 0
        self.output_token_ids = []
        self.num_tokens = 0


ROW_R0 = [3, 8, 2, 7, 1, 5, 9, 4]
ROW_R1 = [7, 1, 5, 2, 9, 4, 6, 8]


def run_config(fa_separate: bool, full_cudagraph: bool):
    backend_cls = _FALikeBackend if fa_separate else _DefaultBackend
    cfg = VllmConfigSeam(num_seqs=8, max_batched_tokens=128, max_model_len=512)
    kv_cfg = make_kv_cache_config(
        groups=[(LAYER0, 16, "full"), (LAYER1, 16, "full")])
    runner = GPUModelRunner(
        vllm_config=cfg, kv_cache_config=kv_cfg,
        attn_groups=[[make_attn_group(backend_cls, [LAYER0], 0)],
                     [make_attn_group(backend_cls, [LAYER1], 0)]],
        device=CPU,
    )
    pool = torch.zeros(32, 1, 16, 16)
    cfg.compilation_config.static_forward_context = {
        LAYER0: _AttentionLayer(pool)}
    runner.input_batch.num_reqs = 2
    runner.input_batch.req_ids = ["r0", "r1"]
    runner.input_batch.req_id_to_index = {"r0": 0, "r1": 1}
    runner.input_batch.num_computed_tokens_cpu[:2] = [0, 100]
    runner.input_batch.num_prompt_tokens_cpu_tensor[:2] = torch.tensor(
        [100, 120], dtype=torch.int32)
    runner.input_batch.block_table[0].add_row(ROW_R0, 0)
    runner.input_batch.block_table[1].add_row(ROW_R0, 0)
    runner.input_batch.block_table[0].add_row(ROW_R1, 1)
    runner.input_batch.block_table[1].add_row(ROW_R1, 1)
    runner.requests = {
        "r0": _ReqState("r0", [list(ROW_R0), list(ROW_R0)]),
        "r1": _ReqState("r1", [list(ROW_R1), list(ROW_R1)]),
    }
    if full_cudagraph:
        runner.seam_batch_desc = BatchDescriptor(num_tokens=128, num_reqs=8)
        runner.seam_cudagraph_mode = CUDAGraphMode.FULL
    else:
        runner.seam_batch_desc = BatchDescriptor(num_tokens=120, num_reqs=2)
        runner.seam_cudagraph_mode = CUDAGraphMode.NONE
    so = SchedulerOutputSeam(
        total_num_scheduled_tokens=120,
        num_scheduled_tokens={"r0": 100, "r1": 20},
        req_ids=["r0", "r1"],
        new_block_ids=[[[], []], [[], []]],
        num_computed_tokens=[0, 100],
        num_output_tokens=[0, 100],
    )
    assert runner.execute_model(so) is None
    sm = runner.execute_model_state.slot_mappings[LAYER0]
    # 重算裁决两开关（真实代码 L4307-L4318 / L4367-L4376 的判据）
    has_separate = not all(
        all(g.backend.forward_includes_kv_cache_update
            for g in runner.attn_groups[gid])
        for gid, _ in enumerate(kv_cfg.kv_cache_groups))
    pad_attn = runner.seam_cudagraph_mode == CUDAGraphMode.FULL
    return {
        "backend": "FlashAttention(separate_kv_update)" if fa_separate else "default(integrated)",
        "cudagraph_mode": "FULL" if full_cudagraph else "NONE",
        "has_separate_kv_update": has_separate,
        "pad_attn": pad_attn,
        "num_tokens_unpadded": 120,
        "num_tokens_padded_given": runner.seam_batch_desc.num_tokens,
        "slot_mapping_len": int(sm.shape[0]),
        "verdict": ("padded" if (pad_attn or has_separate)
                    else "unpadded"),
        "tail_is_minus1": bool(
            int(sm[-1]) == -1 and int(sm[119]) != -1) if sm.shape[0] == 128
        else None,
        "slot_tail_last8": [int(x) for x in sm[-8:]],
        "pad_slot_id": -1,
        "slot_99": int(sm[99]),
    }


configs = [
    run_config(fa_separate=True, full_cudagraph=False),
    run_config(fa_separate=True, full_cudagraph=True),
    run_config(fa_separate=False, full_cudagraph=True),
    run_config(fa_separate=False, full_cudagraph=False),
]

out = {
    "mechanism": "m7 padded/unpadded 双口径裁决",
    "trace_source": "run（runner 切面全链 execute_model）",
    "params": {
        "num_tokens_unpadded": 120, "capture_shape_tokens": 128,
        "num_reqs": 2, "capture_shape_reqs": 8,
        "fa_forward_includes_kv_cache_update": False,
        "default_forward_includes_kv_cache_update": True,
        "anchor": "vllm/v1/worker/gpu_model_runner.py:L4307-L4318,L4367-L4376",
    },
    "configs": configs,
}
path = os.path.join(os.path.dirname(__file__), "m7.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("m7 trace written:", path)
