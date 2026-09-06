# ch21-m11 驱动：builder.build() 翻译——runner 每拍组装一份 CommonAttentionMetadata
# → FA builder.build() 把 Common 字段改名搬入（block_table_tensor→block_table）
# 并补算 FA 特有字段 → 按 layer_name 铺进 dict（组内共享同一对象）→ 混合组
# 同 (spec, builder) 缓存命中走 update_block_table 浅拷只换表。
# 场景：混布模型两个 KV cache 组（组0: L0/L1 同后端同组；组1: L2 同后端同
# spec——换表复用的最小复刻）；批 = 2 个 prefill 请求 + cudagraph padding。
# 末段附读写两腿数值对拍（m12 读腿图的数字出处）。
# 跑法：cd instances/vllm/artifacts-v3/ch21-attention-backends && python explainer/traces/run_ch21-m11.py
# 产物：explainer/traces/ch21-m11.json（trace_source="run"）
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation._host_seams import (  # noqa: E402
    DeviceCapability,
    VllmConfigSeam,
    _BlockTableSeam,
    _CpuGpuSeam,
    set_current_vllm_config,
)
from implementation.config.attention import AttentionConfig  # noqa: E402
from implementation.forward_context import set_forward_context  # noqa: E402
from implementation.platforms import cuda as cuda_platform  # noqa: E402
from implementation.v1.attention import selector  # noqa: E402
from implementation.v1.attention.backend import CommonAttentionMetadata  # noqa: E402
from implementation.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
)
from implementation.v1.attention.backends.registry import (  # noqa: E402
    AttentionBackendEnum,
    register_backend,
)
from implementation.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID  # noqa: E402
from implementation.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from implementation.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402

SM90 = DeviceCapability(9, 0)
register_backend(
    AttentionBackendEnum.FLASH_ATTN,
    "implementation.v1.attention.backends.flash_attn.FlashAttentionBackend",
)
selector._cached_get_attn_backend.cache_clear()
cuda_platform.CudaPlatform._seam_device_capability = SM90

L0 = "model.layers.0.self_attn"
L1 = "model.layers.1.self_attn"
L2 = "model.layers.2.self_attn"


def make_layer(cfg, name):
    from implementation.model_executor.layers.attention.attention import Attention

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with set_current_vllm_config(cfg):
            return Attention(
                num_heads=4, head_size=64, scale=0.125, num_kv_heads=2,
                prefix=name,
                attn_backend=FlashAttentionBackend,
            )
    finally:
        torch.set_default_dtype(prev)


out: dict = {}

# ── 装配（站 6-7）：两个 KV cache 组共用同一 spec 实例（换表复用的前提） ──
cfg = VllmConfigSeam(max_num_seqs=8, max_batched_tokens=512, max_model_len=256)
cfg.attention_config = AttentionConfig()
spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16)
runner = GPUModelRunner(cfg, device=torch.device("cpu"))
for name in (L0, L1, L2):
    cfg.compilation_config.static_forward_context[name] = make_layer(cfg, name)
size = 64 * spec.page_size_bytes
runner.kv_cache_config = KVCacheConfig(
    num_blocks=64,
    kv_cache_tensors=[KVCacheTensor(size=size, shared_by=[n]) for n in (L0, L1, L2)],
    kv_cache_groups=[
        KVCacheGroupSpec([L0, L1], spec),
        KVCacheGroupSpec([L2], spec),
    ],
)
runner.initialize_kv_cache(runner.kv_cache_config)

out["setup"] = {
    "kv_cache_groups": [
        {"gid": 0, "layer_names": [L0, L1], "backend": "FlashAttentionBackend"},
        {"gid": 1, "layer_names": [L2], "backend": "FlashAttentionBackend（同 spec 同 builder——换表复用的场景）"},
    ],
    "kv_tensor_shape_after_station7": list(runner.seam_kv_caches[L0].shape),
}

# ── 一拍的批状态（ch18/ch22 域产物的消费面，seam 直供） ───────────────────
# req0：prefill 4 个新 token（位置 12..15，历史 12，seq_len=16，块 [0,1]）
# req1：prefill 2 个新 token（位置 18..19，历史 18，seq_len=20，块 [2,3]）
# cudagraph padding：请求垫到 4 行、token 垫到 8 个。
qsl = torch.tensor([0, 4, 6, 6, 6], dtype=torch.int32)  # 前缀和 + 非递减尾（FA 要求）
seq_lens = torch.tensor([16, 20, 16, 20], dtype=torch.int32)  # 尾 2 行是上拍残留
table_g0 = torch.tensor([[0, 1], [2, 3], [9, 9], [9, 9]], dtype=torch.int32)  # 尾 2 行 stale
table_g1 = torch.tensor([[4, 5], [6, 7], [9, 9], [9, 9]], dtype=torch.int32)
# 槽位按 ch22 口径 slot = 块表[pos//16]×16+pos%16 直供（req1 位置 18/19 已越过块界、
# 落表行第 1 项）：req0 → g0 块 0（12..15）/ g1 块 4（76..79）；req1 → g0 块 3（50/51）/ g1 块 7（114/115）
slots_g0 = torch.tensor([12, 13, 14, 15, 50, 51, -1, -1], dtype=torch.int64)
slots_g1 = torch.tensor([76, 77, 78, 79, 114, 115, -1, -1], dtype=torch.int64)
NUM_REQS, NUM_TOKENS, PAD_REQS, PAD_TOKENS = 2, 6, 4, 8

runner.optimistic_seq_lens_cpu = seq_lens.clone()
runner.query_start_loc = _CpuGpuSeam(qsl.clone())
runner.seq_lens = seq_lens.clone()
runner.input_batch.block_tables[0] = _BlockTableSeam(table_g0.clone(), slots_g0.clone())
runner.input_batch.block_tables[1] = _BlockTableSeam(table_g1.clone(), slots_g1.clone())

out["batch"] = {
    "req0": {"new_tokens": 4, "positions": "12..15", "history": 12, "seq_len": 16, "blocks": [0, 1], "slots": [12, 13, 14, 15]},
    "req1": {"new_tokens": 2, "positions": "18..19", "history": 18, "seq_len": 20, "blocks": [2, 3], "slots": [50, 51]},
    "padding": {"num_reqs": NUM_REQS, "num_reqs_padded": PAD_REQS, "num_tokens": NUM_TOKENS, "num_tokens_padded": PAD_TOKENS,
                "mechanism": "块表尾行填 NULL_BLOCK_ID=0、槽位尾部填 -1（PAD_SLOT_ID）——cudagraph 捕获的是 max 形状，尾必须每次重填"},
    "NULL_BLOCK_ID": NULL_BLOCK_ID,
    "PAD_SLOT_ID": PAD_SLOT_ID,
}

# ── 站 8-9：_build_attention_metadata（cm_base → build 翻译 → 铺设 → 换表） ─
attn_metadata, _ = runner._build_attention_metadata(
    num_tokens=NUM_TOKENS,
    num_reqs=NUM_REQS,
    max_query_len=4,
    num_tokens_padded=PAD_TOKENS,
    num_reqs_padded=PAD_REQS,
    slot_mappings={0: slots_g0, 1: slots_g1},
)

m0 = attn_metadata[L0]
m2 = attn_metadata[L2]
out["translation"] = {
    "cm_base_shared_fields": {
        "query_start_loc": qsl.tolist(),
        "seq_lens": seq_lens.tolist(),
        "num_reqs": PAD_REQS,
        "num_actual_tokens": PAD_TOKENS,
        "max_query_len": 4,
        "max_seq_len": 20,
        "causal": True,
        "note": "所有后端所有层共享的字段每拍只算一次；block_table/slot_mapping 先填组 0 的",
    },
    "fa_build_rename": {
        "block_table_tensor→block_table": m0.block_table.tolist(),
        "slot_mapping→slot_mapping（直搬）": m0.slot_mapping.tolist(),
        "query_start_loc→query_start_loc（直搬）": m0.query_start_loc.tolist(),
        "seq_lens→seq_lens（直搬）": m0.seq_lens.tolist(),
    },
    "fa_only_fields": {
        "scheduler_metadata": None,
        "scheduler_metadata_note": "host 无 FA3 → schedule 闭包返回 None（真身 FA3 才产 AOT scheduler_metadata）",
        "use_cascade": m0.use_cascade,
        "sliding_window": list(m0.sliding_window),
        "num_decode_reqs": m0.num_decode_reqs,
        "num_prefill_reqs": m0.num_prefill_reqs,
    },
    "block_table_after_null_fill": {
        "rows_2_3_before": [9, 9],
        "rows_2_3_after": m0.block_table[2:].tolist(),
        "NULL_BLOCK_ID": NULL_BLOCK_ID,
    },
    "layer_name_fanout": {
        "m0_is_m1": attn_metadata[L0] is attn_metadata[L1],
        "note": "组内所有层共享同一份 metadata——attn_metadata[layer_name] = 同一对象",
    },
    "group1_table_swap_reuse": {
        "cm_shallow_copy": "copy(cm_base) 后只换 block_table_tensor/slot_mapping（组 0 之外的组只有这两样不同）",
        "m2_is_m0": m2 is m0,
        "m2_block_table": m2.block_table.tolist(),
        "m2_query_start_loc_is_m0_query_start_loc": m2.query_start_loc is m0.query_start_loc,
        "path": "组 1 同 (KVCacheSpec, builder 类型) 已建过且 builder.supports_update_block_table → update_block_table 浅拷只换表（证据：m2 非 m0 但共享 query_start_loc 张量）",
        "m2_slot_mapping": m2.slot_mapping.tolist(),
    },
}

# ── 读写两腿数值对拍（单请求切片：req0——m12 读腿图的数字出处） ────────────
torch.manual_seed(0)
attn0 = cfg.compilation_config.static_forward_context[L0]
q = torch.randn(4, 4, 64, dtype=torch.float16)
k = torch.randn(4, 2, 64, dtype=torch.float16)
v = torch.randn(4, 2, 64, dtype=torch.float16)
common_single = CommonAttentionMetadata(
    query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
    query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
    seq_lens=torch.tensor([16], dtype=torch.int32),
    num_reqs=1,
    num_actual_tokens=4,
    max_query_len=4,
    max_seq_len=16,
    block_table_tensor=torch.tensor([[0, 1]], dtype=torch.int32),
    slot_mapping=torch.arange(12, 16, dtype=torch.int64),
)
builder = runner.attn_groups[0][0].get_metadata_builder()
fa_meta_single = builder.build(common_prefix_len=0, common_attn_metadata=common_single)
with set_forward_context(
    {L0: fa_meta_single}, cfg, slot_mapping={L0: common_single.slot_mapping}
):
    o = attn0(q.view(4, 256), k.view(4, 128), v.view(4, 128))

kv = attn0.kv_cache
kc, vc = kv.transpose(1, 2).split(64, dim=-1)
write_ok = torch.equal(kc[0, 12:16].float(), k.float())

# 读腿对拍：query i 绝对位置 12+i，causal 只看前 13+i 个键
def manual_row(i, head=0):
    # K/V 历史（块 0 的 0..11 行）+ 本拍 4 个新 token（已由写腿落进 12..15 行）
    k_all = kc[0, : 12 + 4, 0].float()  # 头 0 的 16 个键（历史 12 + 新 4）
    v_all = vc[0, : 12 + 4, 0].float()
    n = 12 + i + 1  # query i 看前 n 个键（13/14/15/16）
    scores = (k_all[:n] @ q[i, 0].float()) * 0.125
    p = torch.softmax(scores.double(), dim=0)
    return p.float() @ v_all[:n]

diffs = {}
for i in (0, 3):
    ref = manual_row(i)
    got = o[i].float().view(4, 64)[0]
    diffs[f"row{i}_max_abs_diff"] = float((ref - got).abs().max())
out["forward_legs"] = {
    "scenario": "req0 单请求切片：4 个新 query（位置 12..15）读 16 个键（历史 12 + 新 4）",
    "write_leg": {
        "op": "reshape_and_cache_flash(k, v, key_cache, value_cache, slot_mapping)",
        "slot_mapping": [12, 13, 14, 15],
        "slot_formula": "slot = 块号×块大小 + 块内偏移（12 = 0×16+12 … 15 = 0×16+15）",
        "check_kc_block0_rows_12_15_equals_k": bool(write_ok),
    },
    "read_leg": {
        "op": "flash_attn_varlen_func(q, k=key_cache, v=value_cache, cu_seqlens_q=query_start_loc, seqused_k=seq_lens, block_table=block_table, causal=True)",
        "kv_view": "kv_cache.transpose(1, 2).split(head_size, dim=-1) 拆 K/V cache",
        "cu_seqlens_q": [0, 4],
        "seqused_k": [16],
        "block_table": [[0, 1]],
        "keys_seen_by_query": {"q0(pos 12)": 13, "q1(pos 13)": 14, "q2(pos 14)": 15, "q3(pos 15)": 16},
        "output_shape": list(o.shape),
        "manual_check_max_abs_diff": diffs,
        "scale": 0.125,
    },
}

path = os.path.join(os.path.dirname(__file__), "ch21-m11.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("wrote", path)
print(json.dumps(out, ensure_ascii=False, indent=2))
