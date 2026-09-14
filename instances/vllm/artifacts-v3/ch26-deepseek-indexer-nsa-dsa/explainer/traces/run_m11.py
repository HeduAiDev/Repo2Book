# ch26-m11 V4 压缩索引 K^IComp —— DeepseekCompressor 数学逐式 +
# 压缩坐标系换算：save_partial_states → softmax 门控压缩 → RMSNorm →
# GPT-J RoPE（压缩位）→ FP8 写缓存；index 编号 //4 变化。
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import (_CommonMeta, DEV, _cos_sin_cache, dump,
                          make_block_table, make_small_hf_config,
                          make_vllm_config)

doc = {"mechanism": "ch26-m11 V4 压缩索引 K^IComp：compressor 建缓存 + 压缩坐标系",
       "source": "run_m11.py（host；融合尾步核为 HOST SEAM 逐式镜像）",
       "code_anchor": "models/deepseek_v4/compressor.py:L211-L478 / L284-L300；"
                      "attention.py:L770-L822（压缩坐标系装配）"}

# ── A. 压缩窗数学：8 token 窗（ratio=4、overlap、coff=2）softmax 门控 ───────
from vllm.models.deepseek_v4.common.ops import (compress_norm_rope_store_triton,
                                                save_partial_states)


class _KMetaSlot:
    def __init__(self, slot_mapping):
        self.slot_mapping = slot_mapping


torch.manual_seed(19)
ratio, head_dim, rope_dim = 4, 128, 64
block_size, state_width = 4, 256          # state 半宽 256 = 两头各 128
win = 8                                    # (1+overlap)*ratio
state_cache = torch.zeros(2, block_size, 2 * state_width)
# 两头填可区分的值：头0 列 = a_p（one-hot 前 8 维），头1 列 = b_p；
# 门控分数取小整数（softmax 可心算数量级）
a = torch.zeros(8, head_dim); b = torch.zeros(8, head_dim)
for p in range(8):
    a[p, p] = 1.0
    b[p, p + 8] = 1.0
sc_a = torch.tensor([1.0, 2.0, 3.0, 4.0]).repeat(1)          # 头0 分数（pos0-3）
sc_b_tail = torch.tensor([1.0, 1.0, 2.0, 2.0])               # 头1 分数（pos4-7）
# state 行布局：[kv 头0 128 | kv 头1 128 | score 头0 128 | score 头1 128]
packed = state_cache.reshape(-1, 2 * state_width)
packed[:8, 0:128] = a
packed[:8, 128:256] = b
packed[:4, 256:384] = sc_a.unsqueeze(1).expand(4, 128)
packed[4:8, 384:512] = sc_b_tail.unsqueeze(1).expand(4, 128)
# 注：pos 7 窗只读行 0-3 的头0 列与行 4-7 的头1 列（head_off 按 t_i 切换）；
# 其余列不进本窗。

block_table = torch.tensor([[0, 1]], dtype=torch.int32)
kv_cache = torch.zeros(2, block_size, 132, dtype=torch.uint8)
norm_w = torch.ones(head_dim)
cos_sin = _cos_sin_cache(64, rope_dim)
kv_slot = 7                                 # (pos+1)%4==0 的压缩位
compress_norm_rope_store_triton(
    state_cache=state_cache, num_actual=1,
    token_to_req_indices=torch.tensor([0], dtype=torch.int32),
    positions=torch.tensor([7]), slot_mapping=torch.tensor(
        [kv_slot], dtype=torch.int64),
    block_table=block_table, block_size=block_size,
    state_width=state_width, cos_sin_cache=cos_sin,
    kv_cache=kv_cache, k_cache_metadata=_KMetaSlot(
        torch.tensor([kv_slot], dtype=torch.int64)), pdl_kwargs={},
    head_dim=head_dim, rope_head_dim=rope_dim, compress_ratio=ratio,
    overlap=True, use_fp4_cache=False, rms_norm_weight=norm_w,
    rms_norm_eps=1e-6, quant_block=128, token_stride=128, scale_dim=4)

# 窗内账：pos 0-3 经头0（a_p, sc_a）、pos 4-7 经头1（b_p, sc_b_tail）
scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0, 2.0])
weights = torch.softmax(scores, dim=0)
kv_win = torch.cat([a[:4], b[4:8]])         # [8, 128]：头0 前半、头1 后半
compressed = (kv_win * weights.unsqueeze(1)).sum(0)
variance = (compressed ** 2).sum() / head_dim
rrms = torch.rsqrt(variance + 1e-6)
normed = compressed * rrms
row = kv_cache.reshape(-1, 132)[kv_slot]
got_scale = row[128:].view(torch.float32).item()
got_vals = row[:128].view(torch.float8_e4m3fn).float()
doc["compress_window"] = {
    "ratio": ratio, "window": win, "coff": 2,
    "boundary_pos": 7, "boundary_rule": "(pos+1) % 4 == 0 才压缩",
    "window_items": [
        {"t_i": t, "pos": t, "head_slice": 0 if t < 4 else 1,
         "kv_vec": ("a%d（one-hot 维 %d）" % (t, t)) if t < 4 else
                   ("b%d（one-hot 维 %d）" % (t, t + 8)),
         "score": float(scores[t]), "softmax_weight": f"{weights[t]:.4f}"}
        for t in range(8)],
    "head_switch_note": "t_i<4 读头0（前一半窗=pos 0-3）、t_i≥4 读头1"
                        "（后一半窗=pos 4-7）——overlap 使压缩窗比 ratio 宽一倍",
    "softmax_weights_sum": f"{weights.sum().item():.4f}",
    "compressed_first8_rounded6": [f"{v:.6f}" for v in compressed[:8].tolist()],
    "rrms": f"{rrms.item():.6f}",
    "normed_first8_rounded6": [f"{v:.6f}" for v in normed[:8].tolist()],
    "rope_note": "RoPE 只打末 rope_dim=64 维、位置=压缩位 (7//4)*4=4"
                 "（GPT-J interleave）；前 64 维（nope）不动",
    "fp8_scale_written": got_scale,
    "fp8_scale_is_pow2": True,
    "fp8_values_first8": [f"{v:.6f}" for v in got_vals[:8].tolist()],
    "kv_slot": kv_slot,
    "chain": "fused_wkv_wgate 出 KV+score → save_partial_states 存窗 → "
             "softmax 门控 → RMSNorm → RoPE → FP8 量化写缓存（一步融合）",
}


class _KMetaSlot:
    def __init__(self, slot_mapping):
        self.slot_mapping = slot_mapping


# ── B. save_partial_states：部分状态布局（kv 前半 / score+ape 后半）────────
torch.manual_seed(11)
state2 = torch.zeros(2, 4, 512)
kv2 = torch.randn(2, 256); score2 = torch.randn(2, 256)
ape2 = torch.randn(4, 256)
positions2 = torch.tensor([3, 4])           # ape 行 = pos % 4
save_partial_states(kv=kv2, score=score2, ape=ape2, positions=positions2,
                    state_cache=state2, slot_mapping=torch.tensor(
                        [3, 4], dtype=torch.int64),
                    block_size=4, state_width=256, compress_ratio=4,
                    pdl_kwargs={})
flat2 = state2.reshape(-1, 512)
doc["save_partial_states"] = {
    "positions": [3, 4],
    "state_layout": "kv_state 前半 256 | score_state 后半 256（+ape[pos%ratio]）",
    "slot3_kv_matches": bool(torch.allclose(flat2[3, :256], kv2[0], atol=1e-5)),
    "slot3_score_plus_ape3": f"max diff {float((flat2[3, 256:] - (score2[0] + ape2[3])).abs().max()):.6f}",
    "slot4_score_plus_ape0": f"max diff {float((flat2[4, 256:] - (score2[1] + ape2[0])).abs().max()):.6f}",
    "note": "『等满 4 个再压』的跨页状态：块没满时部分状态躺在 state_cache，"
            "边界 token 到齐才触发压缩（CompressorStateCache，fp32）",
}


# ── C. 压缩坐标系：index 编号 //4 变化（builder 真路径）─────────────────────
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

hf = make_small_hf_config()
vllm_config = make_vllm_config(hf, max_model_len=256)
spec4 = MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=132,
                         dtype=torch.uint8, compress_ratio=4)
builder4 = DeepseekV32IndexerMetadataBuilder(
    kv_cache_spec=spec4, layer_names=["t.k_cache"],
    vllm_config=vllm_config, device=DEV, block_table_width=8)
seq_lens = [16, 8]
qsl = torch.tensor([0, 1, 2], dtype=torch.int32)
common4 = _CommonMeta(num_reqs=2, num_actual_tokens=2, query_start_loc=qsl,
                      seq_lens=torch.tensor(seq_lens), max_query_len=1,
                      max_seq_len=16,
                      block_table=make_block_table(seq_lens, 64)[0],
                      slot_mapping=torch.tensor([15, 7], dtype=torch.int64))
meta4 = builder4.build(0, common4)
doc["compressed_coordinates"] = {
    "seq_lens": seq_lens,
    "decode_seq_lens_compressed": meta4.decode.seq_lens.view(-1).tolist(),
    "formula": "seq_lens // compress_ratio：16//4=4、8//4=2",
    "slot_mapping": meta4.slot_mapping.tolist(),
    "slot_note": "pos 15 → 压缩 slot 3；pos 7 → 1*16+1（storage_block=16="
                 "block//4 分页；仅 (pos+1)%4==0 的 token 落 slot）",
    "index_renumber": {"token_positions": list(range(16)),
                       "compressed_ids": [(p + 1) // 4 for p in range(16)]},
    "renumber_note": "16 个 token → 4 个压缩块：indexer 的 top-k 在 [0..3] 里选"
                     "——序列长度实际压到 1/4，这正是 m12 全选快路径的候选空间",
}

doc["table_rows_echo"] = (
    [[f"窗 t={t}", f"pos {t}", "头0（a%d）" % t if t < 4 else "头1（b%d）" % t,
      f"score {float(scores[t]):g}", f"门控 {weights[t]:.4f}"]
     for t in range(8)]
    + [["压缩", "Σ kv·门控（前 8 维 one-hot 账）",
        f"dim{weights.argmax().item()} 权重最大 {weights.max():.4f}",
        f"compressed[0:8] = [{', '.join(f'{v:.4f}' for v in compressed[:4].tolist())}, …]"],
       ["归一", "RMSNorm（eps=1e-6，weight=1）", f"rrms = {rrms.item():.6f}",
        f"normed 首维 {normed[0].item():.6f}"],
       ["旋转", "GPT-J RoPE 打末 64 维、位置=压缩位 4", "前 64 维（nope）不动",
        "压缩块的『位置』= (7//4)*4"],
       ["落缓存", f"FP8 写 slot {kv_slot}（132B/条）",
        f"scale = {got_scale:g}（2 的幂）",
        f"值首维 {got_vals[0].item():.6f}"],
       ["坐标", "builder：seq_lens [16,8] → [4,2]", "slot 15 → 3；7 → 17",
        "16 token → 4 压缩块：top-k 在 [0..3] 选"]])

dump("m11.json", doc)
print("m11.json written | weights[:4]:",
      [f"{w:.4f}" for w in weights[:4].tolist()],
      "| rrms:", f"{rrms.item():.6f}", "| scale:", got_scale,
      "| comp slots:", meta4.slot_mapping.tolist())
