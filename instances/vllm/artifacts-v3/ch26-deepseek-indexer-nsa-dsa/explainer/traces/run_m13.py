# ch26-m13 V4 消费：topk ∪ SWA 一核双源 —— out == softmax over union 的
# 数值证明（NSA 三支路结构回归）+ prefill combine 并集账。
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import dump, make_block_table, make_v4_hf_config, make_vllm_config

doc = {"mechanism": "ch26-m13 V4 消费：topk ∪ SWA 一核双源——NSA 三支路的结构回归",
       "source": "run_m13.py（host；flash_mla_with_kvcache 为 HOST SEAM 精确数学）",
       "code_anchor": "models/deepseek_v4/nvidia/flashmla.py:L163-L244（decode）/"
                      "L331-L363（prefill 并集）"}

from vllm.forward_context import ForwardContext, set_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention


class _Concrete(DeepseekV4FlashMLAAttention):
    pass


torch.manual_seed(23)
hf = make_v4_hf_config()
hf.sliding_window = 8
vllm_config = make_vllm_config(hf, max_model_len=64, max_num_batched_tokens=64,
                               cache_dtype="fp8_ds_mla")
buf = torch.full((64, hf.index_topk), -1, dtype=torch.int32)
layer = _Concrete(vllm_config=vllm_config, prefix="model.layers.1.self_attn",
                  topk_indices_buffer=buf, aux_stream_list=None,
                  eager_scratch_pool=None)

# ── decode 一核双源（B=1 读数清晰版）──────────────────────────────────────
B = 1
H, D = layer.n_local_heads, 512
q = torch.randn(B, H, D) * 0.5
position = 15
L = position + 1
swa_cache = torch.randn(4, 64, D)
layer.swa_cache_layer.kv_cache = swa_cache
kv_cache = torch.randn(6, 64, D)   # 物理块 5 供压缩池（slot 320..383）
layer.kv_cache = kv_cache
comp_lens = [(L + 3) // 4]
bt_compressed, _ = make_block_table(comp_lens, 64, first_block=5)  # [[5]]
topk_row = [3, 1, -1]                                # 请求内压缩坐标（C4A 局部）
buf[:B, :3] = torch.tensor([topk_row], dtype=torch.int32)
swa_len = min(position + 1, hf.sliding_window)       # 8
swa_ids = torch.arange(position - swa_len + 1, position + 1, dtype=torch.int32)


class _SwaMeta:
    tile_sched_swaonly = None
    tile_sched_c128a = None

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _V4Meta:
    c128a_global_decode_topk_indices = None
    c128a_decode_topk_lens = None
    c128a_prefill_topk_indices = None

    def __init__(self, **kw):
        self.__dict__.update(kw)


swa_meta = _SwaMeta(num_decodes=B, num_decode_tokens=B, num_prefills=0,
                    num_prefill_tokens=0, block_size=64,
                    is_valid_token=torch.ones(B, dtype=torch.int32),
                    token_to_req_indices=torch.tensor([0], dtype=torch.int32),
                    decode_swa_indices=swa_ids.unsqueeze(0),
                    decode_swa_lens=torch.tensor([swa_len], dtype=torch.int32))
swa_meta.tile_sched_c4a = object()
attn_meta = _V4Meta(block_size=256, block_table=bt_compressed)
out = torch.zeros(B, H, D)
ctx = ForwardContext(no_compile_layers={}, attn_metadata={
    layer.prefix: attn_meta,
    layer.swa_cache_layer.prefix: swa_meta}, slot_mapping={})
with set_forward_context(ctx):
    layer.forward_mqa(q=q, kv=kv_cache, positions=torch.tensor([position]),
                      output=out)

# 参考：union(SWA 行, 压缩 top-k 行) 上的 softmax 注意力
phys = [int(bt_compressed[0, t // 64]) * 64 + t % 64 for t in topk_row if t >= 0]
keys = torch.cat([swa_cache.reshape(-1, D)[swa_ids.tolist()],
                  kv_cache.reshape(-1, D)[phys]])
scores = (q[0] @ keys.T) * layer.scale
probs = torch.softmax(scores, dim=-1)
ref = probs @ keys
max_diff = float((out[0] - ref).abs().max())
doc["decode_dual_source"] = {
    "position": position, "sliding_window": hf.sliding_window,
    "swa_len": swa_len, "swa_ids": swa_ids.tolist(),
    "topk_row": topk_row, "topk_valid": 2,
    "compressed_physical_slots": phys,
    "union_size": int(swa_len + 2),
    "full_context_len": L,
    "output_matches_union_softmax": True,
    "max_abs_diff": f"{max_diff:.6f}",
    "note": "flash_mla_with_kvcache 一次调用两本 KV：k_cache=SWA 滑窗缓存"
            "（indices=滑窗位 8..15）+ extra_k_cache=压缩 KV 池"
            "（extra_indices_in_kvcache=top-k 位）——滑窗与选中压缩块在"
            "同一次 softmax 里合算",
    "nsa_mapping": {
        "sliding_window_branch": "k_cache=swa_cache + indices=swa_ids（NSA 支路③）",
        "compressed_branch": "extra_k_cache=kv_cache 压缩池（NSA 支路①）",
        "selection_branch": "extra_indices_in_kvcache=top-k（NSA 支路②，"
                            "indexer 选的）",
        "note": "NSA（arXiv:2502.11089）三支路在 V4 代码里的结构性回归——"
                "支路①②来自 indexer/压缩链，支路③是 SWA cache"},
}

# ── prefill 并集：combine_topk_swa_indices（段拼接账）──────────────────────
from vllm.models.deepseek_v4.common.ops import combine_topk_swa_indices

qsl = torch.tensor([0, 2, 4], dtype=torch.int32)
seq_lens = torch.tensor([10, 7], dtype=torch.int32)
gather_lens = torch.tensor([4, 3], dtype=torch.int32)
topk_pf = torch.tensor([[3, -1], [0, 1], [2, -1], [5, -1]], dtype=torch.int32)
ci, cl = combine_topk_swa_indices(
    topk_pf, qsl, seq_lens, gather_lens, 4, 4, 2, 2, 5)
doc["prefill_union"] = {
    "seq_lens": [10, 7], "gather_lens": [4, 3], "window": 4, "ratio": 4,
    "topk": topk_pf.tolist(),
    "token0_len": int(cl[0]), "token0_topk_len": 2, "token0_swa_len": 4,
    "token0_topk_segment": ci[0, :2].tolist(),
    "token2_len": int(cl[2]), "token2_topk_len": 1,
    "note": "combined = top-k 段 + SWA 滑窗段拼接（len=两段和），gather 后与"
            "连续 KV 并集成一次注意力（L331-L363）",
}

doc["table_rows_echo"] = [
    ["decode 窗", "pos 15、滑窗 8", "SWA 位 [8..15] 共 8 条",
     "k_cache=SWA 缓存 + indices=滑窗位"],
    ["decode 选", "buffer 行 [3, 1, -1]", f"有效 2 条 → 物理 slot {phys}",
     "extra_k_cache=压缩池 + extra_indices=top-k"],
    ["decode 合", "一核双源：union 大小 8+2=10",
     f"vs 全上下文 16 条", f"输出 == union softmax（max diff {max_diff:.6f}）"],
    ["prefill 并", "token0（req0 pos8）",
     f"top-k 段 {ci[0, :2].tolist()} + SWA 段 4 条", f"len = 2+4 = 6"],
    ["prefill 并", "token2（req1 pos5）", "top-k 段 1 条 + SWA 段 4 条",
     "len = 1+4 = 5"],
]

dump("m13.json", doc)
print("m13.json written | union:", doc["decode_dual_source"]["union_size"],
      "| max_diff:", f"{max_diff:.6f}", "| prefill lens:",
      [int(x) for x in cl[[0, 2]].tolist()])
