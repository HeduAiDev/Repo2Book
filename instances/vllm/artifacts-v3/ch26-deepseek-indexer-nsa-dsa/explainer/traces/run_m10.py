# ch26-m10 V4 三类层 —— compress_ratios 逐层表的真实装配证据：
# SWAonly(1)/C4A(4)/C128A(128) 三类层实例化，只有 C4A 建 indexer；
# C128A 压缩后 candidates≤topk → metadata 期直算全选；MTP 层恒 1。
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import dump, make_v4_hf_config, make_vllm_config

doc = {"mechanism": "ch26-m10 V4 三类层：SWAonly(1)/C4A(4)/C128A(128) 与 "
                    "compress_ratios 逐层表",
       "source": "run_m10.py（host；真实装配路径实例化）",
       "code_anchor": "models/deepseek_v4/attention.py:L206-L297（逐层表+建 indexer）"
                      " / backends/mla/sparse_swa.py:L38-L53（三类常量）"}

from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.v1.attention.backends.mla.sparse_swa import _layer_type_for


class _Concrete(DeepseekV4Attention):
    @classmethod
    def get_padded_num_q_heads(cls, num_heads):
        return num_heads

    def forward_mqa(self, q, kv, positions, output):
        raise NotImplementedError

    def _o_proj(self, o, positions):
        raise NotImplementedError


ratios = [1, 4, 128]
kinds, layers = [], {}
for i, r in enumerate(ratios):
    hf = make_v4_hf_config()
    hf.compress_ratios = [r, r, r]
    vllm_config = make_vllm_config(hf, max_model_len=1000,
                                   max_num_batched_tokens=64,
                                   cache_dtype="fp8_ds_mla")
    buf = torch.zeros(64, hf.index_topk, dtype=torch.int32)
    layer = _Concrete(vllm_config=vllm_config,
                      prefix=f"model.layers.{i}.self_attn",
                      topk_indices_buffer=buf, aux_stream_list=None,
                      eager_scratch_pool=None)
    kinds.append(layer.indexer is not None)
    layers[r] = layer

c4a = layers[4].indexer
doc["layer_types"] = {
    "ratios_tried": ratios,
    "indexer_built": kinds,
    "mapping": {f"ratio_{r}": _layer_type_for(r) for r in ratios},
    "c4a_max_model_len_compressed": int(c4a.max_model_len),
    "c4a_max_model_len_compressed_note": "max_model_len 1000 → 1000//4 = 250："
                                         "indexer 活在 //4 压缩坐标系",
    "c4a_skip_k_cache_insert": bool(c4a.indexer_op.skip_k_cache_insert),
    "c4a_skip_note": "skip_k_cache_insert=True：K^IComp 插入归 compressor"
                     "（attention.py:L820；m11）",
    "only_c4a_note": "『Only C4A uses sparse attention and hence has indexer』"
                     "（attention.py:L277-L278 注释原话）",
    "swa_block_size": int(layers[1].swa_cache_layer.block_size),
    "swa_note": "三类层同层并存 SWA 滑窗缓存（SlidingWindowMLASpec）——SWA 与 "
                "C4A 压缩块共享物理张量页（sparse_swa.py:L81-L83 注释："
                "C4A KV 块 [256//4, head_dim]=[64, head_dim] 与 SWA 块同页宽）",
    "per_layer_table_note": "compress_ratios 是 checkpoint 的逐层表"
                            "（attention.py:L207-L213：layer_id < num_hidden_layers "
                            "按表取、MTP 层恒 1）——具体Pattern 随模型发布，"
                            "vLLM 不设默认值",
}

# MTP 护栏：layer_id >= num_hidden_layers → compress_ratio = 1
hf_mtp = make_v4_hf_config()
hf_mtp.compress_ratios = [4, 4, 4]
vllm_mtp = make_vllm_config(hf_mtp, max_model_len=1000, max_num_batched_tokens=64,
                            cache_dtype="fp8_ds_mla")
layer_mtp = _Concrete(vllm_config=vllm_mtp, prefix="model.layers.5.self_attn",
                      topk_indices_buffer=torch.zeros(64, 8, dtype=torch.int32),
                      aux_stream_list=None, eager_scratch_pool=None)
doc["layer_types"]["mtp_layer_id"] = 5
doc["layer_types"]["mtp_num_hidden_layers"] = 3
doc["layer_types"]["mtp_compress_ratio"] = int(layer_mtp.compress_ratio)

# C128A 全选账：压缩后 candidates ≤ topk → metadata 期直算全选
topk_real = 2048
doc["c128a_full_select"] = {
    "topk": topk_real,
    "max_model_len_163840": 163840,
    "c128a_candidates": 163840 // 128,
    "c128a_full_select": bool(163840 // 128 <= topk_real),
    "c4a_candidates": 163840 // 4,
    "c4a_needs_scoring": bool(163840 // 4 > topk_real),
    "c4a_boundary_len": topk_real * 4,
    "note": "C128A：163840//128 = 1280 ≤ 2048 → 全选即最优，metadata 期直算"
            "（_build_c128a_metadata，不经 indexer）；C4A：40960 > 2048 → "
            "真打分；边界：上下文 ≤ 8192（=4×2048）时 C4A 也全选（m12 快路径）",
}

dump("m10.json", doc)
print("m10.json written | kinds:", kinds,
      "| c128a candidates:", doc["c128a_full_select"]["c128a_candidates"],
      "| mtp ratio:", layer_mtp.compress_ratio)
