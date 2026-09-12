# ch25 m12 —— DSV4 第三代：上下文压缩 compress_ratio=4 + fp8_ds_mla 584B 布局
# （vllm/models/deepseek_v4/attention.py:L210-L213 逐层解析 / L655-L674 spec 自报；
#   vllm/v1/kv_cache_interface.py:L403-L426 storage÷compress 与 584B/656B 特账）
# 字节账对照：DSV3 bf16 1152 B/token ↔ DSV3.2 fp8_ds_mla 656 B ↔ DSV4 584B/4=146 B
# + compress_ratios 逐层表与 MTP 恒 1 护栏 + SWA 子缓存另报。
from __future__ import annotations

import torch
from types import SimpleNamespace

from trace_common import T, dump

from vllm.v1.kv_cache_interface import MLAAttentionSpec

BLOCK = 64


def _make(compress_ratios, cache_dtype="auto"):
    hf = SimpleNamespace(
        model_type="deepseek_v4", hidden_size=512,
        num_attention_heads=8, q_lora_rank=192, o_lora_rank=128,
        head_dim=512, qk_rope_head_dim=64, o_groups=4,
        sliding_window=4096, rms_norm_eps=1e-6,
        max_position_embeddings=512,
        num_hidden_layers=len(compress_ratios),
        compress_ratios=list(compress_ratios),
        index_topk=2048,
    )
    return T.enable_ref_backends(T.make_vllm_config(hf_cfg=hf,
                                                    cache_dtype=cache_dtype)), hf


def _build_attention(vllm_config, hf, layer_id, fp8_layout=None):
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    class _NvidiaLike(DeepseekV4Attention):
        if fp8_layout is None:
            use_fp8_ds_mla_layout = not vllm_config.cache_config.cache_dtype == "auto"
        else:
            use_fp8_ds_mla_layout = fp8_layout

        @classmethod
        def get_padded_num_q_heads(cls, num_heads):
            return num_heads

        def forward_mqa(self, q, kv, positions, output):
            raise NotImplementedError

        def _o_proj(self, o, positions):
            raise NotImplementedError

    with T.set_current_vllm_config(vllm_config):
        return _NvidiaLike(vllm_config, f"model.layers.{layer_id}.self_attn")


# ── 字节账三布局（真实 MLAAttentionSpec.real_page_size_bytes）──
bf16_dsv3 = MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=576,
                             dtype=torch.bfloat16)
v32 = MLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=576,
                       dtype=torch.uint8, cache_dtype_str="fp8_ds_mla")
v4 = MLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=512,
                      dtype=torch.uint8, cache_dtype_str="fp8_ds_mla",
                      compress_ratio=4, alignment=576,
                      model_version="deepseek_v4")

per_token = {
    "dsv3_bf16": bf16_dsv3.real_page_size_bytes // BLOCK,       # 73728/64=1152
    "dsv32_fp8_ds_mla": v32.real_page_size_bytes // 256,         # 167936/256=656
    "dsv4_fp8_ds_mla": v4.real_page_size_bytes,                  # 64×584=37376
    "dsv4_per_stored_slot": v4.real_page_size_bytes // v4.storage_block_size,  # 584
    "dsv4_per_token_effective": v4.real_page_size_bytes / v4.storage_block_size / 4,  # 146
    "dsv4_swa_uncompressed_bf16": 512 * 2,                      # 1024
}
ratio_v32 = per_token["dsv3_bf16"] / per_token["dsv32_fp8_ds_mla"]
ratio_v4 = per_token["dsv3_bf16"] / per_token["dsv4_per_token_effective"]

# ── compress_ratios 逐层解析 + MTP 护栏（attention.py:L210-L213）──
# 每层只建一次（static_forward_context 禁止重复注册），spec 自报复用同一实例
ratios = [1, 4, 128, 4, 1]
vc, hf = _make(ratios)
parsed = []
layers = {}
for i in range(len(ratios)):
    layers[i] = _build_attention(vc, hf, i)
    parsed.append({"layer": i, "config_ratio": ratios[i],
                   "compress_ratio": int(layers[i].compress_ratio)})
attn_mtp = _build_attention(vc, hf, len(ratios))               # MTP 层
vc2, hf2 = _make([0, 4])
attn_zero = _build_attention(vc2, hf2, 0)                      # ratio=0 护栏

# ── spec 自报：SWA 层 None / C4 层 uint8+576 对齐（attention.py:L655-L674）──
a_swa = layers[0]                                              # ratio 1 → SWA
a_c4 = layers[1]                                               # ratio 4 → 压缩层
with T.set_current_vllm_config(vc):
    spec_swa_ret = a_swa.get_kv_cache_spec(vc)                 # None
    spec_c4 = a_c4.get_kv_cache_spec(vc)
vc8, hf3 = _make([4], cache_dtype="fp8_ds_mla")
a4 = _build_attention(vc8, hf3, 0, fp8_layout=True)
with T.set_current_vllm_config(vc8):
    spec_c4_fp8 = a4.get_kv_cache_spec(vc8)

# ── SWA 子缓存自报（sparse_swa.py:L87-L102）──
# attention 层 __init__ 已以独立 prefix 装好自己的 DeepseekV4SWACache——直接取用
swa_cache = a_swa.swa_cache_layer
with T.set_current_vllm_config(vc):
    spec_swa_sub = swa_cache.get_kv_cache_spec(vc)

# ── get_mla_dims 双记法（mla_attention.py:L1499-L1530）──
from vllm.model_executor.layers.attention.mla_attention import get_mla_dims
v4cfg = SimpleNamespace(compress_ratios=[1], head_dim=512,
                        qk_rope_head_dim=64, q_lora_rank=1536)
dims4 = get_mla_dims(SimpleNamespace(hf_text_config=v4cfg))

trace = {
    "mechanism": "ch25-m12",
    "source": "run_m12.py @ implementation/（vLLM v0.27.1 只做减法精简版, host CPU）",
    "code_anchor": "vllm/models/deepseek_v4/attention.py:L210-L213（compress_ratios 逐层）· L655-L674（DSV4 spec 自报）· vllm/v1/kv_cache_interface.py:L403-L426（storage÷compress + 584B/656B 特账）· vllm/v1/attention/backends/mla/sparse_swa.py:L87-L102（SWA 子缓存自报）",
    "byte_account": {
        "dsv3_bf16": {
            "per_token_bytes": per_token["dsv3_bf16"],
            "layout": "512 bf16 nope + 64 bf16 rope = 576 元素 × 2B",
            "real_page_size_bytes_block64": bf16_dsv3.real_page_size_bytes,
        },
        "dsv32_fp8_ds_mla": {
            "per_token_bytes": per_token["dsv32_fp8_ds_mla"],
            "layout": "512B fp8 nope + 16B（4 个 fp32 scale）+ 128B bf16 rope = 656B（flashmla_sparse.py:L64-L84 注释）",
            "real_page_size_bytes_block256": v32.real_page_size_bytes,
        },
        "dsv4_fp8_ds_mla": {
            "per_stored_slot_bytes": per_token["dsv4_per_stored_slot"],
            "layout": "448B fp8 nope + 128B bf16 rope（不量化保精度）+ 8B scale（7 个 ue8m0 + 1B pad）= 584B",
            "storage_block_size": int(v4.storage_block_size),     # 256//4=64
            "real_page_size_bytes": v4.real_page_size_bytes,      # 64×584=37376
            "page_size_padded": int(v4.page_size_padded),         # ceil(37376/576)*576
            "per_token_effective_bytes": per_token["dsv4_per_token_effective"],  # 146
            "alignment": int(v4.alignment),                       # 576
        },
        "compression_vs_dsv3": {
            "dsv32_vs_dsv3": round(ratio_v32, 2),                 # 1.75
            "dsv4_vs_dsv3": round(ratio_v4, 1),                   # 7.9
            "note": "584B 是每『存储格』（覆盖 4 个 token）的字节——compress_ratio=4 后每 token 有效字节 146；对照 DSV3 bf16 1152B 压 7.9 倍",
        },
    },
    "compress_ratio_per_layer": {
        "config_ratios": ratios,
        "parsed": parsed,
        "mtp_layer_ratio": int(attn_mtp.compress_ratio),          # 1
        "zero_ratio_guard": int(attn_zero.compress_ratio),        # max(1,0)=1
        "guard_source": "attention.py:L210-L213（layer_id < num_hidden_layers 取 max(1, compress_ratios[layer_id])，MTP 层恒 1——NOTE(zyongye) Compress ratio can't be 0）",
    },
    "spec_self_report": {
        "swa_layer_returns_none": spec_swa_ret is None,
        "swa_none_note": "compress_ratio<=1 层 get_kv_cache_spec 返回 None——SWA 段由 DeepseekV4SWACache 以独立 prefix 另报",
        "c4_layer_spec": {
            "head_size": int(spec_c4.head_size),                  # 512
            "dtype": str(spec_c4.dtype).replace("torch.", ""),    # uint8
            "alignment": int(spec_c4.alignment),                  # 512（非 fp8 布局自然页）
            "compress_ratio": int(spec_c4.compress_ratio),        # 4
            "model_version": spec_c4.model_version,               # deepseek_v4
            "storage_block_size": int(spec_c4.storage_block_size),
        },
        "c4_layer_spec_fp8_layout": {
            "dtype": str(spec_c4_fp8.dtype).replace("torch.", ""),
            "alignment": int(spec_c4_fp8.alignment),              # 576
            "real_page_size_bytes": int(spec_c4_fp8.real_page_size_bytes),
        },
        "swa_subcache_spec": {
            "type": type(spec_swa_sub).__name__,                  # SlidingWindowMLASpec
            "sliding_window": int(spec_swa_sub.sliding_window),   # 4096
            "num_kv_heads": int(spec_swa_sub.num_kv_heads),       # 1
            "block_size": int(spec_swa_sub.block_size),           # 64（与 C4A [256/4] 共页）
            "head_size": int(spec_swa_sub.head_size),             # 512
            "real_page_size_bytes": int(spec_swa_sub.real_page_size_bytes),  # 64×512×2=65536
        },
    },
    "get_mla_dims_dual_format": {
        "v4_unified": {"head_dim": 512, "qk_rope_head_dim": 64,
                       "derived_nope": int(dims4.qk_nope_head_dim),   # 448
                       "kv_lora_rank": int(dims4.kv_lora_rank)},      # 512
        "note": "DSV4 config 用统一 head_dim=512 记法，get_mla_dims 换算 nope=512-64=448——记法漂移的税由框架侧吃（mla_attention.py:L1499-L1530）",
    },
}

assert per_token["dsv3_bf16"] == 1152 and per_token["dsv32_fp8_ds_mla"] == 656
assert per_token["dsv4_per_stored_slot"] == 584
assert v4.storage_block_size == 64 and v4.real_page_size_bytes == 37376
assert int(attn_mtp.compress_ratio) == 1 and int(attn_zero.compress_ratio) == 1
assert spec_swa_ret is None and spec_c4_fp8.alignment == 576
assert int(spec_swa_sub.block_size) == 64
dump("m12.json", trace)
