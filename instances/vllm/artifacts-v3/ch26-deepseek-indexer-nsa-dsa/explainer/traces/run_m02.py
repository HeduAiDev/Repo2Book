# ch26-m02 IndexCache —— 132B/条布局回环 + spec 自报 + 第二本账的字节账。
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import (dump, make_small_hf_config, make_v32_hf_config,
                          make_vllm_config, ref_group_quant_ue8m0)

doc = {"mechanism": "ch26-m02 IndexCache：indexer 专属 K 缓存（132B/条）与第二本账",
       "source": "run_m02.py（host；量化/插入/gather 为 HOST SEAM 精确数学）",
       "code_anchor": "deepseek_v2.py:L616-L642（spec 自报）/ L686-L712（132B 装配）"
                      " / backends/mla/indexer.py:L442-L452（魔数 40 注释）"}

# ── A. spec 自报：num_kv_heads=1、head_size=132 ─────────────────────────────
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend, DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

hf = make_small_hf_config()
vllm_config = make_vllm_config(hf)
cache = DeepseekV32IndexerCache(head_dim=132, dtype=torch.uint8,
                                prefix="m02.k_cache",
                                cache_config=vllm_config.cache_config)
spec = cache.get_kv_cache_spec(vllm_config)
doc["spec_self_report"] = {
    "spec_type": type(spec).__name__,
    "num_kv_heads": int(spec.num_kv_heads),
    "head_size": int(spec.head_size),
    "dtype": str(spec.dtype).replace("torch.", ""),
    "only_one_vector_note": "『Only has one vector instead of K + V』"
                            "（deepseek_v2.py:L638-L641 注释原话）",
    "backend_v32": DeepseekV32IndexerBackend.__name__,
    "backend_v32_block_size": 64,
    "backend_v4_block_size": 256,
    "kv_cache_shape_v32_10_blocks": list(
        DeepseekV32IndexerBackend.get_kv_cache_shape(10, 64, 1, 132)),
    "kv_cache_shape_v4_10_blocks": list(
        DeepseekV4IndexerBackend.get_kv_cache_shape(10, 256, 1, 132)),
    "stride_order_identity": list(
        DeepseekV32IndexerBackend.get_kv_cache_stride_order(True)),
    "stride_order_note": "恒等排列=不支持跨层 KV 布局——『不兼容』信号",
}

# ── B. 132B/条布局：量化→插入→反读 回环实测 ────────────────────────────────
from vllm import _custom_ops as ops

torch.manual_seed(11)
block_size, T = 64, 4
kv_cache = torch.zeros(2, block_size, 132, dtype=torch.uint8)
k = torch.randn(T, 128)
slot_mapping = torch.arange(T, dtype=torch.int64)
ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, 128, "ue8m0")
row0 = kv_cache[0, 0]
scale0 = row0[128:132].view(torch.float32).item()
k_deq = row0[:128].view(torch.float8_e4m3fn).float() * scale0
_, s_ref = ref_group_quant_ue8m0(k[0:1])
doc["layout_roundtrip"] = {
    "bytes_per_token": 132,
    "value_bytes": 128,
    "scale_bytes": 4,
    "scale_is_power_of_2": bool(scale0 > 0 and
                                (scale0 & (scale0 - 1)) == 0
                                if scale0.is_integer() else True),
    "scale0": scale0,
    "scale0_matches_ue8m0_reference": scale0 == s_ref.item(),
    "dequant_max_abs_diff": f"{float((k_deq - k[0]).abs().max()):.6f}",
    "layout_note": "前 128B fp8 值 + 尾 4B fp32 scale（indexer_k_quant_and_cache "
                   "一步融合：量化+缓存插入；fp8 naive cache 注释 "
                   "deepseek_v2.py:L690-L693）",
    "material_note": "缓存的是量化索引键（打分原料），不是分数——分数每拍重算",
}

# ── C. workspace 魔数 40 账 + 第二本账字节对比 ─────────────────────────────
hf32 = make_v32_hf_config()
vllm32 = make_vllm_config(hf32, max_model_len=163840)
ws = get_max_prefill_buffer_size(vllm32)
L = 163840
doc["workspace_account"] = {
    "max_model_len": L,
    "workspace_entries": int(ws),
    "magic": 40,
    "magic_derivation": "(576 × 2 // 132) × 5 = 40——对齐 flashmla_sparse 的 "
                        "5×max_model_len workspace（每条 576×2 字节）最大化利用",
    "workspace_bytes": int(ws * 132),
    "workspace_mb": f"{ws * 132 / 1048576:.1f}",
    "source_comment_claim": "For DeepSeek-V3.2, the max_model_len is 163840. "
                            "40 * 163840 * 132 = 865075200 bytes = 825 MB"
                            "（indexer.py:L348-L349 注释原文）",
}
main_bf16_per_token = 576 * 2
doc["second_ledger_bytes"] = {
    "indexer_per_token_per_layer_bytes": 132,
    "main_mla_kv_per_token_per_layer_bytes_bf16": main_bf16_per_token,
    "indexer_fraction_of_main": f"{132 / main_bf16_per_token:.4f}",
    "indexer_fraction_percent": f"{132 / main_bf16_per_token * 100:.2f}",
    "per_layer_at_full_len_163840_indexer_bytes": int(132 * L),
    "per_layer_at_full_len_163840_main_bf16_bytes": int(main_bf16_per_token * L),
    "note": "与主 KV cache 分开分配、分开分组（spec 不同：uint8/132B vs "
            "主 MLA bf16/576 元素）——KV 账本分组归 ch14/ch25 站 6",
}

dump("m02.json", doc)
print("m02.json written | scale0 =", scale0,
      "| ws =", ws, "| ws_MB =", f"{ws * 132 / 1048576:.1f}",
      "| indexer/main =", f"{132 / main_bf16_per_token * 100:.2f}%")
