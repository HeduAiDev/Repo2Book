# SOURCE: vllm/models/deepseek_v4/common/ops/__init__.py
# ch26 切面：本章消费面子集（MXFP4_BLOCK_SIZE / fused_indexer_q_rope_quant /
# compute_global_topk_indices_and_lens / combine_topk_swa_indices /
# dequantize_and_gather_k_cache / save_partial_states /
# compress_norm_rope_store_triton）。
from .cache_utils import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from .fused_compress_quant_cache import compress_norm_rope_store_triton
from .fused_indexer_q import MXFP4_BLOCK_SIZE, fused_indexer_q_rope_quant
from .save_partial_states import save_partial_states

__all__ = [
    "MXFP4_BLOCK_SIZE",
    "combine_topk_swa_indices",
    "compress_norm_rope_store_triton",
    "compute_global_topk_indices_and_lens",
    "dequantize_and_gather_k_cache",
    "fused_indexer_q_rope_quant",
    "save_partial_states",
]
