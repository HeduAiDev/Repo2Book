# SOURCE: vllm/envs.py
# HOST SEAM：vllm.envs 面的最小承载——本章消费面（真实默认值）：
#   VLLM_SPARSE_INDEXER_MAX_LOGITS_MB（indexer prefill logits 峰值预算，默认 512MB
#   ——envs.py:L60 与 L1053-L1054 的 lambda 定义处）
#   VLLM_BATCH_INVARIANT（MLAAttention 的 prefix-caching 联动位，默认 False）
import os

# SOURCE: vllm/envs.py:L60 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB —— 默认 512
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = int(
    os.getenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
)
# SOURCE: vllm/envs.py VLLM_BATCH_INVARIANT —— 默认 False（HOST SEAM 取默认）
VLLM_BATCH_INVARIANT = False
# SOURCE: vllm/envs.py VLLM_DCP_Q_REPLICATE —— 默认 False（HOST SEAM 取默认；
#   deepseek_v2.py L1026-L1033 的 qrep 探测消费位——单进程恒 False 死支）
VLLM_DCP_Q_REPLICATE = False
# SOURCE: vllm/envs.py VLLM_CUSTOM_OPS —— CustomOp 显式启停面（默认空=全走
#   禁用→forward_native 派发；本章 host 测试与此一致）
VLLM_CUSTOM_OPS = ""
