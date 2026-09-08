# SOURCE: vllm/envs.py
# HOST SEAM：vllm.envs 面的最小承载——本章消费面只剩 VLLM_BATCH_INVARIANT
# （UnquantizedLinearMethod.apply / RMSNorm.forward_cuda 的分支开关）与
# VLLM_PP_LAYER_PARTITION（get_pp_indices 的手工分区覆盖）。真实默认值
# 均为 False/None（vllm/envs.py 的 envs.environment_dict 定义处）。
from __future__ import annotations

# SOURCE: vllm/envs.py VLLM_BATCH_INVARIANT —— 默认 False（HOST SEAM 取默认）
VLLM_BATCH_INVARIANT = False
# SOURCE: vllm/envs.py VLLM_PP_LAYER_PARTITION —— 默认 None（HOST SEAM 取默认）
VLLM_PP_LAYER_PARTITION = None
