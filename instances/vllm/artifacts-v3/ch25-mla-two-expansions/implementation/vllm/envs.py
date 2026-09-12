# SOURCE: vllm/envs.py
# HOST SEAM：vllm.envs 面的最小承载——本章消费面（真实默认值）：
#   VLLM_MLA_DISABLE（config/model.py:L1792 use_mla 的否决位，默认 False）
#   VLLM_BATCH_INVARIANT（delete[5] 已删消费分支，保常量位防回归）
#   Q/K/V_SCALE_CONSTANT（set_default_quant_scales 的量程常数，真实 208.0/
#   300.0/360.0——vllm/envs.py 定义处）
from __future__ import annotations

# SOURCE: vllm/envs.py VLLM_MLA_DISABLE —— 默认 False（HOST SEAM 取默认）
VLLM_MLA_DISABLE = False
# SOURCE: vllm/envs.py VLLM_BATCH_INVARIANT —— 默认 False（HOST SEAM 取默认）
VLLM_BATCH_INVARIANT = False
# SOURCE: vllm/envs.py Q_SCALE_CONSTANT —— 默认 208.0（HOST SEAM 取默认）
Q_SCALE_CONSTANT = 208.0
# SOURCE: vllm/envs.py K_SCALE_CONSTANT —— 默认 300.0（HOST SEAM 取默认）
K_SCALE_CONSTANT = 300.0
# SOURCE: vllm/envs.py V_SCALE_CONSTANT —— 默认 360.0（HOST SEAM 取默认）
V_SCALE_CONSTANT = 360.0
