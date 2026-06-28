"""Subtract-only companion — patch/__init__.py 两段式总纲 + 三条样本台账.

Faithful subset of `vllm_ascend/patch/__init__.py`. This file has no runtime
code — it is the human-readable "patch 台账"。保留两段式总纲注释 + 与三个样本
对应的台账，其余几十条业务 patch 台账（minimax / glm / deepseek / qwen ...）删去。
"""

# SOURCE: vllm_ascend/patch/__init__.py:L17-L27
# ----------------------------------------------------------------------------------
# This module manage the patch for vllm. There are two folders in this module:
# - platform: contains the patches applied before worker starts. It's called by
#             `vllm_ascend.utils.adapt_patch(is_global_patch=True)` in
#             `vllm_ascend.platform.NPUPlatform.pre_register_and_update()` function.
# - worker: contains the patches applied when worker starts. It's called by
#           `vllm_ascend.utils.adapt_patch(is_global_patch=False)` in
#           each worker's `__init__` function.
#
# Once a new patch is added in vllm-ascend, please add the patch description into this file as well.
# ----------------------------------------------------------------------------------

# SUBTRACTED: 此文件其余 800+ 行逐条 patch 的 Why/How/Related PR/Future Plan 台账
#             （platform 段 ~16 项、worker 段 ~28 项）。仍正确：纯注释文档，不影响
#             控制流。下面只保留与本章三个样本对应的台账作对照。
#             原 vllm_ascend/patch/__init__.py:L28-EOF（节选保留如下）

# SOURCE: vllm_ascend/patch/__init__.py:L33-L43
# ** 1. File: platform/patch_distributed.py**
#   1. `torch.distributed.all_reduce`, `torch.distributed.broadcast`
#    Why:  tensor alignment for 310p
#    How:  rewrite all_reduce and broadcast in torch.distributed

# SOURCE: vllm_ascend/patch/__init__.py:L57-L68
# ** 3. File: platform/patch_multiproc_executor.py**
#   1. `vllm.v1.executor.multiproc_executor.MultiprocExecutor`
#    Why:  vLLM create child process with daemon=True, which doesn't work with EPLB
#          case, since EPLB will create a new process which is not allowed by daemon=True.
#    How:  Set daemon=False in MultiprocExecutor.

# SOURCE: vllm_ascend/patch/__init__.py:L383-L394
# ** 14. File: platform/patch_scheduler.py**
#   1. `vllm.v1.core.sched.scheduler.Scheduler._mamba_block_aligned_split`
#    Why:  Upstream vLLM has an assert logic, cause it fails when external KV connector hit
#    How:  remove the assert

# SOURCE: vllm_ascend/patch/__init__.py:L398-L409
# ** 1. File: worker/patch_distributed.py**
#   1. `vllm.distributed.parallel_state.GroupCoordinator`
#    Why:  vllm doesn't support all_to_all for GroupCoordinator.
#    How:  Add all_to_all implementation for GroupCoordinator.
