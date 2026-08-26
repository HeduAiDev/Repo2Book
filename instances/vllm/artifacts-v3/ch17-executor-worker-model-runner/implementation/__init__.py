# ch17 精简版包：只做减法的执行三层（对应 vllm/v1/executor/ + vllm/v1/worker/）。
#
# 铁律：与 pin v0.27.1 (6e448d0ea) 的真实源码同名、同结构、同控制流；只删
# dossier.subtraction_plan.delete 批准的分支，每处删除带 `# SUBTRACTED:`。
# 宿主替身（HOST SEAM）集中在 _host_seams.py / _shm_broadcast_seam.py，逐个登记在
# impl-notes.md 的 Seam 清单。

from . import _host_seams

# HOST SEAM: 真实 vllm 包缺席时，把 vLLM 的规范 qualname（如
# "vllm.v1.worker.gpu_worker.Worker"）映射到本精简包的同名模块——
# resolve_obj_by_qualname 得以逐字保留（vllm/utils/import_utils.py:L104-L110）。
_host_seams.install_vllm_module_aliases()
