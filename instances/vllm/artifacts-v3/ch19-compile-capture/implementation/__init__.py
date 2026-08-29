# ch19 精简版包：只做减法的执行形态层（对应 vllm/config + vllm/compilation +
# vllm/model_executor/{custom_op,layers/{layernorm,attention}} + vllm/forward_context +
# vllm/v1/{cudagraph_dispatcher,worker} + vllm/utils/torch_utils）。
#
# 铁律：与 pin v0.27.1 (6e448d0ea) 的真实源码同名、同结构、同控制流；只删
# dossier.subtraction_plan.delete 批准的分支（+ 章范围外的域按 impl-notes
# §范围裁剪 定点收窄），每处删除带 `# SUBTRACTED:`。
# 宿主替身（HOST SEAM）集中在 _host_seams.py，逐个登记在 impl-notes.md。

from . import _host_seams

# HOST SEAM: 把本章保留代码按规范 qualname 解析的 vllm.* 路径代理到本包
# （resolve_obj_by_qualname 本体在 utils/import_utils.py 逐字保留）。
_host_seams.install_vllm_module_aliases()
