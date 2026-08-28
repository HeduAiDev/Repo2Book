# SOURCE: vllm/utils/torch_utils.py
# 本章消费面：PIN_MEMORY——CpuGpuBuffer / InputBatch / runner 持久缓冲的
# pinned 默认开关。
# SUBTRACTED: get_dtype_size / async_tensor_h2d / is_quantized_kv_cache 等其余
#   工具（ch13/ch14/ch17 域；消费点均在已删除段）。

# SOURCE: vllm/utils/torch_utils.py:L72 PIN_MEMORY = is_pin_memory_available()
# HOST SEAM：CPU host 无 pinned memory（真实在 CUDA 主机上为 True；容器内真
# GPU 环境为 True）。pinned 与否不影响本包任何行为分支——只影响拷贝速度。
PIN_MEMORY = False
