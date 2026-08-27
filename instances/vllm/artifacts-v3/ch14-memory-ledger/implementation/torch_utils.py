# SOURCE: vllm/utils/torch_utils.py
# 本章消费面：get_dtype_size（页物理公式 real_page_size_bytes 的 dtype
# 字节项）。PIN_MEMORY 本章不用（块表缓冲 host 镜像直供 False）。
# SUBTRACTED: async_tensor_h2d / PIN_MEMORY 等其余工具（ch13 精简版消费面）。
import torch


# SOURCE: vllm/utils/torch_utils.py:L212 get_dtype_size
def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    # SOURCE: vllm/utils/torch_utils.py:L214
    return torch.tensor([], dtype=dtype).element_size()
