# vllm_ascend/attention/attention_mask.py —— subtract-only 精简版（注意力 mask 单例工厂）
#
# 本章 MHA 只用 get_attention_mask → get_splitfuse_attn_mask：一次造好 2048×2048 上三角
# splitfuse mask（int8）全程复用，喂给 npu_fused_infer_attention_score 的因果/滑窗分支。
# pooling 模型走 bool mask；MLA 专用 mask 是 ch20 内容，整体折叠。
import torch

from vllm_ascend.platform import ModelConfig
from vllm_ascend.utils import singleton


# SOURCE: vllm_ascend/attention/attention_mask.py:L22-L31
def _generate_attn_mask(max_seq_len, dtype):
    # Construct lower triangle matrix.
    mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
    # Create upper triangle matrix used to mark mask positions.
    mask_flag = ~mask_flag
    # Currently for fp16 dtype, the mask value should be set to -inf.
    mask_value = float("-inf") if dtype == torch.float16 else 1
    attn_mask = torch.zeros(size=(max_seq_len, max_seq_len), dtype=dtype).masked_fill_(mask_flag, mask_value)
    return attn_mask


# SOURCE: vllm_ascend/attention/attention_mask.py:L34-L35
@singleton  # 全局唯一：mask 缓存一次复用，避免每步重建。
class AttentionMaskBuilder:
    # SOURCE: vllm_ascend/attention/attention_mask.py:L36-L42
    def __init__(self, device: torch.device):
        self.attn_mask_cache = None
        self._seq_len_cached = 0
        self.device = device
        self.mla_mask = None
        self.chunked_prefill_attn_mask = None
        self.pcp_mla_mask = None

    # SOURCE: vllm_ascend/attention/attention_mask.py:L44-L51
    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype):
        if self.attn_mask_cache is None or max_seq_len > self._seq_len_cached:
            self.attn_mask_cache = _generate_attn_mask(max_seq_len, dtype)
            self._seq_len_cached = max_seq_len
        assert self.attn_mask_cache is not None, "Something is wrong in generate_attn_mask."
        if self.attn_mask_cache.dtype != dtype:
            self.attn_mask_cache = self.attn_mask_cache.to(dtype)
        return self.attn_mask_cache[:max_seq_len, :max_seq_len].contiguous().to(self.device, non_blocking=True)

    # SOURCE: vllm_ascend/attention/attention_mask.py:L53-L58
    def get_splitfuse_attn_mask(self) -> torch.Tensor:
        if self.chunked_prefill_attn_mask is None:
            # 2048×2048 上三角（int8），一次造好复用。
            self.chunked_prefill_attn_mask = (
                torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(self.device)
            )
        return self.chunked_prefill_attn_mask

    # SUBTRACTED: get_mla_mask / get_pcp_mla_mask / get_final_mla_mask（attention_mask.py:L60-L85）
    #   —— MLA 专用 mask，ch20 内容，非本章 MHA 主线。

    # SOURCE: vllm_ascend/attention/attention_mask.py:L75-L79
    def get_attention_mask(self, causal: bool, model_config: ModelConfig):
        if model_config.runner_type == "pooling":
            return self.get_attn_mask(2048, torch.bool)

        return self.get_splitfuse_attn_mask()
