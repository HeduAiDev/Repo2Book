# SpecDecodeMetadata 契约容器（vllm/v1/spec_decode/metadata.py:L9-L66）。
# 真实行为基准：
#   - __post_init__ 算 max_spec_len = max(num_draft_tokens)（L26-L27）
#   - make_dummy（L29-L66）：num_draft/num_sampled 累积和 int32、
#     draft_token_ids 摊平 int32、三组 index 全零占位（warmup/cudagraph 预热用）
import numpy as np
import torch

from conftest import make_sampling_metadata  # noqa: F401  (确保 sys.path 就位)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def test_post_init_max_spec_len(device):
    md = SpecDecodeMetadata(
        draft_token_ids=torch.tensor([5, 6, 7], dtype=torch.int32, device=device),
        num_draft_tokens=[2, 1],
        cu_num_draft_tokens=torch.tensor([2, 3], dtype=torch.int32, device=device),
        cu_num_sampled_tokens=torch.tensor([3, 5], dtype=torch.int32, device=device),
        target_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        bonus_logits_indices=torch.tensor([3, 4], dtype=torch.int32, device=device),
        logits_indices=torch.tensor(
            [0, 1, 2, 3, 4], dtype=torch.int32, device=device
        ),
    )
    assert md.max_spec_len == 2


def test_make_dummy_layout(device):
    md = SpecDecodeMetadata.make_dummy([[10, 11, 12], [20]], device=device)
    assert md.num_draft_tokens == [3, 1]
    assert md.max_spec_len == 3
    assert md.draft_token_ids.tolist() == [10, 11, 12, 20]
    assert md.draft_token_ids.dtype == torch.int32
    # 含末项累积和：draft [3,4]、sampled（每请求 +1 bonus）[4,6]
    assert md.cu_num_draft_tokens.tolist() == [3, 4]
    assert md.cu_num_sampled_tokens.tolist() == [4, 6]
    # warmup/cudagraph 预热位：三组 index 全零占位（真实 L51-L57；宽度
    # = num_tokens / batch / num_tokens+batch）
    assert md.target_logits_indices.tolist() == [0, 0, 0, 0]
    assert md.bonus_logits_indices.tolist() == [0, 0]
    assert md.logits_indices.tolist() == [0] * 6


def test_make_dummy_empty_batch(device):
    md = SpecDecodeMetadata.make_dummy([[], []], device=device)
    assert md.num_draft_tokens == [0, 0]
    assert md.max_spec_len == 0
    assert md.draft_token_ids.numel() == 0
    assert md.cu_num_draft_tokens.tolist() == [0, 0]
    assert md.cu_num_sampled_tokens.tolist() == [1, 2]
