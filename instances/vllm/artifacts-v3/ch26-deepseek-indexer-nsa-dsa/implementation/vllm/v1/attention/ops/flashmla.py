# SOURCE: vllm/v1/attention/ops/flashmla.py
# HOST SEAM（B1 CUDA kernel 镜像·精确数学，ch25 同款骨架 + 本章 sparse 面）：
# 真实文件在无 FlashMLA 扩展时把全部入口挂成 _raise_flashmla_unavailable
#（else 支，L87-L107——host 与真实无扩展平台同型入口面）；数值由
# vllm/third_party/flashmla/flash_mla_interface.py 的 HOST SEAM 镜像承载：
#   flash_mla_sparse_fwd     576/512 维潜向量上只对选中条目算（O(Lk)）——
#     softmax(q·k^T·scale)·v，-1 跳过、topk_length 截断
#   flash_mla_with_kvcache   分页缓存 + indices/topk_length 稀疏读 +
#     extra_k_cache/extra_indices_in_kvcache 双源（V4 一核双源的字面调用面）
#     + attn_sink（sink logit 进 softmax、无 value 贡献）
#   get_mla_metadata / FlashMLASchedMeta   tile scheduler 契约容器
from __future__ import annotations

from typing import NamedTuple

import torch

from vllm.logger import init_logger
from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401
    FlashMLASchedMeta,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.utils.import_utils import has_cutedsl  # noqa: F401  (cutedsl 探针位)

logger = init_logger(__name__)

# HOST SEAM：真实文件按 vllm._flashmla_C 可用性切换；host 走镜像（见上）
