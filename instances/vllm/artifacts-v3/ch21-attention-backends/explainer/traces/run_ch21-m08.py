# ch21-m08 驱动：逐 KV 组混布——initialize_kv_cache 全链（内含站 6 归组 +
# 最弱链 CG 降级 + kernel 块协商 256→4×64 + 站 7 定形 bind）。
# 层配置：L0/L1 用 FlashAttentionBackend、L2 用 TRITON_ATTN 桩——同一模型
# 同一 KV cache 组内逐层混布不同后端，看 (后端类全名, spec, num_heads_q)
# 等价类怎么把 3 层聚成 2 个 AttentionGroup。
# 跑法：cd instances/vllm/artifacts-v3/ch21-attention-backends && python explainer/traces/run_ch21-m08.py
# 产物：explainer/traces/ch21-m08.json（trace_source="run"）
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation._host_seams import (  # noqa: E402
    DeviceCapability,
    VllmConfigSeam,
    set_current_vllm_config,
)
from implementation.config.attention import AttentionConfig  # noqa: E402
from implementation.v1.attention.backend import MultipleOf  # noqa: E402
from implementation.platforms import cuda as cuda_platform  # noqa: E402
from implementation.v1.attention import selector  # noqa: E402
from implementation.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
)
from implementation.v1.attention.backends.registry import (  # noqa: E402
    AttentionBackendEnum,
    register_backend,
)
from implementation.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from implementation.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402

SM90 = DeviceCapability(9, 0)


class TritonStubBackend(FlashAttentionBackend):
    """TRITON_ATTN 桩：继承 FA 探针面，只改块大小声明为固定 [24, 64]。"""

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [24, 64]


register_backend(
    AttentionBackendEnum.FLASH_ATTN,
    "implementation.v1.attention.backends.flash_attn.FlashAttentionBackend",
)
register_backend(
    AttentionBackendEnum.TRITON_ATTN,
    __name__ + ".TritonStubBackend",
)
selector._cached_get_attn_backend.cache_clear()
cuda_platform.CudaPlatform._seam_device_capability = SM90

L0 = "model.layers.0.self_attn"
L1 = "model.layers.1.self_attn"
L2 = "model.layers.2.self_attn"


def make_layer(cfg, name, backend_cls):
    from implementation.model_executor.layers.attention.attention import Attention

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)  # 真实路径：load 期默认 dtype = 模型 dtype
    try:
        with set_current_vllm_config(cfg):
            return Attention(
                num_heads=4,
                head_size=64,
                scale=0.125,
                num_kv_heads=2,
                prefix=name,
                attn_backend=backend_cls,
            )
    finally:
        torch.set_default_dtype(prev)


out: dict = {}
spec = FullAttentionSpec(block_size=256, num_kv_heads=2, head_size=64, dtype=torch.float16)
out["params"] = {
    "layers": [
        {"name": L0, "backend": "FlashAttentionBackend", "spec": "FullAttentionSpec(block_size=256, num_kv_heads=2, head_size=64)"},
        {"name": L1, "backend": "FlashAttentionBackend", "spec": "同 L0（等价类同键）"},
        {"name": L2, "backend": "TritonStubBackend（TRITON_ATTN 桩）", "spec": "同 L0（spec 相同、后端类全名不同）"},
    ],
    "num_heads_q": 4,
    "kv_cache_groups": 1,
    "note_backend_per_kind": "同一模型逐层混布的用户入口是 AttentionConfig.backend_per_kind（按 KVCacheSpecKind 覆写全局后端）；本驱动经 Attention(attn_backend=...) 直传构造（__init__ 的另一半入口），归组逻辑与真实 load 是同一条 initialize_attn_backend",
}

cfg = VllmConfigSeam(max_num_seqs=8, max_batched_tokens=512, max_model_len=256)
cfg.attention_config = AttentionConfig()
layers = [
    (L0, FlashAttentionBackend),
    (L1, FlashAttentionBackend),
    (L2, TritonStubBackend),
]
runner = GPUModelRunner(cfg, device=torch.device("cpu"))
for name, backend_cls in layers:
    cfg.compilation_config.static_forward_context[name] = make_layer(cfg, name, backend_cls)
size = 64 * spec.page_size_bytes
runner.kv_cache_config = KVCacheConfig(
    num_blocks=64,
    kv_cache_tensors=[KVCacheTensor(size=size, shared_by=[n]) for (n, _) in layers],
    kv_cache_groups=[KVCacheGroupSpec([n for (n, _) in layers], spec)],
)

# ── 站 6-7 全链：归组 → 最弱链 → kernel 块协商 → 定形 bind ────────────────
runner.initialize_kv_cache(runner.kv_cache_config)

groups = runner.attn_groups[0]
out["grouping"] = {
    "num_kv_cache_groups": 1,
    "num_attn_groups": len(groups),
    "rule": "AttentionGroupKey = (attn_backend.full_cls_name(), kv_cache_spec, num_heads_q)",
    "groups": [
        {
            "backend": g.backend.__name__,
            "full_cls_name": ".".join(g.backend.full_cls_name()),
            "layer_names": list(g.layer_names),
            "kv_cache_group_id": g.kv_cache_group_id,
            "num_heads_q": 4,
        }
        for g in groups
    ],
}

# 最弱链：各后端 builder 的 CG 档（host 无 FA3 → FA2 档）
fa_support = FlashAttentionBackend.get_builder_cls().get_cudagraph_support(cfg, spec)
triton_support = TritonStubBackend.get_builder_cls().get_cudagraph_support(cfg, spec)
out["weakest_link"] = {
    "fa_builder_cudagraph_support": {
        "name": fa_support.name,
        "value": fa_support.value,
        "note": "host 无 vllm_flash_attn 库 → get_flash_attn_version 恒 FA2 → UNIFORM_BATCH（真身 FA3 才是 ALWAYS；flash_attn.py:L352-L356 二选一）",
    },
    "triton_builder_cudagraph_support": {"name": triton_support.name, "value": triton_support.value},
    "min_cg_support": {"name": runner.seam_min_cg_support.name, "value": runner.seam_min_cg_support.value},
    "rule": "min over 全部组全部后端的 get_cudagraph_support —— 一个 UNIFORM_BATCH 后端就把全模型 FULL 档压掉",
    "cg_scale": {"ALWAYS": 3, "UNIFORM_BATCH": 2, "UNIFORM_SINGLE_TOKEN_DECODE": 1, "NEVER": 0},
}

def _fmt_declared(backend_cls, gloss):
    sizes = backend_cls.get_supported_kernel_block_sizes()
    lst = "[" + ", ".join(
        f"MultipleOf({s.base})" if isinstance(s, MultipleOf) else str(s) for s in sizes
    ) + "]"
    return f"get_supported_kernel_block_sizes() = {lst}{gloss}"


out["negotiation"] = {
    "manager_block_size": 256,
    "fa_declared_sizes": _fmt_declared(FlashAttentionBackend, "（16 的倍数块）"),
    "triton_declared_sizes": _fmt_declared(TritonStubBackend, "（固定块）"),
    "kernel_block_sizes": list(runner._kernel_block_sizes),
    "arithmetic": "256 = 4 x 64：管理块拆成 4 个 64-token kernel 块（24 不能整除 256，64 是 [24,64] 中能整除 256 的最大者，且 64 = 4×16 合 FA 的倍数约束）",
}

# ── 站 7：定形产物（m07 图数字出处） ─────────────────────────────────────
kv0 = runner.seam_kv_caches[L0]

# HND 对照：同一逻辑形、恒等置换 (0,1,2,3)——走与 NHD 同一条 _reshape_attention_kv_cache
# （物理连续视图 + permute 回逻辑形），唯一差别是 stride_order 换成恒等。
from implementation.v1.worker.gpu_model_runner import (  # noqa: E402
    _reshape_attention_kv_cache,
)

hnd_order = (0, 1, 2, 3)
hnd_raw = torch.empty(kv0.numel() * kv0.element_size(), dtype=torch.uint8)
hnd_view = _reshape_attention_kv_cache(
    hnd_raw, spec, tuple(kv0.shape), hnd_order, kv0.shape[0], None
)

out["kv_tensor_shaping"] = {
    "logical_shape_rule": "FlashAttentionBackend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size) = (num_blocks, num_kv_heads, block_size, 2*head_size)",
    "logical_shape": list(kv0.shape),
    "shape_explanation": "256 kernel 块（64 管理块 × 256/64=4）× 2 KV 头 × 64 token/块 × 2*64=128（K 与 V 打包进 content 维）",
    "kernel_num_blocks": 64 * (256 // 64),
    "stride_order_nhd_default": list(FlashAttentionBackend.get_kv_cache_stride_order()),
    "stride_order_meaning": "NHD=(0,2,1,3)：逻辑形→物理内存序的置换，块内 token 维提前——同 token 的 heads 内存连续；HND=(0,1,2,3) 恒等",
    "strides_nhd": list(kv0.stride()),
    "strides_nhd_explanation": "物理内存序 (B,N,H,2D) 连续：每块 64 token × 2 头 × 128 = 16384 元素",
    "strides_hnd": list(hnd_view.stride()),
    "strides_hnd_explanation": "恒等置换物理序 (B,H,N,2D) 连续：头维一步 = N×2D = 64×128 = 8192（同一头的全部 token），块维一步仍 16384（与 NHD 同）",
    "block_dim_sentinel": {
        "value": FlashAttentionBackend.get_kv_cache_block_dim(64, 2, 64),
        "probe": "get_kv_cache_block_dim：_S=1234567 当 num_blocks 代入 shape 再 .index(_S) 反查块维",
    },
    "layer_bind": {
        name: cfg.compilation_config.static_forward_context[name].kv_cache is t
        for name, t in runner.seam_kv_caches.items()
    },
}

path = os.path.join(os.path.dirname(__file__), "ch21-m08.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("wrote", path)
print(json.dumps(out, ensure_ascii=False, indent=2))
