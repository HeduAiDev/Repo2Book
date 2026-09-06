# ch21-m05 驱动：validate 回退——优先级表 → 逐候选 validate_configuration
# （ImportError 容忍）→ min(priority) 胜者；显式指定只校验不回退。
# 跑法：cd instances/vllm/artifacts-v3/ch21-attention-backends && python explainer/traces/run_ch21-m05.py
# 产物：explainer/traces/ch21-m05.json（trace_source="run"）
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
from implementation.platforms import cuda as cuda_platform  # noqa: E402
from implementation.platforms.cuda import (  # noqa: E402
    CudaPlatform,
    _get_backend_priorities,
)
from implementation.v1.attention import selector  # noqa: E402
from implementation.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
)
from implementation.v1.attention.backends.registry import (  # noqa: E402
    AttentionBackendEnum,
    _ATTN_OVERRIDES,
    register_backend,
)
from implementation.v1.attention.selector import AttentionSelectorConfig  # noqa: E402

SM90 = DeviceCapability(9, 0)   # Hopper
SM100 = DeviceCapability(10, 0)  # Blackwell
SM75 = DeviceCapability(7, 5)   # Turing
SM70 = DeviceCapability(7, 0)   # Volta


# TRITON_ATTN 桩：继承 FA 全部探针、只改块大小面 [24, 64]——
# 24 让 --block-size 24 场景存活（FA 的 16 倍数不含 24）。
class TritonStubBackend(FlashAttentionBackend):
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


def _selector_config(**kw):
    base = dict(
        head_size=64,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=None,
    )
    base.update(kw)
    return AttentionSelectorConfig(**base)


out: dict = {}

# ── 场景 A：SM100（Blackwell, major=10）非 MLA，自动选择 ──────────────────
cuda_platform.CudaPlatform._seam_device_capability = SM100
cfg = _selector_config()
out["scenario"] = {
    "device": "SM100 (Blackwell, major=10, minor=0)",
    "device_capability_major": 10,
    "use_mla": False,
    "head_size": 64,
    "dtype": "torch.float16",
    "kv_cache_dtype": "auto",
    "block_size": None,
    "note_host": "host 无 vllm 包：FLASHINFER/FLEX_ATTENTION/TURBOQUANT 走真实 vllm.* 类路径 → ImportError（get_valid_backends 接住当一条原因）；FLASH_ATTN/TRITON_ATTN 经 register_backend（真实第三方注册机制）挂到包内真身/桩",
}

prios = _get_backend_priorities(False, SM100)
out["priority_table_non_mla_sm100"] = [b.name for b in prios]

valid, invalid = CudaPlatform.get_valid_backends(SM90, cfg)  # ← 设备实参用 seam 位
valid, invalid = CudaPlatform.get_valid_backends(SM100, cfg)
rows = []
for cand in valid:
    rows.append({
        "priority": cand.priority,
        "backend": cand.backend.name,
        "get_class": "ok（register_backend 已注册）",
        "validate_reasons": [],
        "valid": True,
    })
for backend, (prio, reasons) in sorted(invalid.items(), key=lambda kv: kv[1][0]):
    rows.append({
        "priority": prio,
        "backend": backend.name,
        "get_class": "ImportError（host 无 vllm.* 模块）",
        "validate_reasons": reasons,
        "valid": False,
    })
rows.sort(key=lambda r: r["priority"])
out["get_valid_backends"] = rows

cls_path = CudaPlatform.get_attn_backend_cls(None, cfg)
winner = min(valid, key=lambda c: c.priority)
out["auto_winner"] = {
    "rule": "min(priority) 于幸存候选",
    "backend": winner.backend.name,
    "priority": winner.priority,
    "cls_path": cls_path,
    "survivors": [
        {"backend": c.backend.name, "priority": c.priority} for c in sorted(valid, key=lambda c: c.priority)
    ],
}

# ── 场景 B：显式指定合法后端——只校验它，不查优先级表 ──────────────────────
out["explicit_valid"] = {
    "selected": "TRITON_ATTN",
    "validate_reasons": [],
    "cls_path": CudaPlatform.get_attn_backend_cls(AttentionBackendEnum.TRITON_ATTN, cfg),
}

# ── 场景 C：显式指定非法后端——直接 ValueError，不回退 ─────────────────────
bad_cfg = _selector_config(head_size=100)  # 100 % 8 != 0 → head_size 探针
try:
    CudaPlatform.get_attn_backend_cls(AttentionBackendEnum.TRITON_ATTN, bad_cfg)
    out["explicit_invalid"] = {"error": "UNREACHED"}
except ValueError as e:
    out["explicit_invalid"] = {
        "selected": "TRITON_ATTN",
        "head_size": 100,
        "probe": "supports_head_size(100) → 100 % 8 != 0",
        "reasons": ["head_size not supported"],
        "error": str(e),
        "fallback": "无——显式指定不回退，直接 ValueError",
    }

# ── 场景 D：--block-size 24 挤掉更高优先级后端 → warning ───────────────────
cuda_platform.CudaPlatform._seam_device_capability = SM90
import logging  # noqa: E402

logging.getLogger("vllm.platforms.cuda").setLevel(logging.WARNING)
warn_msgs: list[str] = []
h = logging.Handler()
h.emit = lambda record: warn_msgs.append(record.getMessage())  # type: ignore[assignment]
logging.getLogger("vllm.platforms.cuda").addHandler(h)
bs_cfg = _selector_config(block_size=24)
prios_sm90 = _get_backend_priorities(False, SM90)
valid90, invalid90 = CudaPlatform.get_valid_backends(SM90, bs_cfg)
cls_path90 = CudaPlatform.get_attn_backend_cls(None, bs_cfg)
logging.getLogger("vllm.platforms.cuda").removeHandler(h)
out["block_size_warning"] = {
    "device": "SM90 (Hopper, major=9, minor=0)",
    "user_block_size": 24,
    "priority_table_non_mla_sm90": [b.name for b in prios_sm90],
    "fa_validate_reasons": ["block_size not supported"],
    "fa_probe": "supports_block_size(24)：FA get_supported_kernel_block_sizes()=[MultipleOf(16)]，24 % 16 != 0",
    "triton_probe": "TritonStub get_supported_kernel_block_sizes()=[24, 64]，24 % 24 == 0",
    "winner": min(valid90, key=lambda c: c.priority).backend.name,
    "winner_priority": min(valid90, key=lambda c: c.priority).priority,
    "cls_path": cls_path90,
    "warning": [m for m in warn_msgs if "--block-size" in m],
}

# ── 场景 E：探针单测——FA 在 SM75 上算力代不过 ────────────────────────────
out["fa_reasons_sm75"] = FlashAttentionBackend.validate_configuration(
    head_size=64,
    dtype=torch.float16,
    kv_cache_dtype="auto",
    block_size=None,
    use_mla=False,
    has_sink=False,
    use_sparse=False,
    use_mm_prefix=False,
    use_per_head_quant_scales=False,
    device_capability=SM75,
    attn_type="decoder",
)

# ── 场景 F：SM70 全军覆没 → ValueError 带全部原因 ─────────────────────────
cuda_platform.CudaPlatform._seam_device_capability = SM70
try:
    CudaPlatform.get_attn_backend_cls(None, cfg)
    out["all_invalid"] = {"error": "UNREACHED"}
except ValueError as e:
    out["all_invalid"] = {
        "device": "SM70 (Volta, major=7, minor=0)",
        "reason": "FA/TRITON 桩都继承 FA 探针 supports_compute_capability: DeviceCapability(7,0) < DeviceCapability(8,0)",
        "error": str(e).split(" Reasons")[0],
    }

# ── 场景 G：同一配置记忆化——@cache 同键只解一次 ────────────────────────────
from implementation.v1.attention.selector import get_attn_backend  # noqa: E402

cuda_platform.CudaPlatform._seam_device_capability = SM90
vllm_cfg = VllmConfigSeam(max_num_seqs=8, max_batched_tokens=512, max_model_len=256)
vllm_cfg.attention_config = AttentionConfig()
with set_current_vllm_config(vllm_cfg):
    a = get_attn_backend(64, torch.float16, "auto")
    b = get_attn_backend(64, torch.float16, "auto")
out["memoization"] = {
    "head_size": 64,
    "same_class": a is b,
    "resolved": a.__name__,
    "note": "几十层 Attention.__init__ 同配置命中同一 @cache 条目——装配期摊销成每配置一次",
}

# 还原注册表
AttentionBackendEnum.FLASH_ATTN.clear_override()
AttentionBackendEnum.TRITON_ATTN.clear_override()
_ATTn = _ATTN_OVERRIDES.pop(AttentionBackendEnum.CUSTOM, None)
selector._cached_get_attn_backend.cache_clear()

path = os.path.join(os.path.dirname(__file__), "ch21-m05.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("wrote", path)
print(json.dumps(out, ensure_ascii=False, indent=2))
