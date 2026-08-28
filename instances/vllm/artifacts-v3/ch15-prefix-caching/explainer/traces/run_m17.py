# ch15 m17 稀疏驻留 retention interval 驱动：VLLM_PREFIX_CACHE_RETENTION_INTERVAL
# 三态（None=稠密 / 0=只留 replay 边界 / 正数=每段一条）；校验只对 SWA/Mamba 组
# 有意义、必须非负且整除 scheduler_block_size（kv_cache_coordinator.py:L30-L57）；
# mask 三态以 MambaManager.reachable_block_mask（single_type:L1358-L1414）为例。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
import implementation.kv_cache_coordinator as kcc  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec,
    SlidingWindowSpec)
from implementation.kv_cache_manager import KVCacheManager  # noqa: E402
from implementation.single_type_kv_cache_manager import (  # noqa: E402
    MambaManager, SlidingWindowManager)
import torch  # noqa: E402

kcu.init_none_hash(sha256)


def full_spec(bs):
    return FullAttentionSpec(block_size=bs, num_kv_heads=2, head_size=8,
                             dtype=torch.float16)


def swa_spec(bs, window):
    return SlidingWindowSpec(block_size=bs, num_kv_heads=2, head_size=8,
                             dtype=torch.float16, sliding_window=window)


def mamba_spec(bs):
    return MambaSpec(block_size=bs, shapes=((8, 8),), dtypes=(torch.float32,),
                     mamba_cache_mode="align")


def kv_config(specs, num_blocks=32):
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[],
                         kv_cache_groups=[KVCacheGroupSpec([f"l.{i}"], s)
                                          for i, s in enumerate(specs)])


def validate(interval, sbs, specs):
    try:
        kcc._validate_prefix_cache_retention_interval(
            interval, sbs, kv_config(specs))
        return "OK"
    except ValueError as e:
        return e.args[0].split(" (")[0][:60]


out = {"params": {"scheduler_block_size": 16, "env_var":
                  "VLLM_PREFIX_CACHE_RETENTION_INTERVAL"}}

# --- 1) 校验三态 ---
out["validation"] = [
    {"interval": None, "config": "纯 full（无 SWA/Mamba）", "verdict":
        validate(None, 16, [full_spec(16)])},
    {"interval": 16, "config": "纯 full（无 SWA/Mamba）", "verdict":
        validate(16, 16, [full_spec(16)]),
     "reason": "只对 SWA/Mamba 组有意义——设了但组不匹配直接 raise"},
    {"interval": -16, "config": "full+swa", "verdict":
        validate(-16, 16, [full_spec(16), swa_spec(16, 48)])},
    {"interval": 24, "config": "full+swa", "verdict":
        validate(24, 16, [full_spec(16), swa_spec(16, 48)]),
     "reason": "24 不整除 scheduler_block_size 16 → raise"},
    {"interval": 32, "config": "full+swa", "verdict":
        validate(32, 16, [full_spec(16), swa_spec(16, 48)])},
]

# --- 2) env 旋钮真被 coordinator 读走（模块级常量注入后构造） ---
saved = kcc.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
kcc.VLLM_PREFIX_CACHE_RETENTION_INTERVAL = 32
mgr = KVCacheManager(kv_cache_config=kv_config(
    [full_spec(16), swa_spec(16, 48)]), max_model_len=512,
    scheduler_block_size=16, hash_block_size=16)
kcc.VLLM_PREFIX_CACHE_RETENTION_INTERVAL = saved
out["env_read"] = {
    "injected_interval": 32,
    "coordinator_retention_interval": mgr.coordinator.retention_interval,
}

# --- 3) Mamba mask 三态（block 16、8 块、replay 边界 79） ---
spec16 = mamba_spec(16)
mask_none = MambaManager.reachable_block_mask(
    0, 8, alignment_tokens=16, kv_cache_spec=spec16, use_eagle=False)
mask_0 = MambaManager.reachable_block_mask(
    0, 8, alignment_tokens=16, kv_cache_spec=spec16, use_eagle=False,
    retention_interval=0, reachable_boundaries=[79])
mask_32 = MambaManager.reachable_block_mask(
    0, 8, alignment_tokens=16, kv_cache_spec=spec16, use_eagle=False,
    retention_interval=32, reachable_boundaries=[79])
out["mamba_mask"] = {
    "block_size": 16, "num_blocks_range": 8, "replay_boundary_tokens": 79,
    "none_is_dense_none": mask_none is None,
    "interval0_true_positions": [i for i, v in enumerate(mask_0) if v],
    "interval0_replay_boundary_block": 79 // 16 - 1,
    "interval0_replay_boundary_aligned_tokens": 79 // 16 * 16,
    "interval0_num_true": sum(mask_0),
    "interval32_true_positions": [i for i, v in enumerate(mask_32) if v],
    "interval32_per_segment_blocks": 32 // 16,
    "interval32_num_true": sum(mask_32),
    "note": "正数=每 32 token（2 块）留一个段尾状态（位置 1,3,5,7），"
            "replay 边界 79 → 对齐 64 → 块 3 的特赦恰好重合在段尾上",
}

# --- 4) SWA mask 对照（need=3 连续块尾） ---
swa16 = swa_spec(16, 48)
swa_mask = SlidingWindowManager.reachable_block_mask(
    0, 8, alignment_tokens=16, kv_cache_spec=swa16, use_eagle=False,
    retention_interval=0, reachable_boundaries=[79])
out["swa_mask"] = {
    "window": 48, "need_contiguous_blocks": 3,
    "interval0_true_positions": [i for i, v in enumerate(swa_mask) if v],
    "num_true": sum(swa_mask),
    "note": "SWA 命中需要窗口连续块——特赦留的是边界前的 need 块尾（79→块 0-2），"
            "稀疏驻留不掉复用点",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m17.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
