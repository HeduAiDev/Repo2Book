"""TDD tests for ch04 — KV-cache / scheduling / spec Ascend-ization patches.

These tests exercise the *observable behavior* of the real vllm-ascend patch
code (faithfully subtracted into ../implementation) for the chapter's five
cases. Ascend NPU/CANN code cannot run on host, but the patch rebinding /
config-rewriting logic is pure Python — so we stub the vLLM target namespaces
in sys.modules, import the (subtracted) patch modules, and assert behavior +
rebinds match the real repo.

Cases:
  1. block_size 16→128         (patch_mamba_config.py)
  2. MLAAttentionSpec 子类化     (patch_kv_cache_interface.py)
  3. CP+hybrid 前缀缓存          (patch_kv_cache_coordinator.py / patch_kv_cache_utils.py)
  4. bind_kv_cache 跳 NPU raise  (patch_qwen3_next_mtp.py)
  5. int32 slot_mapping         (block_table.py)
"""

import importlib.util
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class ModRegistry:
    """Register fake dotted modules in sys.modules, with auto-cleanup.

    Never clobbers an already-present real module (e.g. torch); only creates
    the missing links and records what it added so the fixture can pop them.
    """

    def __init__(self):
        self.added = []

    def module(self, dotted):
        parts = dotted.split(".")
        for i in range(len(parts)):
            name = ".".join(parts[: i + 1])
            if name not in sys.modules:
                m = types.ModuleType(name)
                sys.modules[name] = m
                self.added.append(name)
                if i > 0:
                    setattr(sys.modules[".".join(parts[:i])], parts[i], m)
        return sys.modules[dotted]

    def cleanup(self):
        for n in reversed(self.added):
            sys.modules.pop(n, None)


@pytest.fixture
def reg():
    r = ModRegistry()
    yield r
    r.cleanup()


_counter = [0]


def load_fresh(filename):
    """Import implementation/<filename> as a brand-new module (re-executed)."""
    _counter[0] += 1
    name = f"_impl_{_counter[0]}_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, IMPL_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cdiv(a, b):
    return -(-a // b)


# byte sizes for the dtypes used across the chapter
_DTYPE_BYTES = {
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float32: 4,
    torch.int8: 1,
    torch.int32: 4,
    torch.int64: 8,
    torch.float8_e4m3fn: 1,
}


class _Cfg:
    """Tiny attribute bag for building fake configs."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# --------------------------------------------------------------------------- #
# case 1: block_size 16→128 (patch_mamba_config) — kernel_block_size pinned 128
# --------------------------------------------------------------------------- #
def test_mamba_config_pins_kernel_block_size_128(reg):
    cfg_mod = reg.module("vllm.model_executor.models.config")

    class HybridAttentionMambaModelConfig:
        @classmethod
        def verify_and_update_config(cls, vllm_config):
            return "ORIGINAL"

    class MambaModelConfig:
        called = []

        @classmethod
        def verify_and_update_config(cls, vllm_config):
            cls.called.append(vllm_config)

    cfg_mod.HybridAttentionMambaModelConfig = HybridAttentionMambaModelConfig
    cfg_mod.MambaModelConfig = MambaModelConfig

    reg.module("vllm.logger").logger = _Cfg(info=lambda *a, **k: None)

    models_mod = reg.module("vllm.model_executor.models")

    class _ModelCls:
        @staticmethod
        def get_mamba_state_shape_from_config(vllm_config):
            return [(65536,), (1024,)]  # ssm (max) + conv (min)

        @staticmethod
        def get_mamba_state_dtype_from_config(vllm_config):
            return [torch.bfloat16, torch.bfloat16]

    models_mod.ModelRegistry = _Cfg(resolve_model_cls=lambda arch, model_config=None: (_ModelCls, None))

    reg.module("vllm.utils.math_utils").cdiv = _cdiv
    tu = reg.module("vllm.utils.torch_utils")
    tu.STR_DTYPE_TO_TORCH_DTYPE = {"bf16": torch.bfloat16}
    tu.get_dtype_size = lambda dt: _DTYPE_BYTES[dt]

    mod = load_fresh("patch_mamba_config.py")

    # technique ③: the module-level classmethod replaced the original
    assert HybridAttentionMambaModelConfig.verify_and_update_config.__func__ is mod.verify_and_update_config.__func__

    # build a fake vllm_config (MLA path), call the patched classmethod
    cache_config = _Cfg(cache_dtype="auto", block_size=None, mamba_page_size_padded=None)
    model_config = _Cfg(
        dtype=torch.bfloat16,
        architecture="FakeMamba",
        use_mla=True,
        hf_text_config=_Cfg(kv_lora_rank=512, qk_rope_head_dim=64),
        get_num_kv_heads=lambda parallel_config: 1,
        get_head_size=lambda: 0,
    )
    vllm_config = _Cfg(cache_config=cache_config, model_config=model_config, parallel_config=_Cfg())

    HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)

    # FULL_AND_PIECEWISE default still enabled via MambaModelConfig
    assert MambaModelConfig.called == [vllm_config]
    # ssm_block_page_size = 65536*2 = 131072; attn_single_token_k = 512*1*2 = 1024
    # attn_block_size = 128 * cdiv(131072, 128*1024) = 128 * 1 = 128  (NOT 16)
    assert cache_config.block_size == 128
    # the alignment identity holds: attn_single_token_k * attn_block_size == ssm_page_size
    assert 1024 * cache_config.block_size == 131072


# --------------------------------------------------------------------------- #
# case 2: MLAAttentionSpec subclass — Sparse-C8 4-tuple page_size_bytes
# --------------------------------------------------------------------------- #
def _setup_kv_cache_interface_stubs(reg, device_type="A3"):
    reg.module("vllm.config").VllmConfig = type("VllmConfig", (), {})
    reg.module("vllm.utils.math_utils").cdiv = _cdiv
    reg.module("vllm.utils.torch_utils").get_dtype_size = lambda dt: _DTYPE_BYTES[dt]

    kvi = reg.module("vllm.v1.kv_cache_interface")

    @dataclass(frozen=True)
    class MLAAttentionSpec:
        block_size: int = 0
        num_kv_heads: int = 0
        head_size: int = 0
        dtype: Any = torch.bfloat16
        cache_dtype_str: Any = None
        compress_ratio: int = 1

    @dataclass(frozen=True, kw_only=True)
    class SlidingWindowMLASpec:
        block_size: int = 0
        num_kv_heads: int = 0
        head_size: int = 0
        dtype: Any = torch.bfloat16
        page_size_padded: Any = None
        sliding_window: int = 0

    kvi.MLAAttentionSpec = MLAAttentionSpec
    kvi.SlidingWindowMLASpec = SlidingWindowMLASpec

    # technique ⑤ target: mla_attention holds its own from-import alias
    reg.module("vllm.model_executor.layers.attention.mla_attention").MLAAttentionSpec = MLAAttentionSpec

    util = reg.module("vllm_ascend.utils")

    class AscendDeviceType:
        A5 = "A5"
        A3 = "A3"

    util.AscendDeviceType = AscendDeviceType
    util.get_ascend_device_type = lambda: getattr(AscendDeviceType, device_type)
    return kvi


def test_mla_spec_sparse_c8_page_size_bytes(reg):
    kvi = _setup_kv_cache_interface_stubs(reg, device_type="A3")
    mod = load_fresh("patch_kv_cache_interface.py")

    spec = mod.AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        sparse_head_dim=(512, 64, 128),  # kv_lora_rank, qk_rope_head_dim, index_head_dim
        cache_sparse_c8=True,
    )
    # A3 (non-A5): qk_rope_head_dim != 0 → kv_dim = 512+64 = 576 (bf16, 2B)
    # num_heads_per_page = 128*1 = 128
    # kv_bytes  = 128 * 576 * 2 = 147456
    # qli_bytes = 128 * 128 * 1 (int8) = 16384
    # qli_scale = 128 * 1 * 2 (fp16) = 256
    assert spec.c8_k_cache_dtype is torch.int8
    assert spec.c8_k_scale_cache_dtype is torch.float16
    assert spec.page_size_bytes == 147456 + 16384 + 256

    # non-sparse path: block_size*num_kv_heads*(head_size*size(dtype) + scale_dim*...)
    plain = mod.AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    assert plain.cache_sparse_c8 is False
    assert plain.page_size_bytes == 128 * 1 * (576 * 2)


def test_mla_spec_rebinds_three_namespaces(reg):
    kvi = _setup_kv_cache_interface_stubs(reg)
    mla_attention = sys.modules["vllm.model_executor.layers.attention.mla_attention"]
    mod = load_fresh("patch_kv_cache_interface.py")

    # technique ①: whole-class replacement on two kv_cache_interface names
    assert kvi.MLAAttentionSpec is mod.AscendMLAAttentionSpec
    assert kvi.SlidingWindowMLASpec is mod.AscendSlidingWindowMLASpec
    assert issubclass(mod.AscendMLAAttentionSpec, mod.AscendMLAAttentionSpec.__mro__[1])
    # technique ⑤: the from-import alias in mla_attention must be rebound too
    assert mla_attention.MLAAttentionSpec is mod.AscendMLAAttentionSpec


def test_sliding_window_mla_real_page_size(reg):
    _setup_kv_cache_interface_stubs(reg)
    mod = load_fresh("patch_kv_cache_interface.py")
    spec = mod.AscendSlidingWindowMLASpec(
        block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16,
        page_size_padded=None, sliding_window=4096,
    )
    # storage_block_size == block_size → 128*1*576*2
    assert spec.storage_block_size == 128
    assert spec.real_page_size_bytes == 128 * 1 * 576 * 2


# --------------------------------------------------------------------------- #
# case 3a: coordinator — _get_effective_block_size + factory fallback + cache trap
# --------------------------------------------------------------------------- #
def _setup_coordinator_stubs(reg):
    reg.module("vllm")
    reg.module("vllm.v1.core.block_pool").BlockPool = type("BlockPool", (), {})

    coord = reg.module("vllm.v1.core.kv_cache_coordinator")
    coord.HybridKVCacheCoordinator = type("HybridKVCacheCoordinator", (), {})
    coord.KVCacheCoordinator = type("KVCacheCoordinator", (), {})
    coord.get_kv_cache_coordinator = lambda *a, **k: "ORIG"  # captured as _orig

    reg.module("vllm.v1.core.kv_cache_metrics").KVCacheMetricsCollector = type("M", (), {})

    kcu = reg.module("vllm.v1.core.kv_cache_utils")
    kcu.BlockHash = type("BlockHash", (), {})
    kcu.BlockHashList = list
    kcu.BlockHashListWithBlockSize = type("BHL", (), {})
    kcu.KVCacheBlock = type("KVCacheBlock", (), {})

    reg.module("vllm.v1.core.single_type_kv_cache_manager").SingleTypeKVCacheManager = type("S", (), {})

    kvi = reg.module("vllm.v1.kv_cache_interface")
    kvi.FullAttentionSpec = type("FullAttentionSpec", (), {})
    kvi.KVCacheConfig = type("KVCacheConfig", (), {})
    kvi.KVCacheSpec = type("KVCacheSpec", (), {})

    class MambaSpec:
        def __init__(self, block_size):
            self.block_size = block_size

    kvi.MambaSpec = MambaSpec

    reg.module("vllm_ascend.core.single_type_kv_cache_manager").get_manager_for_kv_cache_spec = (
        lambda **k: object()
    )
    return coord, kvi


def test_coordinator_effective_block_size_cp_and_compress(reg):
    coord, kvi = _setup_coordinator_stubs(reg)
    mod = load_fresh("patch_kv_cache_coordinator.py")

    # build an instance without running __init__ (it needs the full runtime)
    co = object.__new__(mod.AscendHybridKVCacheCoordinator)
    co.dcp_world_size = 2
    co.pcp_world_size = 2
    co.enable_caching = True

    # non-mamba spec with compress_ratio: block_size * (dcp*pcp) * compress_ratio
    spec = _Cfg(block_size=128, compress_ratio=4)
    assert co._get_effective_block_size(spec) == 128 * (2 * 2) * 4

    # mamba spec with caching short-circuits to the raw block size
    mspec = kvi.MambaSpec(block_size=64)
    assert co._get_effective_block_size(mspec) == 64


def test_coordinator_factory_falls_back_to_upstream(reg):
    coord, kvi = _setup_coordinator_stubs(reg)
    mod = load_fresh("patch_kv_cache_coordinator.py")

    # single (non-deepseek-v4) group, no CP → must keep upstream coordinator
    group = _Cfg(kv_cache_spec=_Cfg(block_size=128))
    kv_cache_config = _Cfg(kv_cache_groups=[group])
    out = mod.get_kv_cache_coordinator(
        kv_cache_config, max_model_len=1024, max_num_batched_tokens=1024,
        use_eagle=False, enable_caching=True, enable_kv_cache_events=False,
        dcp_world_size=1, pcp_world_size=1, hash_block_size=128,
    )
    assert out == "ORIG"  # _orig_get_kv_cache_coordinator was called


def test_coordinator_rebinds_and_cache_trap(reg):
    coord, kvi = _setup_coordinator_stubs(reg)
    # the from-import cache trap target must already be loaded
    kvm = reg.module("vllm.v1.core.kv_cache_manager")
    kvm.get_kv_cache_coordinator = coord.get_kv_cache_coordinator  # the stale binding
    stale = kvm.get_kv_cache_coordinator

    mod = load_fresh("patch_kv_cache_coordinator.py")

    # technique ②/③: factory rebound on the origin module
    assert coord.get_kv_cache_coordinator is mod.get_kv_cache_coordinator
    # technique ⑤: stale from-import binding in kv_cache_manager updated too
    assert kvm.get_kv_cache_coordinator is mod.get_kv_cache_coordinator
    assert kvm.get_kv_cache_coordinator is not stale
    # the captured original is preserved for fallback
    assert mod._orig_get_kv_cache_coordinator is not mod.get_kv_cache_coordinator


# --------------------------------------------------------------------------- #
# case 3b: kv_cache_utils — PR#40860 raise replaced by lcm/gcd computation
# --------------------------------------------------------------------------- #
def _setup_kv_cache_utils_stubs(reg):
    reg.module("vllm.config").VllmConfig = type("VllmConfig", (), {})
    mu = reg.module("vllm.utils.math_utils")
    mu.cdiv = _cdiv
    mu.round_up = lambda x, m: -(-x // m) * m

    kcu = reg.module("vllm.v1.core.kv_cache_utils")
    kcu._approximate_gcd = lambda xs, lower_bound=1: lower_bound
    kcu.may_override_num_blocks = lambda cfg, n: n
    kcu.resolve_kv_cache_block_sizes = lambda cfg, vc: ("ORIG", "ORIG")

    kvi = reg.module("vllm.v1.kv_cache_interface")
    for name in ("KVCacheConfig", "KVCacheGroupSpec", "KVCacheSpec", "KVCacheTensor",
                 "MLAAttentionSpec", "SlidingWindowMLASpec", "UniformTypeKVCacheSpecs"):
        setattr(kvi, name, type(name, (), {}))

    reg.module("vllm.v1.engine.core")
    return kcu


def test_resolve_block_sizes_cp_multi_group_lcm_gcd(reg):
    kcu = _setup_kv_cache_utils_stubs(reg)
    mod = load_fresh("patch_kv_cache_utils.py")

    groups = [_Cfg(kv_cache_spec=_Cfg(block_size=128)), _Cfg(kv_cache_spec=_Cfg(block_size=256))]
    kv_cache_config = _Cfg(kv_cache_groups=groups)
    vllm_config = _Cfg(
        cache_config=_Cfg(block_size=128, enable_prefix_caching=True),
        parallel_config=_Cfg(decode_context_parallel_size=2, prefill_context_parallel_size=1),
    )
    # multi-group + CP>1 → no raise (upstream raises ValueError here).
    # scheduler_block_size = lcm(128,256)*2*1 = 512; hash_block_size = gcd(128,256) = 128
    sched, hash_bs = mod._ascend_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    assert (sched, hash_bs) == (512, 128)


def test_resolve_block_sizes_single_group_and_no_cp_fallback(reg):
    kcu = _setup_kv_cache_utils_stubs(reg)
    mod = load_fresh("patch_kv_cache_utils.py")

    # single group → block_size * dcp * pcp
    single = _Cfg(kv_cache_groups=[_Cfg(kv_cache_spec=_Cfg(block_size=128))])
    vc = _Cfg(cache_config=_Cfg(block_size=128, enable_prefix_caching=True),
              parallel_config=_Cfg(decode_context_parallel_size=2, prefill_context_parallel_size=1))
    assert mod._ascend_resolve_kv_cache_block_sizes(single, vc) == (256, 256)

    # multi-group but no CP → fall back to upstream resolver
    multi = _Cfg(kv_cache_groups=[_Cfg(kv_cache_spec=_Cfg(block_size=128)),
                                  _Cfg(kv_cache_spec=_Cfg(block_size=256))])
    vc_nocp = _Cfg(cache_config=_Cfg(block_size=128, enable_prefix_caching=True),
                   parallel_config=_Cfg(decode_context_parallel_size=1, prefill_context_parallel_size=1))
    assert mod._ascend_resolve_kv_cache_block_sizes(multi, vc_nocp) == ("ORIG", "ORIG")

    # technique ③/⑤: rebound on both kv_cache_utils and the engine.core from-import
    assert kcu.resolve_kv_cache_block_sizes is mod._ascend_resolve_kv_cache_block_sizes
    assert sys.modules["vllm.v1.engine.core"].resolve_kv_cache_block_sizes is mod._ascend_resolve_kv_cache_block_sizes


# --------------------------------------------------------------------------- #
# case 4: bind_kv_cache — bypass NotImplementedError, take layer_names[0]
# --------------------------------------------------------------------------- #
def test_bind_kv_cache_takes_first_layer_per_index(reg):
    utils = reg.module("vllm.v1.worker.utils")
    utils.defaultdict = defaultdict
    utils.extract_layer_index = lambda name, num_attn_module=1: int(name[0])
    reg.module("vllm.model_executor.layers.attention").Attention = type("Attention", (), {})

    mod = load_fresh("patch_qwen3_next_mtp.py")

    # technique ③: rebound on the utils module
    assert utils.bind_kv_cache is mod.bind_kv_cache

    t0a, t0b, t1a = torch.tensor([0]), torch.tensor([1]), torch.tensor([2])
    kv_caches = {"0a": t0a, "0b": t0b, "1a": t1a}  # index 0 has TWO layer_names
    fctx = {name: _Cfg(kv_cache=None) for name in kv_caches}
    runner_kv_caches = []

    # original (CUDA-only) raises NotImplementedError on index 0; this must NOT
    mod.bind_kv_cache(kv_caches, fctx, runner_kv_caches)

    # per index, only layer_names[0] is appended → t0b is intentionally skipped
    assert [t.item() for t in runner_kv_caches] == [t0a.item(), t1a.item()]
    # every layer in forward context still gets its own kv_cache bound
    assert fctx["0a"].kv_cache is t0a
    assert fctx["0b"].kv_cache is t0b
    assert fctx["1a"].kv_cache is t1a


# --------------------------------------------------------------------------- #
# case 5: int32 slot_mapping — del int64 parent tensor, rebuild as int32
# --------------------------------------------------------------------------- #
def test_block_table_slot_mappings_is_int32(reg):
    reg.module("vllm.triton_utils").tl = object()
    sys.modules["vllm.triton_utils"].triton = _Cfg(jit=lambda fn: fn)
    reg.module("vllm.v1.attention.backends.utils").PAD_SLOT_ID = -1

    gpu_bt = reg.module("vllm.v1.worker.gpu.block_table")

    class BlockTables:
        def __init__(self, block_sizes, max_num_reqs, max_num_batched_tokens,
                     max_num_blocks_per_group, device, cp_size=1, cp_rank=0, cp_interleave=1):
            self.num_kv_cache_groups = len(block_sizes)
            self.max_num_batched_tokens = max_num_batched_tokens
            self.device = device
            # upstream default dtype is int64
            self.slot_mappings = torch.zeros(
                self.num_kv_cache_groups, max_num_batched_tokens, dtype=torch.int64, device=device,
            )

    gpu_bt.BlockTables = BlockTables
    gpu_bt._load_ptr = lambda *a, **k: None

    reg.module("vllm_ascend.utils").vllm_version_is = lambda v: v == "0.21.0"

    mod = load_fresh("block_table.py")

    bt = mod.AscendBlockTables(
        block_sizes=[128],
        max_num_reqs=4,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[8],
        device=torch.device("cpu"),
    )
    # the parent int64 slot_mappings was deleted and rebuilt as int32
    assert bt.slot_mappings.dtype == torch.int32
    assert bt.slot_mappings.shape == (1, 16)
