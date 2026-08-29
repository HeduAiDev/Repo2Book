# HOST SEAMS for the ch19 subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).
#
# vLLM itself does not install on this Windows host, so every vllm.* name the
# kept (subtract-only) code touches *outside this chapter's subtract-only
# surface* is mirrored here by a stdlib/torch-backed stand-in with the SAME
# observable interface subset. Each seam carries a `# SOURCE:` anchor into the
# pinned tree; none of them invents behavior the real module does not have on
# the paths this chapter exercises. Full inventory: impl-notes.md §Seam 清单.
#
# Companion files import these via package-relative imports (ch17 v3
# convention): real `from vllm.x import y` becomes `from .._host_seams import y`
# when x is outside this chapter's subtract-only surface.

from __future__ import annotations

import contextlib
import logging
import sys
import types as _types
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# logger — vllm/logger.py init_logger + the *_once wrappers
# ---------------------------------------------------------------------------


# SOURCE: vllm/logger.py init_logger — logging seam with the *_once helpers
def init_logger(name: str):
    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    seen: set = set()

    # SOURCE: vllm/logger.py once-messaging wrapper (info_once/warning_once)
    class _Once:  # HOST SEAM
        # SOURCE: vllm/logger.py once-wrapper construction
        def __init__(self, fn):
            # HOST SEAM (real: functools-based dedup in vllm.logger)
            self._fn = fn

        # SOURCE: vllm/logger.py once-wrapper call
        def __call__(self, msg, *args):
            key = (self._fn.__name__, msg)
            if key not in seen:
                seen.add(key)
                self._fn(msg, *args)

    log.info_once = _Once(log.info)
    log.warning_once = _Once(log.warning)
    log.debug_once = _Once(log.debug)
    return log


# ---------------------------------------------------------------------------
# envs — vllm/envs.py flag seam (defaults per pin v0.27.1)
# ---------------------------------------------------------------------------


# SOURCE: vllm/envs.py env-backed flag table — HOST SEAM (class attribute
# stand-in; defaults are the pin's on the paths this chapter exercises)
class envs:  # HOST SEAM
    # SOURCE: vllm/envs.py VLLM_LOGGING_LEVEL ("" → INFO default)
    VLLM_LOGGING_LEVEL: str = ""
    # SOURCE: vllm/envs.py VLLM_USE_LAYERNAME (default True on torch>=2.11)
    VLLM_USE_LAYERNAME: bool = True
    # SOURCE: vllm/envs.py:L170 VLLM_USE_BREAKABLE_CUDAGRAPH (default 0)
    VLLM_USE_BREAKABLE_CUDAGRAPH: bool = False
    # SOURCE: vllm/envs.py VLLM_ENABLE_CUDAGRAPH_GC (default False → freeze)
    VLLM_ENABLE_CUDAGRAPH_GC: bool = False
    # SOURCE: vllm/envs.py VLLM_COMPILE_CACHE_SAVE_FORMAT (default "binary")
    VLLM_COMPILE_CACHE_SAVE_FORMAT: str = "binary"
    # SOURCE: vllm/envs.py VLLM_DISABLE_COMPILE_CACHE (default False)
    VLLM_DISABLE_COMPILE_CACHE: bool = False
    # SOURCE: vllm/envs.py VLLM_USE_STANDALONE_COMPILE (default False)
    VLLM_USE_STANDALONE_COMPILE: bool = False
    # SOURCE: vllm/envs.py VLLM_USE_MEGA_AOT_ARTIFACT (default False)
    VLLM_USE_MEGA_AOT_ARTIFACT: bool = False
    # SOURCE: vllm/envs.py VLLM_USE_AOT_COMPILE (default False)
    VLLM_USE_AOT_COMPILE: bool = False
    # SOURCE: vllm/envs.py VLLM_USE_BYTECODE_HOOK (default False)
    VLLM_USE_BYTECODE_HOOK: bool = False
    # SOURCE: vllm/envs.py VLLM_LOG_BATCHSIZE_INTERVAL (default -1 → off)
    VLLM_LOG_BATCHSIZE_INTERVAL: int = -1
    # SOURCE: vllm/envs.py VLLM_CACHE_ROOT
    VLLM_CACHE_ROOT: str = "~/.cache/vllm"
    # SOURCE: vllm/envs.py VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE (default False)
    VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE: bool = False
    # SOURCE: vllm/envs.py VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING
    VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING: bool = False
    # SOURCE: vllm/envs.py VLLM_GPU_SYNC_CHECK (None | "warn" | "error")
    VLLM_GPU_SYNC_CHECK = None
    # SOURCE: vllm/envs.py VLLM_BATCH_INVARIANT (default False; branch deleted)
    VLLM_BATCH_INVARIANT: bool = False

    # SOURCE: vllm/envs.py compile_factors() — env factor list for the
    # compile-cache hash (deleted cache block's input; seam returns the
    # documented-factor subset)
    @staticmethod
    def compile_factors() -> list:  # SOURCE: vllm/envs.py
        # HOST SEAM — the deleted cache block consumed this; kept only for
        # interface parity of VllmConfig.compute_hash paths that call it.
        return [envs.VLLM_USE_STANDALONE_COMPILE, envs.VLLM_USE_MEGA_AOT_ARTIFACT]


# ---------------------------------------------------------------------------
# platforms — vllm/platforms/interface.py current_platform seam
# ---------------------------------------------------------------------------


# SOURCE: vllm/platforms/__init__.py current_platform — HOST SEAM platform
# interface subset; device predicates follow the host torch installation.
class _PlatformSeam:
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # SOURCE: vllm/platforms/interface.py Platform.device_plugin dispatch_key
    @property
    def dispatch_key(self) -> str:  # SOURCE: vllm/platforms/interface.py
        # real: CUDA-alike → "CUDA", CPU → "CPU". HOST SEAM: the companion
        # exercises every tensor on host CPU (direct_register_custom_op needs
        # the CPU dispatch key for the op to be callable in-process); the
        # "CUDA" face is the vllm-container domain.
        return "CPU"

    # SOURCE: vllm/platforms/interface.py Platform.is_cuda / is_cpu / ...
    def is_cuda(self) -> bool:
        return torch.cuda.is_available()

    def is_cuda_alike(self) -> bool:  # SOURCE: vllm/platforms/interface.py
        return torch.cuda.is_available()

    def is_rocm(self) -> bool:  # SOURCE: vllm/platforms/interface.py
        return False

    def is_cpu(self) -> bool:  # SOURCE: vllm/platforms/interface.py
        return not torch.cuda.is_available()

    def is_tpu(self) -> bool:  # SOURCE: vllm/platforms/interface.py
        return False

    def is_xpu(self) -> bool:  # SOURCE: vllm/platforms/interface.py Platform.is_xpu
        return False

    def is_out_of_tree(self) -> bool:  # SOURCE: vllm/platforms/interface.py Platform.is_out_of_tree
        return False

    # SOURCE: vllm/platforms/cuda.py CudaPlatform.opaque_attention_op (True on
    # cuda-alike) — HOST SEAM: the companion exercises the direct-call branch
    # on host CPU tensors; the torch.ops branch is real and identical on
    # opaque platforms (both call the same op functions).
    def opaque_attention_op(self) -> bool:  # SOURCE: vllm/platforms/cuda.py
        return False

    # SOURCE: vllm/platforms/interface.py Platform.get_compile_backend —
    # "" → "inductor" on cuda-alike, "eager" otherwise (pin default)
    def get_compile_backend(self) -> str:  # SOURCE: vllm/platforms/interface.py
        return "inductor" if torch.cuda.is_available() else "eager"

    # SOURCE: vllm/platforms/interface.py Platform.simple_compile_backend —
    # HOST SEAM "eager" (real: the platform's minimal JIT-free compile
    # backend used by CustomOp.maybe_compile; host torch runs the eager
    # face, control flow identical)
    simple_compile_backend = "eager"

    # SOURCE: vllm/platforms/interface.py Platform.get_static_graph_wrapper_cls
    # — canonical qualname of CUDAGraphWrapper (resolved through sys.modules
    # aliases installed below)
    def get_static_graph_wrapper_cls(self) -> str:  # SOURCE: vllm/platforms/interface.py
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    # SOURCE: vllm/platforms/interface.py Platform.get_global_graph_pool /
    # graph_pool_handle — None means torch.cuda.graph allocates its own pool
    def get_global_graph_pool(self):  # SOURCE: vllm/platforms/interface.py
        return None

    def graph_pool_handle(self):  # SOURCE: vllm/platforms/interface.py
        return None

    # SOURCE: vllm/platforms/interface.py Platform.apply_config_platform_defaults
    def apply_config_platform_defaults(self, vllm_config) -> None:
        return None

    # SOURCE: vllm/platforms/interface.py Platform.set_additional_forward_context
    # — returns additional kwargs dict ({}` on cuda-alike)
    def set_additional_forward_context(self, **kwargs) -> dict:  # SOURCE: vllm/platforms/interface.py
        return {}

    # SOURCE: vllm/platforms/interface.py Platform.synchronize (None on cuda)
    synchronize = None

    # SOURCE: vllm/platforms/interface.py Platform.manual_seed_all
    def manual_seed_all(self, seed) -> None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


current_platform = _PlatformSeam()  # HOST SEAM


# ---------------------------------------------------------------------------
# distributed — vllm/distributed seam (ch34 domain)
# ---------------------------------------------------------------------------


# SOURCE: vllm/distributed/parallel_state.py graph_capture — HOST SEAM (real:
# torch.cuda.synchronize + capture-safe allocator context; nullcontext is the
# non-CUDA no-op branch of the real context manager)
@contextlib.contextmanager
def graph_capture(device=None):  # HOST SEAM  # SOURCE: vllm/distributed/parallel_state.py
    yield


# SOURCE: vllm/distributed/parallel_state.py get_world_group().local_rank
def get_world_group():  # HOST SEAM
    return _types.SimpleNamespace(local_rank=0)


# SOURCE: vllm/distributed/parallel_state.py is_global_first_rank
def is_global_first_rank() -> bool:  # HOST SEAM
    return True


# SOURCE: vllm/v1/worker/dp_utils.py coordinate_batch_across_dp — HOST SEAM
# (DP>1 collective coordination is ch34 domain; single-DP deployments never
# call it — structural hole by design)
def coordinate_batch_across_dp(*args, **kwargs):  # SOURCE: vllm/v1/worker/dp_utils.py
    raise NotImplementedError(
        "coordinate_batch_across_dp (data-parallel > 1) is out of the ch19 "
        "companion's surface — see impl-notes.md §Seam 清单 (ch34 domain)."
    )


# ---------------------------------------------------------------------------
# ubatch — vllm/v1/worker/ubatch_utils.py UBatchSlices (DBO 扩展态，占位类型)
# ---------------------------------------------------------------------------


# SOURCE: vllm/v1/worker/ubatch_utils.py UBatchSlices — HOST SEAM annotation
# placeholder (DBO microbatching is out of the first-read canon)
class UBatchSlices:  # HOST SEAM  # SOURCE: vllm/v1/worker/ubatch_utils.py
    pass


# ---------------------------------------------------------------------------
# offloader — vllm/model_executor/offloader/base.py get_offloader seam
# (calls deleted per subtraction_plan delete #5)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# kernel warmup / GC freeze — startup-tail call faces (m18; ch17 seams)
# ---------------------------------------------------------------------------


# SOURCE: vllm/v1/worker/gpu_model_runner.py kernel_warmup — HOST SEAM no-op
# (kernel autotuning domain; called for its place in the startup orchestration)
def kernel_warmup(worker) -> None:  # HOST SEAM  # SOURCE: vllm/v1/worker/gpu_model_runner.py
    return None


# ---------------------------------------------------------------------------
# ir — vllm/ir native reference math for RMSNorm (IrOp machinery out of scope)
# ---------------------------------------------------------------------------


class _IrOpSeam:
    """IrOp lookalike exposing only `__call__`/`maybe_inplace` to the native
    math — the vllm.ir IrOp wrapper (torch-library registration, provider
    priority dispatch) is out of this chapter's surface; the math bodies below
    are verbatim from vllm/ir/ops/layernorm.py."""

    # SOURCE: vllm/ir/ops/layernorm.py:L10-L21 rms_norm native implementation
    def rms_norm(self, x, weight, epsilon, variance_size=None):
        """Weighted root-mean-square layer normalization"""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        x_var = x if variance_size is None else x[..., :variance_size]
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + epsilon)
        if weight is not None:
            x = x.to(weight.dtype) * weight
        return x.to(orig_dtype)

    # SOURCE: vllm/ir/ops/layernorm.py:L44-L62 fused_add_rms_norm native
    # implementation (allow_inplace op; maybe_inplace call site gets the
    # functional body's semantics)
    def _fused_add(self, x, x_residual, weight, epsilon, variance_size=None):  # SOURCE: vllm/ir/ops/layernorm.py:L44-L62
        """Fused add and weighted root-mean-square layer normalization"""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        x = x + x_residual.to(torch.float32)
        x_residual = x.to(orig_dtype)

        x_var = x if variance_size is None else x[..., :variance_size]
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + epsilon)
        if weight is not None:
            x = x.to(weight.dtype) * weight
        return x.to(orig_dtype), x_residual

    # SOURCE: vllm/ir/op.py IrOpInplaceOverload.__call__ (maybe_inplace face)
    def maybe_inplace(self, *args, **kwargs):
        return self._fused_add(*args, **kwargs)


# SOURCE: vllm/ir/__init__.py `from . import ops` + vllm/ir/ops/layernorm.py —
# HOST SEAM namespace: IrOp machinery out of scope, native math verbatim
ir = _types.SimpleNamespace(
    ops=_types.SimpleNamespace(
        rms_norm=_IrOpSeam().rms_norm,
        fused_add_rms_norm=_IrOpSeam(),
    )
)


# ---------------------------------------------------------------------------
# weak_ref_tensors — host patch for the vLLM C++ op torch.ops._C.weak_ref_tensor
# ---------------------------------------------------------------------------

# SOURCE: vllm/utils/torch_utils.py weak_ref_tensor uses torch.ops._C.
# weak_ref_tensor (registered by the vLLM C++ extension). HOST SEAM: without
# the extension we approximate with detach() — same storage/data_ptr sharing
# observable in tests; the real op additionally lets the original tensor die
# so the graph pool can reclaim memory (only affects memory reclamation).
if not hasattr(torch.ops._C, "weak_ref_tensor"):  # HOST SEAM
    torch.ops._C.weak_ref_tensor = lambda t: t.detach()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# pynccl allocator — set_graph_pool_id (shared graph-pool injection seam)
# ---------------------------------------------------------------------------


# SOURCE: vllm/distributed/device_communicators/pynccl_allocator.py
# set_graph_pool_id — HOST SEAM no-op (real: tells the CUDA caching allocator
# to allocate the next cudagraph's private pool from the shared pool; the
# allocator plumbing is ch14 memory-ledger domain, not observable here).
def set_graph_pool_id(pool):  # HOST SEAM  # SOURCE: vllm/distributed/device_communicators/pynccl_allocator.py
    return None


# ---------------------------------------------------------------------------
# breakable cudagraph / kv-transfer decorators (extension-state seams)
# ---------------------------------------------------------------------------


# SOURCE: vllm/compilation/breakable_cudagraph.py is_breakable_cudagraph_enabled
# — env gate VLLM_USE_BREAKABLE_CUDAGRAPH (envs.py:L170, default 0)
def is_breakable_cudagraph_enabled() -> bool:  # HOST SEAM  # SOURCE: vllm/compilation/breakable_cudagraph.py
    return bool(envs.VLLM_USE_BREAKABLE_CUDAGRAPH)


# SOURCE: vllm/compilation/breakable_cudagraph.py eager_break_during_capture —
# HOST SEAM identity (real: when breakable cudagraph capture is active it
# ends the current stream-capture segment around the op; disabled path is
# the plain decorator passthrough, which is the branch this chapter pins)
def eager_break_during_capture(fn):  # HOST SEAM  # SOURCE: vllm/compilation/breakable_cudagraph.py
    return fn


# SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py
# maybe_transfer_kv_layer — HOST SEAM identity (kv-connector transfer is
# ch16 domain; the no-connector path is the identity decorator)
def maybe_transfer_kv_layer(fn):  # HOST SEAM  # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py
    return fn


# ---------------------------------------------------------------------------
# startup-tail faces: jit monitor / random seed / pp group / env patch
# ---------------------------------------------------------------------------


# SOURCE: vllm/utils/jit_monitor.py activate — HOST SEAM no-op (real: with
# mode=None, activate is a no-op; the warn/raise monitors are observability
# domain — m18 keeps only the call's place in the orchestration)
def activate_jit_monitor(mode=None, verbose: bool = False):  # HOST SEAM  # SOURCE: vllm/utils/jit_monitor.py
    return None


# SOURCE: vllm/utils/__init__.py set_random_seed — HOST SEAM no-op (real:
# seeds random/numpy/torch; ch09 domain — only the call site is pinned here)
def set_random_seed(seed: int):  # HOST SEAM  # SOURCE: vllm/utils/__init__.py
    return None


# SOURCE: vllm/distributed/parallel_state.py get_pp_group — HOST SEAM (the
# companion's single-process path is always "last rank")
def get_pp_group():  # HOST SEAM  # SOURCE: vllm/distributed/parallel_state.py
    return _types.SimpleNamespace(is_last_rank=True)


# SOURCE: vllm/env_override.py _apply_constrain_to_fx_strides_patch —
# HOST SEAM no-op (real: patches dynamo stride handling pre-compile)
def _apply_constrain_to_fx_strides_patch():  # HOST SEAM  # SOURCE: vllm/env_override.py
    return None


# SOURCE: vllm/model_executor/utils.py maybe_disable_graph_partition —
# HOST SEAM returning empty options (real: platform-specific compile options
# dict that may disable the inductor graph partition)
def maybe_disable_graph_partition(backend: str):  # HOST SEAM  # SOURCE: vllm/model_executor/utils.py
    return {}


# ---------------------------------------------------------------------------
# aiter / flashinfer predicates (O2 preset lambdas reference these)
# ---------------------------------------------------------------------------


# SOURCE: vllm/_aiter_ops/__init__.py rocm_aiter_ops — HOST SEAM (ROCm AITER
# kernels are not present on this host; is_enabled()=False is the pinned
# behavior of the non-ROCm path)
rocm_aiter_ops = _types.SimpleNamespace(is_enabled=lambda: False)  # HOST SEAM


# SOURCE: vllm/utils/flashinfer.py has_flashinfer — HOST SEAM (no flashinfer
# install on this host; False is the pinned non-flashinfer behavior)
def has_flashinfer() -> bool:  # HOST SEAM  # SOURCE: vllm/utils/flashinfer.py
    return False


# ---------------------------------------------------------------------------
# small math / batch-invariance / spec-quant faces
# ---------------------------------------------------------------------------


# SOURCE: vllm/utils/math_utils.py round_up
def round_up(value: int, multiple: int) -> int:  # HOST SEAM (verbatim math)
    return ((value + multiple - 1) // multiple) * multiple


# ---------------------------------------------------------------------------
# runner-span faces: NULL_BLOCK_ID / record_function_or_nullcontext
# ---------------------------------------------------------------------------


# SOURCE: vllm/v1/attention/backends/utils.py:L46 NULL_BLOCK_ID —— Block 0 is
# reserved for padding（值 0 逐字；同文件 PAD_SLOT_ID=-1 归 ch22 slot 域）
NULL_BLOCK_ID = 0  # HOST SEAM (verbatim constant)


# SOURCE: vllm/v1/utils.py:L758-L766 record_function_or_nullcontext ——
# HOST SEAM：默认（两个 scopes-for-profiling env 均关）路径即 nullcontext；
# 真源的 _PROFILER_FUNC 快路径/env 分支是 profiling 观测域
@contextlib.contextmanager
def record_function_or_nullcontext(name: str):  # HOST SEAM  # SOURCE: vllm/v1/utils.py:L758-L766
    yield


# SOURCE: vllm/model_executor/layers/batch_invariant.py
# rms_norm_batch_invariant — HOST SEAM stub (only reachable with
# VLLM_BATCH_INVARIANT=1, an experimental path this chapter never enables;
# the kernel itself is batch-invariance domain)
def rms_norm_batch_invariant(*args, **kwargs):  # HOST SEAM  # SOURCE: vllm/model_executor/layers/batch_invariant.py
    raise NotImplementedError(
        "rms_norm_batch_invariant requires VLLM_BATCH_INVARIANT=1 "
        "(experimental batch-invariance domain, off the ch19 companion path)"
    )


# SOURCE: vllm/v1/kv_cache_interface.py EncoderOnlyAttentionSpec — HOST SEAM
# marker class (KV-cache spec taxonomy is ch13 domain; kept for the isinstance
# branches in the runner padding spans)
class EncoderOnlyAttentionSpec:  # HOST SEAM  # SOURCE: vllm/v1/kv_cache_interface.py
    pass


# ---------------------------------------------------------------------------
# quant faces for the Attention.query_quant branch (F10 seedling, kept
# verbatim; ch27 owns the real quant machinery)
# ---------------------------------------------------------------------------


# SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py
# GroupShape — HOST SEAM (real: NamedTuple of dim0/dim1 quant group shape)
class GroupShape:  # HOST SEAM  # SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py
    PER_TENSOR = None  # real: GroupShape(-1, -1)

    def __init__(self, dim0, dim1):  # SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py
        self.dim0 = dim0
        self.dim1 = dim1


# SOURCE: vllm/model_executor/layers/quantization/input_quant_fp8.py QuantFP8
# — HOST SEAM constructor face (real: per-tensor/per-group fp8 quantizer whose
# __call__ returns (quantized, scale); constructing it needs the ch27 quant
# domain — this chapter keeps the branch text, not the kernel)
class QuantFP8:  # HOST SEAM  # SOURCE: vllm/model_executor/layers/quantization/input_quant_fp8.py
    def __init__(self, static: bool = False, group_shape=None):  # SOURCE: vllm/model_executor/layers/quantization/input_quant_fp8.py
        self.static = static
        self.group_shape = group_shape

    def __call__(self, tensor, scale=None):  # pragma: no cover - ch27 domain  # SOURCE: vllm/model_executor/layers/quantization/input_quant_fp8.py
        raise NotImplementedError(
            "QuantFP8 quantization math is ch27 domain; the ch19 companion "
            "keeps the query_quant branch text (F10 seedling) only."
        )


# ---------------------------------------------------------------------------
# vllm-module aliases — canonical qualnames resolve into this package
# ---------------------------------------------------------------------------


# SOURCE: vllm/utils/import_utils.py resolve_obj_by_qualname imports the
# module by qualname — HOST SEAM installer pre-registers PEP 562 proxy
# modules for the canonical vllm.* paths this chapter's kept code resolves
# (e.g. current_platform.get_static_graph_wrapper_cls()); each proxy lazily
# forwards attribute access to the companion's real module.
def install_vllm_module_aliases() -> None:  # SOURCE: vllm/utils/import_utils.py
    aliases = {
        "vllm.compilation.cuda_graph": "implementation.compilation.cuda_graph",
    }
    for canonical, ours in aliases.items():
        if canonical in sys.modules and getattr(
            sys.modules[canonical], "__vllm_companion_proxy__", False
        ):
            continue
        proxy = _types.ModuleType(canonical)
        proxy.__vllm_companion_proxy__ = True

        def _make_lazy(target):  # SOURCE: vllm/utils/import_utils.py resolve_obj_by_qualname（HOST SEAM 代理装配）
            def __getattr__(name):  # SOURCE: vllm/utils/import_utils.py resolve_obj_by_qualname（HOST SEAM 代理装配）
                import importlib

                return getattr(importlib.import_module(target), name)

            return __getattr__

        proxy.__getattr__ = _make_lazy(ours)  # type: ignore[attr-defined]
        sys.modules[canonical] = proxy
