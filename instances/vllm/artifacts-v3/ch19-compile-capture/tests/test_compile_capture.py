# TDD tests for the v3 ch19 subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).
#
# These assert the *observable pinned vLLM behavior* this chapter teaches — the
# execution-shape dimension of the GPU execution arm: piecewise compilation +
# CUDA graph capture/replay. Mirrors of upstream behavior come from the pinned
# tree itself (notably tests/v1/cudagraph/test_cudagraph_dispatch.py, whose
# dispatcher cases and CUDA-gated CUDAGraphWrapper cases are re-asserted here
# against the subtract-only companion).
#
# Unit / contract, in-process (host CPU, no vllm package):
#   - CUDAGraphMode enum algebra (compilation.py:L53-L103)
#   - OptimizationLevel -O0..-O3 presets + VllmConfig post_init 落账 (vllm.py)
#   - set_splitting_ops_for_v1 main branch (compilation.py:L1133-L1184)
#   - CustomOp constructor-time dispatch / enabled() / default_on() /
#     maybe_compile() (custom_op.py) + RMSNorm as the worked instance
#   - forward context: BatchDescriptor key, thread-local set/get, assert-crash
#   - Attention registration + out-variant op trio (attention.py:L437-L846)
#   - LayerName opaque encode/resolve (torch_utils.py:L845-L888)
#   - should_split + split_graph piecewise algorithm (partition_rules/backends)
#   - CudagraphDispatcher size table / key init / dispatch / capture descs
#   - CUDAGraphWrapper dispatch head (no-context & mode-mismatch passthrough)
#   - capture window tripwire (monitor.set/validate)
#   - runner spans: _determine_batch_execution_and_padding, padding 四件套,
#     load_model FULL wrapper mount, _check_and_update_cudagraph_mode weakest link
#   - Worker.compile_or_warm_up_model startup orchestration order
# End-to-end, real dynamo (host CPU, eager compiler adaptor):
#   - torch.compile(backend=VllmBackend) → split → interpreter → stitched
#     callable; guards dropped (one trace, no retrace on new shapes)
# CUDA-gated (skipped without a GPU, mirroring upstream test gating):
#   - real capture → replay through CUDAGraphWrapper + DEBUG data_ptr assert
#
# Run:  cd instances/vllm/artifacts-v3/ch19-compile-capture
#       python -m pytest tests/ -q
#
# Host: Windows, real torch (CUDA present); no vllm package.

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

_TESTS_DIR = Path(__file__).resolve().parent
_CHAPTER_DIR = _TESTS_DIR.parent
sys.path.insert(0, str(_CHAPTER_DIR))

implementation = importlib.import_module("implementation")

from implementation.config.compilation import (  # noqa: E402
    CUDAGraphMode,
    CompilationConfig,
    CompilationMode,
    PassConfig,
)
from implementation.config import set_current_vllm_config  # noqa: E402
from implementation.config.vllm import (  # noqa: E402
    IS_QUANTIZED,
    OPTIMIZATION_LEVEL_00,
    OPTIMIZATION_LEVEL_01,
    OPTIMIZATION_LEVEL_02,
    OPTIMIZATION_LEVEL_03,
    OPTIMIZATION_LEVEL_TO_CONFIG,
    OptimizationLevel,
    VllmConfig,
)
from implementation.compilation.backends import (  # noqa: E402
    VllmBackend,
    split_graph,
    wrap_with_cudagraph_if_needed,
)
from implementation.compilation.counter import compilation_counter  # noqa: E402
from implementation.compilation.cuda_graph import (  # noqa: E402
    CUDAGraphEntry,
    CUDAGraphOptions,
    CUDAGraphWrapper,
)
from implementation.compilation.monitor import (  # noqa: E402
    set_cudagraph_capturing_enabled,
    validate_cudagraph_capturing_enabled,
)
from implementation.compilation.partition_rules import should_split  # noqa: E402
from implementation.compilation.wrapper import (  # noqa: E402
    TorchCompileWithNoGuardsWrapper,
)
from implementation.forward_context import (  # noqa: E402
    BatchDescriptor,
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
    set_forward_context,
)
from implementation.model_executor.custom_op import CustomOp, op_registry  # noqa: E402
from implementation.model_executor.layers.attention.attention import (  # noqa: E402
    Attention,
    get_attention_context,
    unified_attention_with_output,
    unified_kv_cache_update,
)
from implementation.model_executor.layers.layernorm import RMSNorm  # noqa: E402
from implementation.utils.torch_utils import (  # noqa: E402
    LayerName,
    _encode_layer_name,
    _resolve_layer_name,
)
from implementation.v1.cudagraph_dispatcher import CudagraphDispatcher  # noqa: E402

IS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# helpers — mirror upstream tests/v1/cudagraph/test_cudagraph_dispatch.py
# ---------------------------------------------------------------------------


def make_seam_vllm_config(
    compilation_config: CompilationConfig,
    max_num_seqs: int = 8,
) -> SimpleNamespace:
    """Upstream `_create_vllm_config` (MagicMock + real sub-configs), with the
    ch03-domain sub-configs as attribute carriers (dispatcher only reads
    max_num_seqs / num_speculative_tokens / lora_config / parallel flags)."""
    cfg = SimpleNamespace(
        compilation_config=compilation_config,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            tensor_parallel_size=1,
            use_sequence_parallel_moe=False,
            is_moe_model=None,
            use_ubatching=False,
            all2all_backend="deepep_low_latency",
        ),
        speculative_config=None,
        num_speculative_tokens=0,
        lora_config=None,
        observability_config=SimpleNamespace(cudagraph_metrics=False),
    )
    # Mimic the behavior of VllmConfig.__post_init__()
    if compilation_config.mode == CompilationMode.VLLM_COMPILE:
        compilation_config.set_splitting_ops_for_v1(
            all2all_backend=cfg.parallel_config.all2all_backend,
            data_parallel_size=cfg.parallel_config.data_parallel_size,
        )
    # mimic VllmConfig.__post_init__: O2-default 档位落账 (mode / custom_ops
    # base / cudagraph_mode preset)，供裸 CompilationConfig() 的构造面使用
    if compilation_config.mode is None:
        compilation_config.mode = CompilationMode.VLLM_COMPILE
    if all(s not in compilation_config.custom_ops for s in ("all", "none")):
        if (
            compilation_config.backend == "inductor"
            and compilation_config.mode != CompilationMode.NONE
        ):
            compilation_config.custom_ops.append("none")
        else:
            compilation_config.custom_ops.append("all")
    if compilation_config.cudagraph_mode is None:
        compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    # mimic VllmConfig.__post_init__
    if compilation_config.cudagraph_capture_sizes:
        compilation_config.max_cudagraph_capture_size = (
            compilation_config.cudagraph_capture_sizes[-1]
        )
        compilation_config.post_init_cudagraph_sizes()
    return cfg


def _cfg_ctx(cc: CompilationConfig | None = None):
    """Real construction contract: CustomOp/Attention layers are built inside
    a `set_current_vllm_config` context (custom_op.py's dispatch query face)."""
    return set_current_vllm_config(
        make_seam_vllm_config(cc if cc is not None else CompilationConfig())
    )


# ---------------------------------------------------------------------------
# m11 — CUDAGraphMode enum algebra (config/compilation.py:L53-L103)
# ---------------------------------------------------------------------------


class TestCUDAGraphMode:
    def test_runtime_and_combined_values(self):
        assert CUDAGraphMode.NONE.value == 0
        assert CUDAGraphMode.PIECEWISE.value == 1
        assert CUDAGraphMode.FULL.value == 2
        # combined levels are tuple-valued (python enum bakes the class-body
        # member refs as their raw values: plain Enum, not IntEnum members)
        assert CUDAGraphMode.FULL_DECODE_ONLY.value == (
            CUDAGraphMode.FULL.value,
            CUDAGraphMode.NONE.value,
        )
        assert CUDAGraphMode.FULL_AND_PIECEWISE.value == (
            CUDAGraphMode.FULL.value,
            CUDAGraphMode.PIECEWISE.value,
        )
        # and the tuple members decode back through CUDAGraphMode(...)
        assert CUDAGraphMode(CUDAGraphMode.FULL_DECODE_ONLY.value[0]) is (
            CUDAGraphMode.FULL
        )

    def test_decode_and_mixed_mode_decompose_combined_levels(self):
        m = CUDAGraphMode.FULL_AND_PIECEWISE
        assert m.separate_routine()
        assert m.decode_mode() == CUDAGraphMode.FULL
        assert m.mixed_mode() == CUDAGraphMode.PIECEWISE
        assert CUDAGraphMode.FULL_DECODE_ONLY.mixed_mode() == CUDAGraphMode.NONE
        # non-combined levels decompose to themselves
        assert CUDAGraphMode.PIECEWISE.decode_mode() == CUDAGraphMode.PIECEWISE
        assert not CUDAGraphMode.PIECEWISE.separate_routine()

    def test_has_mode_and_derived_predicates(self):
        m = CUDAGraphMode.FULL_AND_PIECEWISE
        assert m.has_mode(CUDAGraphMode.FULL)
        assert m.has_mode(CUDAGraphMode.PIECEWISE)
        assert not m.has_mode(CUDAGraphMode.NONE)
        assert m.requires_piecewise_compilation()
        assert m.has_piecewise_cudagraphs()
        assert m.max_cudagraph_mode() == CUDAGraphMode.FULL
        assert m.has_full_cudagraphs()
        assert not CUDAGraphMode.PIECEWISE.has_full_cudagraphs()
        assert not CUDAGraphMode.FULL.requires_piecewise_compilation()

    def test_valid_runtime_modes_excludes_combined_levels(self):
        assert CUDAGraphMode.valid_runtime_modes() == frozenset(
            {CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL}
        )
        assert CUDAGraphMode.FULL.is_valid_runtime_mode()
        assert not CUDAGraphMode.FULL_AND_PIECEWISE.is_valid_runtime_mode()

    def test_bool_and_str(self):
        assert bool(CUDAGraphMode.PIECEWISE)
        assert not bool(CUDAGraphMode.NONE)
        assert str(CUDAGraphMode.FULL_AND_PIECEWISE) == "FULL_AND_PIECEWISE"


# ---------------------------------------------------------------------------
# m11 — -O0..-O3 presets (config/vllm.py:L104-L327) and post_init 落账
# ---------------------------------------------------------------------------


class TestOptimizationLevels:
    def test_enum_docs(self):
        # pinned docstring facts (enum member docstrings live in the class
        # body source — plain Enum does not attach them as __doc__)
        import inspect

        src = inspect.getsource(OptimizationLevel)
        assert "no compilation, no cudagraphs" in src
        assert "Dynamo+Inductor" in src
        assert "Full and Piecewise cudagraphs" in src

    def test_preset_table(self):
        assert set(OPTIMIZATION_LEVEL_TO_CONFIG) == set(OptimizationLevel)
        assert OPTIMIZATION_LEVEL_TO_CONFIG[OptimizationLevel.O0] is OPTIMIZATION_LEVEL_00
        assert OPTIMIZATION_LEVEL_TO_CONFIG[OptimizationLevel.O2] is OPTIMIZATION_LEVEL_02
        assert (
            OPTIMIZATION_LEVEL_00["compilation_config"]["cudagraph_mode"]
            is CUDAGraphMode.NONE
        )
        assert (
            OPTIMIZATION_LEVEL_01["compilation_config"]["cudagraph_mode"]
            is CUDAGraphMode.PIECEWISE
        )
        for lvl in (OPTIMIZATION_LEVEL_02, OPTIMIZATION_LEVEL_03):
            assert (
                lvl["compilation_config"]["cudagraph_mode"]
                is CUDAGraphMode.FULL_AND_PIECEWISE
            )
            assert not lvl["compilation_config"]["use_inductor_graph_partition"]
        # every O0 pass is pinned off; O2 carries the hardware-predicate fuses
        assert not any(OPTIMIZATION_LEVEL_00["compilation_config"]["pass_config"].values())
        o2_passes = OPTIMIZATION_LEVEL_02["compilation_config"]["pass_config"]
        assert o2_passes["fuse_attn_quant"] is IS_QUANTIZED
        assert callable(o2_passes["fuse_norm_quant"])

    def test_default_level_is_o2(self):
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(VllmConfig)}
        assert fields["optimization_level"].default is OptimizationLevel.O2

    def make_vllm_config(self, **kw):
        return VllmConfig._for_tests(**kw)

    def test_post_init_mode_and_custom_ops_accounting(self):
        # O2 (default level) + inductor backend → VLLM_COMPILE + custom_ops 'none'
        cfg = self.make_vllm_config(optimization_level=OptimizationLevel.O2)
        assert cfg.compilation_config.mode == CompilationMode.VLLM_COMPILE
        assert cfg.compilation_config.custom_ops == ["none"]
        # O0 → NONE
        cfg = self.make_vllm_config(optimization_level=OptimizationLevel.O0)
        assert cfg.compilation_config.mode == CompilationMode.NONE
        # O1 → VLLM_COMPILE
        cfg = self.make_vllm_config(optimization_level=OptimizationLevel.O1)
        assert cfg.compilation_config.mode == CompilationMode.VLLM_COMPILE
        # non-inductor backend → base mode 'all'
        cfg = self.make_vllm_config(
            optimization_level=OptimizationLevel.O0, backend="eager"
        )
        assert cfg.compilation_config.custom_ops == ["all"]

    def test_post_init_blocked_weights_force_quant_fp8(self):
        # F10 seedling: blocked-quant weights force the +quant_fp8 CUDA op
        quant = SimpleNamespace(weight_block_size=[128, 128])
        cfg = self.make_vllm_config(quant_config=quant)
        assert "+quant_fp8" in cfg.compilation_config.custom_ops
        # unquantized / attribute-less quant config does not
        cfg = self.make_vllm_config(quant_config=SimpleNamespace())
        assert "+quant_fp8" not in cfg.compilation_config.custom_ops

    def test_post_init_applies_o2_defaults_without_overriding_user(self):
        cfg = self.make_vllm_config(optimization_level=OptimizationLevel.O2)
        # preset pass values were applied onto the (None-defaulted) PassConfig
        assert cfg.compilation_config.pass_config.fuse_attn_quant is False
        assert (
            cfg.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
        )
        assert cfg.kernel_config.enable_flashinfer_autotune is True
        # a user-set value is not overridden by the preset
        cc = CompilationConfig(pass_config=PassConfig(fuse_norm_quant=True))
        cfg = self.make_vllm_config(compilation_config=cc)
        assert cfg.compilation_config.pass_config.fuse_norm_quant is True

    def test_ir_enable_torch_wrap_default(self):
        cfg = self.make_vllm_config(optimization_level=OptimizationLevel.O2)
        assert cfg.compilation_config.ir_enable_torch_wrap is True
        cfg = self.make_vllm_config(optimization_level=OptimizationLevel.O0)
        assert cfg.compilation_config.ir_enable_torch_wrap is False


# ---------------------------------------------------------------------------
# m07 — splitting_ops 切图点账本 (config/compilation.py:L764-L784, L1133-L1184)
# ---------------------------------------------------------------------------


def _vllm_compile_cc(**kw) -> CompilationConfig:
    kw.setdefault("mode", CompilationMode.VLLM_COMPILE)
    return CompilationConfig(**kw)


class TestSplittingOps:
    def test_attention_ops_list_is_13_ops(self):
        ops = CompilationConfig._attention_ops
        assert len(ops) == 13
        assert ops[0] == "vllm::unified_attention_with_output"
        assert ops[1] == "vllm::unified_mla_attention_with_output"
        assert "vllm::hpc_rope_norm_forward" == ops[-1]
        # PyTorch operator format: "namespace::name"
        assert all("::" in op for op in ops)

    def test_set_splitting_ops_for_v1_main_branch(self):
        cc = _vllm_compile_cc()
        cc.set_splitting_ops_for_v1(all2all_backend="deepep_low_latency")
        assert cc.splitting_ops is not None
        assert "vllm::unified_attention_with_output" in cc.splitting_ops
        # issue #33267: kv-cache update ops moved out of the compiled graph
        assert "vllm::unified_kv_cache_update" in cc.splitting_ops
        assert "vllm::unified_mla_kv_cache_update" in cc.splitting_ops
        assert len(cc.splitting_ops) == len(CompilationConfig._attention_ops) + 2

    def test_set_splitting_ops_non_vllm_compile_empties(self):
        cc = CompilationConfig(mode=CompilationMode.NONE)
        cc.set_splitting_ops_for_v1(all2all_backend="deepep_low_latency")
        assert cc.splitting_ops == []

    def test_set_splitting_ops_does_not_mutate_classvar(self):
        cc = _vllm_compile_cc()
        cc.set_splitting_ops_for_v1(all2all_backend="deepep_low_latency")
        cc.splitting_ops.append("vllm::sentinel")
        assert "vllm::sentinel" not in CompilationConfig._attention_ops

    def test_fuse_rope_kvcache_downgrade_when_splitting_none(self):
        cc = _vllm_compile_cc(
            pass_config=PassConfig(fuse_rope_kvcache=True),
        )
        cc.set_splitting_ops_for_v1(all2all_backend="deepep_low_latency")
        assert cc.pass_config.fuse_rope_kvcache is False

    def test_containment_predicates(self):
        cc = _vllm_compile_cc()
        # before assembly: kv-update containment reports True (will be added)
        assert cc.splitting_ops_contain_kv_cache_update()
        cc.set_splitting_ops_for_v1(all2all_backend="deepep_low_latency")
        assert cc.splitting_ops_contain_attention()
        assert cc.splitting_ops_contain_kv_cache_update()
        assert cc.is_attention_compiled_piecewise()
        # empty splitting ops → not piecewise-compiled attention
        cc2 = _vllm_compile_cc(splitting_ops=[])
        assert not cc2.splitting_ops_contain_attention()


# ---------------------------------------------------------------------------
# m01/m02/m03 — CustomOp dispatch protocol (model_executor/custom_op.py)
# ---------------------------------------------------------------------------


class TestCustomOp:
    def test_rms_norm_registered(self):
        assert op_registry["rms_norm"] is RMSNorm
        assert RMSNorm.name == "rms_norm"

    def test_duplicate_registration_rejected(self):
        with pytest.raises(AssertionError, match="Duplicate op name"):

            @CustomOp.register("rms_norm")
            class RMSNorm2(CustomOp):
                def forward_native(self, x):
                    return x

    def test_default_on_base_mode_rules(self):
        cc = _vllm_compile_cc()
        with _cfg_ctx(cc):
            cc.custom_ops = ["none"]
            assert CustomOp.default_on() is False
            cc.custom_ops = ["all"]
            assert CustomOp.default_on() is True
            cc.custom_ops = []
            with pytest.raises(ValueError, match="exactly one base mode"):
                CustomOp.default_on()
            cc.custom_ops = ["all", "none"]
            with pytest.raises(ValueError, match="exactly one base mode"):
                CustomOp.default_on()

    def test_enabled_plus_minus_name_protocol(self):
        cc = _vllm_compile_cc()
        cc.custom_ops = ["none", "+rms_norm"]
        with _cfg_ctx(cc):
            assert RMSNorm.enabled() is True
        cc.custom_ops = ["all", "-rms_norm"]
        with _cfg_ctx(cc):
            assert RMSNorm.enabled() is False
        cc.custom_ops = ["all", "+rms_norm", "-rms_norm"]
        with _cfg_ctx(cc), pytest.raises(
            ValueError, match="cannot both enable and disable"
        ):
            RMSNorm.enabled()

    def test_dispatch_forward_binds_once_and_counts(self):
        cc = _vllm_compile_cc()
        cc.custom_ops = ["none"]
        with _cfg_ctx(cc):
            layer = RMSNorm(8)
            # disabled under 'none': bound to (maybe_compile of) forward_native
            assert layer._forward_method is not None
            assert cc.disabled_custom_ops.get("rms_norm") == 1
            # the real enable face: +name exact switch on top of 'none'
            cc.custom_ops = ["none", "+rms_norm"]
            layer2 = RMSNorm(8)
        assert cc.enabled_custom_ops.get("rms_norm") == 1
        # forward is a single attribute forward — same bound callable each call
        xx = torch.randn(2, 8)
        torch.testing.assert_close(layer2.forward(xx), layer2._forward_method(xx))

    def test_dispatch_forward_platform_branches(self):
        seams = implementation._host_seams
        cc = _vllm_compile_cc()
        cc.custom_ops = ["all"]
        layer = RMSNorm.__new__(RMSNorm)
        layer._enforce_enable = True
        with _cfg_ctx(cc), \
             patch.object(seams.current_platform, "is_cpu", return_value=True), \
             patch.object(seams.current_platform, "is_rocm", return_value=False), \
             patch.object(seams.current_platform, "is_tpu", return_value=False), \
             patch.object(seams.current_platform, "is_xpu", return_value=False), \
             patch.object(seams.current_platform, "is_out_of_tree", return_value=False):
            # bound methods: == (each attribute access builds a new object)
            assert layer.dispatch_forward(compile_native=False) == layer.forward_cpu
        with _cfg_ctx(cc), \
             patch.object(seams.current_platform, "is_cpu", return_value=False), \
             patch.object(seams.current_platform, "is_rocm", return_value=True):
            assert layer.dispatch_forward(compile_native=False) == layer.forward_hip
        # default fall-through: CUDA
        with _cfg_ctx(cc), \
             patch.object(seams.current_platform, "is_cpu", return_value=False), \
             patch.object(seams.current_platform, "is_rocm", return_value=False), \
             patch.object(seams.current_platform, "is_tpu", return_value=False), \
             patch.object(seams.current_platform, "is_xpu", return_value=False), \
             patch.object(seams.current_platform, "is_out_of_tree", return_value=False):
            assert layer.dispatch_forward(compile_native=False) == layer.forward_cuda

    def test_maybe_compile_no_compile_gates(self):
        cc = _vllm_compile_cc()  # backend defaults to inductor, mode VLLM_COMPILE
        layer = RMSNorm.__new__(RMSNorm)
        layer._enforce_enable = False
        # mode NONE → identity
        cc.mode = CompilationMode.NONE
        with _cfg_ctx(cc):
            fn = layer.maybe_compile(RMSNorm.forward_native)
        assert fn is RMSNorm.forward_native
        # eager backend → identity
        cc.mode = CompilationMode.VLLM_COMPILE
        cc.backend = "eager"
        with _cfg_ctx(cc):
            assert layer.maybe_compile(RMSNorm.forward_native) is RMSNorm.forward_native
        # enable=False → identity
        cc.backend = "inductor"
        with _cfg_ctx(cc):
            assert (
                layer.maybe_compile(RMSNorm.forward_native, enable=False)
                is RMSNorm.forward_native
            )


# ---------------------------------------------------------------------------
# m01 — RMSNorm as the worked CustomOp instance (layernorm.py:L36-L127)
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_forward_native_matches_ir_reference_math(self):
        torch.manual_seed(0)
        with _cfg_ctx():
            layer = RMSNorm(hidden_size=16, eps=1e-6)
        x = torch.randn(4, 16)
        out = layer.forward_native(x)
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * layer.weight
        torch.testing.assert_close(out, ref)

    def test_forward_native_fused_add_variant(self):
        torch.manual_seed(1)
        with _cfg_ctx():
            layer = RMSNorm(hidden_size=16, eps=1e-6)
        x = torch.randn(4, 16)
        residual = torch.randn(4, 16)
        out, resid_out = layer.forward_native(x, residual)
        x32 = x.float() + residual.float()
        ref_resid = x32.to(x.dtype)
        var = x32.pow(2).mean(-1, keepdim=True)
        ref_out = (x32 * torch.rsqrt(var + 1e-6)).to(layer.weight.dtype) * layer.weight
        torch.testing.assert_close(resid_out, ref_resid)
        torch.testing.assert_close(out, ref_out)

    def test_forward_cuda_falls_back_to_native_by_default(self):
        with _cfg_ctx():
            layer = RMSNorm(hidden_size=8)
        x = torch.randn(2, 8)
        torch.testing.assert_close(layer.forward_cuda(x), layer.forward_native(x))

    def test_forward_xpu_forwards_to_cuda(self):
        with _cfg_ctx():
            layer = RMSNorm(hidden_size=8)
        x = torch.randn(2, 8)
        torch.testing.assert_close(layer.forward_xpu(x), layer.forward_cuda(x))

    def test_weightless_variant_passes_none(self):
        with _cfg_ctx():
            layer = RMSNorm(hidden_size=8, has_weight=False)
        assert layer.pass_weight is False
        x = torch.randn(2, 8)
        out = layer.forward_native(x)
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(out, ref)

    def test_variance_size_override(self):
        with _cfg_ctx():
            layer = RMSNorm(hidden_size=8, var_hidden_size=4)
        assert layer.variance_size_override == 4
        x = torch.randn(2, 8)
        out = layer.forward_native(x)
        var = x[..., :4].pow(2).mean(-1, keepdim=True)
        ref = x * torch.rsqrt(var + 1e-6) * layer.weight
        torch.testing.assert_close(out, ref)


# ---------------------------------------------------------------------------
# m04 — forward context (forward_context.py, minus observation tail)
# ---------------------------------------------------------------------------


class TestForwardContext:
    def teardown_method(self):
        # make sure a failed assertion cannot leak a context into other tests
        implementation.forward_context._forward_context = None

    def test_get_without_context_asserts(self):
        assert not is_forward_context_available()
        with pytest.raises(AssertionError, match="Forward context is not set"):
            get_forward_context()

    def test_set_forward_context_scopes_thread_local(self):
        cfg = make_seam_vllm_config(CompilationConfig())
        with set_forward_context(
            attn_metadata={"layer": "md"},
            vllm_config=cfg,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(num_tokens=3),
            slot_mapping={"layer": torch.zeros(1, dtype=torch.int64)},
        ):
            ctx = get_forward_context()
            assert ctx.attn_metadata == {"layer": "md"}
            assert ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL
            assert ctx.batch_descriptor == BatchDescriptor(num_tokens=3)
            assert ctx.slot_mapping == {"layer": torch.zeros(1, dtype=torch.int64)}
            assert ctx.no_compile_layers is cfg.compilation_config.static_forward_context
        assert not is_forward_context_available()

    def test_batch_descriptor_convenience_creation(self):
        cfg = make_seam_vllm_config(CompilationConfig())
        with set_forward_context(
            attn_metadata=None,
            vllm_config=cfg,
            num_tokens=5,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        ):
            assert get_forward_context().batch_descriptor == BatchDescriptor(
                num_tokens=5
            )
        # mode NONE → no descriptor invented
        with set_forward_context(attn_metadata=None, vllm_config=cfg, num_tokens=5):
            assert get_forward_context().batch_descriptor is None

    def test_invalid_runtime_mode_rejected(self):
        with pytest.raises(AssertionError, match="Invalid cudagraph runtime mode"):
            ForwardContext(
                no_compile_layers={},
                attn_metadata=None,
                slot_mapping={},
                cudagraph_runtime_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            )

    def test_override_forward_context_restores(self):
        outer = ForwardContext(no_compile_layers={}, attn_metadata="outer", slot_mapping={})
        inner = ForwardContext(no_compile_layers={}, attn_metadata="inner", slot_mapping={})
        with override_forward_context(outer):
            assert get_forward_context() is outer
            with override_forward_context(inner):
                assert get_forward_context() is inner
            assert get_forward_context() is outer
        assert not is_forward_context_available()

    def test_dp1_leaves_dp_metadata_none(self):
        cfg = make_seam_vllm_config(CompilationConfig())
        with set_forward_context(attn_metadata=None, vllm_config=cfg, num_tokens=2):
            assert get_forward_context().dp_metadata is None


# ---------------------------------------------------------------------------
# m04/m05/m06 — Attention registration + the unified op trio + LayerName
# ---------------------------------------------------------------------------


class _FakeImpl:
    """Test double for the ch21-domain AttentionImpl (recorded calls)."""

    supports_quant_query_input = False
    forward_includes_kv_cache_update = False  # backend.py:L67 real default True; FA flips

    def __init__(self):
        self.kv_updates = []
        self.forwards = []

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        self.kv_updates.append((key.shape, value.shape, kv_cache, slot_mapping))

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, *, output,
                output_scale=None, output_block_scale=None):
        self.forwards.append((query.shape, output.shape, attn_metadata))
        # deterministic observable write into the pre-allocated output
        output.copy_(query.sum(dim=-1, keepdim=True).expand_as(output))


def _make_attention(prefix: str, cc: CompilationConfig, impl=None):
    cfg = make_seam_vllm_config(cc)
    with set_current_vllm_config(cfg):
        attn = Attention(
            num_heads=2,
            head_size=4,
            scale=0.25,
            num_kv_heads=2,
            prefix=prefix,
            vllm_config=cfg,
        )
    attn.impl = impl or _FakeImpl()
    attn.attn_backend = types.SimpleNamespace(
        forward_includes_kv_cache_update=attn.impl.forward_includes_kv_cache_update
    )
    return attn, cfg


class TestAttentionLayer:
    def test_constructor_self_registers_into_static_forward_context(self):
        cc = _vllm_compile_cc()
        attn, _ = _make_attention("model.layers.0.self_attn.attn", cc)
        assert cc.static_forward_context["model.layers.0.self_attn.attn"] is attn

    def test_duplicate_layer_name_rejected(self):
        cc = _vllm_compile_cc()
        _make_attention("model.layers.0.self_attn.attn", cc)
        with pytest.raises(ValueError, match="Duplicate layer name"):
            _make_attention("model.layers.0.self_attn.attn", cc)

    def test_use_direct_call_on_non_opaque_platform(self):
        cc = _vllm_compile_cc()
        attn, _ = _make_attention("m.l0", cc)
        # host seam platform is not opaque-attention → direct python calls
        assert attn.use_direct_call is (
            not implementation._host_seams.current_platform.opaque_attention_op()
        )

    def test_forward_out_variant_and_kv_update_order(self):
        cc = _vllm_compile_cc()
        impl = _FakeImpl()
        attn, cfg = _make_attention("m.l0", cc, impl)
        slot = torch.arange(4, dtype=torch.int64)
        with set_forward_context(
            attn_metadata={"m.l0": "MD"},
            vllm_config=cfg,
            slot_mapping={"m.l0": slot},
        ):
            q = torch.randn(4, 8)
            k = torch.randn(4, 8)
            v = torch.randn(4, 8)
            out = attn.forward(q, k, v)
        assert out.shape == (4, 8)
        torch.testing.assert_close(out, q.view(4, 2, 4).sum(-1, keepdim=True)
                                   .expand(4, 2, 4).reshape(4, 8))
        # kv update happened before attention (call order)
        assert len(impl.kv_updates) == 1
        assert impl.kv_updates[0][2] is attn.kv_cache
        assert impl.kv_updates[0][3] is slot
        assert len(impl.forwards) == 1
        assert impl.forwards[0][2] == "MD"

    def test_get_attention_context_branches(self):
        cc = _vllm_compile_cc()
        attn, cfg = _make_attention("m.l0", cc)
        slot = torch.zeros(2, dtype=torch.int64)
        md = {"m.l0": "MD0"}
        with set_forward_context(
            attn_metadata=md, vllm_config=cfg, slot_mapping={"m.l0": slot}
        ):
            got_md, got_layer, kv, got_slot = get_attention_context("m.l0")
            assert got_md == "MD0"
            assert got_layer is attn
            assert kv is attn.kv_cache
            assert got_slot is slot
            # non-dict (single) metadata passes through
            with override_forward_context(
                replace(get_forward_context(), attn_metadata="SINGLE")
            ):
                assert get_attention_context("m.l0")[0] == "SINGLE"
            # DBO list form: [0] is the base-model dict
            with override_forward_context(
                replace(get_forward_context(), attn_metadata=[{"m.l0": "BASE"}])
            ):
                assert get_attention_context("m.l0")[0] == "BASE"

    def test_get_attention_context_requires_dict_slot_mapping(self):
        cc = _vllm_compile_cc()
        attn, cfg = _make_attention("m.l0", cc)
        with set_forward_context(
            attn_metadata={"m.l0": "MD"}, vllm_config=cfg, slot_mapping=[{"m.l0": "x"}]
        ):
            with pytest.raises(AssertionError, match="Expected slot_mapping to be a dict"):
                get_attention_context("m.l0")

    def test_unified_kv_cache_update_returns_dummy_dep(self):
        cc = _vllm_compile_cc()
        impl = _FakeImpl()
        attn, cfg = _make_attention("m.l0", cc, impl)
        slot = torch.arange(4, dtype=torch.int64)
        k = torch.randn(4, 8)
        v = torch.randn(4, 8)
        with set_forward_context(
            attn_metadata={"m.l0": "MD"}, vllm_config=cfg, slot_mapping={"m.l0": slot}
        ):
            dummy = unified_kv_cache_update(k, v, "m.l0")
        # the dummy signals the data dependency: empty tensor, key dtype/device
        assert dummy.numel() == 0
        assert dummy.dtype == k.dtype
        assert len(impl.kv_updates) == 1
        # layer without slot mapping: no update, still returns dummy
        with set_forward_context(
            attn_metadata={"m.l0": "MD"}, vllm_config=cfg, slot_mapping={"other": slot}
        ):
            dummy2 = unified_kv_cache_update(k, v, "m.l0")
        assert dummy2.numel() == 0
        assert len(impl.kv_updates) == 1
        # impl without do_kv_cache_update → pinned assertion
        with set_forward_context(
            attn_metadata={"m.l0": "MD"}, vllm_config=cfg, slot_mapping={"m.l0": slot}
        ):
            attn.impl = types.SimpleNamespace()  # no do_kv_cache_update
            with pytest.raises(AssertionError, match="does not support kv cache update"):
                unified_kv_cache_update(k, v, "m.l0")

    def test_unified_attention_with_output_calls_impl(self):
        cc = _vllm_compile_cc()
        impl = _FakeImpl()
        attn, cfg = _make_attention("m.l0", cc, impl)
        slot = torch.arange(4, dtype=torch.int64)
        q = torch.randn(4, 2, 4)
        out = torch.empty(4, 2, 4)
        dummy = torch.empty(0)
        with set_forward_context(
            attn_metadata={"m.l0": "MD"},
            vllm_config=cfg,
            slot_mapping={"m.l0": slot},
        ):
            unified_attention_with_output(q, None, None, out, "m.l0",
                                          kv_cache_dummy_dep=dummy)
        assert len(impl.forwards) == 1
        assert impl.forwards[0][1] == out.shape
        assert not torch.equal(out, torch.zeros_like(out))  # impl wrote into output

    def test_ops_registered_into_torch_vllm_namespace(self):
        # direct_register_custom_op registered the trio (fake impls make them
        # traceable); calling the python functions is the direct-call path.
        assert hasattr(torch.ops.vllm, "unified_attention_with_output")
        assert hasattr(torch.ops.vllm, "unified_kv_cache_update")


class TestLayerName:
    def test_encode_roundtrip(self):
        # on torch < 2.11 encode is the identity (str); on >= 2.11 it wraps
        enc = _encode_layer_name("m.l0")
        assert _resolve_layer_name(enc) == "m.l0"

    def test_layername_equality_and_hash(self):
        a, b = LayerName("x"), LayerName("x")
        assert a == b
        assert hash(a) == hash(b)
        assert a != LayerName("y")
        assert a != "x"

    def test_resolve_plain_str_unchanged(self):
        assert _resolve_layer_name("m.l0") == "m.l0"


# ---------------------------------------------------------------------------
# m08 — should_split + split_graph (partition_rules.py / backends.py:L553-L627)
# ---------------------------------------------------------------------------


def _attention_op(q, k, v, out, layer):
    # resolved through torch.ops so the FX target is an OpOverload, like the
    # dynamo-captured graph in real vLLM
    return torch.ops.vllm.unified_attention_with_output(
        q, k, v, out, layer, kv_cache_dummy_dep=None
    )


class _AttnFn(torch.nn.Module):
    """toy 'model': linear → attention op → linear, CPU tensors."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 4, bias=False)
        self.fc2 = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x).view(-1, 1, 4)
        out = torch.empty_like(h)
        _attention_op(h, None, None, out, "toy.attn")
        y = out.view(-1, 4)
        return self.fc2(y)


def _trace(module) -> torch.fx.GraphModule:
    gm = torch.fx.symbolic_trace(module)
    # symbolic_trace does not populate example_value meta; split_graph itself
    # does not need it (only the interpreter does — dynamo provides it there).
    return gm


class TestSplitGraph:
    def test_should_split_matches_qualified_names(self):
        gm = _trace(_AttnFn())
        # symbolic_trace resolves torch.ops.vllm.unified_attention_with_output
        # through the OpOverload(PACKET) machinery — filter with should_split's
        # own qualified-name logic (packet `name`/overload `_overloadname`)
        attn_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and should_split(n, ["vllm::unified_attention_with_output"])
        ]
        assert len(attn_nodes) == 1
        node = attn_nodes[0]
        assert should_split(node, ["vllm::unified_attention_with_output"])
        assert not should_split(node, [])
        assert not should_split(node, ["vllm::something_else"])
        # non-call_function nodes never split
        for ph in [n for n in gm.graph.nodes if n.op == "placeholder"]:
            assert not should_split(ph, ["vllm::unified_attention_with_output"])

    def test_split_graph_three_pieces(self):
        gm = _trace(_AttnFn())
        split_ops = ["vllm::unified_attention_with_output"]
        split_gm, items = split_graph(gm, split_ops)
        assert [it.graph_id for it in items] == [0, 1, 2]
        flags = {it.graph_id: it.is_splitting_graph for it in items}
        assert flags == {0: False, 1: True, 2: False}
        assert items[1].submod_name == "submod_1"

    def test_split_graph_empty_splitting_ops_no_split(self):
        gm = _trace(_AttnFn())
        split_gm, items = split_graph(gm, [])
        assert len(items) == 1
        assert not items[0].is_splitting_graph

    def test_split_pieces_run_in_order_on_cpu(self):
        cc = _vllm_compile_cc()
        attn, cfg = _make_attention("toy.attn", cc)
        attn.impl = _FakeImpl()
        attn.attn_backend = types.SimpleNamespace(
            forward_includes_kv_cache_update=True  # kv update folded into forward
        )
        gm = _trace(_AttnFn())
        split_gm, items = split_graph(
            gm, ["vllm::unified_attention_with_output"]
        )
        x = torch.randn(3, 4)
        slot = torch.zeros(3, dtype=torch.int64)
        with set_forward_context(
            attn_metadata={"toy.attn": "MD"},
            vllm_config=cfg,
            slot_mapping={"toy.attn": slot},
        ):
            out = split_gm(x)
        ref = _AttnFn()
        ref.load_state_dict(gm.state_dict())  # same weights as traced module
        with set_forward_context(
            attn_metadata={"toy.attn": "MD"},
            vllm_config=cfg,
            slot_mapping={"toy.attn": slot},
        ):
            expected = ref(x)
        torch.testing.assert_close(out, expected)

    def test_wrap_with_cudagraph_if_needed(self):
        cc = _vllm_compile_cc(cudagraph_mode=CUDAGraphMode.PIECEWISE)
        cfg = make_seam_vllm_config(cc)
        wrapped = wrap_with_cudagraph_if_needed(
            lambda *a: None, cfg, cc, is_first_graph=True, is_last_graph=False
        )
        assert isinstance(wrapped, CUDAGraphWrapper)
        assert wrapped.runtime_mode == CUDAGraphMode.PIECEWISE  # always PIECEWISE
        assert wrapped.cudagraph_options.debug_log_enable is True
        assert wrapped.cudagraph_options.gc_disable is False  # first graph: gc runs
        assert wrapped.cudagraph_options.weak_ref_output is False
        # no piecewise cudagraphs → unwrapped passthrough
        cc_none = _vllm_compile_cc(cudagraph_mode=CUDAGraphMode.NONE)
        cfg_none = make_seam_vllm_config(cc_none)
        fn = lambda *a: None  # noqa: E731
        assert (
            wrap_with_cudagraph_if_needed(fn, cfg_none, cc_none, True, True) is fn
        )
        # inductor-partition route does not wrap here either
        cc_ip = _vllm_compile_cc(
            cudagraph_mode=CUDAGraphMode.PIECEWISE, use_inductor_graph_partition=True
        )
        cfg_ip = make_seam_vllm_config(cc_ip)
        assert wrap_with_cudagraph_if_needed(fn, cfg_ip, cc_ip, True, True) is fn


# ---------------------------------------------------------------------------
# m12 — CudagraphDispatcher (v1/cudagraph_dispatcher.py; upstream test mirror)
# ---------------------------------------------------------------------------


class TestCudagraphDispatcher:
    @pytest.mark.parametrize(
        "cudagraph_mode_str,compilation_mode",
        [
            ("FULL", CompilationMode.NONE),
            ("FULL_AND_PIECEWISE", CompilationMode.NONE),
            ("FULL_DECODE_ONLY", CompilationMode.NONE),
            ("PIECEWISE", CompilationMode.VLLM_COMPILE),
        ],
    )
    def test_dispatcher_key_init_and_lookup(
        self, cudagraph_mode_str, compilation_mode
    ):
        comp_config = CompilationConfig(
            cudagraph_mode=cudagraph_mode_str,
            mode=compilation_mode,
            cudagraph_capture_sizes=[1, 8],
        )
        config = make_seam_vllm_config(comp_config, max_num_seqs=8)
        # upstream guard: PIECEWISE mode without VLLM_COMPILE is rejected at
        # dispatcher construction (mirrors upstream test_dispatcher's early
        # return-with-AssertionError for this combo)
        if (
            cudagraph_mode_str == "FULL_AND_PIECEWISE"
            and compilation_mode == CompilationMode.NONE
        ):
            with pytest.raises(AssertionError):
                CudagraphDispatcher(config)
            return
        dispatcher = CudagraphDispatcher(config)
        dispatcher.initialize_cudagraph_keys(
            cudagraph_mode=comp_config.cudagraph_mode, uniform_decode_query_len=1
        )

        if cudagraph_mode_str in ["FULL_AND_PIECEWISE", "PIECEWISE"]:
            assert len(dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]) == 2
        else:
            assert len(dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]) == 0
        if cudagraph_mode_str not in ["NONE", "PIECEWISE"]:
            assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == 2
        else:
            assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == 0

        # 1. non-uniform batch, size in capture list
        desc_full_with_reqs = BatchDescriptor(num_tokens=8, num_reqs=8, uniform=False)
        desc_piecewise = BatchDescriptor(num_tokens=8, num_reqs=None, uniform=False)
        rt_mode, key = dispatcher.dispatch(num_tokens=8, uniform_decode=False)
        if cudagraph_mode_str == "FULL":
            assert rt_mode == CUDAGraphMode.FULL
            assert key == desc_full_with_reqs
        elif cudagraph_mode_str in ["FULL_AND_PIECEWISE", "PIECEWISE"]:
            assert rt_mode == CUDAGraphMode.PIECEWISE
            assert key == desc_piecewise
        else:
            assert rt_mode == CUDAGraphMode.NONE

        # 2. uniform decode batch
        desc_uniform_exact = BatchDescriptor(num_tokens=8, num_reqs=8, uniform=True)
        rt_mode, key = dispatcher.dispatch(num_tokens=8, uniform_decode=True)
        if cudagraph_mode_str == "FULL":
            assert rt_mode == CUDAGraphMode.FULL
            assert key == desc_full_with_reqs
        elif cudagraph_mode_str in ["FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"]:
            assert rt_mode == CUDAGraphMode.FULL
            assert key == desc_uniform_exact
        elif cudagraph_mode_str == "PIECEWISE":
            assert rt_mode == CUDAGraphMode.PIECEWISE
            assert key == replace(desc_uniform_exact, num_reqs=None, uniform=False)
        else:
            assert rt_mode == CUDAGraphMode.NONE

        # 3. no key match (out of capture range) → NONE with raw descriptor
        rt_mode, key = dispatcher.dispatch(num_tokens=15)
        assert rt_mode == CUDAGraphMode.NONE
        assert key == BatchDescriptor(num_tokens=15)

        # 4. invalid_modes={FULL} falls back (cascade-attention shape)
        rt_mode, key = dispatcher.dispatch(num_tokens=8, invalid_modes={CUDAGraphMode.FULL})
        if "PIECEWISE" in cudagraph_mode_str:
            assert rt_mode == CUDAGraphMode.PIECEWISE
            assert key == replace(BatchDescriptor(num_tokens=8), num_reqs=None,
                                  uniform=False)
        else:
            assert rt_mode == CUDAGraphMode.NONE

        # 5. valid_modes={NONE} always NONE even when keys exist
        rt_mode, key = dispatcher.dispatch(num_tokens=8, valid_modes={CUDAGraphMode.NONE})
        assert rt_mode == CUDAGraphMode.NONE
        assert key == BatchDescriptor(num_tokens=8)

    def test_piecewise_requires_vllm_compile(self):
        comp_config = CompilationConfig(
            cudagraph_mode="FULL_AND_PIECEWISE",
            mode=CompilationMode.NONE,  # no piecewise compilation at all
        )
        config = make_seam_vllm_config(comp_config)
        with pytest.raises(AssertionError):
            CudagraphDispatcher(config)

    def test_bs_to_padded_size_table(self):
        comp_config = CompilationConfig(
            cudagraph_mode="FULL",
            mode=CompilationMode.NONE,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        )
        config = make_seam_vllm_config(comp_config)
        dispatcher = CudagraphDispatcher(config)
        dispatcher.cudagraph_mode = CUDAGraphMode.FULL
        dispatcher._compute_bs_to_padded_graph_size()
        table = dispatcher._bs_to_padded_graph_size
        # segment heads keep their shape; interior rounds up to segment end
        assert table[1] == 1 and table[2] == 2 and table[4] == 4 and table[8] == 8
        assert table[3] == 4 and table[9] == 16 and table[15] == 16 and table[16] == 16

    def test_compile_sizes_must_not_be_padded(self):
        comp_config = CompilationConfig(
            cudagraph_mode="FULL",
            mode=CompilationMode.NONE,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
            compile_sizes=[9],  # would be padded to 16
        )
        config = make_seam_vllm_config(comp_config)
        dispatcher = CudagraphDispatcher(config)
        dispatcher.cudagraph_mode = CUDAGraphMode.FULL
        with pytest.raises(ValueError, match="would be padded to 16"):
            dispatcher._compute_bs_to_padded_graph_size()

    def test_padded_batch_descriptor_uniform_full(self):
        comp_config = CompilationConfig(
            cudagraph_mode="FULL",
            mode=CompilationMode.NONE,
            cudagraph_capture_sizes=[1, 8],
        )
        config = make_seam_vllm_config(comp_config, max_num_seqs=8)
        dispatcher = CudagraphDispatcher(config)
        dispatcher.cudagraph_mode = CUDAGraphMode.FULL
        dispatcher._compute_bs_to_padded_graph_size()
        desc = dispatcher._create_padded_batch_descriptor(5, True, False)
        assert desc == BatchDescriptor(num_tokens=8, num_reqs=8, uniform=True)
        desc = dispatcher._create_padded_batch_descriptor(5, False, False)
        assert desc == BatchDescriptor(num_tokens=8, num_reqs=8, uniform=False)

    @pytest.mark.parametrize(
        "cudagraph_mode_str,compilation_mode,expected_modes",
        [
            ("FULL", CompilationMode.NONE, [CUDAGraphMode.FULL]),
            ("PIECEWISE", CompilationMode.VLLM_COMPILE, [CUDAGraphMode.PIECEWISE]),
            ("FULL_DECODE_ONLY", CompilationMode.NONE, [CUDAGraphMode.FULL]),
            ("NONE", CompilationMode.NONE, []),
        ],
    )
    def test_get_capture_descs_ordering(
        self, cudagraph_mode_str, compilation_mode, expected_modes
    ):
        comp_config = CompilationConfig(
            cudagraph_mode=cudagraph_mode_str,
            mode=compilation_mode,
            cudagraph_capture_sizes=[1, 4, 8, 16],
        )
        config = make_seam_vllm_config(comp_config, max_num_seqs=16)
        dispatcher = CudagraphDispatcher(config)
        dispatcher.initialize_cudagraph_keys(
            cudagraph_mode=comp_config.cudagraph_mode, uniform_decode_query_len=1
        )
        capture_descs = dispatcher.get_capture_descs()
        assert [mode for mode, _ in capture_descs] == expected_modes
        for mode, descs in capture_descs:
            tokens = [d.num_tokens for d in descs]
            assert tokens == sorted(tokens, reverse=True)
            assert len({d.uniform for d in descs}) == 1

    def test_get_capture_descs_empty_when_not_initialized(self):
        comp_config = CompilationConfig(
            cudagraph_mode="FULL",
            mode=CompilationMode.NONE,
            cudagraph_capture_sizes=[1, 8],
        )
        dispatcher = CudagraphDispatcher(make_seam_vllm_config(comp_config))
        assert dispatcher.get_capture_descs() == []

    def test_none_mode_initializes_to_no_keys(self):
        comp_config = CompilationConfig(cudagraph_mode="NONE")
        dispatcher = CudagraphDispatcher(make_seam_vllm_config(comp_config))
        dispatcher.initialize_cudagraph_keys(CUDAGraphMode.NONE)
        assert dispatcher.keys_initialized
        assert dispatcher.get_capture_descs() == []
        mode, key = dispatcher.dispatch(num_tokens=8)
        assert mode == CUDAGraphMode.NONE and key.num_tokens == 8


# ---------------------------------------------------------------------------
# m14 — CUDAGraphWrapper dispatch head (compilation/cuda_graph.py:L233-L261)
# ---------------------------------------------------------------------------


class TestCUDAGraphWrapperHead:
    def _wrapper(self):
        cc = CompilationConfig()
        cfg = make_seam_vllm_config(cc)
        return CUDAGraphWrapper(
            lambda *a, **k: ("ran", a), cfg, runtime_mode=CUDAGraphMode.FULL
        )

    def test_none_runtime_mode_rejected(self):
        cc = CompilationConfig()
        cfg = make_seam_vllm_config(cc)
        with pytest.raises(AssertionError):
            CUDAGraphWrapper(lambda: None, cfg, runtime_mode=CUDAGraphMode.NONE)

    def test_no_forward_context_passes_through(self):
        w = self._wrapper()
        out = w(torch.ones(2))
        assert out[0] == "ran"
        assert isinstance(out[1], tuple) and out[1][0].shape == (2,)
        assert not w.concrete_cudagraph_entries

    def test_mode_mismatch_passes_through(self):
        w = self._wrapper()  # FULL wrapper
        cfg = make_seam_vllm_config(CompilationConfig())
        called = []
        w.runnable = lambda *a, **k: called.append(1)
        with set_forward_context(
            attn_metadata=None,
            vllm_config=cfg,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,  # not ours
            batch_descriptor=BatchDescriptor(num_tokens=1),
        ):
            w(torch.ones(1))
        assert called == [1]
        assert not w.concrete_cudagraph_entries
        with set_forward_context(
            attn_metadata=None,
            vllm_config=cfg,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=1),
        ):
            w(torch.ones(1))
        assert called == [1, 1]

    def test_getattr_forwards_to_runnable_and_raises(self):
        class R:
            marker = 7

            def __call__(self, *a):
                return None

        cc = CompilationConfig()
        cfg = make_seam_vllm_config(cc)
        w = CUDAGraphWrapper(R(), cfg, runtime_mode=CUDAGraphMode.FULL)
        assert w.marker == 7
        assert w.unwrap() is w.runnable
        assert w.cudagraph_wrapper is w
        with pytest.raises(AttributeError):
            _ = w.missing_attr

    def test_capturing_window_tripwire(self):
        set_cudagraph_capturing_enabled(False)
        try:
            with pytest.raises(RuntimeError, match="inappropriate"):
                validate_cudagraph_capturing_enabled()
        finally:
            set_cudagraph_capturing_enabled(True)
        validate_cudagraph_capturing_enabled()  # re-enabled: no raise

    def test_debug_replay_address_assert(self):
        w = self._wrapper()
        entry = CUDAGraphEntry(batch_descriptor=BatchDescriptor(num_tokens=2))
        entry.cudagraph = MagicMock()
        entry.input_addresses = [123]
        w.concrete_cudagraph_entries[BatchDescriptor(num_tokens=2)] = entry
        w.is_debugging_mode = True
        cfg = make_seam_vllm_config(CompilationConfig())
        t = torch.ones(2)
        t.data_ptr = lambda: 123  # type: ignore[method-assign]
        with set_forward_context(
            attn_metadata=None,
            vllm_config=cfg,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(num_tokens=2),
        ):
            out = w(t)
        entry.cudagraph.replay.assert_called_once()
        other = torch.ones(2)
        other.data_ptr = lambda: 999  # type: ignore[method-assign]
        with set_forward_context(
            attn_metadata=None,
            vllm_config=cfg,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(num_tokens=2),
        ):
            with pytest.raises(
                AssertionError, match="Input addresses for cudagraphs are different"
            ):
                w(other)

    def test_clear_all_graphs(self):
        w = self._wrapper()
        w.concrete_cudagraph_entries[BatchDescriptor(num_tokens=1)] = CUDAGraphEntry(
            batch_descriptor=BatchDescriptor(num_tokens=1)
        )
        CUDAGraphWrapper.clear_all_graphs()
        assert not w.concrete_cudagraph_entries


@pytest.mark.cuda
class TestCUDAGraphWrapperCaptureReplay:
    """Upstream TestCUDAGraphWrapper mirror — needs a real CUDA device."""

    def _setup(self):
        cc = CompilationConfig()
        cfg = make_seam_vllm_config(cc)
        model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 10)).cuda()
        wrapper = CUDAGraphWrapper(model, cfg, runtime_mode=CUDAGraphMode.FULL)
        return wrapper, cfg, model

    def test_capture_and_replay(self):
        wrapper, cfg, model = self._setup()
        batch_descriptor = BatchDescriptor(num_tokens=10)
        persistent = torch.zeros(1, 10, device="cuda")
        input_tensor = persistent.clone().normal_()

        with set_forward_context(
            attn_metadata=None,
            vllm_config=cfg,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=None,
        ):
            wrapper(input_tensor)

        with (
            set_forward_context(
                attn_metadata=None,
                vllm_config=cfg,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=batch_descriptor,
            ),
            patch("torch.cuda.graph", wraps=torch.cuda.graph) as mock_graph,
        ):
            output1 = wrapper(input_tensor)
            assert torch.allclose(output1, torch.zeros_like(output1))
            mock_graph.assert_called_once()
        assert batch_descriptor in wrapper.concrete_cudagraph_entries
        entry = wrapper.concrete_cudagraph_entries[batch_descriptor]
        assert entry.cudagraph is not None

        with (
            set_forward_context(
                attn_metadata=None,
                vllm_config=cfg,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=batch_descriptor,
            ),
            patch.object(entry.cudagraph, "replay", wraps=entry.cudagraph.replay),
        ):
            output2 = wrapper(input_tensor)
        torch.testing.assert_close(model(input_tensor), output2)

    def test_capture_blocked_after_window_closes(self):
        wrapper, cfg, model = self._setup()
        input_tensor = torch.zeros(1, 10, device="cuda")
        set_cudagraph_capturing_enabled(False)
        try:
            with set_forward_context(
                attn_metadata=None,
                vllm_config=cfg,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=BatchDescriptor(num_tokens=1),
            ):
                with pytest.raises(RuntimeError, match="inappropriate"):
                    wrapper(input_tensor)
        finally:
            set_cudagraph_capturing_enabled(True)


# ---------------------------------------------------------------------------
# m10 — TorchCompileWithNoGuardsWrapper (compilation/wrapper.py)
# ---------------------------------------------------------------------------


class TestNoGuardsWrapper:
    def test_mode_none_rejected(self):
        cc = CompilationConfig()
        cfg = make_seam_vllm_config(cc)
        cc.mode = None  # simulate un-set mode reaching the wrapper
        with patch.object(
            implementation.compilation.wrapper, "get_current_vllm_config", return_value=cfg
        ):
            with pytest.raises(RuntimeError, match="mode cannot be NO_COMPILATION"):
                class _W(TorchCompileWithNoGuardsWrapper):
                    def forward(self, *a, **k):
                        return None

                _W()

    def test_eager_backend_compiles_and_runs_once(self):
        cc = _vllm_compile_cc(backend="eager", compile_ranges_endpoints=[8])
        cfg = make_seam_vllm_config(cc)

        class _Model(TorchCompileWithNoGuardsWrapper):
            def forward(self, x):
                return x * 2

        with patch.object(
            implementation.compilation.wrapper,
            "get_current_vllm_config",
            return_value=cfg,
        ):
            m = _Model()
        assert m.first_compile
        x = torch.randn(2, 3)
        assert torch.allclose(m(x), x * 2)
        assert not m.first_compile
        # guards dropped: a different shape does not retrace through the
        # wrapper's compilation context (no recompile assertion machinery)
        y = torch.randn(5, 3)
        assert torch.allclose(m(y), y * 2)


# ---------------------------------------------------------------------------
# m07/m09 — VllmBackend end-to-end on dynamo + eager adaptor (host CPU)
# ---------------------------------------------------------------------------


class TestVllmBackendE2E:
    def _compile_toy(self, cudagraph_mode=CUDAGraphMode.NONE):
        cc = _vllm_compile_cc(
            backend="eager",
            cudagraph_mode=cudagraph_mode,
            compile_ranges_endpoints=[16],
        )
        cfg = make_seam_vllm_config(cc)
        attn, _ = _make_attention("toy.attn", cc)
        attn.attn_backend = types.SimpleNamespace(
            forward_includes_kv_cache_update=True
        )
        model = _AttnFn()
        backend = VllmBackend(cfg)
        compiled = torch.compile(
            model, fullgraph=True, dynamic=False, backend=backend
        )
        return compiled, backend, cfg, attn, cc

    def test_split_and_piecewise_interpreter_and_stitched_callable(self):
        compiled, backend, cfg, attn, cc = self._compile_toy()
        x = torch.randn(3, 4)
        slot = torch.zeros(3, dtype=torch.int64)
        with set_forward_context(
            attn_metadata={"toy.attn": "MD"},
            vllm_config=cfg,
            slot_mapping={"toy.attn": slot},
        ):
            out = compiled(x)
        assert backend._called
        # graph was split at the attention op into pre/split/post pieces
        names = [n for n, _ in backend.split_gm.named_children()]
        assert len(names) == 3
        assert all(hasattr(backend.split_gm, n) for n in names)
        # the interpreter replaced compiled pieces; splitting piece stays eager
        compiled_names = [
            it.submod_name
            for it in backend.piecewise_graphs
            if not it.is_splitting_graph
        ]
        for n in compiled_names:
            attr = backend.split_gm.__dict__[n]
            assert not isinstance(attr, torch.fx.GraphModule)
        # second call replays the stitched callable without a new dynamo trace
        with set_forward_context(
            attn_metadata={"toy.attn": "MD"},
            vllm_config=cfg,
            slot_mapping={"toy.attn": slot},
        ):
            out2 = compiled(x)
        torch.testing.assert_close(out, out2)
        assert compilation_counter.num_graphs_seen >= 1

    def test_backend_can_only_be_called_once(self):
        compiled, backend, cfg, attn, cc = self._compile_toy()
        x = torch.randn(3, 4)
        with set_forward_context(
            attn_metadata={"toy.attn": "MD"},
            vllm_config=cfg,
            slot_mapping={"toy.attn": torch.zeros(3, dtype=torch.int64)},
        ):
            compiled(x)
        with pytest.raises(AssertionError, match="can only be called once"):
            backend(MagicMock(), [])

    def test_piecewise_mode_wraps_pieces(self):
        compiled, backend, cfg, attn, cc = self._compile_toy(
            cudagraph_mode=CUDAGraphMode.PIECEWISE
        )
        x = torch.randn(2, 4)
        with set_forward_context(
            attn_metadata={"toy.attn": "MD"},
            vllm_config=cfg,
            slot_mapping={"toy.attn": torch.zeros(2, dtype=torch.int64)},
            cudagraph_runtime_mode=CUDAGraphMode.NONE,  # runtime NONE → passthrough
        ):
            out = compiled(x)
        # every compiled piece is wrapped in a PIECEWISE CUDAGraphWrapper
        for it in backend.piecewise_graphs:
            if not it.is_splitting_graph:
                assert isinstance(
                    backend.split_gm.__dict__[it.submod_name], CUDAGraphWrapper
                )


# ---------------------------------------------------------------------------
# m13/m15 — runner spans (gpu_model_runner.py scoped)
# ---------------------------------------------------------------------------


class _CpuGpuDouble:
    """Test double for the ch18-domain CpuGpuBuffer (np + gpu + copy_to_gpu)."""

    def __init__(self, np_arr):
        import numpy as np

        self.np = np_arr
        self.gpu = torch.from_numpy(np_arr.copy()).to(torch.int32)
        self.copies = 0

    def copy_to_gpu(self):
        self.copies += 1
        self.gpu = torch.from_numpy(self.np.copy()).to(torch.int32)


def _make_runner(cc: CompilationConfig, max_num_seqs=8):
    from implementation.v1.worker.gpu_model_runner import GPUModelRunner

    cfg = make_seam_vllm_config(cc, max_num_seqs=max_num_seqs)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = cfg
    runner.compilation_config = cc
    runner.parallel_config = cfg.parallel_config
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.input_batch = SimpleNamespace(lora_id_to_lora_request={})
    runner.uniform_decode_query_len = 1
    runner.cudagraph_dispatcher = CudagraphDispatcher(cfg)
    runner.observability_config = SimpleNamespace(cudagraph_metrics=False)
    runner.device = torch.device("cpu")
    return runner, cfg


class TestDetermineBatchExecutionAndPadding:
    def _init_full(self, runner):
        runner.cudagraph_dispatcher.initialize_cudagraph_keys(
            CUDAGraphMode.FULL_AND_PIECEWISE, uniform_decode_query_len=1
        )

    def test_uniform_decode_hits_full(self):
        cc = CompilationConfig(
            cudagraph_mode="FULL_AND_PIECEWISE",
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_capture_sizes=[1, 8],
        )
        runner, _ = _make_runner(cc)
        self._init_full(runner)
        import numpy as np

        mode, desc, ubatch, dp, stats = runner._determine_batch_execution_and_padding(
            num_tokens=8,
            num_reqs=8,
            num_scheduled_tokens_np=np.ones(8, dtype=np.int32),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
        )
        assert mode == CUDAGraphMode.FULL
        assert desc == BatchDescriptor(num_tokens=8, num_reqs=8, uniform=True)
        assert ubatch is False and dp is None and stats is None

    def test_cascade_attn_disables_full(self):
        cc = CompilationConfig(
            cudagraph_mode="FULL_AND_PIECEWISE",
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_capture_sizes=[1, 8],
        )
        runner, _ = _make_runner(cc)
        self._init_full(runner)
        import numpy as np

        mode, desc, *_ = runner._determine_batch_execution_and_padding(
            num_tokens=8,
            num_reqs=8,
            num_scheduled_tokens_np=np.ones(8, dtype=np.int32),
            max_num_scheduled_tokens=1,
            use_cascade_attn=True,  # invalid_modes={FULL}
        )
        assert mode == CUDAGraphMode.PIECEWISE
        assert desc.num_reqs is None  # relaxed key

    def test_encoder_output_disables_full(self):
        cc = CompilationConfig(
            cudagraph_mode="FULL_AND_PIECEWISE",
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_capture_sizes=[1, 8],
        )
        runner, _ = _make_runner(cc)
        runner.model_config = SimpleNamespace(is_encoder_decoder=True)
        self._init_full(runner)
        import numpy as np

        mode, desc, *_ = runner._determine_batch_execution_and_padding(
            num_tokens=8,
            num_reqs=8,
            num_scheduled_tokens_np=np.ones(8, dtype=np.int32),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
            num_encoder_reqs=1,
        )
        assert mode == CUDAGraphMode.PIECEWISE

    def test_non_uniform_batch_pads_and_falls_to_piecewise(self):
        cc = CompilationConfig(
            cudagraph_mode="FULL_AND_PIECEWISE",
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_capture_sizes=[1, 8],
        )
        runner, _ = _make_runner(cc)
        self._init_full(runner)
        import numpy as np

        # 5 tokens over 3 requests, max sched 3 → not uniform → no FULL key for
        # a mixed batch under FULL_AND_PIECEWISE (mixed mode is PIECEWISE)
        mode, desc, *_ = runner._determine_batch_execution_and_padding(
            num_tokens=5,
            num_reqs=3,
            num_scheduled_tokens_np=np.array([1, 1, 3], dtype=np.int32),
            max_num_scheduled_tokens=3,
            use_cascade_attn=False,
        )
        assert mode == CUDAGraphMode.PIECEWISE
        assert desc.num_tokens == 8  # padded up to the captured shape

    def test_force_eager_yields_none(self):
        cc = CompilationConfig(
            cudagraph_mode="FULL_AND_PIECEWISE",
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_capture_sizes=[1, 8],
        )
        runner, _ = _make_runner(cc)
        self._init_full(runner)
        import numpy as np

        mode, desc, *_ = runner._determine_batch_execution_and_padding(
            num_tokens=8,
            num_reqs=8,
            num_scheduled_tokens_np=np.ones(8, dtype=np.int32),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
            force_eager=True,
        )
        assert mode == CUDAGraphMode.NONE
        assert desc.num_tokens == 8

    def test_is_uniform_decode_static(self):
        from implementation.v1.worker.gpu_model_runner import GPUModelRunner

        f = GPUModelRunner._is_uniform_decode
        assert f(1, 1, 8, 8) is True
        assert f(1, 1, 8, 5) is False
        assert f(3, 1, 6, 2) is False
        assert f(1, 1, 8, 8, force_uniform_decode=True) is True
        assert f(1, 1, 8, 8, force_uniform_decode=False) is False


class TestPaddingFour:
    """m13 — the four padding spans, each verbatim in its scoped method."""

    def _runner_with_buffers(self):
        import numpy as np

        from implementation.v1.worker.gpu_model_runner import (
            NULL_BLOCK_ID,
            GPUModelRunner,
        )

        cc = _vllm_compile_cc()
        cfg = make_seam_vllm_config(cc)
        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.vllm_config = cfg
        runner.query_start_loc = _CpuGpuDouble(np.zeros(9, dtype=np.int32))
        runner.positions = torch.arange(16, dtype=torch.float32)
        runner.device = torch.device("cpu")
        # block-table / slot-mapping carriers (ch18-domain block table double:
        # get_device_tensor → (reqs, 2) table; slot_mapping.gpu → per-token slots)
        runner.input_batch = SimpleNamespace(
            block_table=[
                SimpleNamespace(
                    get_device_tensor=lambda padded: torch.arange(padded * 2,
                                                                 dtype=torch.int32)
                    .view(padded, 2)
                    .clone(),
                    slot_mapping=SimpleNamespace(
                        gpu=torch.arange(16, dtype=torch.int64)
                    ),
                )
            ]
        )
        runner.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(block_size=2),
                    layer_names=["m.l0", "m.l1"],
                )
            ]
        )
        return runner

    def test_query_start_loc_non_decreasing_tail(self):
        runner = self._runner_with_buffers()
        import numpy as np

        cu = np.array([0, 3, 5], dtype=np.int32)  # 3 reqs, inclusive cumsum
        num_reqs = 3
        # — the pinned span (gpu_model_runner.py:L2073-L2078) —
        runner.query_start_loc.np[0] = 0
        runner.query_start_loc.np[1 : num_reqs + 1] = cu
        runner.query_start_loc.np[num_reqs + 1 :].fill(cu[-1])
        runner.query_start_loc.copy_to_gpu()
        assert runner.query_start_loc.copies == 1
        tail = runner.query_start_loc.np[num_reqs + 1 :]
        assert (tail == 5).all()
        assert list(runner.query_start_loc.np) == [0, 0, 3, 5, 5, 5, 5, 5, 5]

    def test_block_table_null_fill(self):
        from implementation.v1.worker.gpu_model_runner import NULL_BLOCK_ID

        runner = self._runner_with_buffers()
        num_reqs, num_reqs_padded = 2, 4
        # exercise the pinned closure span through the scoped method
        blk = runner.input_batch.block_table[0].get_device_tensor(num_reqs_padded)
        blk[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)
        assert (blk[2:] == NULL_BLOCK_ID).all()
        assert NULL_BLOCK_ID == 0  # Block 0 is reserved for padding

    def test_slot_mapping_minus_one_tail(self):
        runner = self._runner_with_buffers()
        by_gid, by_layer = runner._get_slot_mappings(
            num_tokens_padded=8,
            num_reqs_padded=4,
            num_tokens_unpadded=5,
            ubatch_slices=None,
        )
        sm = by_gid[0]
        assert (sm[5:8] == -1).all()  # KV-write kernel skips padded slots
        assert (sm[:5] >= 0).all()
        assert by_layer == {"m.l0": sm, "m.l1": sm}

    def test_positions_zero_tail(self):
        runner = self._runner_with_buffers()
        num_input_tokens, num_scheduled_tokens = 8, 5
        positions = runner.positions[:num_input_tokens]
        # — the pinned span (gpu_model_runner.py:L3663-L3664) —
        if num_input_tokens > num_scheduled_tokens:
            runner.positions[num_scheduled_tokens:num_input_tokens].zero_()
        assert (positions[5:] == 0).all()
        assert not (positions[:5] == 0).all()


class TestRunnerCaptureSpans:
    def _capture_runner(self, cc):
        from implementation.v1.worker.gpu_model_runner import GPUModelRunner

        cfg = make_seam_vllm_config(cc)
        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.vllm_config = cfg
        runner.compilation_config = cc
        runner.parallel_config = cfg.parallel_config
        runner.cudagraph_dispatcher = CudagraphDispatcher(cfg)
        runner.cudagraph_dispatcher.initialize_cudagraph_keys(
            cc.cudagraph_mode, uniform_decode_query_len=1
        )
        runner.device = torch.device("cpu")
        runner.encoder_cudagraph_manager = None
        runner.uniform_decode_query_len = 1
        return runner

    def test_capture_model_skips_when_none_mode(self, caplog):
        import logging

        cc = CompilationConfig(cudagraph_mode="NONE")
        runner = self._capture_runner(cc)
        with caplog.at_level(logging.WARNING):
            assert runner.capture_model() == 0
        assert "Skipping CUDA graph capture" in caplog.text

    def test_warmup_then_capture_order(self):
        cc = _vllm_compile_cc(
            cudagraph_mode="PIECEWISE", cudagraph_capture_sizes=[1, 4],
            cudagraph_num_of_warmups=2,
        )
        runner = self._capture_runner(cc)
        calls = []

        def dummy_run(num_tokens, **kw):
            calls.append((num_tokens, kw.get("cudagraph_runtime_mode"),
                          kw.get("is_graph_capturing", False)))
            return None, None

        runner._dummy_run = dummy_run
        from implementation.compilation.monitor import (
            set_cudagraph_capturing_enabled,
        )

        set_cudagraph_capturing_enabled(True)
        try:
            runner._warmup_and_capture(
                BatchDescriptor(num_tokens=4, num_reqs=None, uniform=False),
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            )
        finally:
            set_cudagraph_capturing_enabled(False)
        # 2 eager warmups (mode NONE) then one capturing run
        assert calls == [
            (4, CUDAGraphMode.NONE, False),
            (4, CUDAGraphMode.NONE, False),
            (4, CUDAGraphMode.PIECEWISE, True),
        ]

    def test_capture_model_large_first_and_window_closes(self):
        cc = _vllm_compile_cc(
            cudagraph_mode="PIECEWISE", cudagraph_capture_sizes=[1, 4],
        )
        runner = self._capture_runner(cc)
        modes_seen = []

        def capture(batch_descriptors, cudagraph_runtime_mode, profiler=None):
            modes_seen.append(
                (cudagraph_runtime_mode, [d.num_tokens for d in batch_descriptors])
            )

        runner._capture_cudagraphs = capture
        import implementation.compilation.monitor as monitor

        with patch.object(
            torch.accelerator, "synchronize"
        ), patch.object(torch.accelerator, "empty_cache"), patch.object(
            torch.accelerator,
            "get_memory_info",
            return_value=(1000, 2000),
        ), patch.object(
            implementation.v1.worker.gpu_model_runner, "graph_capture",
            lambda **kw: __import__("contextlib").nullcontext(),
        ):
            used = runner.capture_model()
        # PIECEWISE group first, largest-first inside the group
        assert modes_seen[0][0] == CUDAGraphMode.PIECEWISE
        assert modes_seen[0][1] == [4, 1]
        # the capture window is closed afterwards: unexpected capture raises
        with pytest.raises(RuntimeError, match="inappropriate"):
            monitor.validate_cudagraph_capturing_enabled()
        monitor.set_cudagraph_capturing_enabled(True)
        assert used == 0  # mocked memory info: no delta


class TestLoadModelWrapperMount:
    def test_full_mode_mounts_full_wrapper(self):
        from implementation.v1.worker.gpu_model_runner import GPUModelRunner

        cc = _vllm_compile_cc(
            cudagraph_mode="FULL", cudagraph_capture_sizes=[1, 2]
        )
        cfg = make_seam_vllm_config(cc)
        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.vllm_config = cfg
        runner.compilation_config = cc
        runner.parallel_config = cfg.parallel_config
        runner.model = torch.nn.Linear(2, 2)
        runner.load_model()
        assert isinstance(runner.model, CUDAGraphWrapper)
        assert runner.model.runtime_mode == CUDAGraphMode.FULL  # FULL outside model

    def test_no_cudagraph_leaves_model_unwrapped(self):
        from implementation.v1.worker.gpu_model_runner import GPUModelRunner

        cc = _vllm_compile_cc(cudagraph_mode="NONE")
        cfg = make_seam_vllm_config(cc)
        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.vllm_config = cfg
        runner.compilation_config = cc
        runner.parallel_config = cfg.parallel_config
        runner.model = torch.nn.Linear(2, 2)
        runner.load_model()
        assert not isinstance(runner.model, CUDAGraphWrapper)


class TestCheckAndUpdateCudagraphMode:
    def test_weakest_link_downgrade_and_key_init(self):
        from implementation.v1.attention.backend import AttentionCGSupport

        cc = _vllm_compile_cc(
            cudagraph_mode="FULL_AND_PIECEWISE", cudagraph_capture_sizes=[1, 8]
        )
        cfg = make_seam_vllm_config(cc)
        from implementation.v1.worker.gpu_model_runner import GPUModelRunner

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.vllm_config = cfg
        runner.compilation_config = cc
        runner.parallel_config = cfg.parallel_config
        runner.uniform_decode_query_len = 1
        runner.cudagraph_dispatcher = CudagraphDispatcher(cfg)
        runner.kv_cache_config = None
        runner.max_num_reqs = 8
        runner.speculative_config = None

        class _BackendDouble:
            """Hashable attention-backend double (sets hold them)."""

            def __init__(self, name: str, support):
                self.__name__ = name
                self._support = support

            def get_builder_cls(self):
                support = self._support
                return SimpleNamespace(
                    get_cudagraph_support=lambda vllm_cfg, spec: support
                )

        always = _BackendDouble("AlwaysBackend", AttentionCGSupport.ALWAYS)
        never = _BackendDouble("NeverBackend", AttentionCGSupport.NEVER)
        group = SimpleNamespace(kv_cache_spec=SimpleNamespace())
        runner._check_and_update_cudagraph_mode(
            [{always}, {never}], [group, group]
        )
        # one NEVER backend drags the whole model's mixed-batch mode down:
        # FULL_AND_PIECEWISE → mixed FULL unsupported → PIECEWISE
        assert cc.cudagraph_mode == CUDAGraphMode.PIECEWISE
        assert runner.cudagraph_dispatcher.keys_initialized
        assert runner.cudagraph_dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]


# ---------------------------------------------------------------------------
# m18 — Worker.compile_or_warm_up_model startup orchestration (gpu_worker.py)
# ---------------------------------------------------------------------------


class TestCompileOrWarmUpModel:
    def _worker(self, cc):
        from implementation.v1.worker.gpu_worker import Worker

        cfg = make_seam_vllm_config(cc)
        worker = Worker.__new__(Worker)
        worker.vllm_config = cfg
        worker.compilation_config = cc
        worker.model_config = SimpleNamespace(enforce_eager=False, seed=0)
        worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=1)
        worker.scheduler_config = SimpleNamespace(
            max_num_seqs=4, max_num_batched_tokens=16
        )
        worker.observability_config = SimpleNamespace(
            jit_monitor_mode=None, jit_monitor_verbose=False
        )
        worker.device = torch.device("cpu")
        worker.parallel_config = cfg.parallel_config
        events = []
        worker.model_runner = SimpleNamespace(
            # real call sites use both positional (warmup loop) and keyword
            # `num_tokens=` (sampler warmup) — record whichever shape arrives
            _dummy_run=lambda *args, **kw: events.append(
                ("dummy_run", kw.get("num_tokens", args[0] if args else None))
            )
            or (None, None),
            maybe_remove_all_loras=lambda cfg_: events.append(("loras",)),
            lora_config=None,
            capture_model=lambda: events.append(("capture",)) or 0,
            is_pooling_model=False,
            _dummy_sampler_run=lambda **kw: events.append(("sampler",)),
        )
        return worker, events

    def test_orchestration_order_v1(self):
        cc = _vllm_compile_cc(
            compile_sizes=[8, 16], cudagraph_mode="NONE"
        )
        cfg = make_seam_vllm_config(cc)
        cc.compile_sizes = [8, 16]
        worker, events = self._worker(cc)
        with patch.object(
            implementation.v1.worker.gpu_worker, "kernel_warmup",
            lambda w: events.append(("kernel_warmup",)),
        ), patch.object(
            implementation.v1.worker.gpu_worker, "get_pp_group",
            lambda: SimpleNamespace(is_last_rank=True),
        ), patch.object(
            implementation.v1.worker.gpu_worker, "freeze_gc_heap",
            lambda: events.append(("freeze_gc",)),
        ), patch.object(
            implementation.v1.worker.gpu_worker, "enable_gpu_sync_check",
            lambda: events.append(("sync_check",)),
        ):
            times = worker.compile_or_warm_up_model()
        kinds = [e[0] for e in events]
        # warmup runs large-to-small, then kernels, capture, then the V1
        # sampler warmup (dummy_run at max_num_seqs + sampler), freeze, gate
        assert kinds[:4] == ["dummy_run", "dummy_run", "loras", "kernel_warmup"]
        assert events[0][1] == 16 and events[1][1] == 8  # sorted reverse
        assert kinds.index("capture") < kinds.index("sampler")
        assert kinds.index("sampler") < kinds.index("freeze_gc") < kinds.index(
            "sync_check"
        )
        assert times.language_model == cc.compilation_time

    def test_vllm_compile_warmup_excludes_cg_sizes_and_pads_ranges(self):
        cc = _vllm_compile_cc(
            compile_sizes=[8, 20],
            cudagraph_mode="PIECEWISE",
            cudagraph_capture_sizes=[4, 8],
            compile_ranges_endpoints=[8, 32],
        )
        cfg = make_seam_vllm_config(cc)
        worker, events = self._worker(cc)
        with patch.object(
            implementation.v1.worker.gpu_worker, "kernel_warmup", lambda w: None
        ), patch.object(
            implementation.v1.worker.gpu_worker, "get_pp_group",
            lambda: SimpleNamespace(is_last_rank=True),
        ), patch.object(
            implementation.v1.worker.gpu_worker, "freeze_gc_heap", lambda: None
        ), patch.object(
            implementation.v1.worker.gpu_worker, "enable_gpu_sync_check", lambda: None
        ), patch.object(
            implementation.v1.worker.gpu_worker, "activate_jit_monitor",
            lambda **kw: None,
        ):
            worker.compile_or_warm_up_model()
        warmup_sizes = [e[1] for e in events if e[0] == "dummy_run"]
        # 20 kept (not a cg size); 8 dropped (in cg sizes); both compile ranges
        # covered ((1,8] by cg {4,8}, (9,32] by 20) → no range-end appended;
        # the trailing dummy_run is the V1 sampler warmup (max_num_seqs=4)
        assert warmup_sizes[0] == 20
        assert 8 not in warmup_sizes
        assert 32 not in warmup_sizes
        assert warmup_sizes[-1] == 4

    def test_eager_skips_capture(self):
        cc = CompilationConfig(mode=CompilationMode.NONE, cudagraph_mode="NONE")
        worker, events = self._worker(cc)
        # enforce_eager is the capture gate in compile_or_warm_up_model
        worker.model_config = SimpleNamespace(enforce_eager=True, seed=0)
        with patch.object(
            implementation.v1.worker.gpu_worker, "kernel_warmup", lambda w: None
        ), patch.object(
            implementation.v1.worker.gpu_worker, "get_pp_group",
            lambda: SimpleNamespace(is_last_rank=True),
        ), patch.object(
            implementation.v1.worker.gpu_worker, "freeze_gc_heap", lambda: None
        ), patch.object(
            implementation.v1.worker.gpu_worker, "enable_gpu_sync_check", lambda: None
        ), patch.object(
            implementation.v1.worker.gpu_worker, "activate_jit_monitor",
            lambda **kw: None,
        ):
            worker.compile_or_warm_up_model()
        assert "capture" not in [e[0] for e in events]
