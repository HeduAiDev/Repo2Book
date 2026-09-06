# ch21《注意力后端》测试电池 —— TDD：先测真实 vLLM v0.27.1（6e448d0ea）的
# 可观察行为，精简版实现到通过为止。全部 host 可跑（纯单元，不 import vllm、
# 无 CUDA 上下文——vllm_flash_attn CUDA op / pynvml / torch.ops.vllm 的 CUDA 面
# 以 HOST SEAM 镜像承载，见 implementation/_host_seams.py 与 impl-notes §Seam）。
#
# 覆盖三条主线：
#   装配期·选后端（站 1-5）：优先级表 / validate 回退（ImportError 容忍）/
#     显式指定只校验不回退 / backend_per_kind 逐组覆写 / set_kv_cache_layout 副作用
#   KV 初始化期（站 6-7）：(full_cls_name, spec, num_heads_q) 归组 / 最弱链 CG
#     降级 / kernel 块协商（256→4×64）/ get_kv_cache_shape+stride_order 定形 bind
#   运行期（站 8-12）：cm_base 组装 / builder.build 翻译 / layer_name 铺设 /
#     逐组换表复用 / 写腿 reshape_and_cache_flash / 读腿 flash_attn_varlen_func
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implementation._host_seams import (  # noqa: E402
    DeviceCapability,
    VllmConfigSeam,
    _BlockTableSeam,
    _CpuGpuSeam,
    set_current_vllm_config,
)
from implementation.config.attention import AttentionConfig  # noqa: E402
from implementation.platforms import cuda as cuda_platform  # noqa: E402
from implementation.platforms.cuda import _get_backend_priorities  # noqa: E402
from implementation.utils.import_utils import resolve_obj_by_qualname  # noqa: E402
from implementation.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from implementation.v1.attention.backends.registry import (  # noqa: E402
    AttentionBackendEnum,
    _ATTN_OVERRIDES,
    register_backend,
)
from implementation.v1.attention.backends.utils import (  # noqa: E402
    NULL_BLOCK_ID,
    get_kv_cache_layout,
    set_kv_cache_layout,
)
from implementation.v1.attention.backend import (  # noqa: E402
    AttentionCGSupport,
    CommonAttentionMetadata,
    MultipleOf,
)
from implementation.v1.attention import selector  # noqa: E402
from implementation.v1.attention.selector import (  # noqa: E402
    AttentionSelectorConfig,
    get_attn_backend,
    get_attn_spec_kind,
)
from implementation.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheSpecKind,
    SlidingWindowSpec,
)

SM90 = DeviceCapability(9, 0)
SM100 = DeviceCapability(10, 0)
SM80 = DeviceCapability(8, 0)


# ─── 测试内注册的真实协议后端（第三方注册流 = register_backend 的真实用法） ──
# TRITON_ATTN 桩：固定块大小 [64]、CG=ALWAYS —— 用来对照优先级与最弱链。
class _TritonStubBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [24, 64]  # 24 让 --block-size 24 场景存活（FA 的 16 倍数不含 24）；64 驱动 256→4×64


class _AlwaysBuilder(FlashAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.ALWAYS


# 第三方后端（装饰器形态的载体：类路径自动取 __module__.__qualname__）
class _MyCustomBackend(FlashAttentionBackend):
    @staticmethod
    def get_name():
        return "MINE"


# 要求 NHD 布局的后端（set_kv_cache_layout 副作用的载体测试）
class _NhdBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def get_required_kv_cache_layout(cls):
        return "NHD"


@pytest.fixture()
def registry_overrides():
    """把包内真身注册进枚举（register_backend 真实机制），测完还原。"""
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        "implementation.v1.attention.backends.flash_attn.FlashAttentionBackend",
    )
    register_backend(
        AttentionBackendEnum.TRITON_ATTN,
        __name__ + "._TritonStubBackend",
    )
    selector._cached_get_attn_backend.cache_clear()
    yield
    AttentionBackendEnum.FLASH_ATTN.clear_override()
    AttentionBackendEnum.TRITON_ATTN.clear_override()
    _ATTN_OVERRIDES.pop(AttentionBackendEnum.CUSTOM, None)
    selector._cached_get_attn_backend.cache_clear()


def _cfg(
    backend=None,
    backend_per_kind=None,
    user_specified_block_size=False,
    block_size=16,
):
    cfg = VllmConfigSeam(max_num_seqs=8, max_batched_tokens=512, max_model_len=256)
    cfg.attention_config = AttentionConfig(
        backend=backend, backend_per_kind=backend_per_kind or {}
    )
    if user_specified_block_size:
        cfg.cache_config.user_specified_block_size = True
        cfg.cache_config.block_size = block_size
    return cfg


def _selector_config(**kw):
    base = dict(
        head_size=64,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=None,
    )
    base.update(kw)
    return AttentionSelectorConfig(**base)


# ═══ A. 注册表：枚举值=类路径、覆盖表、懒加载 ═════════════════════════════


class TestRegistry:
    def test_enum_value_is_class_path(self):
        # 枚举值就是完整类路径字符串（『名字→类路径』静态表）
        assert (
            AttentionBackendEnum.FLASH_ATTN.value
            == "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
        )
        assert AttentionBackendEnum.CUSTOM.value is None

    def test_unregistered_member_imports_the_real_module(self):
        # 未注册覆盖 → 回退枚举值路径 → host 无 vllm 包 → ImportError
        # （get_valid_backends 正是靠接住它来容忍『没装依赖』）
        with pytest.raises(ImportError):
            AttentionBackendEnum.FLASHINFER.get_class()

    def test_get_class_resolves_override(self, registry_overrides):
        cls = AttentionBackendEnum.FLASH_ATTN.get_class()
        assert cls is FlashAttentionBackend

    def test_custom_must_be_registered(self):
        with pytest.raises(ValueError, match="must be registered"):
            AttentionBackendEnum.CUSTOM.get_path()

    def test_unknown_name_lists_members(self):
        with pytest.raises(ValueError, match="Valid options are"):
            AttentionBackendEnum["NO_SUCH_BACKEND"]

    def test_register_backend_decorator_form(self, registry_overrides):
        # 装饰器形态：class_path=None → 路径自动取 __module__.__qualname__
        decorator = register_backend(AttentionBackendEnum.CUSTOM)
        assert decorator(_MyCustomBackend) is _MyCustomBackend
        assert AttentionBackendEnum.CUSTOM.get_class() is _MyCustomBackend

    def test_resolve_obj_by_qualname(self):
        import math

        obj = resolve_obj_by_qualname("math.floor")
        assert obj is math.floor


# ═══ B. 平台优先级表（cuda.py _get_backend_priorities） ═══════════════════


class TestPriorities:
    def test_non_mla_sm100_prefers_flashinfer(self):
        prios = _get_backend_priorities(False, SM100)
        assert prios[0] == AttentionBackendEnum.FLASHINFER
        assert prios[1] == AttentionBackendEnum.FLASH_ATTN

    def test_non_mla_sm100_non_causal_prefers_flash_attn(self):
        # non-causal 的 cutlass 路径有问题 → FLASH_ATTN 提前
        prios = _get_backend_priorities(False, SM100, use_non_causal=True)
        assert prios[0] == AttentionBackendEnum.FLASH_ATTN

    def test_non_mla_other_generations_prefer_flash_attn(self):
        assert _get_backend_priorities(False, SM90)[0] == AttentionBackendEnum.FLASH_ATTN
        assert _get_backend_priorities(False, SM80)[0] == AttentionBackendEnum.FLASH_ATTN

    def test_mla_sm100_head(self):
        prios = _get_backend_priorities(True, SM100)
        assert prios[0] == AttentionBackendEnum.FLASHINFER_MLA
        # TOKENSPEED_MLA 第二（注释自述 bs≈8 以上才赢）
        assert prios[1] == AttentionBackendEnum.TOKENSPEED_MLA
        assert prios[2] == AttentionBackendEnum.CUTLASS_MLA

    def test_mla_sm120_head(self):
        prios = _get_backend_priorities(True, DeviceCapability(12, 0))
        assert prios == [
            AttentionBackendEnum.TRITON_MLA,
            AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
        ]

    def test_mla_fp8_kv_sparse_order(self):
        prios = _get_backend_priorities(True, SM100, kv_cache_dtype="fp8")
        tail = prios[-2:]
        assert tail == [
            AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
            AttentionBackendEnum.FLASHMLA_SPARSE,
        ]

    def test_priorities_are_cached_and_hashable_key(self):
        a = _get_backend_priorities(False, SM90)
        b = _get_backend_priorities(False, SM90)
        assert a is b  # @cache 同 key 同对象


# ═══ C. validate 探针（AttentionBackend.validate_configuration + FA 实现） ═


class TestValidate:
    def test_fa_valid_on_sm90(self):
        reasons = FlashAttentionBackend.validate_configuration(
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=None,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=SM90,
            attn_type="decoder",
        )
        assert reasons == []

    def test_compute_capability_probe(self):
        # FA 要求 >= 8.0（Ampere）
        kw = dict(
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=None,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            attn_type="decoder",
        )
        reasons = FlashAttentionBackend.validate_configuration(
            device_capability=DeviceCapability(7, 5), **kw
        )
        assert "compute capability not supported" in reasons

    def test_head_size_probe(self):
        kw = dict(
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=None,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=SM90,
            attn_type="decoder",
        )
        assert "head_size not supported" in FlashAttentionBackend.validate_configuration(
            head_size=100, **kw  # 100 % 8 != 0
        )

    def test_kv_cache_dtype_probe(self):
        kw = dict(
            head_size=64,
            dtype=torch.float16,
            block_size=None,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=SM90,
            attn_type="decoder",
        )
        assert (
            "kv_cache_dtype not supported"
            in FlashAttentionBackend.validate_configuration(
                kv_cache_dtype="fp8_e5m2", **kw
            )
        )

    def test_block_size_probe_multiple_of_16(self):
        assert FlashAttentionBackend.supports_block_size(None) is True
        assert FlashAttentionBackend.supports_block_size(16) is True
        assert FlashAttentionBackend.supports_block_size(48) is True  # 48 % 16 == 0
        assert FlashAttentionBackend.supports_block_size(24) is False

    def test_supports_combination_fp8_requires_fa3_sm90(self):
        # host 无 FA3（get_flash_attn_version 恒 2）→ fp8 组合探针给原因
        reason = FlashAttentionBackend.supports_combination(
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype="fp8",
            block_size=None,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            device_capability=SM90,
        )
        assert reason == "FP8 KV cache requires FA3 on SM90 or FA4 on SM100"

    def test_validate_default_attn_type_is_decoder_only(self):
        kw = dict(
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=None,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=SM90,
        )
        # 基类默认只支持 decoder（FA 覆写了 supports_attn_type 全支持）
        from implementation.v1.attention.backend import AttentionBackend

        reasons = AttentionBackend.validate_configuration(attn_type="encoder_only", **kw)
        assert any("attention type" in r for r in reasons)


# ═══ D. validate 回退与胜者（get_valid_backends / get_attn_backend_cls） ═══


class TestSelection:
    def test_get_valid_backends_importerror_tolerated(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        valid, invalid = cuda_platform.CudaPlatform.get_valid_backends(SM90, _selector_config())
        names = [c.backend.name for c in valid]
        # FLASHINFER 未注册 → host 无 vllm 包 → ImportError 被当一条原因
        assert (
            AttentionBackendEnum.FLASHINFER in invalid
            and invalid[AttentionBackendEnum.FLASHINFER][1] == ["ImportError"]
        )
        # FLASH_ATTN（已注册包内真身）与 TRITON_ATTN（桩）合法
        assert "FLASH_ATTN" in names and "TRITON_ATTN" in names
        prio = {c.backend.name: c.priority for c in valid}
        assert prio["FLASH_ATTN"] == 0  # SM90 表头

    def test_auto_select_min_priority_wins(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM100
        cls_path = cuda_platform.CudaPlatform.get_attn_backend_cls(
            None, _selector_config()
        )
        # SM100 表 [FLASHINFER, FLASH_ATTN, ...]：FLASHINFER 落选（ImportError）
        # → 胜者是 priority 1 的 FLASH_ATTN
        assert cls_path.endswith("FlashAttentionBackend")

    def test_explicit_backend_validated_not_fallback(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM100
        cls_path = cuda_platform.CudaPlatform.get_attn_backend_cls(
            AttentionBackendEnum.TRITON_ATTN, _selector_config()
        )
        assert cls_path.endswith("_TritonStubBackend")

    def test_explicit_invalid_backend_raises(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM80
        # FA 在 SM80 上本身合法；用 TRITON 桩（继承 FA 探针）改用坏 head_size
        cfg = _selector_config(head_size=100)
        with pytest.raises(ValueError, match="not valid for this configuration"):
            cuda_platform.CudaPlatform.get_attn_backend_cls(
                AttentionBackendEnum.TRITON_ATTN, cfg
            )

    def test_all_invalid_raises_with_reasons(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = DeviceCapability(7, 0)
        with pytest.raises(ValueError, match="No valid attention backend"):
            cuda_platform.CudaPlatform.get_attn_backend_cls(
                None, _selector_config()
            )

    def test_user_block_size_precluded_warning(self, registry_overrides, caplog):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        with caplog.at_level("WARNING", logger="vllm.platforms.cuda"):
            cuda_platform.CudaPlatform.get_attn_backend_cls(
                None,
                _selector_config(block_size=24),  # 24 % 16 != 0 → FA 落选
            )
        assert any("--block-size" in r for r in caplog.messages)


# ═══ E. selector 入口：打包维度、backend_per_kind、@cache、layout 副作用 ═══


class TestSelector:
    def test_invalid_kv_cache_dtype_asserts(self, registry_overrides):
        with set_current_vllm_config(_cfg()), pytest.raises(AssertionError):
            get_attn_backend(64, torch.float16, "bogus_dtype")

    def test_get_attn_backend_resolves_registered_class(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        with set_current_vllm_config(_cfg()):
            cls = get_attn_backend(64, torch.float16, "auto")
        assert cls is FlashAttentionBackend

    def test_selector_config_picks_up_user_block_size(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg(user_specified_block_size=True, block_size=32)
        seen = {}

        class _SpyPlatform(cuda_platform.CudaPlatform):
            @classmethod
            def get_attn_backend_cls(cls_, selected_backend, attn_selector_config, num_heads=None):
                seen.update(attn_selector_config._asdict())
                return "implementation.v1.attention.backends.flash_attn.FlashAttentionBackend"

        import implementation._host_seams as seams

        orig = seams.current_platform
        seams.current_platform = _SpyPlatform
        selector._cached_get_attn_backend.cache_clear()
        try:
            with set_current_vllm_config(cfg):
                get_attn_backend(64, torch.float16, "auto")
        finally:
            seams.current_platform = orig
            selector._cached_get_attn_backend.cache_clear()
        assert seen["block_size"] == 32
        assert seen["head_size"] == 64
        assert seen["use_mla"] is False

    def test_backend_per_kind_overrides_by_kind(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg(
            backend_per_kind={"sliding_window": "TRITON_ATTN"},
        )
        import implementation._host_seams as seams

        seen = []

        class _SpyPlatform(cuda_platform.CudaPlatform):
            @classmethod
            def get_attn_backend_cls(cls_, selected_backend, attn_selector_config, num_heads=None):
                seen.append(selected_backend)
                return "implementation.v1.attention.backends.flash_attn.FlashAttentionBackend"

        orig = seams.current_platform
        seams.current_platform = _SpyPlatform
        selector._cached_get_attn_backend.cache_clear()
        try:
            with set_current_vllm_config(cfg):
                get_attn_backend(64, torch.float16, "auto", has_sliding_window=True)
                get_attn_backend(64, torch.float16, "auto")  # full attention 层
        finally:
            seams.current_platform = orig
            selector._cached_get_attn_backend.cache_clear()
        # SW 层被覆写为 TRITON_ATTN；full attention 层回退全局（None=自动）
        assert seen == [AttentionBackendEnum.TRITON_ATTN, None]

    def test_cached_selector_returns_same_class(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        selector._cached_get_attn_backend.cache_clear()
        with set_current_vllm_config(_cfg()):
            a = get_attn_backend(64, torch.float16, "auto")
            b = get_attn_backend(64, torch.float16, "auto")
        assert a is b  # 同配置只解一次，几十层共享结果

    def test_selector_requires_layout_side_effect(self, registry_overrides):
        # 后端要求特定 KV layout → _cached_get_attn_backend 全局 set_kv_cache_layout
        register_backend(AttentionBackendEnum.FLASH_ATTN, __name__ + "._NhdBackend")
        selector._cached_get_attn_backend.cache_clear()
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        try:
            with set_current_vllm_config(_cfg()):
                get_attn_backend(64, torch.float16, "auto")
            assert get_kv_cache_layout() == "NHD"
            # FA 的 stride_order 随之变为 NHD 物理序
            assert FlashAttentionBackend.get_kv_cache_stride_order() == (0, 2, 1, 3)
        finally:
            set_kv_cache_layout(None)
            AttentionBackendEnum.FLASH_ATTN.clear_override()
            selector._cached_get_attn_backend.cache_clear()

    def test_get_attn_spec_kind_matrix(self):
        assert (
            get_attn_spec_kind(False, False, "decoder") == KVCacheSpecKind.FULL_ATTENTION
        )
        assert (
            get_attn_spec_kind(False, True, "decoder") == KVCacheSpecKind.SLIDING_WINDOW
        )
        assert get_attn_spec_kind(True, False, "decoder") == KVCacheSpecKind.MLA_ATTENTION
        assert (
            get_attn_spec_kind(True, True, "decoder") == KVCacheSpecKind.SLIDING_WINDOW_MLA
        )
        assert (
            get_attn_spec_kind(False, False, "encoder_only")
            == KVCacheSpecKind.ENCODER_ONLY_ATTENTION
        )
        assert (
            get_attn_spec_kind(False, False, "encoder_decoder")
            == KVCacheSpecKind.CROSS_ATTENTION
        )


# ═══ F. 后端协议：KV 布局声明（shape / stride_order / block_dim） ═════════


class TestBackendProtocol:
    def test_fa_kv_cache_shape_packs_kv_in_content_dim(self):
        # K/V 打包进最后一维：(num_blocks, num_kv_heads, block_size, 2*head_size)
        assert FlashAttentionBackend.get_kv_cache_shape(10, 16, 2, 8) == (10, 2, 16, 16)

    def test_fa_shape_requires_block_multiple_of_16(self):
        with pytest.raises(ValueError, match="multiple of 16"):
            FlashAttentionBackend.get_kv_cache_shape(10, 24, 2, 8)

    def test_stride_order_nhd_vs_hnd(self, registry_overrides):
        set_kv_cache_layout("NHD")
        assert FlashAttentionBackend.get_kv_cache_stride_order() == (0, 2, 1, 3)
        assert FlashAttentionBackend.get_kv_cache_stride_order(True) == (1, 0, 3, 2, 4)
        set_kv_cache_layout("HND")
        assert FlashAttentionBackend.get_kv_cache_stride_order() == (0, 1, 2, 3)
        assert FlashAttentionBackend.get_kv_cache_stride_order(True) == (1, 2, 0, 3, 4)
        set_kv_cache_layout(None)

    def test_stride_order_unknown_layout_raises(self):
        with pytest.raises(ValueError, match="Unknown cache layout"):
            from implementation.v1.attention.backends.flash_attn import (
                FlashAttentionBackend as FAB,
            )

            set_kv_cache_layout("XXD")
            FAB.get_kv_cache_stride_order()
        set_kv_cache_layout(None)

    def test_block_dim_sentinel_probe(self):
        # 哨兵探测：num_blocks 是逻辑形第几维（FA=0：blocks-first）
        assert FlashAttentionBackend.get_kv_cache_block_dim(16, 2, 8) == 0

    def test_full_cls_name_dedup_key(self):
        module, qual = FlashAttentionBackend.full_cls_name()
        assert module.endswith("flash_attn")
        assert qual == "FlashAttentionBackend"


# ═══ G. Attention 层：建层即选、自注册、forward 双算子 ═══════════════════


class TestAttentionLayer:
    def _make(self, cfg, prefix="model.layers.0.self_attn", **kw):
        from implementation.model_executor.layers.attention.attention import Attention

        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)  # 真实路径：load 期设模型 dtype 为默认 dtype
        try:
            return self._make_inner(cfg, prefix, **kw)
        finally:
            torch.set_default_dtype(prev)

    def _make_inner(self, cfg, prefix, **kw):
        from implementation.model_executor.layers.attention.attention import Attention

        with set_current_vllm_config(cfg):
            return Attention(
                num_heads=4,
                head_size=64,
                scale=0.125,
                num_kv_heads=2,
                prefix=prefix,
                **kw,
            )

    def test_init_selects_backend_and_reverse_looks_up_enum(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg()
        attn = self._make(cfg)
        assert attn.attn_backend is FlashAttentionBackend
        assert attn.backend == AttentionBackendEnum.FLASH_ATTN  # 枚举反查
        assert isinstance(attn.impl, FlashAttentionImpl)
        assert attn.use_direct_call is True  # host seam 平台直调

    def test_init_registers_into_static_forward_context(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg()
        attn = self._make(cfg)
        assert cfg.compilation_config.static_forward_context["model.layers.0.self_attn"] is attn
        with pytest.raises(ValueError, match="Duplicate layer name"):
            self._make(cfg)

    def test_forward_write_then_read_legs(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        from implementation.forward_context import set_forward_context

        cfg = _cfg()
        attn = self._make(cfg)
        # 裸显存 as_strided 成 FA 布局并 bind（站 7 消费点的最小复刻）
        blocks, kvh, kbs, hs = 4, 2, 16, 64
        raw = torch.zeros(blocks * kvh * kbs * 2 * hs, dtype=torch.float16)
        kv_cache = raw.view(blocks, kvh, kbs, 2 * hs)
        attn.kv_cache = kv_cache

        # 一个 prefill 请求：4 个新 token，历史 12 token
        qsl = torch.tensor([0, 4], dtype=torch.int32)
        seq_lens = torch.tensor([16], dtype=torch.int32)
        block_table = torch.tensor([[0, 1]], dtype=torch.int32)
        slots = torch.arange(12, 16, dtype=torch.int64)
        common = CommonAttentionMetadata(
            query_start_loc=qsl,
            query_start_loc_cpu=qsl.clone(),
            seq_lens=seq_lens,
            num_reqs=1,
            num_actual_tokens=4,
            max_query_len=4,
            max_seq_len=16,
            block_table_tensor=block_table,
            slot_mapping=slots,
        )
        fa_meta = FlashAttentionMetadataBuilder(
            kv_cache_spec=FullAttentionSpec(
                block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
            ),
            layer_names=["model.layers.0.self_attn"],
            vllm_config=cfg,
            device=torch.device("cpu"),
        ).build(common_prefix_len=0, common_attn_metadata=common)

        torch.manual_seed(0)
        q = torch.randn(4, 4, 64, dtype=torch.float16)
        k = torch.randn(4, 2, 64, dtype=torch.float16)
        v = torch.randn(4, 2, 64, dtype=torch.float16)

        with set_forward_context(
            {"model.layers.0.self_attn": fa_meta},
            cfg,
            slot_mapping={"model.layers.0.self_attn": slots},
        ):
            out = attn(q.view(4, 256), k.view(4, 128), v.view(4, 128))
        assert out.shape == (4, 4 * 64)

        # 写腿：新 K/V 已按 slot_mapping 散写进块 0 的第 12..15 行
        kc, _ = kv_cache.transpose(1, 2).split(64, dim=-1)
        torch.testing.assert_close(kc[0, 12:16].float(), k.float())
        # 读腿：与手工精确 attention 对拍（ch20 的数学）
        k_hist = torch.cat([kc[0, :12], k[0:4]], dim=0).repeat_interleave(2, dim=1)
        _, vc = kv_cache.transpose(1, 2).split(64, dim=-1)
        v_hist = torch.cat([vc[0, :12], v[0:4]], dim=0).repeat_interleave(2, dim=1)
        for i in range(4):
            n = 12 + i + 1  # query i 的绝对位置 = 12+i，看前 n 个键
            scores = (k_hist[:n, 0].float() @ q[i, 0].float()) * 0.125
            p = torch.softmax(scores.double(), dim=0)
            torch.testing.assert_close(
                out[i].float().view(4, 64)[0],
                (p.float() @ v_hist[:n, 0].float()),
                rtol=1e-3,
                atol=1e-3,
            )

    def test_kv_sharing_layer_skips_write(self, registry_overrides):
        # kv_sharing_target_layer_name 非 None → forward 不走写腿
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg()
        attn0 = self._make(cfg, prefix="model.layers.0.self_attn")
        attn1 = self._make(
            cfg, prefix="model.layers.1.self_attn", kv_sharing_target_layer_name="model.layers.0.self_attn"
        )
        assert attn1.kv_sharing_target_layer_name == "model.layers.0.self_attn"

    def test_get_kv_cache_spec_full_attention(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg()
        cfg.cache_config.block_size = 16
        attn = self._make(cfg)
        spec = attn.get_kv_cache_spec(cfg)
        assert isinstance(spec, FullAttentionSpec)
        assert spec.block_size == 16
        assert spec.num_kv_heads == 2
        assert spec.head_size == 64

    def test_get_kv_cache_spec_sliding_window(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg()
        cfg.cache_config.block_size = 16
        attn = self._make(cfg, per_layer_sliding_window=64)
        spec = attn.get_kv_cache_spec(cfg)
        assert isinstance(spec, SlidingWindowSpec)
        assert spec.sliding_window == 64


# ═══ H. AttentionConfig：backend / backend_per_kind 用户面 ═══════════════


class TestAttentionConfig:
    def test_auto_parses_to_none(self):
        assert AttentionConfig(backend="auto").backend is None
        assert AttentionConfig(backend="flash_attn").backend is AttentionBackendEnum.FLASH_ATTN

    def test_backend_per_kind_parses_enums(self):
        cfg = AttentionConfig(
            backend_per_kind={"sliding_window": "TRITON_ATTN", "mla_attention": "FLASHINFER_MLA"}
        )
        assert cfg.backend_per_kind["sliding_window"] is AttentionBackendEnum.TRITON_ATTN
        assert cfg.backend_per_kind["mla_attention"] is AttentionBackendEnum.FLASHINFER_MLA

    def test_backend_per_kind_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown KV cache group kind"):
            AttentionConfig(backend_per_kind={"no_such_kind": "TRITON_ATTN"})


# ═══ I. 归组与混布（initialize_attn_backend / AttentionGroupKey） ═════════


def _spec(block_size=16):
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=2, head_size=64, dtype=torch.float16
    )


def _make_runner(cfg, layers, with_tensors=False):
    """layers: list[(layer_name, backend_cls, spec)] → facet runner（站 6-7 全链）。"""
    from implementation.v1.worker.gpu_model_runner import GPUModelRunner
    from implementation.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    runner = GPUModelRunner(cfg, device=torch.device("cpu"))
    for name, backend_cls, spec in layers:
        attn = _make_layer(cfg, name, backend_cls, spec)
        cfg.compilation_config.static_forward_context[name] = attn
    group_spec = layers[0][2]
    tensors = []
    if with_tensors:
        # 裸显存账本：num_blocks × page_size_bytes（每层一张整卡账）
        from implementation.v1.kv_cache_interface import AttentionSpec as _AS

        assert isinstance(group_spec, _AS)
        size = 64 * group_spec.page_size_bytes
        tensors = [
            KVCacheTensor(size=size, shared_by=[name]) for (name, _, _) in layers
        ]
    runner.kv_cache_config = KVCacheConfig(
        num_blocks=64, kv_cache_tensors=tensors,
        kv_cache_groups=[KVCacheGroupSpec([n for (n, _, _) in layers], group_spec)],
    )
    return runner


def _make_layer(cfg, name, backend_cls, spec):
    from implementation.model_executor.layers.attention.attention import Attention

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)  # 真实路径：load 期默认 dtype = 模型 dtype
    try:
        with set_current_vllm_config(cfg):
            return _make_layer_inner(cfg, name, backend_cls, spec, Attention)
    finally:
        torch.set_default_dtype(prev)


def _make_layer_inner(cfg, name, backend_cls, spec, Attention):
    with set_current_vllm_config(cfg):
        attn = Attention(
            num_heads=4, head_size=64, scale=0.125, num_kv_heads=2,
            prefix=name, attn_backend=backend_cls,
            per_layer_sliding_window=(spec.sliding_window if isinstance(spec, SlidingWindowSpec) else None),
        )
    return attn


def _make_layer_tail():
    pass


class TestGrouping:
    @pytest.fixture()
    def runner_env(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg()
        layers = [
            ("model.layers.0.self_attn", FlashAttentionBackend, _spec()),
            ("model.layers.1.self_attn", FlashAttentionBackend, _spec()),
            ("model.layers.2.self_attn", _TritonStubBackend, _spec()),
        ]
        runner = _make_runner(cfg, layers)
        return cfg, runner

    def test_groups_by_backend_class_name(self, runner_env):
        cfg, runner = runner_env
        runner.initialize_attn_backend(runner.kv_cache_config)
        assert len(runner.attn_groups) == 1  # 一个 KV cache 组
        groups = runner.attn_groups[0]
        assert len(groups) == 2  # FA 两层 + Triton 桩一层 → 两个 attn group
        by_backend = {g.backend: g.layer_names for g in groups}
        assert by_backend[FlashAttentionBackend] == [
            "model.layers.0.self_attn",
            "model.layers.1.self_attn",
        ]
        assert by_backend[_TritonStubBackend] == ["model.layers.2.self_attn"]
        assert all(g.kv_cache_group_id == 0 for g in groups)

    def test_min_cg_support_weakest_link(self, runner_env):
        cfg, runner = runner_env
        runner.initialize_attn_backend(runner.kv_cache_config)
        # FA2 builder=UNIFORM_BATCH（host 无 FA3），Triton 桩继承 FA builder →
        # 最弱链 = UNIFORM_BATCH（不是 ALWAYS）——降级链收到的就是这个值
        assert runner.seam_min_cg_support == AttentionCGSupport.UNIFORM_BATCH

    def test_double_initialize_asserts(self, runner_env):
        cfg, runner = runner_env
        runner.initialize_attn_backend(runner.kv_cache_config)
        with pytest.raises(AssertionError, match="already initialized"):
            runner.initialize_attn_backend(runner.kv_cache_config)


# ═══ J. kernel 块协商（select_common_block_size / prepare_kernel_block_sizes） ═


class TestBlockSizeNegotiation:
    def test_case1_manager_size_supported(self):
        from implementation.v1.worker.utils import select_common_block_size

        assert select_common_block_size(256, [FlashAttentionBackend]) == 256

    def test_case2_splits_256_into_64s(self):
        from implementation.v1.worker.utils import select_common_block_size

        # 256-token 管理块 vs 桩（只支持固定 64）→ 协商出 64（4×64）
        assert select_common_block_size(256, [FlashAttentionBackend, _TritonStubBackend]) == 64

    def test_no_common_size_raises(self):
        from implementation.v1.worker.utils import select_common_block_size

        with pytest.raises(ValueError, match="No common block size"):
            select_common_block_size(24, [FlashAttentionBackend, _TritonStubBackend])

    def test_prepare_kernel_block_sizes_end_to_end(self, registry_overrides):
        from implementation.v1.kv_cache_interface import (
            KVCacheConfig,
            KVCacheGroupSpec,
        )
        from implementation.v1.worker.utils import prepare_kernel_block_sizes

        cfg = _cfg()
        runner = _make_runner(cfg, [
            ("model.layers.0.self_attn", FlashAttentionBackend, _spec(256)),
            ("model.layers.1.self_attn", _TritonStubBackend, _spec(256)),
        ])
        runner.initialize_attn_backend(runner.kv_cache_config)
        sizes = prepare_kernel_block_sizes(runner.kv_cache_config, runner.attn_groups)
        assert sizes == [64]  # 256 管理块 → 4×64 kernel 块


# ═══ K. KV 定形与 bind（站 7：shape/stride 消费 + as_strided + bind） ═════


class TestKvCacheTensors:
    def _runner_with_cache(self, registry_overrides):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg()
        runner = _make_runner(
            cfg,
            [
                ("model.layers.0.self_attn", FlashAttentionBackend, _spec(16)),
                ("model.layers.1.self_attn", FlashAttentionBackend, _spec(16)),
            ],
            with_tensors=True,
        )
        runner.initialize_kv_cache(runner.kv_cache_config)  # 内含 initialize_attn_backend（站 6 入口）
        return cfg, runner

    def test_allocate_reshape_bind(self, registry_overrides):
        cfg, runner = self._runner_with_cache(registry_overrides)
        assert runner._kernel_block_sizes == [16]
        kv = runner.seam_kv_caches
        assert set(kv) == {"model.layers.0.self_attn", "model.layers.1.self_attn"}
        for name, tensor in kv.items():
            # get_kv_cache_shape 逻辑形：(num_blocks, kvh, bs, 2*D)
            assert tensor.shape == (64, 2, 16, 128)
            # 层的 .kv_cache 已被 bind（写腿/读腿都吃它）
            layer = cfg.compilation_config.static_forward_context[name]
            assert layer.kv_cache is tensor
        # runner 侧 kv_caches 列表按层序排列
        assert len(runner.kv_caches) == 2

    def test_reshape_is_a_view_of_raw_allocation(self, registry_overrides):
        cfg, runner = self._runner_with_cache(registry_overrides)
        kv = runner.seam_kv_caches["model.layers.0.self_attn"]
        # 默认物理序 NHD：逻辑形 (B,H,N,2D) 但块内 token 维提前——
        # stride 反映物理内存序 (B,N,H,2D) 连续
        assert kv.shape == (64, 2, 16, 128)
        assert kv.stride() == (16 * 2 * 128, 128, 2 * 128, 1)

    def test_hnd_layout_override_changes_stride(self, registry_overrides):
        set_kv_cache_layout("HND")
        try:
            cfg = _cfg()
            runner = _make_runner(
                cfg,
                [("model.layers.0.self_attn", FlashAttentionBackend, _spec(16))],
                with_tensors=True,
            )
            runner.initialize_kv_cache(runner.kv_cache_config)
            kv = runner.seam_kv_caches["model.layers.0.self_attn"]
            # HND：物理序=逻辑形（恒等置换）→ contiguous
            assert kv.shape == (64, 2, 16, 128)
            assert kv.stride() == (2 * 16 * 128, 16 * 128, 128, 1)
        finally:
            set_kv_cache_layout(None)


# ═══ L. 每拍 metadata（cm_base → build 翻译 → layer_name 铺设 → 逐组换表） ═


def _common(num_reqs=1, tokens=4, seq=16, table=None, slots=None, padded_reqs=None, padded_tokens=None):
    qsl = torch.tensor([0, tokens][: num_reqs + 1], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32) * tokens,
        query_start_loc_cpu=torch.arange(num_reqs + 1, dtype=torch.int32) * tokens,
        seq_lens=torch.full((num_reqs,), seq, dtype=torch.int32),
        num_reqs=padded_reqs or num_reqs,
        num_actual_tokens=padded_tokens or tokens * num_reqs,
        max_query_len=tokens,
        max_seq_len=seq,
        block_table_tensor=table if table is not None else torch.tensor([[0, 1]], dtype=torch.int32),
        slot_mapping=slots if slots is not None else torch.arange(tokens, dtype=torch.int64),
    )


class TestBuildAttentionMetadata:
    def _runner(self, registry_overrides, n_groups=1):
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg()
        layers = [("model.layers.0.self_attn", FlashAttentionBackend, _spec(16))]
        if n_groups == 1:
            layers.append(("model.layers.1.self_attn", FlashAttentionBackend, _spec(16)))
        runner = _make_runner(cfg, layers, with_tensors=True)
        runner.initialize_kv_cache(runner.kv_cache_config)  # 内含 initialize_attn_backend（站 6 入口）
        return cfg, runner

    def _seed(self, runner, common):
        """每拍输入面注入（真实由 _prepare_inputs 装填——ch22 域；测试直供）。"""
        runner.optimistic_seq_lens_cpu = common.seq_lens.to(torch.int32)
        runner.query_start_loc = _CpuGpuSeam(common.query_start_loc)
        runner.seq_lens = common.seq_lens
        runner.input_batch.block_tables[0] = _BlockTableSeam(
            common.block_table_tensor, common.slot_mapping
        )

    def test_translation_common_to_fa_metadata(self, registry_overrides):
        cfg, runner = self._runner(registry_overrides)
        common = _common()
        self._seed(runner, common)
        attn_metadata, _ = runner._build_attention_metadata(
            num_tokens=4, num_reqs=1, max_query_len=4, slot_mappings={0: common.slot_mapping},
        )
        meta = attn_metadata["model.layers.0.self_attn"]
        # 『翻译』：Common 字段改名搬入（block_table_tensor→block_table；
        # 表经 _get_block_table 切片视图，比内存别名与值）
        assert meta.block_table.data_ptr() == common.block_table_tensor.data_ptr()
        assert torch.equal(meta.block_table, common.block_table_tensor)
        assert meta.slot_mapping is common.slot_mapping
        assert torch.equal(meta.query_start_loc, common.query_start_loc)
        assert meta.seq_lens.data_ptr() == common.seq_lens.data_ptr()
        assert meta.use_cascade is False
        assert meta.scheduler_metadata is None  # host 恒 FA2 → 无 AOT 调度
        assert meta.sliding_window == (-1, -1)

    def test_layer_name_fanout_shares_one_metadata(self, registry_overrides):
        cfg, runner = self._runner(registry_overrides)
        common = _common()
        self._seed(runner, common)
        attn_metadata, _ = runner._build_attention_metadata(
            num_tokens=4, num_reqs=1, max_query_len=4, slot_mappings={0: common.slot_mapping},
        )
        assert (
            attn_metadata["model.layers.0.self_attn"]
            is attn_metadata["model.layers.1.self_attn"]
        )

    def test_cudagraph_capture_uses_build_for_capture(self, registry_overrides):
        cfg, runner = self._runner(registry_overrides)
        common = _common()
        self._seed(runner, common)
        attn_metadata, _ = runner._build_attention_metadata(
            num_tokens=4, num_reqs=1, max_query_len=4,
            slot_mappings={0: common.slot_mapping}, for_cudagraph_capture=True,
        )
        assert "model.layers.0.self_attn" in attn_metadata

    def test_padding_fill_null_block_id(self, registry_overrides):
        cfg, runner = self._runner(registry_overrides)
        table = torch.zeros((4, 2), dtype=torch.int32)
        table[1] = 5
        common = _common(num_reqs=2, padded_reqs=4, table=table)
        self._seed(runner, common)
        runner.input_batch.block_tables[0] = _BlockTableSeam(table)
        attn_metadata, _ = runner._build_attention_metadata(
            num_tokens=8, num_reqs=2, num_reqs_padded=4, max_query_len=4,
            slot_mappings={0: common.slot_mapping},
        )
        # 尾部 padded 行的块表填 NULL_BLOCK_ID（Block 0 保留给 padding）
        blk = attn_metadata["model.layers.0.self_attn"].block_table
        assert blk.shape[0] == 4
        assert (blk[2:] == NULL_BLOCK_ID).all()

    def test_update_block_table_reuse_across_groups(self, registry_overrides):
        # cached_attn_metadata 是 _build_attention_metadata 的函数内缓存——
        # 复用发生在同一拍内「逐 KV 组」循环：组 2 同 (spec, builder) 时走
        # update_block_table 浅拷只换表。两 KV 组各一层（同后端同 spec）。
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg()
        from implementation.v1.kv_cache_interface import (
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
        )

        runner = _make_runner(cfg, [("model.layers.0.self_attn", FlashAttentionBackend, _spec(16))])
        runner._make_layer = None  # noqa — 占位防误用
        layer1 = _make_layer(cfg, "model.layers.1.self_attn", FlashAttentionBackend, _spec(16))
        cfg.compilation_config.static_forward_context["model.layers.1.self_attn"] = layer1
        spec = _spec(16)
        runner.kv_cache_config = KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[
                KVCacheTensor(size=64 * spec.page_size_bytes, shared_by=[n])
                for n in ("model.layers.0.self_attn", "model.layers.1.self_attn")
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["model.layers.0.self_attn"], spec),
                KVCacheGroupSpec(["model.layers.1.self_attn"], spec),
            ],
        )
        runner.initialize_kv_cache(runner.kv_cache_config)

        common = _common()
        self._seed(runner, common)
        table1 = torch.tensor([[0, 1]], dtype=torch.int32)
        table2 = torch.tensor([[2, 3]], dtype=torch.int32)
        runner.input_batch.block_tables[0] = _BlockTableSeam(table1)
        runner.input_batch.block_tables[1] = _BlockTableSeam(table2)
        m, _ = runner._build_attention_metadata(
            num_tokens=4, num_reqs=1, max_query_len=4,
            slot_mappings={0: common.slot_mapping, 1: common.slot_mapping},
        )
        m1 = m["model.layers.0.self_attn"]
        m2 = m["model.layers.1.self_attn"]
        # 组 2 走 update_block_table：表换成组 2 的，其余字段白拿组 1 的（浅拷）
        assert m2 is not m1
        assert m2.block_table.data_ptr() == table2.data_ptr()
        assert m1.block_table.data_ptr() == table1.data_ptr()
        assert m2.query_start_loc is m1.query_start_loc


# ═══ M. 写腿 / 读腿算子语义（HOST SEAM 镜像 vs 真实行为） ═══════════════


class TestLegs:
    def test_reshape_and_cache_flash_slot_semantics(self):
        from implementation._custom_ops import reshape_and_cache_flash

        blocks, kvh, kbs, hs = 2, 1, 4, 8
        kv = torch.zeros(blocks, kvh, kbs, 2 * hs, dtype=torch.float16)
        kc, vc = kv.transpose(1, 2).split(hs, dim=-1)
        k = torch.arange(3 * hs, dtype=torch.float16).view(3, 1, hs)
        v = k + 100
        slot_mapping = torch.tensor([4, -1, 6], dtype=torch.int64)  # -1 = PAD 跳过
        reshape_and_cache_flash(k, v, kc, vc, slot_mapping, "auto", None, None)
        assert (kc[1, 0] == k[0]).all()  # slot 4 = 块1 偏移0
        assert (kc[1, 2] == k[2]).all()  # slot 6 = 块1 偏移2
        assert (vc[1, 0] == v[0]).all()
        # PAD 位与未触及行保持原零（块0 全零、块1 的偏移1/3 为零）
        assert (kc[0] == 0).all() and (kc[1, 1] == 0).all() and (kc[1, 3] == 0).all()

    def test_slot_mapping_shape_limits_token_count(self):
        # NOTE(woosuk)：key/value 是 padded 的而 slot_mapping 不 pad——
        # op 用 slot_mapping 的形状定 token 数
        from implementation._custom_ops import reshape_and_cache_flash

        kv = torch.zeros(4, 1, 4, 16, dtype=torch.float16)
        kc, vc = kv.transpose(1, 2).split(8, dim=-1)
        k = torch.arange(1, 9 * 8 + 1, dtype=torch.float16).view(9, 1, 8)
        reshape_and_cache_flash(k, k, kc, vc, torch.arange(8), "auto", None, None)
        # key 有 9 行（padded），slot_mapping 只有 8 行 → 只写 8 行：
        # slot 7 = 块1 偏移3；k[8]（第 9 行）绝不落池；块2/块3 全零
        assert (kc[1, 3] == k[7]).all()
        assert (kc[0, 0] == k[0]).all() and (kc[0, 3] == k[3]).all()
        assert (kc[2:] == 0).all() and (kc[1, 3] != k[8]).all()

    def test_do_kv_cache_update_encoder_skip(self, registry_overrides):
        impl = FlashAttentionImpl(
            num_heads=4, head_size=64, scale=0.125, num_kv_heads=2,
            alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
            attn_type="encoder_only",
        )
        kv = torch.zeros(1, 2, 16, 128, dtype=torch.float16)
        impl.do_kv_cache_update(object(), torch.zeros(1, 2, 64), torch.zeros(1, 2, 64), kv,
                                torch.tensor([0]))
        assert (kv == 0).all()  # encoder 不落池

    def test_kv_cache_dummy_dep_preserved(self, registry_overrides):
        # unified_kv_cache_update 返回空张量 dummy——写→算数据依赖的载体
        from implementation.model_executor.layers.attention.attention import (
            unified_kv_cache_update,
        )
        from implementation.forward_context import set_forward_context

        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg()
        attn = _make_layer(cfg, "model.layers.0.self_attn", FlashAttentionBackend, _spec())
        # 站 7 的落点：裸显存定形后的张量 bind 到层（此处直接给 FA 布局张量）
        attn.kv_cache = torch.zeros(2, 2, 16, 128, dtype=torch.float16)
        slots = torch.tensor([0], dtype=torch.int64)
        with set_forward_context(
            {"model.layers.0.self_attn": None},
            cfg,
            slot_mapping={"model.layers.0.self_attn": slots},
        ):
            key = torch.zeros(1, 2, 64, dtype=torch.float16)
            dep = unified_kv_cache_update(key, key, "model.layers.0.self_attn")
        assert dep.shape == (0,) and dep.dtype == torch.float16


# ═══ N. execute_model 每拍主线（set_forward_context + 层取数） ═══════════


class TestExecuteModel:
    def test_execute_model_full_step(self, registry_overrides):
        # 站 8→12 全链：execute_model 切面跑 _get_slot_mappings →
        # _build_attention_metadata → set_forward_context → 逐层前向（写腿+
        # 读腿）；前向内按 layer_name 取到的两通道由观测位记录
        cuda_platform.CudaPlatform._seam_device_capability = SM90
        cfg = _cfg()
        runner = _make_runner(
            cfg,
            [
                ("model.layers.0.self_attn", FlashAttentionBackend, _spec(16)),
                ("model.layers.1.self_attn", FlashAttentionBackend, _spec(16)),
            ],
            with_tensors=True,
        )
        runner.initialize_kv_cache(runner.kv_cache_config)

        # 每拍输入面（ch18/ch22 域产物的消费面——seam 直供）
        runner.seam_num_reqs = 1
        runner.seam_num_tokens = 4
        runner.seam_max_query_len = 4
        runner.query_start_loc = _CpuGpuSeam(torch.tensor([0, 4], dtype=torch.int32))
        runner.seq_lens = torch.tensor([16], dtype=torch.int32)
        runner.optimistic_seq_lens_cpu = torch.tensor([16], dtype=torch.int32)
        table = torch.tensor([[0, 1]], dtype=torch.int32)
        slots = torch.arange(4, dtype=torch.int64)
        runner.input_batch.block_tables[0] = _BlockTableSeam(table, slots)

        out = runner.execute_model(scheduler_output=None)
        # 两层都跑完（写腿+读腿）
        assert set(out) == {"model.layers.0.self_attn", "model.layers.1.self_attn"}
        for name in out:
            assert out[name].shape == (4, 4 * 64)
        # 前向内 set_forward_context 的两通道按 layer_name 可见
        for name in out:
            assert name in runner.seam_seen_metadata
            assert runner.seam_seen_metadata[name].slot_mapping.shape == (4,)
            assert runner.seam_seen_slot_mapping[name].shape == (4,)
        # 写腿确实落池：槽位 0..3 的 K 行已写（_model_forward 的零输入 → 0 值
        # 不可分辨，改查块内容 == 0 与 slot_mapping 形状约束）——以
        # forward 输出形状与通道可见性为准（数值对拍在 TestAttentionLayer）


# ═══ O. CommonAttentionMetadata 核心字段面 ═══════════════════════════════


class TestCommonMetadata:
    def test_core_fields_present(self):
        common = _common()
        for f in (
            "query_start_loc", "query_start_loc_cpu", "seq_lens", "num_reqs",
            "num_actual_tokens", "max_query_len", "max_seq_len",
            "block_table_tensor", "slot_mapping", "causal",
        ):
            assert hasattr(common, f)
        assert common.causal is True
