"""ch24 测试 — 复现真实 vLLM 的可观察行为（纯 host，不 import vllm）。

覆盖：
  1. KV cache shape / stride_order 约定（FlashAttentionBackend.get_kv_cache_shape /
     get_kv_cache_stride_order / get_kv_cache_block_dim）。
  2. validate_configuration 能力探针聚合（空列表=合法）。
  3. 注册表懒加载 get_class() + register_backend 覆盖 + CUSTOM 占位。
  4. selector 选后端：Hopper 默认选 FLASH_ATTN；显式指定不合法后端报错；@cache 命中。
  5. Common → FlashAttentionMetadata 的『翻译』（block_table=block_table_tensor 等）。
  6. PagedAttention 写（reshape_and_cache_flash 照 slot_mapping 散写）。
  7. PagedAttention 读（flash_attn_varlen_func 照 block_table 读，单请求对照稠密注意力）。
  8. f18 端到端：Attention 层经 forward_context 按 layer_name 取 kv_cache/metadata 写+读。
"""

import torch

import backend as backend_mod
from backend import AttentionType, CommonAttentionMetadata
from registry import AttentionBackendEnum, register_backend, resolve_obj_by_qualname
from selector import (
    AttentionSelectorConfig,
    _cached_get_attn_backend,
    get_attn_backend,
)
from platform_cuda import CudaPlatform, DeviceCapability
from flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
    flash_attn_varlen_func,
    get_kv_cache_layout,
    reshape_and_cache_flash,
    set_kv_cache_layout,
)


class _OverrideBackend(FlashAttentionBackend):
    """模块级 importable 后端类，供 register_backend 覆盖测试用。"""

    @staticmethod
    def get_name() -> str:
        return "MY_FA"


# ---------------------------------------------------------------------------
# 1. KV cache shape / stride 约定
# ---------------------------------------------------------------------------
def test_kv_cache_shape_is_2_nblocks_bsize_kvheads_headsize():
    shape = FlashAttentionBackend.get_kv_cache_shape(
        num_blocks=10, block_size=16, num_kv_heads=4, head_size=64
    )
    assert shape == (2, 10, 16, 4, 64)


def test_kv_cache_shape_rejects_non_multiple_of_16_block():
    try:
        FlashAttentionBackend.get_kv_cache_shape(10, 17, 4, 64)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_stride_order_nhd_identity_hnd_swaps_block_and_head():
    set_kv_cache_layout("NHD")
    assert FlashAttentionBackend.get_kv_cache_stride_order() == (0, 1, 2, 3, 4)
    set_kv_cache_layout("HND")
    # HND 把 block_size(dim2) 与 num_kv_heads(dim3) 互换 → 同 head 的 token 连续。
    assert FlashAttentionBackend.get_kv_cache_stride_order() == (0, 1, 3, 2, 4)
    set_kv_cache_layout("NHD")  # 还原


def test_block_dim_is_discovered_as_dim1():
    # 逻辑 shape 的 num_blocks 维在 index 1。
    assert FlashAttentionBackend.get_kv_cache_block_dim(16, 4, 64) == 1


# ---------------------------------------------------------------------------
# 2. validate_configuration
# ---------------------------------------------------------------------------
def _cfg(**kw) -> dict:
    base = dict(
        head_size=64,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        attn_type=AttentionType.DECODER,
    )
    base.update(kw)
    return base


def test_validate_configuration_empty_means_valid():
    reasons = FlashAttentionBackend.validate_configuration(
        device_capability=DeviceCapability(9, 0), **_cfg()
    )
    assert reasons == []


def test_validate_configuration_flags_unsupported_head_size():
    reasons = FlashAttentionBackend.validate_configuration(
        device_capability=DeviceCapability(9, 0), **_cfg(head_size=65)
    )
    assert "head_size not supported" in reasons


def test_validate_configuration_flags_non_mla_mismatch_and_low_capability():
    # use_mla=True 但 FA 是非 MLA → "MLA not supported"
    reasons = FlashAttentionBackend.validate_configuration(
        device_capability=DeviceCapability(9, 0), **_cfg(use_mla=True)
    )
    assert "MLA not supported" in reasons
    # compute capability 7.5 < 8.0 → 不支持
    reasons2 = FlashAttentionBackend.validate_configuration(
        device_capability=DeviceCapability(7, 5), **_cfg()
    )
    assert "compute capability not supported" in reasons2


# ---------------------------------------------------------------------------
# 3. 注册表懒加载 + 覆盖
# ---------------------------------------------------------------------------
def test_get_class_lazy_resolves_flash_attn():
    cls = AttentionBackendEnum.FLASH_ATTN.get_class()
    assert cls is FlashAttentionBackend
    assert cls.get_name() == "FLASH_ATTN"


def test_register_backend_overrides_then_clears():
    # 用 register_backend 把 FLASH_ATTN 指向另一个 importable 类（这里复用 _Override 模块级类），
    # 验证 get_class() 走覆盖表而非枚举默认值；clear_override 后回退默认。
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        f"{_OverrideBackend.__module__}.{_OverrideBackend.__qualname__}",
    )
    try:
        assert AttentionBackendEnum.FLASH_ATTN.is_overridden()
        assert AttentionBackendEnum.FLASH_ATTN.get_class() is _OverrideBackend
        assert AttentionBackendEnum.FLASH_ATTN.get_class().get_name() == "MY_FA"
    finally:
        AttentionBackendEnum.FLASH_ATTN.clear_override()
    assert not AttentionBackendEnum.FLASH_ATTN.is_overridden()
    assert AttentionBackendEnum.FLASH_ATTN.get_class() is FlashAttentionBackend


def test_custom_backend_requires_registration():
    try:
        AttentionBackendEnum.CUSTOM.get_path()
        assert False, "expected ValueError for unregistered CUSTOM"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 4. selector / platform 选后端
# ---------------------------------------------------------------------------
def test_hopper_auto_selects_flash_attn():
    _cached_get_attn_backend.cache_clear()
    CudaPlatform._device_capability = DeviceCapability(9, 0)
    backend = get_attn_backend(
        head_size=64, dtype=torch.bfloat16, kv_cache_dtype="auto"
    )
    assert backend is FlashAttentionBackend


def test_explicit_invalid_backend_raises():
    _cached_get_attn_backend.cache_clear()
    CudaPlatform._device_capability = DeviceCapability(9, 0)
    # 显式指定 FLASH_ATTN 但 head_size 非法 → 直接报错（不回退自动选择）。
    try:
        get_attn_backend(
            head_size=65,
            dtype=torch.bfloat16,
            kv_cache_dtype="auto",
            backend=AttentionBackendEnum.FLASH_ATTN,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_selector_config_is_hashable_cache_key():
    c1 = AttentionSelectorConfig(64, torch.bfloat16, "auto", 16)
    c2 = AttentionSelectorConfig(64, torch.bfloat16, "auto", 16)
    assert hash(c1) == hash(c2)
    assert c1 == c2


# ---------------------------------------------------------------------------
# 5. Common → FlashAttentionMetadata 翻译
# ---------------------------------------------------------------------------
def _make_common(num_reqs, q_lens, kv_lens, block_table, slot_mapping):
    qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)))
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(kv_lens),
        num_reqs=num_reqs,
        num_actual_tokens=int(sum(q_lens)),
        max_query_len=max(q_lens),
        max_seq_len=max(kv_lens),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
    )


def test_build_translates_common_to_fa_metadata():
    block_table = torch.tensor([[0, 1]])
    slot_mapping = torch.tensor([0, 1, 2])
    common = _make_common(1, [3], [3], block_table, slot_mapping)
    builder = FlashAttentionMetadataBuilder()
    md = builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert isinstance(md, FlashAttentionMetadata)
    # 共享字段直接搬：block_table_tensor → block_table，slot_mapping 原样搬。
    assert torch.equal(md.block_table, block_table)
    assert torch.equal(md.slot_mapping, slot_mapping)
    assert torch.equal(md.seq_lens, common.seq_lens)
    # FA 特有字段被新增（默认走标准非 cascade 路径）。
    assert md.use_cascade is False
    assert md.scheduler_metadata is None


# ---------------------------------------------------------------------------
# 6. PagedAttention 写
# ---------------------------------------------------------------------------
def test_reshape_and_cache_flash_writes_by_slot_mapping():
    num_blocks, block_size, num_kv_heads, head_size = 4, 16, 2, 8
    kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_size)
    key_cache, value_cache = kv_cache.unbind(0)
    num_tokens = 3
    key = torch.randn(num_tokens, num_kv_heads, head_size)
    value = torch.randn(num_tokens, num_kv_heads, head_size)
    # slot 0→block0 off0, slot 17→block1 off1, slot -1→跳过(padding)。
    slot_mapping = torch.tensor([0, 17, -1])
    reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, "auto", None, None
    )
    assert torch.equal(key_cache[0, 0], key[0])
    assert torch.equal(key_cache[1, 1], key[1])
    assert torch.equal(value_cache[1, 1], value[1])
    # slot==-1 的 token 不写入，块 0 off1 仍为零。
    assert torch.count_nonzero(key_cache[0, 1]) == 0


# ---------------------------------------------------------------------------
# 7. PagedAttention 读 — 单请求对照稠密注意力
# ---------------------------------------------------------------------------
def test_flash_attn_varlen_matches_dense_attention_single_request():
    torch.manual_seed(0)
    block_size, num_kv_heads, head_size = 16, 2, 8
    num_heads = 2  # MHA(queries_per_kv=1) 便于对照
    seq_len = 20   # 跨两个块
    num_blocks = 4

    # 造连续的 K/V，再按块写进 paged cache（block_table=[0,1]）。
    key = torch.randn(seq_len, num_kv_heads, head_size)
    value = torch.randn(seq_len, num_kv_heads, head_size)
    kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_size)
    key_cache, value_cache = kv_cache.unbind(0)
    slot_mapping = torch.arange(seq_len)  # slot i → block i//16, off i%16
    reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, "auto", None, None
    )

    query = torch.randn(seq_len, num_heads, head_size)
    out = torch.empty(seq_len, num_heads, head_size)
    scale = 1.0 / (head_size ** 0.5)
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=torch.tensor([0, seq_len]),
        max_seqlen_q=seq_len,
        seqused_k=torch.tensor([seq_len]),
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
        block_table=torch.tensor([[0, 1, 2, 3]]),
    )

    # 稠密参考：标准 causal 自注意力（prefill，q_len==kv_len）。
    ref = torch.empty(seq_len, num_heads, head_size)
    for h in range(num_heads):
        s = (query[:, h] @ key[:, h].T) * scale
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        ref[:, h] = p @ value[:, h]
    assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. f18 端到端：Attention 层经 forward_context 按 layer_name 写+读
# ---------------------------------------------------------------------------
def test_attention_layer_dispatch_by_layer_name_end_to_end():
    import attention_layer as al

    _cached_get_attn_backend.cache_clear()
    CudaPlatform._device_capability = DeviceCapability(9, 0)

    num_heads = 2
    head_size = 8
    block_size, num_kv_heads, num_blocks = 16, 2, 4
    seq_len = 20

    # 真实 vLLM 在 bf16/fp16 默认 dtype 下构建模型（FA 只支持这两种 dtype），__init__ 读
    # torch.get_default_dtype() 选后端；故在 bfloat16 默认 dtype 下构建层，再用 float32 张量做
    # 前向以便数值稳定地对照。
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        layer = al.Attention(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0 / (head_size ** 0.5),
            num_kv_heads=num_kv_heads,
            kv_cache_dtype="auto",
            prefix="layer.0.attn",
        )
    finally:
        torch.set_default_dtype(old_dtype)
    assert layer.attn_backend is FlashAttentionBackend
    assert isinstance(layer.impl, FlashAttentionImpl)

    kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_size)
    layer.kv_cache = kv_cache

    common = _make_common(
        1, [seq_len], [seq_len],
        torch.tensor([[0, 1, 2, 3]]), torch.arange(seq_len),
    )
    builder = FlashAttentionMetadataBuilder()
    md = builder.build(0, common)

    # model_runner 那头：按 layer_name 把 metadata/slot_mapping 装进 forward_context。
    ctx = al.get_forward_context()
    ctx.attn_metadata = {"layer.0.attn": md}
    ctx.slot_mapping = {"layer.0.attn": common.slot_mapping}

    query = torch.randn(seq_len, num_heads, head_size)
    key = torch.randn(seq_len, num_kv_heads, head_size)
    value = torch.randn(seq_len, num_kv_heads, head_size)
    out = layer.forward(query, key, value)

    # 写半边生效：KV 已照 slot_mapping 进 paged cache（block0 off0 == key[0]）。
    assert torch.equal(kv_cache[0, 0, 0], key[0])
    # 读半边产出形状正确、非全零。
    assert out.shape == (seq_len, num_heads * head_size)
    assert torch.count_nonzero(out) > 0


def test_resolve_obj_by_qualname_roundtrip():
    obj = resolve_obj_by_qualname("flash_attn.FlashAttentionBackend")
    assert obj is FlashAttentionBackend
    # 引用 backend 模块符号确保抽象核心可 import。
    assert backend_mod.AttentionBackend is not None
