"""ch16 — KV cache 在昇腾上的分配 / reshape / 绑定：可观察控制流测试。

测的是「精简版复现真实 vllm-ascend 的可观察行为」，不是测精简版自洽：
  * 对齐算术 _align_up / _align_memory（2MB 边界）
  * int8 裸分配 _allocate_int8_cache_tensor（含 kv_transfer 对齐分支）
  * sparse-c8 indexer 双视图 _allocate_sparse_c8_indexer_tensors（共享一块对齐内存）
  * K/V 字节拆分 calc_split_factor（head 维度占比）
  * NPU 物理布局 _adjust_kv_layout（block 维 stride = page_size_bytes//dtype_size）
  * _allocate→_reshape 标准 attn / MLA 的 int8→typed 还原与 nope/rope 拆分
  * bind 三分支派发（deepseek_v4 层序 / longcat num_attn_module=2 / 普通）
  * may_reinitialize_input_batch 的 kernel_block_sizes 装配与重建判定
  * get_kv_cache_spec 的 MLA → AscendMLAAttentionSpec
"""
import types

import torch


# --------------------------------------------------------------------------- #
# 构造一个「裸」NPUModelRunner：跳过 __init__，只挂本章方法用到的属性。
# --------------------------------------------------------------------------- #
def _bare_runner(mr, **attrs):
    r = mr.NPUModelRunner.__new__(mr.NPUModelRunner)
    defaults = dict(
        device=torch.device("cpu"),
        use_sparse=False,
        use_compress=False,
        use_hybrid_blocks=False,
        hybrid_with_attn_and_mamba=False,
        enable_hamming_sparse=False,
        runner_only_attn_layers=set(),
        shared_kv_cache_layers={},
        kv_caches=[],
    )
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(r, k, v)
    return r


def _vllm_config(kv_transfer=None, speculative=None):
    return types.SimpleNamespace(
        kv_transfer_config=kv_transfer,
        speculative_config=speculative,
        quant_config=None,
    )


# --------------------------------------------------------------------------- #
# 对齐原语
# --------------------------------------------------------------------------- #
def test_align_up_ceils_to_multiple(env):
    up = env.mr.NPUModelRunner._align_up
    assert up(0, 4) == 0
    assert up(1, 4) == 4
    assert up(4, 4) == 4
    assert up(5, 4) == 8
    # 2MB 对齐：恰好一个大页 → 不变；多 1 字节 → 抬一整页
    M = 2 * 1024 * 1024
    assert up(M, M) == M
    assert up(M + 1, M) == 2 * M


def test_align_memory_lands_on_alignment_boundary(env):
    r = _bare_runner(env.mr)
    alignment = 4096
    raw = torch.zeros(1024 + alignment, dtype=torch.int8)
    aligned = r._align_memory(raw, alignment)
    assert aligned.data_ptr() % alignment == 0
    # 起点抬高后仍留有 >= 原始可用长度
    assert aligned.numel() >= 1024


# --------------------------------------------------------------------------- #
# int8 裸分配
# --------------------------------------------------------------------------- #
def test_allocate_int8_no_kv_transfer(env):
    r = _bare_runner(env.mr, vllm_config=_vllm_config(kv_transfer=None))
    t = r._allocate_int8_cache_tensor(640, 2 * 1024 * 1024)
    assert t.dtype == torch.int8
    assert t.shape == (640,)


def test_allocate_int8_with_kv_transfer_aligns_addr(env):
    r = _bare_runner(env.mr, vllm_config=_vllm_config(kv_transfer=object()))
    alignment = 2 * 1024 * 1024
    t = r._allocate_int8_cache_tensor(777, alignment)
    assert t.numel() == 777
    assert t.dtype == torch.int8
    assert t.data_ptr() % alignment == 0


def test_allocate_int8_rejects_nonpositive(env):
    r = _bare_runner(env.mr, vllm_config=_vllm_config())
    import pytest
    with pytest.raises(ValueError):
        r._allocate_int8_cache_tensor(0, 64)


# --------------------------------------------------------------------------- #
# sparse-c8 indexer 双视图
# --------------------------------------------------------------------------- #
def test_sparse_c8_indexer_two_views_share_storage(env):
    r = _bare_runner(env.mr, vllm_config=_vllm_config(kv_transfer=None))
    dsa_k_size, scale_size = 100, 16
    scale_dtype = torch.float16  # 2 bytes
    dsa_k, dsa_k_scale = r._allocate_sparse_c8_indexer_tensors(
        dsa_k_tensor_size=dsa_k_size,
        dsa_k_scale_tensor_size=scale_size,
        alignment=2 * 1024 * 1024,
        scale_dtype=scale_dtype,
    )
    assert dsa_k.numel() == dsa_k_size
    assert dsa_k_scale.numel() == scale_size
    # 两段是同一块裸内存的视图：scale 起点 = base + _align_up(dsa_k_size, 2)
    base = dsa_k.data_ptr()
    expected_scale_offset = env.mr.NPUModelRunner._align_up(dsa_k_size, 2)
    assert dsa_k_scale.data_ptr() == base + expected_scale_offset
    # scale 视图首地址按 scale_dtype 大小对齐
    assert dsa_k_scale.data_ptr() % 2 == 0
    assert dsa_k.is_contiguous() and dsa_k_scale.is_contiguous()


# --------------------------------------------------------------------------- #
# K/V 字节拆分
# --------------------------------------------------------------------------- #
def test_calc_split_factor_reflects_head_ratio(env):
    calc = env.utils.calc_split_factor
    # GQA: k_dim == v_dim → 对半
    assert calc([64, 64]) == [2.0, 2.0]
    # MLA: nope(512) / rope(64) 不等
    fk, fv = calc([512, 64])
    assert fk == 576 / 512 and fv == 576 / 64
    # size // factor 给出各自字节占比
    size = 5760
    assert int(size // fk) == size * 512 // 576
    assert int(size // fv) == size * 64 // 576


# --------------------------------------------------------------------------- #
# NPU 物理布局重排
# --------------------------------------------------------------------------- #
def test_adjust_kv_layout_sets_page_stride(env):
    r = _bare_runner(env.mr)
    raw = torch.zeros(8192, dtype=torch.int8)
    shape = (4, 2, 1, 8)          # (num_blocks, block_size, num_kv_heads, head_size)
    dtype = torch.float16          # 2 bytes
    # page_size_bytes 选成「每 block 16 个 float16 元素」 → stride[0] 应 = 16
    page_size_bytes = 16 * 2
    out = r._adjust_kv_layout(raw, [shape], [dtype], page_size_bytes)
    assert len(out) == 1
    t = out[0]
    assert tuple(t.shape) == shape
    assert t.dtype == dtype
    # block(第0)维 stride 被强制为 page_size_bytes // dtype_size
    assert t.stride()[0] == page_size_bytes // 2
    # 其余维保持自然 stride
    natural = torch.empty(shape).stride()
    assert t.stride()[1:] == natural[1:]


# --------------------------------------------------------------------------- #
# _allocate → _reshape 标准 attn 整链（非 sparse）
# --------------------------------------------------------------------------- #
def _make_kv_cache_config(env, layer="model.layers.0.attn", spec=None, num_blocks=4):
    head_size = 8
    block_size = 2
    num_kv_heads = 1
    elem = 2  # float16
    page_size_bytes = block_size * num_kv_heads * head_size * elem * 2  # K+V 同页
    if spec is None:
        spec = env.AttentionSpec(block_size=block_size, num_kv_heads=num_kv_heads,
                                 head_size=head_size, dtype=torch.float16,
                                 page_size_bytes=page_size_bytes)
    total_bytes = num_blocks * page_size_bytes
    kv_tensor = types.SimpleNamespace(size=total_bytes, shared_by=[layer])
    group = types.SimpleNamespace(kv_cache_spec=spec, layer_names=[layer])
    cfg = types.SimpleNamespace(
        kv_cache_tensors=[kv_tensor],
        kv_cache_groups=[group],
        num_blocks=num_blocks,
    )
    return cfg, spec, layer, num_blocks, (num_blocks, block_size, num_kv_heads, head_size)


def test_allocate_then_reshape_standard_attn(env):
    cfg, spec, layer, num_blocks, kv_shape = _make_kv_cache_config(env)
    r = _bare_runner(
        env.mr,
        vllm_config=_vllm_config(kv_transfer=None),
    )
    raw = r._allocate_kv_cache_tensors(cfg)
    # 标准 attn：拆成 (k_tensor, v_tensor) 两块独立 int8
    assert layer in raw
    k_raw, v_raw = raw[layer]
    assert k_raw.dtype == torch.int8 and v_raw.dtype == torch.int8
    # GQA k_dim==v_dim → 对半
    assert k_raw.numel() == v_raw.numel()
    assert k_raw.numel() + v_raw.numel() == num_blocks * spec.page_size_bytes

    # reshape：把裸字节 .view(dtype).view(shape) 还原成带 dtype 的 KV
    # 真实 backend.get_kv_cache_shape 返回前置 size-2 的 K/V 维 (2, nb, bs, nh, hs)；
    # 昇腾把 K、V 拆成独立张量，各取 shape[1:]（去掉那个 size-2 KV 维）。
    backend = types.SimpleNamespace(
        get_kv_cache_shape=lambda nb, bs, nh, hs: (2, nb, bs, nh, hs)
    )
    grp = types.SimpleNamespace(backend=backend, kv_cache_spec=spec, layer_names=[layer])
    r.attn_backend = backend
    r._kv_cache_spec_attn_group_iterator = lambda: [grp]
    kv = r._reshape_kv_cache_tensors(cfg, raw)
    k_cache, v_cache = kv[layer]
    assert k_cache.dtype == torch.float16
    assert tuple(k_cache.shape) == kv_shape       # (num_blocks, block_size, num_kv_heads, head_size)
    assert tuple(v_cache.shape) == kv_shape


# --------------------------------------------------------------------------- #
# bind 三分支派发
# --------------------------------------------------------------------------- #
def _stub_alloc_reshape(r, kv_caches):
    r._allocate_kv_cache_tensors = lambda cfg: {"raw": None}
    r._reshape_kv_cache_tensors = lambda cfg, raw: dict(kv_caches)


def test_bind_normal_model_uses_num_attn_module_1(env):
    calls = {}
    env.wutils.bind_kv_cache = lambda kc, ctx, runner_kc, n: calls.update(n=n, kc=kc)
    r = _bare_runner(
        env.mr,
        model_config=types.SimpleNamespace(hf_text_config=types.SimpleNamespace(model_type="qwen3")),
        compilation_config=types.SimpleNamespace(static_forward_context={}),
    )
    _stub_alloc_reshape(r, {"l0": ("k", "v")})
    r.initialize_kv_cache_tensors(object())
    assert calls["n"] == 1


def test_bind_longcat_uses_num_attn_module_2(env):
    calls = {}
    env.wutils.bind_kv_cache = lambda kc, ctx, runner_kc, n: calls.update(n=n)
    r = _bare_runner(
        env.mr,
        model_config=types.SimpleNamespace(hf_text_config=types.SimpleNamespace(model_type="longcat_flash")),
        compilation_config=types.SimpleNamespace(static_forward_context={}),
    )
    _stub_alloc_reshape(r, {"l0": ("k", "v")})
    r.initialize_kv_cache_tensors(object())
    assert calls["n"] == 2


def test_bind_deepseek_v4_uses_custom_layer_order(env):
    # deepseek_v4 走自定层序：extract_dsv4_layer_index 把 mtp 层排到主模型层之后
    cfg = types.SimpleNamespace(num_hidden_layers=2)
    ctx = {name: types.SimpleNamespace(kv_cache=None)
           for name in ["model.layers.1.attn", "model.layers.0.attn", "mtp.0.attn"]}
    r = _bare_runner(
        env.mr,
        model_config=types.SimpleNamespace(hf_text_config=cfg),
        compilation_config=types.SimpleNamespace(static_forward_context=ctx),
        kv_caches=[],
    )
    cfg.model_type = "deepseek_v4"
    kvs = {
        "model.layers.0.attn": "kv0",
        "model.layers.1.attn": "kv1",
        "mtp.0.attn": "kv_mtp",
    }
    _stub_alloc_reshape(r, kvs)
    # extract_dsv4_layer_index: layer0→0, layer1→1, mtp.0→num_hidden_layers(2)+0=2
    r.initialize_kv_cache_tensors(object())
    assert r.kv_caches == ["kv0", "kv1", "kv_mtp"]
    # 每层的 static_forward_context.kv_cache 被填成 [kv]
    assert ctx["mtp.0.attn"].kv_cache == ["kv_mtp"]


# --------------------------------------------------------------------------- #
# may_reinitialize_input_batch
# --------------------------------------------------------------------------- #
def _runner_for_reinit(env, groups, cache_block_size, attn_groups):
    r = _bare_runner(
        env.mr,
        cache_config=types.SimpleNamespace(block_size=cache_block_size, enable_prefix_caching=False),
        attn_groups=attn_groups,
        max_model_len=128,
        max_encoder_len=0,
        max_num_reqs=8,
        max_num_tokens=256,
        pin_memory=False,
        is_pooling_model=False,
        vllm_config=_vllm_config(speculative=None),
        input_batch=types.SimpleNamespace(logitsprocs=None),
        model_config=types.SimpleNamespace(get_vocab_size=lambda: 1000),
    )
    cfg = types.SimpleNamespace(kv_cache_groups=groups)
    return r, cfg


def test_reinit_skipped_when_uniform_block_size(env):
    # 单 group 且 block_size == cache_config.block_size 且 kernel==cache → 不重建
    spec = env.AttentionSpec(block_size=128)
    grp = types.SimpleNamespace(kv_cache_spec=spec)
    backend = object()
    r, cfg = _runner_for_reinit(env, [grp], 128, [[types.SimpleNamespace(backend=backend)]])
    env.mr.select_common_block_size = lambda kv_block, backends: 128
    sentinel = r.input_batch
    r.may_reinitialize_input_batch(cfg)
    assert r.kernel_block_sizes == [[128]]
    assert r.input_batch is sentinel  # 未重建

def test_reinit_triggered_on_block_size_mismatch(env):
    # kernel block size 与 cache_config.block_size 不一致（如 ch04 16→128）→ 重建 NPUInputBatch
    spec = env.AttentionSpec(block_size=128)
    grp = types.SimpleNamespace(kv_cache_spec=spec)
    backend = object()
    r, cfg = _runner_for_reinit(env, [grp], 128, [[types.SimpleNamespace(backend=backend)]])
    env.mr.select_common_block_size = lambda kv_block, backends: 16
    r.may_reinitialize_input_batch(cfg)
    assert r.kernel_block_sizes == [[16]]
    # 触发重建
    assert isinstance(r.input_batch, env.nib.NPUInputBatch)
    assert r.input_batch.kwargs["kernel_block_sizes"] == [[16]]
    assert r.input_batch.kwargs["block_sizes"] == [128]


def test_reinit_mamba_group_uses_zero_kernel_block(env):
    # 非 attention（mamba）spec → kernel_block_sizes 用 [0] 关闭 slot mapping
    mspec = env.MambaSpec(shapes=[(4,)], dtypes=[torch.float16], page_size_bytes=64, block_size=128)
    grp = types.SimpleNamespace(kv_cache_spec=mspec)
    r, cfg = _runner_for_reinit(env, [grp], 128, [[]])
    r.may_reinitialize_input_batch(cfg)
    assert r.kernel_block_sizes == [[0]]


# --------------------------------------------------------------------------- #
# get_kv_cache_spec：MLA → AscendMLAAttentionSpec
# --------------------------------------------------------------------------- #
def test_get_kv_cache_spec_mla_uses_ascend_spec(env):
    layer = "model.layers.0.mla"

    class _MLAModule(env.attn.MLAAttention):
        def get_kv_cache_spec(self, vllm_config):
            return env.MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576,
                                        dtype=torch.float16, cache_dtype_str="auto")

    mla_mod = _MLAModule()
    mla_mod.impl = types.SimpleNamespace(fa_quant_layer=False)
    # 让 get_layers_from_vllm_config 返回我们的 MLA module
    env.mr.get_layers_from_vllm_config = lambda *_a, **_k: {layer: mla_mod}

    r = _bare_runner(
        env.mr,
        use_sparse=False,
        vllm_config=_vllm_config(),
    )
    spec = r.get_kv_cache_spec()
    assert layer in spec
    built = spec[layer]
    # 重建为（被 patch 的）MLAAttentionSpec，head_size/dtype 透传
    assert isinstance(built, env.MLAAttentionSpec)
    assert built.head_size == 576
    assert built.dtype == torch.float16
