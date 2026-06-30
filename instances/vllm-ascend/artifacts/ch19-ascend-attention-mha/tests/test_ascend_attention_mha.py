"""ch19 —— 标准 MHA 的 NPU 内核与状态机：验证精简版复现 vllm-ascend 的可观察控制流。

测的是「精简版与真仓 attention_v1.py 的控制流一致」，不是精简版自洽：
五态机分流 / split_decodes_and_prefills 拆批 / build 装配 / forward_impl 选 paged vs fused /
reshape_and_cache 写 KV / get_impl_cls 按 enable_cp 收口 f7 / workspace 预取节拍。
真实 torch_npu 算子由记录替身承接（host 无 NPU）。
"""
import torch

from conftest import make_common_metadata


def _state(mods):
    return mods.attention_v1.AscendAttentionState


def _make_impl(mods, *, num_heads=4, head_size=8, num_kv_heads=2, sliding_window=None, attn_type="decoder"):
    return mods.attention_v1.AscendAttentionBackendImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=0.125,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=attn_type,
        kv_sharing_target_layer_name=None,
    )


def _make_kv_cache(num_blocks=8, block_size=128, num_kv_heads=2, head_size=8):
    return torch.zeros(2, num_blocks, block_size, num_kv_heads, head_size)


def _meta(mods, **kw):
    return mods.attention_v1.AscendMetadata(**kw)


# ----------------------------- (1) 五态机 ----------------------------- #

def test_state_machine_five_members(env):
    mods, _ = env
    S = _state(mods)
    assert S.PrefillNoCache.value == 0
    assert S.PrefillCacheHit.value == 1
    assert S.DecodeOnly.value == 2
    assert S.ChunkedPrefill.value == 3
    assert S.SpecDecoding.value == 4
    assert [s.name for s in S] == [
        "PrefillNoCache", "PrefillCacheHit", "DecodeOnly", "ChunkedPrefill", "SpecDecoding",
    ]


# ------------------- (2) split_decodes_and_prefills ------------------- #

def test_split_all_decode(env):
    mods, _ = env
    cm = make_common_metadata(mods, query_lens=[1, 1, 1], attn_state=_state(mods).DecodeOnly)
    out = mods.utils.split_decodes_and_prefills(cm, decode_threshold=1)
    # max_query_len <= threshold → 全是 decode
    assert out == (3, 0, 3, 0)


def test_split_mixed_decode_then_prefill(env):
    mods, _ = env
    cm = make_common_metadata(mods, query_lens=[1, 1, 5, 7], attn_state=_state(mods).ChunkedPrefill)
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        mods.utils.split_decodes_and_prefills(cm, decode_threshold=1)
    )
    assert num_decodes == 2
    assert num_prefills == 2
    assert num_decode_tokens == 2          # query_start_loc[2]
    assert num_prefill_tokens == 12        # 14 - 2


def test_split_all_prefill(env):
    mods, _ = env
    cm = make_common_metadata(mods, query_lens=[5, 7], attn_state=_state(mods).ChunkedPrefill)
    assert mods.utils.split_decodes_and_prefills(cm, decode_threshold=1) == (0, 2, 0, 12)


def test_split_spec_threshold(env):
    mods, _ = env
    # 投机解码：decode_threshold=2，query_len<=2 算 decode
    cm = make_common_metadata(mods, query_lens=[2, 2, 8], attn_state=_state(mods).SpecDecoding)
    num_decodes, num_prefills, num_decode_tokens, _ = mods.utils.split_decodes_and_prefills(cm, decode_threshold=2)
    assert (num_decodes, num_prefills, num_decode_tokens) == (2, 1, 4)


# --------------------- (3) build 元数据装配 --------------------------- #

def _make_builder(mods, knobs):
    return mods.attention_v1.AscendAttentionMetadataBuilder(
        kv_cache_spec=None,
        layer_names=["layer.0"],
        vllm_config=knobs.vllm_config,
        device=torch.device("cpu"),
    )


def test_build_assembles_metadata(env):
    mods, knobs = env
    builder = _make_builder(mods, knobs)
    cm = make_common_metadata(mods, query_lens=[1, 1, 5, 7], attn_state=_state(mods).ChunkedPrefill)
    md = builder.build(common_prefix_len=0, common_attn_metadata=cm)

    assert md.attn_state == _state(mods).ChunkedPrefill
    assert md.num_actual_tokens == 14
    assert md.slot_mapping.tolist() == list(range(14))      # slot_mapping[:num_actual_tokens]
    assert md.block_tables.shape == (4, 4)                   # block_table_tensor 透传
    assert md.num_decodes == 2 and md.num_prefills == 2
    assert md.seq_lens_list == [1, 1, 5, 7]
    assert md.actual_seq_lengths_q == [1, 2, 7, 14]          # query_start_loc[1:]
    assert md.attn_mask.shape == (2048, 2048)                # splitfuse mask 单例


def test_build_for_graph_capture_allowed_states(env):
    mods, knobs = env
    builder = _make_builder(mods, knobs)
    cm = make_common_metadata(mods, query_lens=[1, 1], attn_state=_state(mods).DecodeOnly)
    md = builder.build_for_graph_capture(cm, attn_state=_state(mods).SpecDecoding)
    assert md.attn_state == _state(mods).SpecDecoding


def test_build_for_graph_capture_rejects_prefill_nocache(env):
    mods, knobs = env
    builder = _make_builder(mods, knobs)
    cm = make_common_metadata(mods, query_lens=[1, 1], attn_state=_state(mods).DecodeOnly)
    try:
        builder.build_for_graph_capture(cm, attn_state=_state(mods).PrefillNoCache)
        assert False, "应对 PrefillNoCache 抛 NotImplementedError"
    except NotImplementedError:
        pass


# --------------------- 后端契约 / f7 收口 ---------------------------- #

def test_get_kv_cache_shape(env):
    mods, _ = env
    shape = mods.attention_v1.AscendAttentionBackend.get_kv_cache_shape(8, 128, 2, 16)
    assert shape == (2, 8, 128, 2, 16)


def test_get_impl_builder_cls_default(env):
    mods, _ = env
    B = mods.attention_v1.AscendAttentionBackend
    assert B.get_impl_cls() is mods.attention_v1.AscendAttentionBackendImpl
    assert B.get_builder_cls() is mods.attention_v1.AscendAttentionMetadataBuilder


def test_get_impl_builder_cls_cp_branch_f7(env):
    mods, knobs = env
    import vllm_ascend.attention.context_parallel.attention_cp as cp
    # 运行期 enable_cp() 为真 → 切到 CP 版（f7 收口）
    knobs.vllm_config.parallel_config.prefill_context_parallel_size = 2
    mods.utils.enable_cp.cache_clear()
    B = mods.attention_v1.AscendAttentionBackend
    assert B.get_impl_cls() is cp.AscendAttentionCPImpl
    assert B.get_builder_cls() is cp.AscendAttentionCPMetadataBuilder


# --------------------- (4) KV 写入 reshape_and_cache ----------------- #

def test_reshape_and_cache_writes_kv(env):
    mods, _ = env
    impl = _make_impl(mods)
    kv_cache = _make_kv_cache()
    md = _meta(mods, attn_state=_state(mods).ChunkedPrefill, num_actual_tokens=3,
               slot_mapping=torch.tensor([35, 2, 17, 99]))
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    impl.reshape_and_cache(torch.empty(0), key, value, kv_cache, md, torch.empty(0))

    assert "_npu_reshape_and_cache" in mods.rec.names()
    kw = mods.rec.last("_npu_reshape_and_cache")
    # slot_indices == slot_mapping[:num_actual_tokens]
    assert kw["slot_indices"].tolist() == [35, 2, 17]
    assert kw["key"].shape == (3, 2, 8)
    assert kw["key_cache"].data_ptr() == kv_cache[0].data_ptr()   # 写进 kv_cache 的 key 半张


# --------------------- (5) forward_impl 按状态分流 ------------------- #

def _decode_meta(mods, num_tokens):
    return _meta(
        mods,
        attn_state=_state(mods).DecodeOnly,
        block_tables=torch.zeros(num_tokens, 4, dtype=torch.int32),
        seq_lens=torch.tensor([1] * num_tokens),
        seq_lens_list=[1] * num_tokens,
        actual_seq_lengths_q=list(range(1, num_tokens + 1)),
    )


def test_forward_impl_decode_goes_paged(env):
    mods, _ = env
    impl = _make_impl(mods)
    impl.key_cache, impl.value_cache = _make_kv_cache()[0], _make_kv_cache()[1]
    num_tokens = 2  # 命中 pa_shape_list
    md = _decode_meta(mods, num_tokens)
    query = torch.zeros(num_tokens, 4, 8)
    out = torch.zeros(num_tokens, 4, 8)
    impl.forward_impl(query, None, None, _make_kv_cache(), md, out)
    assert "_npu_paged_attention" in mods.rec.names()
    assert "npu_fused_infer_attention_score" not in mods.rec.names()


def test_forward_impl_decode_fallback_to_fused(env):
    mods, knobs = env
    # cudagraph 模式非 FULL_DECODE_ONLY → using_paged_attention 为假 → decode 回落 fused
    knobs.vllm_config.compilation_config.cudagraph_mode = "NONE"
    impl = _make_impl(mods)
    kv = _make_kv_cache()
    impl.key_cache, impl.value_cache = kv[0], kv[1]
    num_tokens = 2
    md = _decode_meta(mods, num_tokens)
    query = torch.zeros(num_tokens, 4, 8)
    out = torch.zeros(num_tokens, 4, 8)
    impl.forward_impl(query, torch.zeros(num_tokens, 2, 8), torch.zeros(num_tokens, 2, 8), kv, md, out)
    assert "npu_fused_infer_attention_score" in mods.rec.names()
    assert "_npu_paged_attention" not in mods.rec.names()


def _chunked_fused_meta(mods, num_tokens, *, causal=True):
    return _meta(
        mods,
        attn_state=_state(mods).ChunkedPrefill,
        block_tables=torch.zeros(2, 4, dtype=torch.int32),
        seq_lens=torch.tensor([num_tokens]),
        seq_lens_list=[num_tokens],
        actual_seq_lengths_q=[num_tokens],
        attn_mask=torch.ones(4, 4),
        causal=causal,
    )


def test_forward_impl_chunked_prefill_goes_fused_causal(env):
    mods, _ = env
    impl = _make_impl(mods)
    kv = _make_kv_cache()
    impl.key_cache, impl.value_cache = kv[0], kv[1]
    num_tokens = 5
    md = _chunked_fused_meta(mods, num_tokens, causal=True)
    query = torch.zeros(num_tokens, 4, 8)
    out = torch.zeros(num_tokens, 4, 8)
    impl.forward_impl(query, torch.zeros(num_tokens, 2, 8), torch.zeros(num_tokens, 2, 8), kv, md, out)
    assert mods.rec.last("npu_fused_infer_attention_score")["sparse_mode"] == 3  # 因果


def test_fused_sparse_mode_noncausal(env):
    mods, _ = env
    impl = _make_impl(mods)
    kv = _make_kv_cache()
    impl.key_cache, impl.value_cache = kv[0], kv[1]
    num_tokens = 5
    md = _chunked_fused_meta(mods, num_tokens, causal=False)
    query = torch.zeros(num_tokens, 4, 8)
    out = torch.zeros(num_tokens, 4, 8)
    impl.forward_fused_infer_attention(query, torch.zeros(num_tokens, 2, 8),
                                       torch.zeros(num_tokens, 2, 8), md, out, kv)
    assert mods.rec.last("npu_fused_infer_attention_score")["sparse_mode"] == 0  # 非因果


def test_fused_sparse_mode_sliding_window(env):
    mods, _ = env
    impl = _make_impl(mods, sliding_window=128)
    kv = _make_kv_cache()
    impl.key_cache, impl.value_cache = kv[0], kv[1]
    num_tokens = 5
    md = _chunked_fused_meta(mods, num_tokens, causal=True)
    query = torch.zeros(num_tokens, 4, 8)
    out = torch.zeros(num_tokens, 4, 8)
    impl.forward_fused_infer_attention(query, torch.zeros(num_tokens, 2, 8),
                                       torch.zeros(num_tokens, 2, 8), md, out, kv)
    kw = mods.rec.last("npu_fused_infer_attention_score")
    assert kw["sparse_mode"] == 4 and kw["pre_tokens"] == 128  # 滑窗


# --------------------- _get_fia_params 按五态整理 ------------------- #

def test_get_fia_params_prefill_no_cache(env):
    mods, _ = env
    impl = _make_impl(mods)
    md = _meta(mods, attn_state=_state(mods).PrefillNoCache, actual_seq_lengths_q=[5])
    key, value, block_size, block_table, kv_lens = impl._get_fia_params(
        torch.zeros(5, 2, 8), torch.zeros(5, 2, 8), md
    )
    assert block_table is None          # 纯新 prefill：无 block_table
    assert block_size == 128
    assert kv_lens == [5]               # actual_seq_lengths_q


def test_get_fia_params_decode_only_reads_cache(env):
    mods, _ = env
    impl = _make_impl(mods)
    kv = _make_kv_cache(num_blocks=8, block_size=128)
    impl.key_cache, impl.value_cache = kv[0], kv[1]
    md = _meta(mods, attn_state=_state(mods).DecodeOnly,
               block_tables=torch.zeros(3, 4, dtype=torch.int32), seq_lens_list=[1, 1, 1])
    key, value, block_size, block_table, kv_lens = impl._get_fia_params(None, None, md)
    assert block_size == 128
    assert key.shape == (8, 128, 16)    # view(num_block, block_size, -1)
    assert block_table is md.block_tables
    assert kv_lens == [1, 1, 1]


# --------------------- using_paged_attention 门槛 ------------------- #

def test_using_paged_attention_all_gates_pass(env):
    mods, knobs = env
    assert mods.utils.using_paged_attention(2, knobs.vllm_config) is True


def test_using_paged_attention_blocked_by_spec(env):
    mods, knobs = env
    knobs.vllm_config.speculative_config = object()
    assert mods.utils.using_paged_attention(2, knobs.vllm_config) is False


def test_using_paged_attention_blocked_by_cudagraph_mode(env):
    mods, knobs = env
    knobs.vllm_config.compilation_config.cudagraph_mode = "NONE"
    assert mods.utils.using_paged_attention(2, knobs.vllm_config) is False


def test_using_paged_attention_blocked_by_shape(env):
    mods, knobs = env
    assert mods.utils.using_paged_attention(3, knobs.vllm_config) is False  # 3 不在 pa_shape_list


def test_using_paged_attention_blocked_on_a5(env):
    mods, knobs = env
    knobs.device_type = "A5"
    assert mods.utils.using_paged_attention(2, knobs.vllm_config) is False


# --------------------- workspace 预取节拍 --------------------------- #

def test_workspace_prefetch_on_capture(env):
    mods, knobs = env
    mods.forward_ctx.capturing = True   # 图捕获路径
    impl = _make_impl(mods)
    kv = _make_kv_cache()
    impl.key_cache, impl.value_cache = kv[0], kv[1]
    num_tokens = 2
    md = _decode_meta(mods, num_tokens)
    query = torch.zeros(num_tokens, 4, 8)
    out = torch.zeros(num_tokens, 4, 8)
    impl.forward_paged_attention(query, md, out)
    # 先用 *_get_workspace 量出 workspace 并缓存（NPU 算子特有节拍）
    assert "_npu_paged_attention_get_workspace" in mods.rec.names()
    assert num_tokens in mods.graph_params.workspaces
