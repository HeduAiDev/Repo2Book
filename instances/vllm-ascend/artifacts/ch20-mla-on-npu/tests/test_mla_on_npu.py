"""ch20 —— MLA 在 NPU 上：验证精简版复现 vllm-ascend mla_v1.py 的可观察控制流。

测的是「精简版与真仓 mla_v1.py 的控制流/形状代数一致」，不是精简版自洽：
权重吸收 absorb 的 split/permute/bmm 形状 / 三段 metadata 装配 / chunked-context LSE 合并循环 /
forward 按 decode-prefill 派发 / exec_kv 的 is_output_kv 差异。真实 torch_npu 算子由记录替身承接。
"""
import types

import torch

from conftest import make_builder, make_common_metadata, make_mla_impl


# ============================ (A) 后端契约 / f-收口 ============================ #

def test_backend_get_impl_builder_default(env):
    mods, _ = env
    B = mods.mla_v1.AscendMLABackend
    assert B.get_impl_cls() is mods.mla_v1.AscendMLAImpl
    assert B.get_builder_cls() is mods.mla_v1.AscendMLAMetadataBuilder


def test_backend_kv_cache_shape(env):
    mods, _ = env
    assert mods.mla_v1.AscendMLABackend.get_kv_cache_shape(8, 128, 1, 576) == (8, 128, 1, 576)
    assert mods.mla_v1.AscendMLABackend.get_supported_kernel_block_sizes() == [128]


def test_backend_cp_branch(env):
    mods, knobs = env
    import vllm_ascend.attention.context_parallel.mla_cp as cp
    knobs.vllm_config.parallel_config.prefill_context_parallel_size = 2
    import vllm_ascend.attention.utils as autil
    autil.enable_cp.cache_clear()
    B = mods.mla_v1.AscendMLABackend
    assert B.get_impl_cls() is cp.AscendMlaCPImpl
    assert B.get_builder_cls() is cp.AscendMlaCPMetadataBuilder


# ============================ (B) 权重吸收 absorb ============================ #

def test_process_weights_splits_W_UK_W_UV(env):
    mods, _ = env
    impl = make_mla_impl(mods, None, num_heads=2, kv_lora_rank=4, qk_nope_head_dim=3, v_head_dim=3)
    impl.process_weights_after_loading(torch.float32)
    # W_UK_T: (N, P, L)；W_UV: (N, L, V)
    assert impl.W_UK_T.shape == (2, 3, 4)
    assert impl.W_UV.shape == (2, 4, 3)
    # kv_b_proj.weight 先经 npu_format_cast(FRACTAL_ND=2)
    args, _ = mods.rec.last("npu_format_cast")
    assert args[1] == 2


def test_q_proj_and_k_up_proj_absorbs_to_latent(env):
    mods, _ = env
    impl = make_mla_impl(mods, None, num_heads=2, kv_lora_rank=4, qk_nope_head_dim=3,
                         qk_rope_head_dim=2, v_head_dim=3, q_lora_rank=5)
    impl.process_weights_after_loading(torch.float32)
    B = 6
    ql_nope, q_pe = impl._q_proj_and_k_up_proj(torch.zeros(B, 5))
    # q_nope 经 bmm(q_nope, W_UK_T) 吸收进 latent → (B, N, L)
    assert ql_nope.shape == (B, 2, 4)
    assert q_pe.shape == (B, 2, 2)
    # 吸收的实际算子是 torch.bmm（不经 torch_npu）；上投影未显式解压 KV
    assert "npu_transpose_batchmatmul" not in mods.rec.names()


def test_v_up_proj_latent_to_v(env):
    mods, _ = env
    impl = make_mla_impl(mods, None, num_heads=2, kv_lora_rank=4, v_head_dim=3)
    impl.process_weights_after_loading(torch.float32)
    B = 5
    # 输入 latent 注意力输出 (N, B, L)
    out = impl._v_up_proj(torch.zeros(2, B, 4))
    # → (B, N*V)
    assert out.shape == (B, 2 * 3)
    a, kw = mods.rec.last("npu_transpose_batchmatmul")
    assert kw["perm_y"] == (1, 0, 2)


# ============================ (C) 三段 metadata 装配 ============================ #

def test_build_assembles_three_segments(env):
    mods, knobs = env
    builder = make_builder(mods, knobs)
    cm = make_common_metadata(mods, query_lens=[1, 1, 5, 7], attn_state=mods.state.ChunkedPrefill)
    md = builder.build(common_prefix_len=0, common_attn_metadata=cm)
    # split_decodes_and_prefills 切：query_len<=1 → decode
    assert md.num_decodes == 2 and md.num_prefills == 2
    assert md.num_decode_tokens == 2
    assert md.num_actual_tokens == 14
    assert md.slot_mapping.tolist() == list(range(14))
    # 双路装配：prefill 段 + decode 段都建出来
    assert md.prefill is not None
    assert md.decode is not None
    assert md.query_lens == [1, 1, 5, 7]


def test_build_decode_only_no_prefill_segment(env):
    mods, knobs = env
    builder = make_builder(mods, knobs)
    cm = make_common_metadata(mods, query_lens=[1, 1, 1], attn_state=mods.state.DecodeOnly)
    md = builder.build(common_prefix_len=0, common_attn_metadata=cm)
    assert md.num_decodes == 3 and md.num_prefills == 0
    assert md.prefill is None
    assert md.decode is not None


def test_build_prefill_only_no_decode_segment(env):
    mods, knobs = env
    builder = make_builder(mods, knobs)
    cm = make_common_metadata(mods, query_lens=[5, 7], attn_state=mods.state.ChunkedPrefill)
    md = builder.build(common_prefix_len=0, common_attn_metadata=cm)
    assert md.num_decodes == 0 and md.num_prefills == 2
    assert md.decode is None
    assert md.prefill is not None
    # TND 变长右边界 = prefill query_lens 的累积和
    assert md.prefill.actual_seq_lengths_q == [5, 12]


def test_build_prefill_metadata_segments_after_decode(env):
    mods, knobs = env
    builder = make_builder(mods, knobs)
    cm = make_common_metadata(mods, query_lens=[1, 1, 5, 7], attn_state=mods.state.ChunkedPrefill)
    md = builder.build(common_prefix_len=0, common_attn_metadata=cm)
    # prefill 段从 decode 段（num_decodes=2）之后切出：actual_seq_lengths_q = cumsum([5,7])
    assert md.prefill.actual_seq_lengths_q == [5, 12]
    # decode 段 actual_seq_lengths_q = query_start_loc_cpu[1:num_decodes+1] = [1, 2]
    assert md.decode.actual_seq_lengths_q == [1, 2]


# ============================ (D) build_chunked_metadata 分块游标 ============================ #

def test_build_chunked_metadata_num_chunks(env):
    mods, knobs = env
    builder = make_builder(mods, knobs)
    # 直接喂 builder 状态做形状级数学验证
    builder.chunked_prefill_enabled = True
    builder.chunked_prefill_workspace_size = 32
    builder.block_size = 8
    builder.num_decodes = 0
    builder.num_prefills = 2
    builder.seq_lens = torch.tensor([10, 20])
    builder.query_lens = torch.tensor([2, 3])  # context_lens = [8, 17]
    cm = make_common_metadata(mods, query_lens=[2, 3], attn_state=mods.state.ChunkedPrefill)
    ck = builder.build_chunked_metadata(0, cm)
    # max_context_chunk = round_down(32 // 2, 8) = 16；num_chunks = cdiv(17, 16) = 2
    assert builder.max_context_chunk == 16
    assert builder.num_chunks == 2
    assert len(ck.seq_tot) == 2


def test_build_chunked_metadata_disabled_returns_none(env):
    mods, knobs = env
    builder = make_builder(mods, knobs)
    builder.chunked_prefill_enabled = False
    cm = make_common_metadata(mods, query_lens=[5], attn_state=mods.state.ChunkedPrefill)
    builder.num_decodes = 0
    builder.num_prefills = 1
    builder.seq_lens = torch.tensor([5])
    builder.query_lens = torch.tensor([5])
    assert builder.build_chunked_metadata(0, cm) is None


# ============================ (E) 前向派发 _mla_preprocess / forward ============================ #

def _decode_meta(mods, num_decode_tokens):
    R = 2
    return mods.mla_v1.AscendMLADecodeMetadata(
        input_positions=torch.zeros(num_decode_tokens, dtype=torch.int64),
        block_table=torch.zeros(num_decode_tokens, 4, dtype=torch.int32),
        seq_lens=torch.tensor([1] * num_decode_tokens),
        max_seq_lens=1,
        seq_lens_list=[1] * num_decode_tokens,
        cos=torch.zeros(num_decode_tokens, R),
        sin=torch.zeros(num_decode_tokens, R),
    )


def test_mla_preprocess_splits_qc_kv_and_dispatches_decode(env):
    mods, _ = env
    impl = make_mla_impl(mods, None)
    seen = {}

    def _dec(*a):
        seen["decode"] = True
        return "D"

    def _pre(*a):
        seen["prefill"] = True
        return "P"

    impl.mla_preprocess_decode = _dec
    impl.mla_preprocess_prefill = _pre
    md = types.SimpleNamespace(num_decodes=2, num_prefills=0, num_decode_tokens=2, num_actual_tokens=2)
    dres, pres = impl._mla_preprocess("layer", torch.zeros(2, 16), None, md, False)
    # fused_qkv_a_proj 一把拆 q_c/kv_no_split
    assert "proj:fused_qkv_a_proj" in mods.rec.names()
    # has_decode → 走 mla_preprocess_decode；has_prefill=False → 不走 prefill
    assert dres == "D" and pres is None
    assert "prefill" not in seen


def test_mla_preprocess_dispatches_both_on_mixed(env):
    mods, _ = env
    impl = make_mla_impl(mods, None)
    impl.mla_preprocess_decode = lambda *a: "D"
    impl.mla_preprocess_prefill = lambda *a: "P"
    md = types.SimpleNamespace(num_decodes=1, num_prefills=1, num_decode_tokens=1, num_actual_tokens=3)
    dres, pres = impl._mla_preprocess("layer", torch.zeros(3, 16), None, md, False)
    assert dres == "D" and pres == "P"


def test_forward_routes_decode_into_o_proj_slice(env):
    mods, _ = env
    impl = make_mla_impl(mods, None, num_heads=2, v_head_dim=3)
    mods.mla_v1._EXTRA_CTX.num_tokens = 2
    dres = mods.mla_v1.DecodeMLAPreprocessResult(
        ql_nope=torch.zeros(2, 2, 4), q_pe=torch.zeros(2, 2, 2),
        k_nope=torch.zeros(1), k_pe=torch.zeros(1),
    )
    impl._mla_preprocess = lambda *a: (dres, None)
    impl._forward_decode = lambda *a: torch.ones(2, 6)
    md = types.SimpleNamespace(num_decodes=2, num_prefills=0, num_decode_tokens=2, num_actual_tokens=2)
    kv_cache = (torch.zeros(8, 8, 1, 4), torch.zeros(8, 8, 1, 2))
    output = torch.zeros(2, 6)
    out = impl.forward("layer", torch.zeros(2, 16), kv_cache, md, output=output)
    # decode 输出写进 o_proj_input[:num_decode_tokens] → o_proj → output
    assert "proj:o_proj" in mods.rec.names()
    _, kw = mods.rec.last("proj:o_proj")
    assert kw["is_prefill"] is False  # 纯 decode
    assert out is output


def test_forward_prefill_sets_is_prefill_true(env):
    mods, _ = env
    impl = make_mla_impl(mods, None, num_heads=2, v_head_dim=3)
    mods.mla_v1._EXTRA_CTX.num_tokens = 3
    pres = mods.mla_v1.PrefillMLAPreprocessResult(
        q_nope=torch.zeros(3, 2, 3), q_pe=torch.zeros(3, 2, 2),
        k_nope=torch.zeros(3, 2, 3), k_pe=torch.zeros(3, 2, 2), value=torch.zeros(3, 2, 3),
    )
    impl._mla_preprocess = lambda *a: (None, pres)
    impl._forward_prefill = lambda *a: torch.ones(3, 6)
    md = types.SimpleNamespace(num_decodes=0, num_prefills=1, num_decode_tokens=0, num_actual_tokens=3)
    kv_cache = (torch.zeros(8, 8, 1, 4), torch.zeros(8, 8, 1, 2))
    output = torch.zeros(3, 6)
    impl.forward("layer", torch.zeros(3, 16), kv_cache, md, output=output)
    _, kw = mods.rec.last("proj:o_proj")
    assert kw["is_prefill"] is True


def test_forward_profiling_run_fills_zero(env):
    mods, _ = env
    impl = make_mla_impl(mods, None)
    output = torch.ones(2, 6)
    out = impl.forward("layer", torch.zeros(2, 16), None, None, output=output)
    assert torch.all(out == 0)


# ============================ (F) exec_kv：is_output_kv 差异 ============================ #

def test_exec_kv_decode_takes_cache_outputs(env):
    mods, _ = env
    impl = make_mla_impl(mods, None, kv_lora_rank=4, qk_rope_head_dim=2)
    kv_no_split = torch.zeros(2, 6)  # (B, L+R)
    kv_cache = (torch.zeros(8, 8, 1, 4), torch.zeros(8, 8, 1, 2))
    cos = torch.zeros(2, 2)
    k_pe, k_nope = impl.exec_kv_decode(kv_no_split, cos, cos, kv_cache, torch.zeros(2, dtype=torch.int64))
    # decode 取算子第 1/2 个返回值（写进 cache 的隐向量）
    assert k_pe.flatten()[0] == 1 and k_nope.flatten()[0] == 2
    _, kw = mods.rec.last("npu_kv_rmsnorm_rope_cache")
    assert kw.get("cache_mode") == "PA"
    assert "is_output_kv" not in kw  # decode 不带该标志


def test_exec_kv_prefill_is_output_kv(env):
    mods, _ = env
    impl = make_mla_impl(mods, None, kv_lora_rank=4, qk_rope_head_dim=2)
    kv_no_split = torch.zeros(2, 6)
    kv_cache = (torch.zeros(8, 8, 1, 4), torch.zeros(8, 8, 1, 2))
    cos = torch.zeros(2, 2)
    k_pe, k_nope = impl.exec_kv_prefill(kv_no_split, cos, cos, kv_cache, torch.zeros(2, dtype=torch.int64))
    # prefill is_output_kv=True 且取第 3/4 个返回值（未量化输出 KV 供显式解压）
    assert k_pe.flatten()[0] == 3 and k_nope.flatten()[0] == 4
    _, kw = mods.rec.last("npu_kv_rmsnorm_rope_cache")
    assert kw.get("is_output_kv") is True


# ============================ (G) mla_preprocess 两路差异 ============================ #

def test_mla_preprocess_decode_uses_absorb(env):
    mods, _ = env
    impl = make_mla_impl(mods, None)
    impl.process_weights_after_loading(torch.float32)
    md = types.SimpleNamespace(
        num_decode_tokens=2,
        slot_mapping=torch.zeros(2, dtype=torch.int64),
        decode=_decode_meta(mods, 2),
    )
    kv_cache = (torch.zeros(8, 8, 1, 4), torch.zeros(8, 8, 1, 2))
    res = impl.mla_preprocess_decode(torch.zeros(2, 5), torch.zeros(2, 6), kv_cache, md)
    # 吸收路径：q 经 q_b_proj 上投影 + exec_kv_decode 写 cache；不调 kv_b_proj 显式解压
    assert "proj:q_b_proj" in mods.rec.names()
    assert "npu_kv_rmsnorm_rope_cache" in mods.rec.names()
    assert "proj:kv_b_proj" not in mods.rec.names()
    assert res.ql_nope.shape == (2, 2, 4)


def test_mla_preprocess_prefill_explicit_decompress(env):
    mods, _ = env
    impl = make_mla_impl(mods, None, num_heads=2, qk_nope_head_dim=3, v_head_dim=3)
    R = 2
    prefill_md = mods.mla_v1.AscendMLAPrefillMetadata(
        attn_mask=None, query_lens=torch.tensor([3]), seq_lens=[3],
        context_lens=torch.tensor([3]), input_positions=torch.zeros(3, dtype=torch.int64),
        query_start_loc=torch.tensor([0, 3]), block_table=torch.zeros(1, 4, dtype=torch.int32),
        max_query_len=3, max_seq_lens=3, cos=torch.zeros(3, R), sin=torch.zeros(3, R),
    )
    md = types.SimpleNamespace(
        num_decode_tokens=0, num_actual_tokens=3,
        slot_mapping=torch.zeros(3, dtype=torch.int64), prefill=prefill_md,
    )
    kv_cache = (torch.zeros(8, 8, 1, 4), torch.zeros(8, 8, 1, 2))
    res = impl.mla_preprocess_prefill(torch.zeros(3, 5), torch.zeros(3, 6), kv_cache, md)
    # MHA 风格：显式 kv_b_proj 解压 k_nope/value（decode 路不会调）
    assert "proj:kv_b_proj" in mods.rec.names()
    assert res.value is not None


# ============================ (H) chunked-context LSE 合并 ============================ #

def test_compute_prefill_context_merges_chunks(env):
    mods, _ = env
    impl = make_mla_impl(mods, None, num_heads=2, qk_nope_head_dim=3, v_head_dim=3, qk_rope_head_dim=2)
    impl.process_weights_after_loading(torch.float32)
    num_tokens = 3
    chunked = mods.mla_v1.ChunkedContextMetadata(
        cu_seq_lens=None, starts=[torch.zeros(1, dtype=torch.int32)] * 2,
        seq_tot=[4, 5], max_seq_lens=[4, 5],
        chunk_seq_lens=torch.zeros(2, 1), chunk_seq_lens_npu=torch.zeros(2, 1, dtype=torch.int32),
        workspace=torch.zeros(1),
        chunk_actual_seq_lengths_kv_list=[[4], [5]],
    )
    prefill_md = mods.mla_v1.AscendMLAPrefillMetadata(
        attn_mask=None, query_lens=torch.tensor([3]), seq_lens=[3],
        context_lens=torch.tensor([9]), input_positions=torch.zeros(3, dtype=torch.int64),
        query_start_loc=torch.tensor([0, 3]), block_table=torch.zeros(1, 4, dtype=torch.int32),
        max_query_len=3, max_seq_lens=9, chunked_context=chunked,
        cos=torch.zeros(3, 2), sin=torch.zeros(3, 2), actual_seq_lengths_q=[3],
    )
    md = types.SimpleNamespace(prefill=prefill_md)
    cache = (torch.zeros(8, 8, 2, 4), torch.zeros(8, 8, 2, 2))
    q_nope = torch.zeros(num_tokens, 2, 3)
    q_pe = torch.zeros(num_tokens, 2, 2)
    prefix_out = torch.zeros(num_tokens, 2, 3)
    prefix_lse = torch.zeros(2, num_tokens)
    out, lse = impl._compute_prefill_context(q_nope, q_pe, cache, 2, md, prefix_out, prefix_lse)
    # iters=2 块各算一次注意力 + 一次在线合并
    assert mods.rec.names().count("npu_fused_infer_attention_score") == 2
    # 合并入参：lse_list/out_list 长度 = 1(prefix) + iters(2) = 3
    (lse_list, out_list, _dim), _ = mods.rec.last("npu_attention_update")
    assert len(lse_list) == 3 and len(out_list) == 3
    assert lse is None  # 合并后不再返回 lse


def test_compute_prefill_context_no_chunk_returns_prefix(env):
    mods, _ = env
    impl = make_mla_impl(mods, None)
    prefill_md = mods.mla_v1.AscendMLAPrefillMetadata(
        attn_mask=None, query_lens=torch.tensor([3]), seq_lens=[3],
        context_lens=torch.tensor([0]), input_positions=torch.zeros(3, dtype=torch.int64),
        query_start_loc=torch.tensor([0, 3]), block_table=torch.zeros(1, 4, dtype=torch.int32),
        max_query_len=3, max_seq_lens=3, chunked_context=None,
    )
    md = types.SimpleNamespace(prefill=prefill_md)
    cache = (torch.zeros(8, 8, 2, 4), torch.zeros(8, 8, 2, 2))
    po = torch.zeros(3, 2, 3)
    pl = torch.zeros(2, 3)
    out, lse = impl._compute_prefill_context(torch.zeros(3, 2, 3), torch.zeros(3, 2, 2), cache, 2, md, po, pl)
    # chunked_context=None → 直接返回 prefix，不调注意力/合并
    assert out is po and lse is pl
    assert "npu_attention_update" not in mods.rec.names()
