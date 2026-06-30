"""ch21 —— 稀疏注意力 SFA/DSA：验证精简版复现 vllm-ascend sfa_v1.py / dsa_v1.py / device_op.py 的
可观察控制流（不是精简版自洽）：两段式稀疏（选 top-k → 只对 top-k 算）、Lightning Indexer 的
sparse_count=512/2048 与 sparse_mode=3、DeviceOperator 门面按设备代际多态派发、SFA 建在 MLA 之上
vs DSA 自起一套的继承关系。真实稀疏算子由记录替身承接。
"""
import types

import torch

import conftest


# ============================ (A) 后端契约 / 继承关系 ============================ #

def test_sfa_backend_contract_and_inheritance(env):
    sfa, backend = env.sfa, env.backend
    # ch18 路由落点：HACK 名 + builder/impl 选择
    assert sfa.AscendSFABackend.get_name() == "ASCEND_SFA"
    assert sfa.AscendSFABackend.get_builder_cls() is sfa.AscendSFAMetadataBuilder
    assert sfa.AscendSFABackend.get_impl_cls() is sfa.AscendSFAImpl
    # 建在 MLA 之上：builder 继承 MLACommonMetadataBuilder，impl 继承 vLLM MLAAttentionImpl
    import vllm.model_executor.layers.attention.mla_attention as mla
    assert issubclass(sfa.AscendSFAMetadataBuilder, mla.MLACommonMetadataBuilder)
    assert issubclass(sfa.AscendSFAImpl, backend.MLAAttentionImpl)


def test_dsa_backend_contract_and_inheritance(env):
    dsa, backend = env.dsa, env.backend
    assert dsa.AscendDSABackend.get_name() == "ASCEND_DSA"
    assert dsa.AscendDSABackend.get_builder_cls() is dsa.AscendDSAMetadataBuilder
    assert dsa.AscendDSABackend.get_impl_cls() is dsa.AscendDSAImpl
    # DSA 自起一套：builder 继承 vLLM AttentionMetadataBuilder（非 MLA builder）；
    # impl 继承昇腾自有 DSAAttentionImpl（非 vLLM MLAAttentionImpl）。
    assert issubclass(dsa.AscendDSAMetadataBuilder, backend.AttentionMetadataBuilder)
    assert issubclass(dsa.AscendDSAImpl, env.abstract.DSAAttentionImpl)
    assert not issubclass(dsa.AscendDSAImpl, backend.MLAAttentionImpl)


# ==================== (B) DeviceOperator 门面：按设备代际多态派发 ==================== #

def test_device_operator_dispatch_by_generation(env):
    device_op = env.device_op
    # import 期已按 A2 选定 Base
    assert device_op.DeviceOperator is device_op.BaseDeviceAdaptor
    # 改设备代际为 A5 → 门面工厂改选 A5DeviceAdaptor
    conftest.DEVICE_TYPE["value"] = "A5"
    assert device_op.get_device_adaptor() is device_op.A5DeviceAdaptor
    conftest.DEVICE_TYPE["value"] = "A2"
    assert device_op.get_device_adaptor() is device_op.BaseDeviceAdaptor


def test_reshape_and_cache_routes_per_generation(env):
    device_op, rec = env.device_op, env.rec_npu
    t = torch.zeros(2, 1, 4)
    device_op.BaseDeviceAdaptor.reshape_and_cache(t, t, t, t, torch.zeros(2, dtype=torch.int64))
    assert "_npu_reshape_and_cache" in rec.names()  # A2/A3 主路
    device_op.A5DeviceAdaptor.reshape_and_cache(t, t, t, t, torch.zeros(2, dtype=torch.int64))
    assert "npu_scatter_pa_kv_cache" in rec.names()  # A5 override


def test_get_dsa_sparse_attn_op_is_sharedkv(env):
    device_op, rec_c = env.device_op, env.rec_c
    op = device_op.DeviceOperator.get_dsa_sparse_attn_op()
    op(torch.zeros(1))  # 调用以触发记录
    assert "npu_sparse_attn_sharedkv" in rec_c.names()


# ==================== (C) SFA 两段式稀疏：选 top-k(2048) → 只对 top-k 算 ==================== #

def test_sfa_device_indexer_uses_sparse_count_2048(env):
    """阶段一：Lightning Indexer 的 default 分支 sparse_count=2048, sparse_mode=3。"""
    device_op, rec_c = env.device_op, env.rec_c
    sfa_impl = types.SimpleNamespace(use_sparse_c8_indexer=False)
    attn_metadata = types.SimpleNamespace(block_table=torch.zeros(1, 4, dtype=torch.int32))
    kv_cache = (torch.zeros(1, 8, 1, 8),) * 4
    device_op.DeviceOperator.indexer_select_post_process(
        sfa_impl, torch.zeros(1, 64, 128), None, None, torch.zeros(1, 8),
        kv_cache, attn_metadata, torch.tensor([1]), torch.tensor([4]), False, False,
    )
    _, kw = rec_c.last("npu_lightning_indexer")
    assert kw["sparse_count"] == 2048
    assert kw["sparse_mode"] == 3


def test_sfa_sparse_flash_attention_consumes_topk(env):
    """阶段二：稀疏 flash 把 top-k 索引喂给内核（sparse_indices=topk_indices, sparse_mode=3）。"""
    device_op, rec_c = env.device_op, env.rec_c
    sfa_impl = types.SimpleNamespace(scale=0.5)
    attn_metadata = types.SimpleNamespace(block_table=torch.zeros(1, 4, dtype=torch.int32))
    kv_cache = (torch.zeros(1, 8, 1, 8), torch.zeros(1, 8, 1, 8), torch.zeros(1, 8, 1, 8))
    topk = torch.zeros(1, 2048, dtype=torch.int32)
    device_op.DeviceOperator.execute_sparse_flash_attention_process(
        sfa_impl, torch.zeros(1, 1, 8), torch.zeros(1, 1, 8), kv_cache, topk,
        attn_metadata, torch.tensor([1]), torch.tensor([4]),
    )
    _, kw = rec_c.last("npu_sparse_flash_attention")
    assert kw["sparse_indices"] is topk
    assert kw["sparse_mode"] == 3
    assert kw["sparse_block_size"] == 1


def test_sfa_impl_two_stage_methods_delegate(env):
    """AscendSFAImpl 的两段式落点都派到 DeviceOperator 门面；indexer_select_post_process 先投影 q_li。"""
    sfa = env.sfa
    impl = sfa.AscendSFAImpl.__new__(sfa.AscendSFAImpl)
    impl.scale = 0.5
    impl.n_head = 64
    impl.head_dim = 128
    impl.qk_rope_head_dim = 64
    impl.use_sparse_c8_indexer = False
    impl.use_torch_npu_lightning_indexer = False
    calls = {"wq_b": 0, "wk": 0}

    def wq_b(x):
        calls["wq_b"] += 1
        return torch.zeros(x.shape[0], 64 * 128), None

    def wk(x):
        calls["wk"] += 1
        return torch.zeros(x.shape[0], 128 + 8), None

    impl.wq_b = wq_b
    impl.wk_weights_proj = wk
    attn_metadata = types.SimpleNamespace(block_table=torch.zeros(1, 4, dtype=torch.int32))
    kv_cache = (torch.zeros(1, 8, 1, 8),) * 4
    out = impl.indexer_select_post_process(
        x=torch.zeros(1, 16), q_c=torch.zeros(1, 16), kv_cache=kv_cache, attn_metadata=attn_metadata,
        cos=torch.zeros(1, 64), sin=torch.zeros(1, 64),
        actual_seq_lengths_query=torch.tensor([1]), actual_seq_lengths_key=torch.tensor([4]),
    )
    assert calls["wq_b"] == 1 and calls["wk"] == 1
    # 返回的是 indexer 选出的 top-k 索引（经门面 npu_lightning_indexer）
    assert "npu_lightning_indexer" in env.rec_c.names()
    assert out.shape[-1] == 2048


# ==================== (D) DSA Lightning Indexer：top-512 元数据装配 + 选择 ==================== #

def _make_dsa_builder(env, num_decodes=0, num_prefills=1):
    dsa = env.dsa
    b = dsa.AscendDSAMetadataBuilder.__new__(dsa.AscendDSAMetadataBuilder)
    hf = types.SimpleNamespace(
        num_attention_heads=64, index_topk=512, index_n_heads=64, index_head_dim=128, sliding_window=128,
    )
    b.model_config = types.SimpleNamespace(hf_config=hf, get_head_size=lambda: 128)
    b.block_size = 128
    b.num_decodes = num_decodes
    b.num_prefills = num_prefills
    b.num_decode_tokens = 0
    b.num_prefill_tokens = 4
    b.num_actual_tokens = 4
    b.seq_lens = torch.tensor([4])
    b.query_lens = torch.tensor([4])
    b.seqused_q = torch.tensor([])
    b.slot_mapping = torch.zeros(8, 2, dtype=torch.int32)
    b.block_table = torch.zeros(1, 4, dtype=torch.int32)
    b.start_pos_prefill = torch.zeros(8, dtype=torch.int32)
    b.start_pos_decode = torch.zeros(8, dtype=torch.int32)
    b.decode_sas_metadata = torch.zeros(1024, dtype=torch.int32)
    b.decode_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    return b


def _make_common(num_decodes=0):
    return types.SimpleNamespace(
        num_reqs=1,
        query_start_loc=torch.tensor([0, 4]),
        query_start_loc_cpu=torch.tensor([0, 4]),
        positions=torch.arange(4),
        seq_lens=torch.tensor([4]),
        attn_state="ChunkedPrefill",
    )


def test_dsa_build_prefill_metadata_uses_index_topk_512(env):
    """build_prefill_metadata 预建 qli_metadata：sparse_count=index_topk=512, sparse_mode=3（章节核心数值）。"""
    b = _make_dsa_builder(env)
    md = b.build_prefill_metadata(0, _make_common())
    _, kw = env.rec_c.last("npu_quant_lightning_indexer_metadata")
    assert kw["sparse_count"] == 512
    assert kw["sparse_mode"] == 3
    # 同时装配了稀疏注意力元数据 sas_metadata（经门面 npu_sparse_attn_sharedkv_metadata）
    assert "npu_sparse_attn_sharedkv_metadata" in env.rec_c.names()
    assert md.qli_metadata is not None and md.sas_metadata is not None


def test_dsa_build_decode_metadata_uses_index_topk_512(env):
    """build_decode_metadata 同样预建 qli_metadata：sparse_count=512（与 prefill 对称）。"""
    b = _make_dsa_builder(env, num_decodes=1, num_prefills=0)
    b.num_decode_tokens = 4
    b.num_prefill_tokens = 0
    md = b.build_decode_metadata(0, _make_common(num_decodes=1), None)
    _, kw = env.rec_c.last("npu_quant_lightning_indexer_metadata")
    assert kw["sparse_count"] == 512
    assert kw["sparse_mode"] == 3
    assert md.qli_metadata is not None


def test_dsa_indexer_qli_selects_top_index_topk(env):
    """前向期 _indexer_qli → npu_quant_lightning_indexer(sparse_count=self.index_topk=512, sparse_mode=3)。"""
    dsa = env.dsa
    impl = dsa.AscendDSAImpl.__new__(dsa.AscendDSAImpl)
    impl.index_topk = 512
    prefill = types.SimpleNamespace(
        query_start_loc=torch.tensor([0, 4]),
        seq_lens=torch.tensor([4]),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        qli_metadata=torch.zeros(1024, dtype=torch.int32),
    )
    meta = types.SimpleNamespace(prefill=prefill, decode=None)
    idxs = impl._indexer_qli(
        q=torch.zeros(4, 64, 128), weights=torch.zeros(4, 64), q_scale=torch.ones(4),
        indexer_k_cache=torch.zeros(1, 8, 1, 128), indexer_scale_cache=torch.zeros(1, 8, 1, 1),
        indexer_kv_scale_metadata=meta, with_prefill=True,
    )
    _, kw = env.rec_c.last("npu_quant_lightning_indexer")
    assert kw["sparse_count"] == 512
    assert kw["sparse_mode"] == 3
    assert idxs.shape[-1] == 512  # 选出 top-512 个 KV 位置


def test_dsa_indexer_select_qli_chains_to_lightning_indexer(env):
    """indexer_select_qli 整链：投影 q + 压缩 KV → 量化写 indexer cache → npu_quant_lightning_indexer 出 top-512。"""
    dsa = env.dsa
    impl = dsa.AscendDSAImpl.__new__(dsa.AscendDSAImpl)
    impl.index_topk = 512
    impl.indexer_heads = 64
    impl.indexcom_head_dim = 128
    impl.rope_head_dim = 64
    impl.compress_ratio = 4
    impl.compressor_overlap = False
    impl.indexcom_rotate = False
    impl.indexer_softmax_scale = 128 ** -0.5
    impl.compressor_norm_eps = 1e-6
    impl.inderxer_wq_b = lambda qr: torch.zeros(4, 64 * 128)
    impl.weights_proj = lambda x: torch.zeros(4, 64)
    impl.indexcom_wkv = types.SimpleNamespace(weight=torch.zeros(8, 8))
    impl.indexcom_wgate = types.SimpleNamespace(weight=torch.zeros(8, 8))
    impl.indexcom_norm = types.SimpleNamespace(weight=torch.zeros(8))
    impl.indexcom_ape = torch.zeros(8)

    prefill = types.SimpleNamespace(
        query_start_loc=torch.tensor([0, 4]), seq_lens=torch.tensor([4]),
        block_table=torch.zeros(1, 4, dtype=torch.int32), qli_metadata=torch.zeros(1024, dtype=torch.int32),
        start_pos=torch.zeros(1, dtype=torch.int32), slot_mapping=torch.zeros(4, 2, dtype=torch.int32),
    )
    meta = types.SimpleNamespace(prefill=prefill, decode=None, hadamard=torch.zeros(128, 128))
    # attn_metadata 5 元组（compress_ratio==4）：第 3/4 项是 indexer 的 state/scale 元数据
    attn_metadata = [None, None, meta, meta, None]
    # indexer KV cache 解包要 6 元素：(_, _, _, state, k, scale)
    kv_cache = tuple(torch.zeros(1, 8, 1, 128) for _ in range(6))

    idxs = impl.indexer_select_qli(
        x=torch.zeros(4, 16), qr=torch.zeros(4, 16), kv_cache=kv_cache, attn_metadata=attn_metadata,
        cos=torch.zeros(4, 2), sin=torch.zeros(4, 2), compressed_cos=torch.zeros(4, 2), compressed_sin=torch.zeros(4, 2),
        actual_seq_lengths_query=torch.tensor([0, 4]), actual_seq_lengths_key=torch.tensor([4]), with_prefill=True,
    )
    names = env.rec_c.names()
    assert "compressor" in names                      # 先压缩 KV
    assert "npu_scatter_nd_update_v2" in names         # 量化写 indexer cache
    assert "npu_quant_lightning_indexer" in names      # 出 top-512
    assert idxs.shape[-1] == 512
