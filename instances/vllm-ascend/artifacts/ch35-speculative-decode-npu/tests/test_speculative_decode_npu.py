"""ch29 —— 投机解码的 NPU 对位（工厂分发 + 薄壳继承 + 重量级 base 的提议骨架）。

测的是精简版**复现真仓可观察行为**，不是自洽：
  - 工厂 if-elif 的 method→类映射（含非同名）；
  - 薄壳 proposer 的提议控制流（写回/转发/gather 索引/no-op stub）；
  - 重量级 base 的 prepare_inputs 纯 host numpy 索引运算（按拒绝数收缩 query/seq + 构造 token_indices）。
重 NPU 路径（ACLGraph/Triton draft 前向/MLA）不真跑。
"""

import types

import numpy as np
import torch


# ---------------------------------------------------------------------------
# (1) 工厂分发：一处 if-elif 把 method 字符串映射到 8 个 Ascend*Proposer（含非同名映射）
# ---------------------------------------------------------------------------
def test_factory_dispatch_maps_every_method_to_its_proposer(env):
    f = env.factory
    # 把 8 个类替换成哨兵，只验「分发到哪个类 + 传了哪些位置参数」，不触发重量级构造。
    sentinels = {}
    for name in (
        "AscendNgramProposer",
        "AscendNgramProposerNPU",
        "AscendSuffixDecodingProposer",
        "AscendMedusaProposer",
        "AscendEagleProposer",
        "AscendDflashProposer",
        "AscendDraftModelProposer",
        "AscendExtractHiddenStatesProposer",
    ):
        def _mk(n):
            return lambda *args: (n, args)

        sentinels[name] = _mk(name)
        setattr(f, name, sentinels[name])

    cfg, dev, runner = object(), object(), object()
    get = f.get_spec_decode_method

    # method 字符串与类名并非一一同名：这正是工厂吸收差异、对外只暴露一个 method 旋钮的体现。
    assert get("ngram", cfg, dev, runner)[0] == "AscendNgramProposer"
    assert get("ngram_gpu", cfg, dev, runner)[0] == "AscendNgramProposerNPU"
    assert get("suffix", cfg, dev, runner)[0] == "AscendSuffixDecodingProposer"
    assert get("medusa", cfg, dev, runner)[0] == "AscendMedusaProposer"
    assert get("dflash", cfg, dev, runner)[0] == "AscendDflashProposer"
    assert get("draft_model", cfg, dev, runner)[0] == "AscendDraftModelProposer"
    assert get("extract_hidden_states", cfg, dev, runner)[0] == "AscendExtractHiddenStatesProposer"
    # eagle / eagle3 / mtp 三种 method 共用 AscendEagleProposer。
    for m in ("eagle", "eagle3", "mtp"):
        assert get(m, cfg, dev, runner)[0] == "AscendEagleProposer"


def test_factory_arg_signatures_differ_per_proposer(env):
    f = env.factory
    for name in ("AscendNgramProposer", "AscendSuffixDecodingProposer", "AscendMedusaProposer", "AscendEagleProposer"):
        setattr(f, name, (lambda n: (lambda *args: (n, args)))(name))
    cfg, dev, runner = "CFG", "DEV", "RUN"
    # ngram/suffix 不收 device；medusa 不收 runner；eagle 收 (cfg, device, runner)。
    assert f.get_spec_decode_method("ngram", cfg, dev, runner)[1] == (cfg, runner)
    assert f.get_spec_decode_method("suffix", cfg, dev, runner)[1] == (cfg, runner)
    assert f.get_spec_decode_method("medusa", cfg, dev, runner)[1] == (cfg, dev)
    assert f.get_spec_decode_method("eagle", cfg, dev, runner)[1] == (cfg, dev, runner)


def test_factory_unknown_method_raises(env):
    import pytest

    with pytest.raises(ValueError, match="Unknown speculative decoding method"):
        env.factory.get_spec_decode_method("does_not_exist", object(), object(), object())


# ---------------------------------------------------------------------------
# (2) CPU n-gram 薄壳：propose 跳过空/不支持/超长请求、写回 token_ids_cpu、交父类 batch_propose
# ---------------------------------------------------------------------------
def _make_ngram_runner(num_reqs, sampled_lens, unsupported=(), at_max=()):
    req_ids = [f"r{i}" for i in range(num_reqs)]
    token_ids_cpu = np.zeros((num_reqs, 64), dtype=np.int64)
    num_tokens_no_spec = [100 if i in at_max else 5 for i in range(num_reqs)]
    input_batch = types.SimpleNamespace(
        req_ids=req_ids,
        spec_decode_unsupported_reqs=set(unsupported),
        num_tokens_no_spec=num_tokens_no_spec,
        token_ids_cpu=token_ids_cpu,
        max_model_len=100,
    )
    return types.SimpleNamespace(input_batch=input_batch)


def test_ngram_propose_filters_and_writes_back(env):
    runner = _make_ngram_runner(num_reqs=4, sampled_lens=None, unsupported={"r1"}, at_max={3})
    prop = object.__new__(env.ngram.AscendNgramProposer)
    prop.runner = runner

    sampled = [[11, 12], [], [21, 22, 23], [31]]
    #         r0 ok      r1 unsupported(被skip), 但其实 r1 空也skip; r2 ok; r3 已达 max_model_len skip
    # r1 既是空([])又 unsupported——两条 continue 都会跳过。
    valid = prop.propose(sampled)

    # 只有 r0、r2 是 valid：r1 空+不支持、r3 达到 max_model_len 被跳过。
    assert valid == [0, 2]
    # 新采样 token 已写回 token_ids_cpu 的 [start:start+len] 区间（start=num_tokens_no_spec[i]=5）。
    assert runner.input_batch.token_ids_cpu[0, 5:7].tolist() == [11, 12]
    assert runner.input_batch.token_ids_cpu[2, 5:8].tolist() == [21, 22, 23]
    # 被跳过的 r3 未写回。
    assert runner.input_batch.token_ids_cpu[3, 5:6].tolist() == [0]


# ---------------------------------------------------------------------------
# (3) no-op 薄壳极致：AscendNgramProposerNPU.propose 是裸 pass（不复用父类 GPU kernel）
# ---------------------------------------------------------------------------
def test_ngram_npu_propose_is_noop_stub(env):
    prop = object.__new__(env.ngram_npu.AscendNgramProposerNPU)
    # propose / load_model / dummy_run 全 no-op：返回 None，不复用父类那段 GPU 批量 n-gram kernel。
    assert prop.propose(None, None, None, None) is None
    assert prop.load_model() is None
    assert prop.dummy_run(0) is None


# ---------------------------------------------------------------------------
# (4) 最薄薄壳：AscendSuffixDecodingProposer.propose 一行转发父类（补 runner.input_batch）
# ---------------------------------------------------------------------------
def test_suffix_propose_forwards_to_parent_with_input_batch(env):
    runner = types.SimpleNamespace(input_batch="INPUT_BATCH")
    prop = object.__new__(env.suffix.AscendSuffixDecodingProposer)
    prop.runner = runner
    out = prop.propose(["sampled"])
    # 父类桩回显 (tag, input_batch, valid) —— 验证转发时补上了 self.runner.input_batch。
    assert out == ("BASE_suffix", "INPUT_BATCH", ["sampled"])


# ---------------------------------------------------------------------------
# (5) 中等薄壳：AscendMedusaProposer.propose 按已接受 token 数 gather 每请求末位 hidden state
# ---------------------------------------------------------------------------
def test_medusa_propose_gathers_last_accepted_hidden_state(env):
    prop = object.__new__(env.medusa.AscendMedusaProposer)
    prop.device = torch.device("cpu")

    # 两请求，num_draft_tokens=[2,2] → 每请求拼接 (2+1)=3 个 hidden state，共 6 行。
    valid_sampled = [[10, 11], [20]]  # 接受数 = [2, 1]
    spec_meta = types.SimpleNamespace(num_draft_tokens=[2, 2])
    sample_hidden = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)  # 6 行，非「无 draft」分支

    out = prop.propose(valid_sampled, sampling_metadata=None, spec_decode_metadata=spec_meta,
                       sample_hidden_states=sample_hidden)

    # offsets = cumsum(nd+1) - (nd+1) = [0, 3]；indices = offsets + accepted - 1 = [0+2-1, 3+1-1] = [1, 3]
    base, gathered = out
    assert base == "BASE_medusa"
    assert torch.equal(gathered, sample_hidden[torch.tensor([1, 3])])


def test_medusa_propose_passthrough_when_no_draft_tokens(env):
    prop = object.__new__(env.medusa.AscendMedusaProposer)
    prop.device = torch.device("cpu")
    valid_sampled = [[10], [20]]
    sample_hidden = torch.randn(2, 4)  # shape[0] == len(valid) → target 输入不含 draft token
    out = prop.propose(valid_sampled, sampling_metadata=None,
                       spec_decode_metadata=types.SimpleNamespace(num_draft_tokens=[0, 0]),
                       sample_hidden_states=sample_hidden)
    # 直接整块 hidden_states 交父类，不做 gather。
    assert torch.equal(out[1], sample_hidden)


# ---------------------------------------------------------------------------
# (6) 薄入口转调重量级 base：pass_hidden_states_to_model eagle=True / draft=False
# ---------------------------------------------------------------------------
def test_eagle_and_draft_entry_forward_pass_hidden_states_flag(env, monkeypatch):
    recorded = {}

    def _rec_init(self, vllm_config, device, pass_hidden_states_to_model, runner=None):
        recorded["flag"] = pass_hidden_states_to_model

    # 桩掉重量级 AscendSpecDecodeBaseProposer.__init__，只截获薄入口转调时的布尔。
    monkeypatch.setattr(env.llm_base.AscendSpecDecodeBaseProposer, "__init__", _rec_init)

    env.eagle.AscendEagleProposer(object(), object(), runner=None)
    assert recorded["flag"] is True  # eagle 需把 target hidden states 喂进 draft 模型

    env.draft.AscendDraftModelProposer(object(), object(), runner=None)
    assert recorded["flag"] is False  # draft_model 不传 hidden states + 额外校验


# ---------------------------------------------------------------------------
# (7) 重量级 base 的 prepare_inputs：按拒绝 token 数收缩 query/seq + 构造 token_indices（纯 host numpy）
# ---------------------------------------------------------------------------
def test_prepare_inputs_shrinks_by_rejected_and_builds_token_indices(env):
    prop = object.__new__(env.llm_base.AscendSpecDecodeBaseProposer)
    prop.token_arange_np = np.arange(64, dtype=np.int32)
    prop.runner = types.SimpleNamespace(
        actual_seq_lengths_q="ASLQ", attn_state="STATE", decode_token_per_req=1
    )

    # 3 请求，query lens q=[2,4,3] → query_start_loc=[0,2,6,9]
    cad = types.SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 6, 9], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 6, 9], dtype=torch.int32),
        _seq_lens_cpu=None,
        seq_lens_cpu=torch.tensor([10, 20, 15], dtype=torch.int32),
        num_computed_tokens_cpu=None,
        _num_computed_tokens_cpu=None,
        num_reqs=3,
        num_input_tokens=9,
        block_table_tensor="BT",
        slot_mapping=torch.arange(12, dtype=torch.int32),
        slot_mapping_cpu="SM_CPU",
        positions=torch.arange(12, dtype=torch.int32) * 10,
        positions_cpu=None,
    )

    # num_draft_tokens=[1,3,2]，sampled 长度=[2,2,3]
    #   num_rejected = [n+1-len if n>0 else 0] = [1+1-2, 3+1-2, 2+1-3] = [0, 2, 0]
    sampled = [[0, 0], [0, 0], [0, 0, 0]]
    num_draft = [1, 3, 2]

    spec_cad, token_indices = prop.prepare_inputs(cad, sampled, num_draft)

    # new_num_tokens_per_req = q - rejected = [2, 2, 3] → new_query_start_loc=[0,2,4,7]
    assert spec_cad.query_start_loc_cpu.tolist() == [0, 2, 4, 7]
    # token_indices：req0 old_start0→[0,1]；req1 old_start2,len2→[2,3]；req2 old_start6,len3→[6,7,8]
    assert token_indices.tolist() == [0, 1, 2, 3, 6, 7, 8]
    # new_seq_lens = seq_lens - rejected = [10, 18, 15]
    assert spec_cad.seq_lens_cpu.tolist() == [10, 18, 15]
    assert spec_cad.num_actual_tokens == 7
    # positions 按 token_indices 重排。
    assert torch.equal(spec_cad.positions, cad.positions[torch.tensor([0, 1, 2, 3, 6, 7, 8])])
