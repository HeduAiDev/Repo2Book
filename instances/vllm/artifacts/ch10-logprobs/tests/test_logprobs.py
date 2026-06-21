"""Ch10 logprobs 精简版测试 —— 复现真实 vLLM 的可观察行为。

不 import vllm；用桩 tokenizer/张量。覆盖：容器初始化三分支、sample/prompt
双路装配、累计 logprob、上下文感知 UTF-8 字节回退修正、_get_sampled_context_ids
flat/nested 双路径、append rank 链、DELTA pop/切尾。
"""
import itertools
from dataclasses import dataclass

import conftest  # noqa: F401  (sets sys.path)
from conftest import (
    ByteFallbackTokenizer,
    FakeArray,
    FakeDetokenizer,
    IdentityTokenizer,
)

from logprobs import (
    FlatLogprobs,
    Logprob,
    append_logprobs_for_next_position,
    create_prompt_logprobs,
    create_sample_logprobs,
)
from logprobs_processor import LogprobsProcessor
from outputs import LogprobsLists, LogprobsTensors
from output_processor import (
    CompletionOutput,
    RequestOutputKind,
    RequestState,
)


# --- 桩 sampling_params / request ----------------------------------------

@dataclass
class FakeSamplingParams:
    num_logprobs: int | None = None
    prompt_logprobs: int | None = None
    flat_logprobs: bool = False


@dataclass
class FakeRequest:
    sampling_params: FakeSamplingParams


@dataclass
class FakeEngineCoreOutput:
    new_logprobs: object = None
    new_prompt_logprobs_tensors: object = None


def make_proc(tokenizer=None, num_logprobs=None, num_prompt_logprobs=None,
              flat=False):
    req = FakeRequest(FakeSamplingParams(
        num_logprobs=num_logprobs,
        prompt_logprobs=num_prompt_logprobs,
        flat_logprobs=flat,
    ))
    return LogprobsProcessor.from_new_request(tokenizer, req)


# ========================================================================
# from_new_request：三分支语义
# ========================================================================

def test_from_new_request_all_disabled():
    p = make_proc(num_logprobs=None, num_prompt_logprobs=None)
    assert p.logprobs is None
    assert p.prompt_logprobs is None
    assert p.cumulative_logprob is None
    assert p.num_logprobs is None
    assert p.num_prompt_logprobs is None


def test_from_new_request_sample_enabled_inits_cumulative_zero():
    p = make_proc(num_logprobs=2)
    assert p.logprobs == []          # nested 容器，空
    assert p.cumulative_logprob == 0.0
    assert p.prompt_logprobs is None


def test_from_new_request_prompt_first_position_is_none():
    p = make_proc(num_prompt_logprobs=2)
    # create_prompt_logprobs 首位 append(None) 占位
    assert p.prompt_logprobs == [None]
    assert p.cumulative_logprob is None  # sample 关闭


def test_from_new_request_flat_containers():
    p = make_proc(num_logprobs=2, num_prompt_logprobs=2, flat=True)
    assert isinstance(p.logprobs, FlatLogprobs)
    assert isinstance(p.prompt_logprobs, FlatLogprobs)
    # prompt flat 首位也写了 None 占位（空区间）
    assert len(p.prompt_logprobs) == 1
    assert p.prompt_logprobs.start_indices == [0]
    assert p.prompt_logprobs.end_indices == [0]


# ========================================================================
# _update_sample_logprobs：累计、sampled-first、去 token
# ========================================================================

def _lists_one_step(token_ids, logprobs, rank):
    """构造单 step 的 LogprobsLists（外层 1 个位置）。"""
    return LogprobsLists(
        logprob_token_ids=[FakeArray(token_ids)],
        logprobs=[FakeArray(logprobs)],
        sampled_token_ranks=[FakeArray(rank)],
        cu_num_generated_tokens=None,
    )


def test_update_sample_logprobs_accumulates_cumulative():
    p = make_proc(tokenizer=IdentityTokenizer(), num_logprobs=2)
    # sampler 把被采样 token 放第一个；cumulative += logprobs[0]
    p._update_sample_logprobs(_lists_one_step([1065, 1066, 1067],
                                              [-0.5, -1.0, -2.0], 1))
    assert p.cumulative_logprob == -0.5
    p._update_sample_logprobs(_lists_one_step([1097, 1098, 1099],
                                              [-0.25, -1.0, -2.0], 1))
    assert p.cumulative_logprob == -0.75


def test_update_sample_logprobs_nested_structure():
    p = make_proc(tokenizer=IdentityTokenizer(), num_logprobs=2)
    p._update_sample_logprobs(_lists_one_step([1065, 1066, 1067],
                                              [-0.5, -1.0, -2.0], 7))
    assert len(p.logprobs) == 1
    pos0 = p.logprobs[0]
    # 第一个 token 是被采样 token，rank = 真实 vocab rank（7）
    sampled = pos0[1065]
    assert sampled.rank == 7
    assert sampled.logprob == -0.5
    assert sampled.decoded_token == "A"   # chr(1065-1000)=chr(65)='A'
    # 后续候选 rank = 1..num_logprobs
    assert pos0[1066].rank == 1
    assert pos0[1067].rank == 2


def test_update_sample_logprobs_flat_first_entry_is_sampled():
    p = make_proc(tokenizer=IdentityTokenizer(), num_logprobs=2, flat=True)
    p._update_sample_logprobs(_lists_one_step([1065, 1066, 1067],
                                              [-0.5, -1.0, -2.0], 7))
    # FlatLogprobs：每位置第一个 token_id 即被采样 token
    assert p.logprobs.token_ids[p.logprobs.start_indices[0]] == 1065
    assert p.logprobs.ranks[0] == 7


def test_update_sample_logprobs_no_tokenizer_uses_nones():
    p = make_proc(tokenizer=None, num_logprobs=2)
    p._update_sample_logprobs(_lists_one_step([1065, 1066, 1067],
                                              [-0.5, -1.0, -2.0], 1))
    pos0 = p.logprobs[0]
    assert pos0[1065].decoded_token is None


# ========================================================================
# _get_sampled_context_ids：每位置第一个=被选中 token；flat/nested；max 4
# ========================================================================

def test_get_context_ids_nested_first_key():
    p = make_proc(tokenizer=IdentityTokenizer(), num_logprobs=1)
    for tid in (1065, 1066, 1067):
        p._update_sample_logprobs(_lists_one_step([tid, tid + 1],
                                                  [-0.1, -1.0], 1))
    ids = LogprobsProcessor._get_sampled_context_ids(p.logprobs)
    assert ids == [1065, 1066, 1067]


def test_get_context_ids_flat_path():
    p = make_proc(tokenizer=IdentityTokenizer(), num_logprobs=1, flat=True)
    for tid in (1065, 1066, 1067):
        p._update_sample_logprobs(_lists_one_step([tid, tid + 1],
                                                  [-0.1, -1.0], 1))
    ids = LogprobsProcessor._get_sampled_context_ids(p.logprobs)
    assert ids == [1065, 1066, 1067]


def test_get_context_ids_caps_at_max_context():
    p = make_proc(tokenizer=IdentityTokenizer(), num_logprobs=1)
    for tid in range(1065, 1075):   # 10 positions
        p._update_sample_logprobs(_lists_one_step([tid], [-0.1], 1))
    ids = LogprobsProcessor._get_sampled_context_ids(p.logprobs)
    assert len(ids) == 4
    assert ids == [1071, 1072, 1073, 1074]


def test_get_context_ids_empty():
    assert LogprobsProcessor._get_sampled_context_ids(None) == []
    assert LogprobsProcessor._get_sampled_context_ids([]) == []


def test_get_context_ids_flat_skips_empty_positions():
    # prompt flat 首位 append(None) 是空区间，应被跳过
    flat = create_prompt_logprobs(True)  # 含首位 None
    append_logprobs_for_next_position(flat, [1066], [-0.1], ["B"], 1, 1)
    ids = LogprobsProcessor._get_sampled_context_ids(flat)
    assert ids == [1066]   # 首位空区间被跳过


# ========================================================================
# 字节回退 UTF-8 多字节重建（核心）
# ========================================================================

def test_byte_fallback_tokenizer_decodes_partial_to_replacement():
    """前提自检：单个 byte token 解出 U+FFFD。"""
    tok = ByteFallbackTokenizer()
    # "中" = e4 b8 ad
    assert tok.decode([0xE4]).endswith("�")
    assert tok.decode([0xE4, 0xB8, 0xAD]) == "中"


def test_correct_decoded_token_reconstructs_multibyte_char():
    p = make_proc(tokenizer=ByteFallbackTokenizer(), num_logprobs=1)
    # "中" 三字节：前两字节作上下文，最后一字节完成
    e4, b8, ad = 0xE4, 0xB8, 0xAD
    # 第三个字节 token，上下文是前两个字节 token
    result = p._correct_decoded_token(ad, context_token_ids=[e4, b8])
    assert result == "中"


def test_correct_decoded_token_returns_empty_when_incomplete():
    p = make_proc(tokenizer=ByteFallbackTokenizer(), num_logprobs=1)
    e4, b8 = 0xE4, 0xB8
    # 只有两个字节，"中" 还差一个字节，拼不出完整字符 → ''
    assert p._correct_decoded_token(b8, context_token_ids=[e4]) == ""


def test_correct_decoded_token_strips_clean_prefix():
    p = make_proc(tokenizer=ByteFallbackTokenizer(), num_logprobs=1)
    # 干净 ASCII 'X'(ord 88 → byte 0x58) 在前，再接 "中" 的三字节
    x = 0x58
    e4, b8, ad = 0xE4, 0xB8, 0xAD
    # 完成 token = ad，上下文 = [X, e4, b8]
    result = p._correct_decoded_token(ad, context_token_ids=[x, e4, b8])
    # 干净前缀 'X' 应被剥掉，只归属本 token 的 "中"
    assert result == "中"


def test_verify_tokens_corrects_only_replacement_ended():
    p = make_proc(tokenizer=ByteFallbackTokenizer(), num_logprobs=1)
    e4, b8, ad = 0xE4, 0xB8, 0xAD
    # 同一位置的候选 [完成字节 ad, 干净 ASCII 'Y']；ad 解出 � 需修正
    ad_decoded = p.tokenizer.decode([ad])      # '�'
    y_decoded = p.tokenizer.decode([0x59])     # 'Y'
    decoded_list = [ad_decoded, y_decoded]
    out = p._verify_tokens(
        decoded_tokens_list=list(decoded_list),
        tokens=[ad, 0x59],
        context_token_ids=[e4, b8],
    )
    assert out[0] == "中"   # 候选0 用序列上下文修正
    assert out[1] == "Y"    # 候选1 不以 � 结尾，原样保留


def test_verify_tokens_candidates_use_same_context():
    """横向候选 vs 纵向上下文：每个候选用同一份序列上下文独立修正。"""
    p = make_proc(tokenizer=ByteFallbackTokenizer(), num_logprobs=1)
    e4, b8, ad = 0xE4, 0xB8, 0xAD
    # 两个候选都是 'ad'（同字节），都应被同一上下文修成 "中"
    out = p._verify_tokens(
        decoded_tokens_list=[p.tokenizer.decode([ad]), p.tokenizer.decode([ad])],
        tokens=[ad, ad],
        context_token_ids=[e4, b8],
    )
    assert out == ["中", "中"]


# ========================================================================
# _update_sample_logprobs end-to-end 字节回退（用累计上下文）
# ========================================================================

def test_sample_logprobs_multibyte_across_positions():
    """逐位置喂 "中" 的三个字节 token，最后一位应解出 "中"，前两位为 ''。"""
    p = make_proc(tokenizer=ByteFallbackTokenizer(), num_logprobs=0)
    e4, b8, ad = 0xE4, 0xB8, 0xAD
    for tid in (e4, b8, ad):
        p._update_sample_logprobs(_lists_one_step([tid], [-0.1], 1))
    # 各位置第一个（被采样）token 的 decoded_token
    assert p.logprobs[0][e4].decoded_token == ""
    assert p.logprobs[1][b8].decoded_token == ""
    assert p.logprobs[2][ad].decoded_token == "中"


# ========================================================================
# _update_prompt_logprobs：torch 张量 Pythonize + 逐位置切片
# ========================================================================

def test_update_prompt_logprobs_pythonizes_and_slices():
    p = make_proc(tokenizer=IdentityTokenizer(), num_prompt_logprobs=2)
    # 2 个 prompt 位置，每位置 2 个候选 → [2,2] 张量
    token_ids = FakeArray([[1065, 1066], [1067, 1068]])
    logprobs = FakeArray([[-0.5, -1.0], [-0.25, -1.5]])
    ranks = FakeArray([3, 4])
    tensors = LogprobsTensors(token_ids, logprobs, ranks, None)
    p._update_prompt_logprobs(tensors)
    # 首位 None 占位仍在，后接 2 个新位置
    assert p.prompt_logprobs[0] is None
    pos1 = p.prompt_logprobs[1]
    assert pos1[1065].rank == 3       # 被选中 prompt token 的真实 rank
    assert pos1[1065].decoded_token == "A"
    assert pos1[1066].rank == 1
    pos2 = p.prompt_logprobs[2]
    assert pos2[1067].rank == 4
    # prompt 不维护 cumulative
    assert p.cumulative_logprob is None


def test_update_prompt_logprobs_context_from_prompt_chain():
    """prompt 用 self.prompt_logprobs 做上下文：跨位置字节回退也能重建。"""
    p = make_proc(tokenizer=ByteFallbackTokenizer(), num_prompt_logprobs=0)
    e4, b8, ad = 0xE4, 0xB8, 0xAD
    # 3 个 prompt 位置，每位置 1 个候选，依次是 "中" 的三个字节
    token_ids = FakeArray([[e4], [b8], [ad]])
    logprobs = FakeArray([[-0.1], [-0.1], [-0.1]])
    ranks = FakeArray([1, 1, 1])
    p._update_prompt_logprobs(LogprobsTensors(token_ids, logprobs, ranks, None))
    # 首位 None + 3 个 prompt 位置
    assert p.prompt_logprobs[1][e4].decoded_token == ""
    assert p.prompt_logprobs[2][b8].decoded_token == ""
    assert p.prompt_logprobs[3][ad].decoded_token == "中"


# ========================================================================
# append_logprobs_for_next_position：rank 链 / flat vs nested
# ========================================================================

def test_append_rank_chain_nested():
    container = []
    append_logprobs_for_next_position(container, [1065, 1066, 1067],
                                      [-0.5, -1.0, -2.0], ["A", "B", "C"],
                                      rank=9, num_logprobs=2)
    pos = container[0]
    assert pos[1065].rank == 9    # chain 第一个 = 传入 rank
    assert pos[1066].rank == 1
    assert pos[1067].rank == 2


def test_append_full_vocab_minus_one():
    container = []
    append_logprobs_for_next_position(container, [1065, 1066],
                                      [-0.5, -1.0], ["A", "B"],
                                      rank=5, num_logprobs=-1)
    # num_logprobs == -1 ⇒ 用 len(logprobs)=2 顶替
    pos = container[0]
    assert pos[1065].rank == 5
    assert pos[1066].rank == 1


def test_append_flat_vs_nested_equivalent_first_token():
    nested = []
    flat = FlatLogprobs()
    for c in (nested, flat):
        append_logprobs_for_next_position(c, [1065, 1066], [-0.5, -1.0],
                                          ["A", "B"], rank=3, num_logprobs=1)
    assert flat.token_ids[flat.start_indices[0]] == 1065
    assert next(iter(nested[0])) == 1065


# ========================================================================
# pop_prompt_logprobs（DELTA 语义）
# ========================================================================

def test_pop_prompt_logprobs_returns_and_clears():
    p = make_proc(tokenizer=IdentityTokenizer(), num_prompt_logprobs=2)
    token_ids = FakeArray([[1065, 1066]])
    logprobs = FakeArray([[-0.5, -1.0]])
    ranks = FakeArray([1])
    p._update_prompt_logprobs(LogprobsTensors(token_ids, logprobs, ranks, None))
    popped = p.pop_prompt_logprobs()
    assert popped is not None and len(popped) == 2   # None 占位 + 1 位置
    # pop 后清空为 []
    assert p.prompt_logprobs == []


def test_pop_prompt_logprobs_none_when_disabled():
    p = make_proc(num_prompt_logprobs=None)
    assert p.pop_prompt_logprobs() is None


# ========================================================================
# update_from_output：分派（两 if 互不排斥）
# ========================================================================

def test_update_from_output_dispatches_both():
    p = make_proc(tokenizer=IdentityTokenizer(), num_logprobs=1,
                  num_prompt_logprobs=1)
    sample = _lists_one_step([1065, 1066], [-0.5, -1.0], 1)
    prompt = LogprobsTensors(FakeArray([[1080]]), FakeArray([[-0.3]]),
                             FakeArray([1]), None)
    p.update_from_output(FakeEngineCoreOutput(new_logprobs=sample,
                                              new_prompt_logprobs_tensors=prompt))
    assert len(p.logprobs) == 1
    assert len(p.prompt_logprobs) == 2   # None + 1


def test_update_from_output_sample_only():
    p = make_proc(tokenizer=IdentityTokenizer(), num_logprobs=1)
    sample = _lists_one_step([1065, 1066], [-0.5, -1.0], 1)
    p.update_from_output(FakeEngineCoreOutput(new_logprobs=sample))
    assert len(p.logprobs) == 1


# ========================================================================
# _new_completion_output：DELTA 切尾 / cumulative 不切
# ========================================================================

def test_completion_output_delta_tail_slices_logprobs():
    p = make_proc(tokenizer=IdentityTokenizer(), num_logprobs=1)
    for tid in (1065, 1066, 1067):
        p._update_sample_logprobs(_lists_one_step([tid], [-0.5], 1))
    state = RequestState(
        logprobs_processor=p,
        output_kind=RequestOutputKind.DELTA,
        detokenizer=FakeDetokenizer(text="C"),
    )
    out = state._new_completion_output(token_ids=[1067])   # 本批 1 个新 token
    assert isinstance(out, CompletionOutput)
    assert len(out.logprobs) == 1            # 只切尾最后 1 个
    assert next(iter(out.logprobs[0])) == 1067
    # cumulative 始终是整段累计，不随切尾重置
    assert out.cumulative_logprob == -1.5


def test_completion_output_non_delta_keeps_all():
    p = make_proc(tokenizer=IdentityTokenizer(), num_logprobs=1)
    for tid in (1065, 1066, 1067):
        p._update_sample_logprobs(_lists_one_step([tid], [-0.5], 1))
    state = RequestState(
        logprobs_processor=p,
        output_kind=RequestOutputKind.FINAL_ONLY,
        detokenizer=FakeDetokenizer(text="ABC", output_token_ids=[1065, 1066, 1067]),
    )
    out = state._new_completion_output(token_ids=[1067])
    assert len(out.logprobs) == 3   # FINAL：全量
    assert out.token_ids == [1065, 1066, 1067]


def test_completion_output_delta_flat_slice_returns_flat():
    p = make_proc(tokenizer=IdentityTokenizer(), num_logprobs=1, flat=True)
    for tid in (1065, 1066, 1067):
        p._update_sample_logprobs(_lists_one_step([tid], [-0.5], 1))
    state = RequestState(
        logprobs_processor=p,
        output_kind=RequestOutputKind.DELTA,
        detokenizer=FakeDetokenizer(text="BC"),
    )
    out = state._new_completion_output(token_ids=[1066, 1067])   # 2 个新 token
    assert isinstance(out.logprobs, FlatLogprobs)
    assert len(out.logprobs) == 2
    # 0-indexed 平移后仍能取回最后两位置的被选中 token
    assert out.logprobs.token_ids[out.logprobs.start_indices[0]] == 1066
    assert out.logprobs.token_ids[out.logprobs.start_indices[1]] == 1067


# ========================================================================
# FlatLogprobs 不变式：只追加、不可改
# ========================================================================

def test_flat_logprobs_is_append_only():
    flat = FlatLogprobs()
    flat.append({1065: Logprob(logprob=-0.5, rank=1, decoded_token="A")})
    import pytest
    with pytest.raises(TypeError):
        flat[0] = {}
    with pytest.raises(TypeError):
        del flat[0]
    with pytest.raises(TypeError):
        flat.insert(0, {})


def test_flat_logprobs_getitem_rebuilds_dict():
    flat = FlatLogprobs()
    flat.append({1065: Logprob(logprob=-0.5, rank=1, decoded_token="A")})
    pos = flat[0]
    assert isinstance(pos, dict)
    assert pos[1065].decoded_token == "A"
