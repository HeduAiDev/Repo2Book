"""ch05 Stage 1 输入处理 —— 复现真实 vLLM 可观察行为的单元测试。

测的是精简版是否复现真实 vllm/v1/engine/input_processor.py / parallel_sampling.py /
async_llm.py 的**可观察行为**，而非精简版自洽。纯单元测试，不 import vllm，host 直接跑：
    python3 -m pytest instances/vllm/artifacts/ch05-input-processing/tests -q
"""

import os
import re
import sys

import pytest

_IMPL = os.path.join(os.path.dirname(__file__), "..", "implementation")
sys.path.insert(0, os.path.abspath(_IMPL))

from async_llm import AsyncLLM  # noqa: E402
from config import (  # noqa: E402
    LoRAConfig,
    ModelConfig,
    Renderer,
    VllmConfig,
    _FakeTokenizer,
)
from input_processor import InputProcessor  # noqa: E402
from messages import (  # noqa: E402
    CompletionOutput,
    EngineCoreRequest,
    LoRARequest,
    MultiModalFeatureSpec,
    PlaceholderRange,
    PoolingParams,
    RequestOutputKind,
    SamplingParams,
    argsort_mm_positions,
    split_enc_dec_input,
)
from parallel_sampling import ParentRequest  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


def make_processor(*, max_model_len=2048, vocab_size=32000, lora=None,
                   eos_token_id=None, generation_config=None,
                   max_token_id=31999):
    cfg = VllmConfig(
        model_config=ModelConfig(max_model_len=max_model_len, vocab_size=vocab_size),
        lora_config=lora,
    )
    if generation_config is not None:
        cfg.model_config.try_get_generation_config = lambda: generation_config
    tok = _FakeTokenizer(max_token_id=max_token_id)
    renderer = Renderer(tokenizer=tok, eos_token_id=eos_token_id)
    return InputProcessor(cfg, renderer=renderer)


def tokens_prompt(ids, **extra):
    d = {"type": "token", "prompt_token_ids": list(ids)}
    d.update(extra)
    return d


# --------------------------------------------------------------------------- #
# process_inputs 主线
# --------------------------------------------------------------------------- #


def test_rendered_token_prompt_passthrough_to_engine_core_request():
    p = make_processor()
    req = p.process_inputs(
        "req-1", tokens_prompt([1, 2, 3]), SamplingParams(max_tokens=8),
        supported_tasks=("generate",), arrival_time=1.0,
    )
    assert isinstance(req, EngineCoreRequest)
    assert req.request_id == "req-1"          # assign_request_id 尚未调用
    assert req.prompt_token_ids == [1, 2, 3]
    assert req.prompt_embeds is None
    assert req.sampling_params is not None
    assert req.pooling_params is None
    assert req.arrival_time == 1.0


def test_max_tokens_defaulted_to_remaining_context():
    p = make_processor(max_model_len=100)
    req = p.process_inputs(
        "r", tokens_prompt(list(range(10))), SamplingParams(max_tokens=None),
        supported_tasks=("generate",),
    )
    # max_tokens = max_model_len - seq_len = 100 - 10
    assert req.sampling_params.max_tokens == 90


def test_explicit_max_tokens_preserved():
    p = make_processor(max_model_len=100)
    req = p.process_inputs(
        "r", tokens_prompt([1, 2, 3]), SamplingParams(max_tokens=7),
        supported_tasks=("generate",),
    )
    assert req.sampling_params.max_tokens == 7


def test_sampling_params_is_cloned_not_mutated_in_place():
    p = make_processor(max_model_len=100)
    original = SamplingParams(max_tokens=None)
    req = p.process_inputs(
        "r", tokens_prompt([1, 2, 3]), original, supported_tasks=("generate",),
    )
    # 调用方传入对象不被污染
    assert original.max_tokens is None
    assert req.sampling_params is not original
    assert req.sampling_params.max_tokens == 97


def test_cache_salt_passthrough():
    p = make_processor()
    req = p.process_inputs(
        "r", tokens_prompt([1, 2], cache_salt="salt-x"),
        SamplingParams(max_tokens=4), supported_tasks=("generate",),
    )
    assert req.cache_salt == "salt-x"


def test_pooling_params_path():
    p = make_processor()
    req = p.process_inputs(
        "r", tokens_prompt([1, 2, 3]), PoolingParams(task="embed"),
        supported_tasks=("embed",),
    )
    assert req.pooling_params is not None
    assert req.sampling_params is None


# --------------------------------------------------------------------------- #
# 参数 / LoRA 校验
# --------------------------------------------------------------------------- #


def test_generation_unsupported_raises():
    p = make_processor()
    with pytest.raises(ValueError, match="does not support generation"):
        p.process_inputs("r", tokens_prompt([1]), SamplingParams(max_tokens=1),
                         supported_tasks=("embed",))


def test_pooling_unsupported_raises():
    p = make_processor()
    with pytest.raises(ValueError, match="does not support pooling"):
        p.process_inputs("r", tokens_prompt([1]), PoolingParams(task="embed"),
                         supported_tasks=("generate",))


def test_bad_params_type_raises():
    p = make_processor()
    with pytest.raises(TypeError):
        p.process_inputs("r", tokens_prompt([1]), object(),
                         supported_tasks=("generate",))


def test_lora_request_without_lora_enabled_raises():
    p = make_processor(lora=None)
    with pytest.raises(ValueError, match="LoRA is not enabled"):
        p.process_inputs("r", tokens_prompt([1]), SamplingParams(max_tokens=1),
                         supported_tasks=("generate",),
                         lora_request=LoRARequest("adapter"))


def test_data_parallel_rank_out_of_range_raises():
    p = make_processor()
    with pytest.raises(ValueError, match="out of range"):
        p.process_inputs("r", tokens_prompt([1]), SamplingParams(max_tokens=1),
                         supported_tasks=("generate",), data_parallel_rank=5)


# --------------------------------------------------------------------------- #
# 模型输入校验：空 / 超长 / 等长 / token 越界
# --------------------------------------------------------------------------- #


def test_empty_decoder_prompt_raises():
    p = make_processor()
    with pytest.raises(ValueError, match="cannot be empty"):
        p.process_inputs("r", tokens_prompt([]), SamplingParams(max_tokens=1),
                         supported_tasks=("generate",))


def test_prompt_longer_than_max_model_len_raises():
    p = make_processor(max_model_len=4)
    with pytest.raises(ValueError, match="longer than the maximum model length"):
        p.process_inputs("r", tokens_prompt([1, 2, 3, 4, 5]),
                         SamplingParams(max_tokens=1), supported_tasks=("generate",))


def test_prompt_equal_to_max_len_for_generate_raises():
    p = make_processor(max_model_len=3)
    with pytest.raises(ValueError, match="at least 1"):
        p.process_inputs("r", tokens_prompt([1, 2, 3]),
                         SamplingParams(max_tokens=1), supported_tasks=("generate",))


def test_token_id_out_of_vocabulary_raises():
    p = make_processor(vocab_size=10, max_token_id=9)
    with pytest.raises(ValueError, match="out of vocabulary"):
        p.process_inputs("r", tokens_prompt([1, 2, 999]),
                         SamplingParams(max_tokens=1), supported_tasks=("generate",))


def test_token_id_uses_larger_of_tokenizer_and_model_vocab():
    # model_vocab_size-1 = 49, tokenizer.max_token_id = 9 → 取较大者 49，id 40 合法
    p = make_processor(vocab_size=50, max_token_id=9)
    req = p.process_inputs("r", tokens_prompt([40]),
                           SamplingParams(max_tokens=1), supported_tasks=("generate",))
    assert req.prompt_token_ids == [40]


# --------------------------------------------------------------------------- #
# SamplingParams 补全：eos / bad_words
# --------------------------------------------------------------------------- #


def test_update_from_generation_config_injects_eos():
    p = make_processor(eos_token_id=2, generation_config={"eos_token_id": [5, 6]})
    req = p.process_inputs("r", tokens_prompt([1, 2, 3]),
                           SamplingParams(max_tokens=4), supported_tasks=("generate",))
    sp = req.sampling_params
    assert sp.eos_token_id == 2
    assert {5, 6}.issubset(set(sp.stop_token_ids))


def test_update_from_tokenizer_validates_bad_words():
    p = make_processor(max_token_id=9, vocab_size=10)
    # bad word "AB" -> ord('A')%10=5, ord('B')%10=6 都 <=9，合法
    req = p.process_inputs("r", tokens_prompt([1]),
                           SamplingParams(max_tokens=1, bad_words=["AB"]),
                           supported_tasks=("generate",))
    assert req.sampling_params.bad_words_token_ids


# --------------------------------------------------------------------------- #
# embeds 路径
# --------------------------------------------------------------------------- #


def test_embeds_prompt_sets_prompt_embeds():
    p = make_processor(max_model_len=100)
    embeds = [[0.0] * 4 for _ in range(6)]  # 6 行 = seq_len 6
    prompt = {"type": "embeds", "prompt_embeds": embeds}
    req = p.process_inputs("r", prompt, SamplingParams(max_tokens=None),
                           supported_tasks=("generate",))
    assert req.prompt_embeds is embeds
    assert req.prompt_token_ids is None
    assert req.sampling_params.max_tokens == 94  # 100 - 6


# --------------------------------------------------------------------------- #
# 多模态展平
# --------------------------------------------------------------------------- #


def test_argsort_mm_positions_sorts_interleaved_by_offset():
    positions = {
        "image": [PlaceholderRange(offset=0, length=4),
                  PlaceholderRange(offset=20, length=4)],
        "audio": [PlaceholderRange(offset=10, length=2)],
    }
    order = argsort_mm_positions(positions)
    # 按 offset 升序：image[0]@0, audio[0]@10, image[1]@20
    assert order == [("image", 0), ("audio", 0), ("image", 1)]


def test_multimodal_flatten_into_feature_specs():
    p = make_processor(max_model_len=1000)
    prompt = {
        "type": "multimodal",
        "prompt_token_ids": list(range(30)),
        "mm_kwargs": {"image": ["IMG0", "IMG1"], "audio": ["AUD0"]},
        "mm_hashes": {"image": ["h_img0", "h_img1"], "audio": ["h_aud0"]},
        "mm_placeholders": {
            "image": [PlaceholderRange(0, 4), PlaceholderRange(20, 4)],
            "audio": [PlaceholderRange(10, 2)],
        },
    }
    req = p.process_inputs("r", prompt, SamplingParams(max_tokens=4),
                           supported_tasks=("generate",))
    feats = req.mm_features
    assert all(isinstance(f, MultiModalFeatureSpec) for f in feats)
    # 展平后按 offset 升序
    assert [f.mm_hash for f in feats] == ["h_img0", "h_aud0", "h_img1"]
    assert [f.data for f in feats] == ["IMG0", "AUD0", "IMG1"]
    # 无 LoRA tower connector → identifier == mm_hash
    assert [f.identifier for f in feats] == ["h_img0", "h_aud0", "h_img1"]


def test_mm_hashes_must_be_strings():
    p = make_processor(max_model_len=1000)
    prompt = {
        "type": "multimodal",
        "prompt_token_ids": [0, 1, 2, 3],
        "mm_kwargs": {"image": ["IMG0"]},
        "mm_hashes": {"image": [123]},  # 非字符串
        "mm_placeholders": {"image": [PlaceholderRange(0, 4)]},
    }
    with pytest.raises(ValueError, match="mm_hashes must contain only strings"):
        p.process_inputs("r", prompt, SamplingParams(max_tokens=4),
                         supported_tasks=("generate",))


def test_mm_identifier_gets_lora_prefix_when_tower_connector_enabled():
    p = make_processor(max_model_len=1000,
                       lora=LoRAConfig(enable_tower_connector_lora=True))
    prompt = {
        "type": "multimodal",
        "prompt_token_ids": [0, 1, 2, 3],
        "mm_kwargs": {"image": ["IMG0"]},
        "mm_hashes": {"image": ["h0"]},
        "mm_placeholders": {"image": [PlaceholderRange(0, 4)]},
    }
    req = p.process_inputs("r", prompt, SamplingParams(max_tokens=4),
                           supported_tasks=("generate",),
                           lora_request=LoRARequest("myadapter"))
    assert req.mm_features[0].identifier == "myadapter:h0"
    assert req.mm_features[0].mm_hash == "h0"


# --------------------------------------------------------------------------- #
# split_enc_dec_input
# --------------------------------------------------------------------------- #


def test_split_enc_dec_decoder_only():
    inp = tokens_prompt([1, 2])
    enc, dec = split_enc_dec_input(inp)
    assert enc is None
    assert dec is inp


def test_split_enc_dec_compound():
    enc_in = tokens_prompt([9])
    dec_in = tokens_prompt([1, 2])
    inp = {"type": "enc_dec", "encoder_prompt": enc_in, "decoder_prompt": dec_in}
    enc, dec = split_enc_dec_input(inp)
    assert enc is enc_in
    assert dec is dec_in


# --------------------------------------------------------------------------- #
# raw-prompt deprecated 兜底路径
# --------------------------------------------------------------------------- #


def test_raw_prompt_fallback_tokenizes():
    p = make_processor(max_model_len=1000, max_token_id=255)
    # raw prompt（无 'type'）→ InputPreprocessor.preprocess 现场 tokenize
    req = p.process_inputs("r", {"prompt": "Hi"}, SamplingParams(max_tokens=4),
                           supported_tasks=("generate",))
    assert req.prompt_token_ids == [ord("H"), ord("i")]


# --------------------------------------------------------------------------- #
# assign_request_id
# --------------------------------------------------------------------------- #


def _make_req(request_id="req-abc"):
    return EngineCoreRequest(
        request_id=request_id, prompt_token_ids=[1], mm_features=None,
        sampling_params=SamplingParams(max_tokens=1), pooling_params=None,
        arrival_time=0.0, lora_request=None, cache_salt=None,
        data_parallel_rank=None,
    )


def test_assign_request_id_adds_8_char_suffix_and_keeps_external():
    req = _make_req("req-abc")
    InputProcessor.assign_request_id(req)
    assert req.external_req_id == "req-abc"
    assert req.request_id.startswith("req-abc-")
    suffix = req.request_id[len("req-abc-"):]
    assert len(suffix) == 8
    assert re.fullmatch(r"[0-9a-f]{8}", suffix)


def test_assign_request_id_rejects_preset_external():
    req = _make_req()
    req.external_req_id = "already-set"
    with pytest.raises(ValueError, match="external_req_id field should not be set"):
        InputProcessor.assign_request_id(req)


def test_assign_request_id_uniqueness_for_duplicate_external():
    r1, r2 = _make_req("dup"), _make_req("dup")
    InputProcessor.assign_request_id(r1)
    InputProcessor.assign_request_id(r2)
    assert r1.request_id != r2.request_id
    assert r1.external_req_id == r2.external_req_id == "dup"


# --------------------------------------------------------------------------- #
# ParentRequest fan-out
# --------------------------------------------------------------------------- #


def _parent(n=3, seed=None, output_kind=RequestOutputKind.CUMULATIVE):
    req = EngineCoreRequest(
        request_id="par-1", prompt_token_ids=[1], mm_features=None,
        sampling_params=SamplingParams(n=n, seed=seed, output_kind=output_kind,
                                       max_tokens=4),
        pooling_params=None, arrival_time=0.0, lora_request=None,
        cache_salt=None, data_parallel_rank=None, external_req_id="par-1",
    )
    return ParentRequest(req)


def test_child_ids_and_n_one_each():
    parent = _parent(n=3)
    ids = []
    for i in range(3):
        cid, cparams = parent.get_child_info(i)
        ids.append(cid)
        assert cparams.n == 1
    assert ids == ["0_par-1", "1_par-1", "2_par-1"]
    assert parent.child_requests == set(ids)


def test_child_params_shared_when_no_seed():
    parent = _parent(n=3, seed=None)
    _, c0 = parent.get_child_info(0)
    _, c1 = parent.get_child_info(1)
    # 无 seed → 复用同一份缓存克隆
    assert c0 is c1


def test_child_params_unique_seed_when_seed_set():
    parent = _parent(n=3, seed=100)
    seeds = [parent.get_child_info(i)[1].seed for i in range(3)]
    assert seeds == [100, 101, 102]


def test_get_outputs_streaming_forwards_each():
    parent = _parent(n=2, output_kind=RequestOutputKind.CUMULATIVE)
    for i in range(2):
        parent.get_child_info(i)
    out = CompletionOutput(index=0, _finished=False)
    res, finished = parent.get_outputs("0_par-1", out)
    assert res == [out]
    assert finished is False


def test_get_outputs_final_only_aggregates_until_all_done():
    parent = _parent(n=2, output_kind=RequestOutputKind.FINAL_ONLY)
    for i in range(2):
        parent.get_child_info(i)
    o0 = CompletionOutput(index=0, _finished=True)
    res, finished = parent.get_outputs("0_par-1", o0)
    assert res == []           # 还差一个 child
    assert finished is False
    o1 = CompletionOutput(index=1, _finished=True)
    res, finished = parent.get_outputs("1_par-1", o1)
    assert finished is True
    assert res[0] is o0 and res[1] is o1


# --------------------------------------------------------------------------- #
# AsyncLLM.add_request n 路由
# --------------------------------------------------------------------------- #


def test_async_add_request_single_when_n_one():
    p = make_processor()
    llm = AsyncLLM(p)
    llm.add_request("r", tokens_prompt([1, 2]), SamplingParams(n=1, max_tokens=4))
    assert len(llm.added) == 1
    assert llm.added[0][1] is None  # 无 parent_request


def test_async_add_request_fans_out_when_n_gt_one():
    p = make_processor()
    llm = AsyncLLM(p)
    llm.add_request("r", tokens_prompt([1, 2]), SamplingParams(n=3, max_tokens=4))
    assert len(llm.added) == 3
    child_ids = [a[0] for a in llm.added]
    # child id 形如 '{idx}_{request_id}'，request_id 已带随机后缀
    assert all(re.match(r"\d+_r-[0-9a-f]{8}", cid) for cid in child_ids)
    # 都共享同一个 ParentRequest
    parents = {id(a[1]) for a in llm.added}
    assert len(parents) == 1


def test_async_add_request_pooling_single():
    p = make_processor()
    llm = AsyncLLM(p, supported_tasks=("embed",))
    llm.add_request("r", tokens_prompt([1, 2]), PoolingParams(task="embed"))
    assert len(llm.added) == 1
