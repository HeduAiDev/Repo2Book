"""TDD tests for the v3 ch08 subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).

These assert the *observable vLLM behavior* the chapter teaches — the logprobs
branch of the uplink swimlane, riding the same bus as ch07's token->text main
lane: entry declaration (logprobs=True -> SamplingParams.logprobs), engine-batch
registration (num_logprobs dict + batch-level max), the sampler's RAW snapshot
(log_softmax BEFORE any penalties/temperature — NOTE(woosuk)), the gather
triple (topk values+ids, sampled logprob, count-based vocab rank), the async
D2H (torch->numpy), scheduler per-request row slicing, the msgpack crossing
(ndarray/tensor enc hooks), LogprobsProcessor assembly on the API process
(non-incremental detokenization, U+FFFD context byte repair, cumulative sum,
rank chain), the DELTA tail-slicing exit, and the OpenAI three-field record
(token/logprob/bytes). Runs on a plain CPU host without the vllm package; the
only stand-ins are the documented HOST SEAMs in logprobs_lane.py (msgspec,
CUDA streams/events, tokenizer faces, ch7-product detokenizer), each anchored
to the real vLLM interface it mirrors.

Mechanism map (dossier m1-m20, station numbers in parentheses):
- s1  entry declaration: to_sampling_params logprobs triage + echo fallback
      + output_kind by stream; SamplingParams.__post_init__ logprobs=True->1,
      skip_reading_prefix_cache mutual exclusion (m19);
- s2  batch registration: num_logprobs[req_id] (-1 -> vocab_size),
      logprob_token_ids dict, max_num_logprobs = max(all), pop on finish,
      _make_sampling_metadata hands both to SamplingMetadata (m3);
- s3  raw snapshot: compute_logprobs == log_softmax fp32 of the ORIGINAL
      logits (m1); logprobs_mode four states (m16); None -> lane fully off
      (m18); num_logprobs == -1 full-vocab passthrough;
- s4  gather triple: topk+sampled+count-rank, [num_tok, k+1] with the
      sampled token ALWAYS column 0 (m2); batched_count_greater_than ties
      share the upper rank; logprob_token_ids sparse path: column 0 sampled,
      requested ids after, padded slots -inf (m17);
- s5  D2H: AsyncGPUModelRunnerOutput copies on the copy stream, get_output
      event-syncs then tolists() -> LogprobsLists numpy (m4);
- s6  scheduler slicing: slice_request per request rows (cu offset variant),
      prompt tensors from prompt_logprobs_dict, EngineCoreOutput fields;
- s7  crossing: MsgpackEncoder/Decoder ndarray+tensor hooks ride the
      EngineCoreOutputs bus (m5);
- s8  arrival dispatch: process_outputs step 3 -> update_from_output ->
      sample/prompt two-way split (m6);
- s9  sample assembly: per-column tolist -> non-incremental
      convert_ids_list_to_tokens -> cumulative += logprobs[0] (m7);
- s10 U+FFFD repair: replacement-char tail -> re-decode with <=4 context
      tokens, strip the clean prefix (m8; REAL byte-fallback tokenizer);
- s11 rank chain: chain((rank,), range(1, k+1)) + dict dedup, sampled token
      first (m9); per-request k truncation of the batch-uniform k+1 columns;
- s12 prompt branch: hidden_states re-run compute_logits, target =
      prompt[i+1], chunked accumulation into in_progress_prompt_logprobs_cpu,
      last chunk only delivery (m11); first None placeholder + DELTA pop (m12);
- s13 exit loading: DELTA tail logprobs[-len(token_ids):] + cumulative into
      CompletionOutput; prompt_logprobs pop vs direct read (m14);
- s14 OpenAI bytes: _create_chat_logprobs token/logprob(clamped)/bytes,
      _get_top_logprobs truncation (m15);
- m10 FlatLogprobs: 6 parallel primitive lists, O(1) object count,
      __getitem__ int/slice, append(None) empty position;
- m13 prompt assembly: shape recovery + flat one-shot detokenization;
- m20 tokenizer=None: decoded_tokens are NONES, numbers still flow.

Run:  cd instances/vllm/artifacts-v3/ch08-logprobs
      python -m pytest tests/ -q
"""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import tokenizers
import tokenizers.decoders

_IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(_IMPL_DIR))

lane = importlib.import_module("logprobs_lane")

# The host may or may not carry a GPU: the AsyncGPUModelRunnerOutput copy
# stream is a real torch.cuda.Stream when CUDA is up (the module then runs
# the REAL stream/event path), else None rides the HOST SEAM no-op face.
COPY_STREAM = torch.cuda.Stream() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# Harness: tokenizers
# ---------------------------------------------------------------------------


class ByteFallbackFace:
    """decode() face of the real tokenizers the serving layer touches: real
    serving.py calls tokenizer.decode(token_id) with a SCALAR (protocol
    L1216) — the transformers PreTrainedTokenizerFast accepts both scalars
    and lists, so the face widens the Rust decode() the same way."""

    def __init__(self, tk):
        self._tk = tk
        self.backend_tokenizer = tk

    def decode(self, ids):
        return self._tk.decode([ids] if isinstance(ids, int) else list(ids))


def byte_fallback_tokenizer():
    """A REAL Rust tokenizer with byte-fallback vocab: every token <0xXX> is
    one raw byte, so multi-byte chars (中 = E4 B8 AD) split across tokens and
    decode([single byte]) yields the replacement char U+FFFD — the exact
    behavior the real HF byte-fallback tokenizers show (what _verify_tokens
    exists to repair)."""
    vocab = {f"<0x{i:02X}>": i for i in range(256)}
    vocab["hello"] = 256
    vocab[" world"] = 257
    tk = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token=None)
    )
    tk.decoder = tokenizers.decoders.ByteFallback()
    return ByteFallbackFace(tk)


class MetaspaceLikeTokenizer:
    """The transformers PreTrainedTokenizerFast surface the leading-space
    restore path touches: decode() strips the SentencePiece marker (the
    add_dummy_prefix inverse), convert_ids_to_tokens() returns the raw vocab
    pieces, backend_tokenizer.to_str() carries the Metaspace pre_tokenizer
    config (SentencePiece-based Llama/Mistral/T5 shape)."""

    def __init__(self):
        tk = tokenizers.Tokenizer(
            tokenizers.models.WordLevel(
                vocab={"▁Hello": 0, "▁world": 1, "Hello": 2, "▁▁Hi": 3},
                unk_token=None,
            )
        )
        tk.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme="always"
        )
        tk.decoder = tokenizers.decoders.Metaspace(
            replacement="▁", prepend_scheme="always"
        )
        self._tk = tk
        self.backend_tokenizer = tk

    def decode(self, ids):
        return self._tk.decode(list(ids))

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [self._tk.id_to_token(i) for i in ids]


# ---------------------------------------------------------------------------
# Harness: sampling params / requests / batches
# ---------------------------------------------------------------------------


def sp(**kw):
    return lane.SamplingParams(**kw)


def make_request(rid, prompt_ids, params, client_index=0):
    return lane.EngineCoreRequest(
        request_id=rid,
        prompt_token_ids=list(prompt_ids),
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        client_index=client_index,
    )


def metadata(max_num_logprobs=None, logprob_token_ids=None, greedy=True):
    return lane.SamplingMetadata(
        temperature=None if greedy else torch.tensor([1.0]),
        all_greedy=greedy,
        all_random=not greedy,
        max_num_logprobs=max_num_logprobs,
        logprob_token_ids=logprob_token_ids,
    )


def input_batch(vocab_size=8):
    return lane.InputBatch(vocab_size=vocab_size)


def logprobs_lists_rows(rows, ranks=None, k=None):
    """A LogprobsLists built from python rows [[sampled, t1, t2...], ...]."""
    return lane.LogprobsLists(
        logprob_token_ids=np.array(
            [[t[0] for t in row] for row in rows], dtype=np.int32
        ),
        logprobs=np.array(
            [[t[1] for t in row] for row in rows], dtype=np.float64
        ),
        sampled_token_ranks=np.array(
            ranks if ranks is not None else [row[0][2] for row in rows],
            dtype=np.int32,
        ),
    )


def engine_core_output(rid, new_ids, logprobs=None, prompt_tensors=None):
    return lane.EngineCoreOutput(
        request_id=rid,
        new_token_ids=list(new_ids),
        new_logprobs=logprobs,
        new_prompt_logprobs_tensors=prompt_tensors,
    )


# ===========================================================================
# s1 — entry declaration (protocol.py to_sampling_params / SamplingParams)
# ===========================================================================


def chat_req(**kw):
    base = dict(
        messages=[{"role": "user", "content": "hi"}],
        model="tiny",
        logprobs=False,
        top_logprobs=0,
        prompt_logprobs=None,
        echo=False,
        stream=False,
        logprob_token_ids=None,
    )
    base.update(kw)
    return lane.ChatCompletionRequest(**base)


class TestEntryDeclaration:
    def test_logprobs_true_maps_to_top_logprobs(self):
        # protocol.py:L709-713: logprobs=(top_logprobs if logprobs and not
        # logprob_token_ids else None)
        p = chat_req(logprobs=True, top_logprobs=5).to_sampling_params(
            max_tokens=16, default_sampling_params={}
        )
        assert p.logprobs == 5
        assert p.prompt_logprobs is None

    def test_logprobs_false_keeps_lane_silent(self):
        p = chat_req(logprobs=False, top_logprobs=5).to_sampling_params(
            16, {}
        )
        assert p.logprobs is None
        assert p.num_logprobs is None  # the whole branch switch (m18)

    def test_logprob_token_ids_preempts_top_k(self):
        # logprob_token_ids 优先分叉：设它则 logprobs=None（稀疏路）
        p = chat_req(
            logprobs=True, top_logprobs=5, logprob_token_ids=[7, 9]
        ).to_sampling_params(16, {})
        assert p.logprobs is None
        assert p.logprob_token_ids == [7, 9]
        # num_logprobs property folds len(logprob_token_ids) into the account
        assert p.num_logprobs == 2

    def test_echo_defaults_prompt_logprobs_to_top_logprobs(self):
        # protocol.py:L686-688: prompt_logprobs None + echo -> top_logprobs
        # (SamplingParams has NO num_prompt_logprobs property — only the
        # num_logprobs property folds logprob_token_ids in; prompt_logprobs
        # is read raw by LogprobsProcessor.from_new_request)
        p = chat_req(logprobs=True, top_logprobs=3, echo=True).to_sampling_params(
            16, {}
        )
        assert p.prompt_logprobs == 3

    def test_stream_selects_delta_vs_final_only(self):
        assert (
            chat_req(stream=True).to_sampling_params(16, {}).output_kind
            is lane.RequestOutputKind.DELTA
        )
        assert (
            chat_req(stream=False).to_sampling_params(16, {}).output_kind
            is lane.RequestOutputKind.FINAL_ONLY
        )

    def test_post_init_normalizes_bool_logprobs(self):
        # sampling_params.py:L486-490: logprobs=True -> 1
        assert sp(logprobs=True).logprobs == 1
        assert sp(prompt_logprobs=True).prompt_logprobs == 1

    def test_prompt_logprobs_skips_reading_prefix_cache(self):
        # m19: skip_reading_prefix_cache = prompt_logprobs is not None —
        # cached tokens never went through the model, no logprob to give.
        assert sp(prompt_logprobs=2).skip_reading_prefix_cache is True
        assert sp(logprobs=2).skip_reading_prefix_cache is False

    def test_num_logprobs_property_none_when_all_off(self):
        assert sp().num_logprobs is None
        assert sp(logprob_token_ids=None).num_logprobs is None


# ===========================================================================
# s2 — engine batch registration (gpu_input_batch)
# ===========================================================================


class TestBatchRegistration:
    def test_add_request_registers_num_logprobs(self):
        ib = input_batch()
        ib.add_request("r0", 0, sp(logprobs=3))
        ib.add_request("r1", 0, sp())
        assert ib.num_logprobs == {"r0": 3}

    def test_minus_one_means_full_vocab(self):
        ib = input_batch(vocab_size=8192)
        ib.add_request("r0", 0, sp(logprobs=-1))
        assert ib.num_logprobs["r0"] == 8192

    def test_logprob_token_ids_registered(self):
        ib = input_batch()
        ib.add_request("r0", 0, sp(logprob_token_ids=[4, 5]))
        assert ib.logprob_token_ids == {"r0": [4, 5]}

    def test_max_num_logprobs_is_batch_max(self):
        # 批内一人要 20，全批算 20：批级均一化的账
        ib = input_batch()
        ib.add_request("a", 0, sp(logprobs=2))
        ib.add_request("b", 0, sp(logprobs=20))
        ib.add_request("c", 0, sp())
        assert ib.max_num_logprobs == 20
        ib.remove_request("b")
        assert ib.max_num_logprobs == 2
        ib.remove_request("a")
        assert ib.max_num_logprobs is None

    def test_make_sampling_metadata_carries_logprobs(self):
        ib = input_batch()
        ib.add_request("a", 0, sp(logprobs=2))
        ib.add_request("b", 1, sp(logprob_token_ids=[4]))
        md = ib._make_sampling_metadata()
        assert md.max_num_logprobs == 2
        # req_id keyed dict becomes req_index keyed for the sampler
        assert md.logprob_token_ids == {1: [4]}


# ===========================================================================
# s3/s4 — sampler: raw snapshot, four modes, gather triple, sparse path
# ===========================================================================


def ref_logprobs(logits):
    return logits.log_softmax(dim=-1, dtype=torch.float32)


class TestSamplerRawSnapshot:
    def test_compute_logprobs_is_log_softmax_fp32(self):
        logits = torch.randn(3, 11, dtype=torch.bfloat16)
        out = lane.Sampler.compute_logprobs(logits)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref_logprobs(logits.float()))

    def test_raw_snapshot_taken_from_original_logits(self):
        # m1/WC1: NOTE(woosuk) — top-k logprobs come from the logits BEFORE
        # any penalties or temperature scaling. Here the logits pass through
        # fp32 conversion and the (subtract-only placeholder) processor
        # chain; the gathered values must equal log_softmax of the input.
        torch.manual_seed(0)
        logits = torch.randn(2, 10, dtype=torch.float32)
        sampler = lane.Sampler()
        out = sampler.forward(logits.clone(), metadata(max_num_logprobs=3))
        ref = ref_logprobs(logits)
        topv, topi = torch.topk(ref, 3, dim=-1)
        assert torch.allclose(out.logprobs_tensors.logprobs[:, 1:], topv)
        assert torch.equal(
            out.logprobs_tensors.logprob_token_ids[:, 1:].to(torch.int64), topi
        )

    def test_no_logprobs_keeps_lane_off(self):
        # m18: sampler leaves nothing behind when nobody asked
        logits = torch.randn(2, 10)
        out = lane.Sampler().forward(logits, metadata(max_num_logprobs=None))
        assert out.logprobs_tensors is None

    def test_mode_raw_logits_returns_logits_not_logprobs(self):
        # m16 four-state contract: raw_logits keeps the raw SCORES (not
        # log_softmax); gather columns 1: are their top-k, value-sorted
        torch.manual_seed(1)
        logits = torch.randn(2, 10)
        sampler = lane.Sampler(logprobs_mode="raw_logits")
        out = sampler.forward(logits.clone(), metadata(max_num_logprobs=2))
        ref = logits.to(torch.float32)
        topv, _ = torch.topk(ref, 2, dim=-1)
        assert torch.allclose(out.logprobs_tensors.logprobs[:, 1:], topv)

    def test_mode_processed_logprobs_overwrites_on_greedy_path(self):
        # m16: the greedy fast path is the ONLY place processed_* materialize
        # (sampler.py:L261-271) and forward overwrites the raw snapshot with
        # them (L103-104). Placeholder processors leave logits untouched, so
        # processed == raw here — the assertion pins the plumbing.
        torch.manual_seed(2)
        logits = torch.randn(2, 10)
        sampler = lane.Sampler(logprobs_mode="processed_logprobs")
        out = sampler.forward(logits.clone(), metadata(max_num_logprobs=2))
        ref = ref_logprobs(logits)
        topv, _ = torch.topk(ref, 2, dim=-1)
        assert torch.allclose(out.logprobs_tensors.logprobs[:, 1:], topv)

    def test_full_vocab_minus_one_passthrough(self):
        # num_logprobs == -1: full unsorted unranked raw logprobs
        logits = torch.randn(2, 6)
        out = lane.Sampler().forward(
            logits.clone(), metadata(max_num_logprobs=-1)
        )
        lt = out.logprobs_tensors
        assert lt.logprob_token_ids.numel() == 0
        assert lt.selected_token_ranks.numel() == 0
        assert torch.allclose(lt.logprobs, ref_logprobs(logits))


class TestGatherTriple:
    def test_sampled_token_always_column_zero(self):
        torch.manual_seed(3)
        logits = torch.randn(4, 12)
        ref = ref_logprobs(logits)
        # "sampled" = argmax (top-1) for rows 0/2, a mid-rank token elsewhere
        sampled = torch.tensor([5, 11, 2, 0], dtype=torch.int64)
        lt = lane.Sampler.gather_logprobs(ref, 3, sampled)
        assert lt.logprob_token_ids.shape == (4, 4)
        assert lt.logprobs.shape == (4, 4)
        assert lt.selected_token_ranks.shape == (4,)
        assert torch.equal(
            lt.logprob_token_ids[:, 0].to(torch.int64), sampled
        )
        for i in range(4):
            got = ref[i, sampled[i]]
            assert torch.isclose(lt.logprobs[i, 0], got)

    def test_rank_is_count_not_sort(self):
        # (x >= v).sum(-1): 1-based count, no sort; ties share the upper rank
        logprobs = torch.tensor(
            [[-0.1, -0.1, -0.1, -2.0, -3.0]], dtype=torch.float32
        )
        ranks = lane.batched_count_greater_than(
            logprobs, torch.tensor([[-0.1]])
        )
        assert ranks.tolist() == [3]  # three entries >= -0.1 (incl. itself)
        ranks2 = lane.batched_count_greater_than(
            logprobs, torch.tensor([[-2.0]])
        )
        assert ranks2.tolist() == [4]
        ranks3 = lane.batched_count_greater_than(
            logprobs, torch.tensor([[-3.0]])
        )
        assert ranks3.tolist() == [5]

    def test_batched_count_shape_contracts(self):
        x = torch.randn(3, 7)
        v = torch.randn(3, 1)
        out = lane.batched_count_greater_than(x, v)
        assert out.shape == (3,)
        assert out.dtype in (torch.int64, torch.int32)

    def test_sampled_outside_topk_still_first(self):
        # gather with k=1 but a sampled token ranked 3rd: column 0 = the
        # sampled token's logprob even though it is not in the topk
        logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4]])
        lt = lane.Sampler.gather_logprobs(
            logprobs, 1, torch.tensor([3], dtype=torch.int64)
        )
        assert lt.logprob_token_ids[0].tolist() == [3, 0]
        assert lt.logprobs[0].tolist() == pytest.approx([-0.4, -0.1])

    def test_token_ids_must_be_int64(self):
        with pytest.raises(AssertionError):
            lane.Sampler.gather_logprobs(
                torch.randn(1, 5), 1, torch.tensor([0], dtype=torch.int32)
            )


class TestSparseTokenIdsPath:
    def test_column_zero_sampled_then_requested_then_inf(self):
        torch.manual_seed(4)
        logits = torch.randn(2, 10)
        ref = ref_logprobs(logits)
        sampled = torch.tensor([4, 7], dtype=torch.int64)
        sampler = lane.Sampler()
        lt = sampler.gather_specific_token_logprobs(
            ref, {0: [1, 2], 1: [9]}, sampled
        )
        assert lt.logprob_token_ids[:, 0].to(torch.int64).tolist() == [4, 7]
        assert lt.logprob_token_ids[0, 1:].to(torch.int64).tolist() == [1, 2]
        # row 1 is padded to the batch max width: [9, 0(padded)]
        assert lt.logprob_token_ids[1, 1:].to(torch.int64).tolist() == [9, 0]
        # row 1 padded slot masked to -inf
        assert lt.logprobs[1, 2].item() == float("-inf")
        assert torch.isclose(lt.logprobs[0, 0], ref[0, 4])
        assert torch.isclose(lt.logprobs[0, 1], ref[0, 1])

    def test_empty_dict_returns_none(self):
        assert (
            lane.Sampler().gather_specific_token_logprobs(
                torch.randn(2, 5), {}, torch.tensor([0, 1], dtype=torch.int64)
            )
            is None
        )

    def test_forward_prefers_sparse_over_topk(self):
        # L133-136: both set -> logprob_token_ids wins (more specific)
        torch.manual_seed(5)
        logits = torch.randn(2, 10)
        out = lane.Sampler().forward(
            logits.clone(),
            metadata(max_num_logprobs=3, logprob_token_ids={0: [1]}),
        )
        lt = out.logprobs_tensors
        assert lt.logprobs.shape[1] == 2  # sampled + one requested, not k+1


# ===========================================================================
# s5 — D2H: LogprobsTensors/Lists + AsyncGPUModelRunnerOutput
# ===========================================================================


class TestD2H:
    def test_to_cpu_nonblocking_is_noop_on_cpu(self):
        lt = lane.LogprobsTensors(
            torch.tensor([[1, 2]]), torch.tensor([[0.5, 0.2]]), torch.tensor([1])
        )
        assert lane.LogprobsTensors.to_cpu_nonblocking(lt) is lt

    def test_tolist_converts_to_numpy(self):
        lt = lane.LogprobsTensors(
            torch.tensor([[1, 2]], dtype=torch.int32),
            torch.tensor([[0.5, 0.2]]),
            torch.tensor([7], dtype=torch.int32),
        )
        lists = lt.tolists()
        assert isinstance(lists, lane.LogprobsLists)
        assert isinstance(lists.logprobs, np.ndarray)
        assert lists.sampled_token_ranks.tolist() == [7]

    def test_slice_request_rows_and_cu_offset(self):
        lists = logprobs_lists_rows(
            [
                [(10, -0.1, 1), (11, -0.2, 1)],
                [(12, -0.3, 1), (13, -0.4, 1)],
                [(14, -0.5, 1), (15, -0.6, 1)],
            ]
        )
        two = lists.slice_request(1, 2)
        assert two.logprob_token_ids[:, 0].tolist() == [12, 14]
        # speculative decoding: cu offsets relocate the request's row start
        cu = lane.LogprobsLists(
            lists.logprob_token_ids, lists.logprobs, lists.sampled_token_ranks,
            [0, 1],  # request 1 begins at absolute row 1
        )
        one = cu.slice_request(1, 1)
        assert one.logprob_token_ids[:, 0].tolist() == [12]

    def test_empty_cpu_preallocates_full_prompt(self):
        lt = lane.LogprobsTensors.empty_cpu(5, 3)
        assert lt.logprob_token_ids.shape == (5, 3)
        assert lt.logprob_token_ids.dtype == torch.int32
        assert lt.logprobs.dtype == torch.float32
        assert lt.selected_token_ranks.shape == (5,)

    def test_async_output_get_output_yields_lists(self):
        # one decode step: sampled [num_reqs, 1], logprobs copied on the
        # (seam) copy stream, event-synced in get_output, tolists() into
        # ModelRunnerOutput.logprobs — same Future as the sampled tokens
        lt = lane.LogprobsTensors(
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
            torch.tensor([1, 1], dtype=torch.int32),
        )
        mrr = lane.ModelRunnerOutput(
            req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1}
        )
        async_out = lane.AsyncGPUModelRunnerOutput(
            model_runner_output=mrr,
            sampled_token_ids=torch.tensor([[10], [20]], dtype=torch.int32),
            logprobs_tensors=lt,
            invalid_req_indices=[1],
            async_output_copy_stream=COPY_STREAM,
            vocab_size=8,
        )
        out = async_out.get_output()
        assert out.logprobs.logprob_token_ids[:, 0].tolist() == [1, 3]
        # invalid request row cleared from sampled ids (max_gen_len == 1 path)
        assert out.sampled_token_ids == [[10], []]

    def test_async_output_none_logprobs(self):
        mrr = lane.ModelRunnerOutput(
            req_ids=["a"], req_id_to_index={"a": 0}
        )
        async_out = lane.AsyncGPUModelRunnerOutput(
            model_runner_output=mrr,
            sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
            logprobs_tensors=None,
            invalid_req_indices=[],
            async_output_copy_stream=COPY_STREAM,
            vocab_size=8,
        )
        assert async_out.get_output().logprobs is None


# ===========================================================================
# s12 — prompt branch engine side: _get_prompt_logprobs_dict
# ===========================================================================


class FakeLogitsModel:
    """compute_logits face of the two-method model contract (WC2): the
    forward pass yields hidden_states, lm_head projection happens here."""

    def __init__(self, vocab, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(vocab, 4, generator=g)

    def compute_logits(self, hidden_states):
        return hidden_states @ self.w.T


def make_runner(prompt_ids, num_prompt_logprobs, vocab=8, hidden=None, seed=0):
    """A GPUModelRunner field face carrying exactly what
    _get_prompt_logprobs_dict touches."""
    model = FakeLogitsModel(vocab, seed)
    requests = {
        rid: lane.CachedRequestState(
            req_id=rid,
            prompt_token_ids=list(ids),
            sampling_params=None,
            num_computed_tokens=0,
            in_progress_prompt_logprobs_cpu=None,
        )
        for rid, ids in prompt_ids.items()
    }
    runner = lane.GPUModelRunner(
        requests=requests,
        num_prompt_logprobs={r: num_prompt_logprobs for r in prompt_ids},
        model=model,
        model_config=SimpleNamespace(logprobs_mode="raw_logprobs"),
        sampler=lane.Sampler(),
        device="cpu",
        query_start_loc=SimpleNamespace(np=np.array([0, len(next(iter(prompt_ids.values())))])),
        input_batch=SimpleNamespace(
            req_id_to_index={rid: i for i, rid in enumerate(prompt_ids)}
        ),
    )
    return runner, model


class TestPromptLogprobsEngineSide:
    def test_single_chunk_computes_and_delivers(self):
        prompt = [1, 3, 5, 7, 2]
        runner, model = make_runner({"r0": prompt}, num_prompt_logprobs=2)
        hidden = torch.randn(5, 4)
        d = runner._get_prompt_logprobs_dict(hidden, {"r0": 5})
        assert set(d) == {"r0"}
        lt = d["r0"]
        # last chunk: num_logits = 4 (5 scheduled > 4 remaining); the runner
        # slices THIS chunk's hidden rows before compute_logits — mirror it
        ref = model.compute_logits(hidden[0:4]).log_softmax(
            -1, dtype=torch.float32
        )
        # targets are prompt[1:] — prompt[i] predicts prompt[i+1]
        tgt = torch.tensor(prompt[1:], dtype=torch.int64)
        expect = lane.Sampler.gather_logprobs(ref, 2, tgt)
        assert torch.allclose(lt.logprobs, expect.logprobs)
        assert torch.equal(
            lt.logprob_token_ids, expect.logprob_token_ids
        )
        # last chunk: request deregistered + in-progress cleared
        assert "r0" not in runner.num_prompt_logprobs
        assert runner.requests["r0"].in_progress_prompt_logprobs_cpu is None

    def test_chunked_prefill_accumulates_and_delays(self):
        prompt = [1, 2, 3, 4, 5, 6]
        runner, model = make_runner({"r0": prompt}, num_prompt_logprobs=1)
        h1, h2 = torch.randn(3, 4), torch.randn(3, 4)
        hidden = torch.cat([h1, h2])
        # chunk 1: 3 of 5 remaining -> not delivered, rows cached
        d1 = runner._get_prompt_logprobs_dict(hidden, {"r0": 3})
        assert d1 == {}
        in_prog = runner.requests["r0"].in_progress_prompt_logprobs_cpu
        assert in_prog.logprobs.shape == (5, 2)
        # column 0 of each row is the TARGET token's logprob (the prompt
        # path's "sampled" = prompt[i+1])
        assert torch.allclose(in_prog.logprobs[:3, 0:1], ref_logprobs(h1 @ model.w.T).gather(
            -1, torch.tensor([[2], [3], [4]])))
        # the runner's in-flight bookkeeping advances between steps (the real
        # _update_states does this): 3 more tokens computed, chunk 2 starts at
        # hidden offset 3
        runner.requests["r0"].num_computed_tokens = 3
        runner.query_start_loc = SimpleNamespace(np=np.array([0, 3]))
        # chunk 2 (last): delivered with ALL rows
        d2 = runner._get_prompt_logprobs_dict(hidden, {"r0": 3})
        assert set(d2) == {"r0"}
        assert d2["r0"].logprobs.shape == (5, 2)

    def test_preempted_request_skipped(self):
        prompt = [1, 2, 3]
        runner, _ = make_runner({"r0": prompt}, num_prompt_logprobs=1)
        # num_scheduled_tokens missing the req -> preempted in prefill
        d = runner._get_prompt_logprobs_dict(torch.randn(3, 4), {})
        assert d == {}
        assert "r0" in runner.num_prompt_logprobs  # still registered

    def test_logits_mode_raw_logits_uses_raw_scores(self):
        prompt = [1, 2, 3]
        runner, model = make_runner({"r0": prompt}, num_prompt_logprobs=1)
        runner.model_config = SimpleNamespace(logprobs_mode="raw_logits")
        hidden = torch.randn(3, 4)
        d = runner._get_prompt_logprobs_dict(hidden, {"r0": 3})
        # prompt of 3 tokens -> 2 scored positions (runner slices them)
        raw = model.compute_logits(hidden[0:2]).to(torch.float32)
        # prompt logprobs skip sampling processors -> same scores either way;
        # column 1 is the top-1 raw score (gather sorts by value)
        topv, _ = torch.topk(raw, 1, dim=-1)
        assert torch.allclose(d["r0"].logprobs[:, 1:], topv)


# ===========================================================================
# s6 — scheduler slicing into EngineCoreOutput
# ===========================================================================


def fake_sched_request(rid, num_logprobs=None, client_index=0):
    return SimpleNamespace(
        request_id=rid,
        sampling_params=sp(logprobs=num_logprobs) if num_logprobs else sp(),
        client_index=client_index,
    )


class TestSchedulerSlicing:
    def test_slice_per_request_and_route_by_client(self):
        lists = logprobs_lists_rows(
            [
                [(10, -0.1, 1), (11, -0.2, 1)],
                [(12, -0.3, 1), (13, -0.4, 1)],
            ]
        )
        sched = lane.Scheduler(
            {
                "a": fake_sched_request("a", num_logprobs=2),
                "b": fake_sched_request("b", client_index=1),
            }
        )
        mrr = lane.ModelRunnerOutput(
            req_ids=["a", "b"],
            req_id_to_index={"a": 0, "b": 1},
            sampled_token_ids=[[10], [12]],
            logprobs=lists,
        )
        out = sched.update_from_output(
            SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1}), mrr
        )
        eco_a = out[0].outputs[0]
        eco_b = out[1].outputs[0]
        assert eco_a.request_id == "a"
        assert eco_a.new_logprobs.logprob_token_ids[:, 0].tolist() == [10]
        assert eco_a.new_logprobs.logprobs[:, 0].tolist() == pytest.approx([-0.1])
        # request without logprobs carries None — the lane costs it nothing
        assert eco_b.new_logprobs is None
        assert eco_b.new_token_ids == [12]

    def test_prompt_tensors_attach_to_their_request(self):
        pt = lane.LogprobsTensors(
            torch.tensor([[1, 2]], dtype=torch.int32),
            torch.tensor([[-0.1, -0.2]]),
            torch.tensor([1], dtype=torch.int32),
        )
        sched = lane.Scheduler({"a": fake_sched_request("a")})
        mrr = lane.ModelRunnerOutput(
            req_ids=["a"],
            req_id_to_index={"a": 0},
            sampled_token_ids=[[5]],
            prompt_logprobs_dict={"a": pt},
        )
        out = sched.update_from_output(
            SimpleNamespace(num_scheduled_tokens={"a": 1}), mrr
        )
        assert out[0].outputs[0].new_prompt_logprobs_tensors is pt


# ===========================================================================
# s7 — crossing the process boundary (msgpack ndarray/tensor hooks)
# ===========================================================================


class TestCrossing:
    def test_logprobs_fields_survive_msgpack_round_trip(self):
        lists = logprobs_lists_rows([[(10, -0.125, 1), (11, -0.25, 1)]])
        pt = lane.LogprobsTensors(
            torch.tensor([[1, 2]], dtype=torch.int32),
            torch.tensor([[-0.5, -0.75]]),
            torch.tensor([1], dtype=torch.int32),
        )
        eco = lane.EngineCoreOutputs(
            outputs=[
                engine_core_output("a", [10], logprobs=lists, prompt_tensors=pt)
            ]
        )
        wire = lane.MsgpackEncoder().encode(eco)
        assert isinstance(wire[0], (bytes, bytearray, memoryview))
        back = lane.MsgpackDecoder(lane.EngineCoreOutputs).decode(wire)
        got = back.outputs[0]
        assert isinstance(got.new_logprobs, lane.LogprobsLists)
        assert isinstance(got.new_logprobs.logprobs, np.ndarray)
        assert np.allclose(got.new_logprobs.logprobs, lists.logprobs)
        assert got.new_logprobs.logprob_token_ids.tolist() == [[10, 11]]
        assert isinstance(got.new_prompt_logprobs_tensors, lane.LogprobsTensors)
        assert torch.equal(
            got.new_prompt_logprobs_tensors.logprobs, pt.logprobs
        )
        assert got.new_token_ids == [10]

    def test_encoder_hook_shapes_are_native_tuples(self):
        # _encode_ndarray -> (dtype_str, shape, data); _encode_tensor likewise.
        # Both hooks only run inside encode(): set the aux_buffers stash the
        # real encode() sets up (same assert as real serial_utils).
        enc = lane.MsgpackEncoder()
        enc.aux_buffers = [b""]
        arr = np.array([[1, 2]], dtype=np.int32)
        dtype, shape, data = enc._encode_ndarray(arr)
        assert dtype == "<i4" and tuple(shape) == (1, 2)
        t = torch.tensor([[1.5]])
        dtype_t, shape_t, _ = enc._encode_tensor(t)
        assert dtype_t == "float32" and tuple(shape_t) == (1, 1)

    def test_no_logprobs_output_stays_small(self):
        # omit_defaults + all-None logprobs: nothing numeric rides the bus
        eco = lane.EngineCoreOutputs(
            outputs=[engine_core_output("a", [10])]
        )
        wire = lane.MsgpackEncoder().encode(eco)
        back = lane.MsgpackDecoder(lane.EngineCoreOutputs).decode(wire)
        assert back.outputs[0].new_logprobs is None
        assert back.outputs[0].finished is False


# ===========================================================================
# s8 — arrival: process_outputs step 3 -> update_from_output dispatch
# ===========================================================================


class DetokFace:
    """HOST-side face of ch07's product (IncrementalDetokenizer): exactly the
    surface _new_completion_output touches."""

    def __init__(self, vocab=None):
        self.output_token_ids: list[int] = []

    def update(self, new_token_ids, finished):
        self.output_token_ids.extend(new_token_ids)
        return None

    def get_next_output_text(self, finished, delta):
        return "".join(chr(64 + i) for i in self.output_token_ids)


def make_state(rid, params, detokenizer=None, output_kind=None):
    """RequestState via the real from_new_request."""
    req = make_request(rid, [1, 2], params)
    req.external_req_id = "ext-" + rid
    return lane.RequestState.from_new_request(
        tokenizer=None,
        request=req,
        prompt="",
        parent_req=None,
        request_index=0,
        queue=None,
        log_stats=False,
        stream_interval=1,
        detokenizer=detokenizer,
        output_kind=output_kind,
    )


class TestArrivalDispatch:
    def test_step3_dispatches_sample_and_prompt(self):
        params = sp(logprobs=1, prompt_logprobs=1)
        state = make_state("r0", params)
        lp = logprobs_lists_rows([[(10, -0.5, 1), (11, -0.2, 1)]])
        pt = lane.LogprobsTensors(
            torch.tensor([[9, 8]], dtype=torch.int32),
            torch.tensor([[-0.7, -0.9]]),
            torch.tensor([1], dtype=torch.int32),
        )
        state.logprobs_processor.update_from_output(
            engine_core_output("r0", [10], logprobs=lp, prompt_tensors=pt)
        )
        # both containers grew: sample path + prompt path from one output
        assert len(state.logprobs_processor.logprobs) == 1
        assert len(state.logprobs_processor.prompt_logprobs) == 2  # None + 1

    def test_process_outputs_calls_update(self):
        params = sp(logprobs=1)
        detok = DetokFace()
        op = lane.OutputProcessor(tokenizer=None, log_stats=False)
        req = make_request("r0", [1], params)
        req.external_req_id = "ext-r0"
        op.add_request(req, "", None, 0, None, detokenizer=detok)
        lp = logprobs_lists_rows([[(10, -0.5, 1), (11, -0.2, 1)]])
        res = op.process_outputs([engine_core_output("r0", [10], logprobs=lp)])
        state = op.request_states["r0"]
        assert len(state.logprobs_processor.logprobs) == 1
        # no queue -> LLMEngine face: outputs accumulate in the returned list
        assert len(res.request_outputs) == 1
        assert res.request_outputs[0].request_id == "ext-r0"

    def test_aborted_request_ignored(self):
        op = lane.OutputProcessor(tokenizer=None, log_stats=False)
        res = op.process_outputs(
            [engine_core_output("ghost", [1])],
        )
        assert res.request_outputs == []


# ===========================================================================
# s9/s10/s11 — sample assembly, U+FFFD repair, rank chain
# ===========================================================================


def processor(tokenizer=None, num_logprobs=1, flat=False, prompt_k=None):
    params = sp(logprobs=num_logprobs, flat_logprobs=flat)
    if prompt_k is not None:
        params.prompt_logprobs = prompt_k
    req = make_request("r0", [1], params)
    req.external_req_id = "ext-r0"
    return lane.LogprobsProcessor.from_new_request(
        tokenizer=tokenizer, request=req
    )


class TestSampleAssembly:
    def test_tolist_detokenize_cumulative_append(self):
        # REAL byte-fallback tokenizer, ASCII tokens: numbers flow, first
        # entry is the sampled token, cumulative += logprobs[0]
        tk = byte_fallback_tokenizer()
        proc = processor(tk, num_logprobs=1)
        lp = lane.LogprobsLists(
            logprob_token_ids=np.array([[256, 257]], dtype=np.int32),
            logprobs=np.array([[-0.25, -0.5]]),
            sampled_token_ranks=np.array([1], dtype=np.int32),
        )
        proc.update_from_output(engine_core_output("r0", [256], logprobs=lp))
        entry = proc.logprobs[0]
        first_id, first = next(iter(entry.items()))
        assert first_id == 256
        assert first.logprob == pytest.approx(-0.25)
        assert first.rank == 1
        assert first.decoded_token == "hello"
        assert proc.cumulative_logprob == pytest.approx(-0.25)

    def test_second_step_accumulates(self):
        tk = byte_fallback_tokenizer()
        proc = processor(tk, num_logprobs=1)
        for ids, lps in (
            ([256, 257], [-0.25, -0.5]),
            ([257, 256], [-1.5, -2.0]),
        ):
            lp = lane.LogprobsLists(
                logprob_token_ids=np.array([ids], dtype=np.int32),
                logprobs=np.array([lps]),
                sampled_token_ranks=np.array([1], dtype=np.int32),
            )
            proc.update_from_output(
                engine_core_output("r0", [ids[0]], logprobs=lp)
            )
        assert proc.cumulative_logprob == pytest.approx(-1.75)
        assert len(proc.logprobs) == 2

    def test_per_request_k_truncates_batch_uniform_columns(self):
        # batch computed k=2 (3 columns) but this request asked k=1: the
        # extra candidate is silently dropped (theory 6 / m3)
        tk = byte_fallback_tokenizer()
        proc = processor(tk, num_logprobs=1)
        lp = lane.LogprobsLists(
            logprob_token_ids=np.array(
                [[256, 257, 0x41]], dtype=np.int32
            ),
            logprobs=np.array([[-0.25, -0.5, -0.75]]),
            sampled_token_ranks=np.array([1], dtype=np.int32),
        )
        proc.update_from_output(engine_core_output("r0", [256], logprobs=lp))
        assert len(proc.logprobs[0]) == 2  # sampled + top1 only

    def test_tokenizer_none_keeps_numbers_and_nones(self):
        # m20: skip_tokenizer_init / detokenize=False -> NONES decoded tokens
        proc = processor(None, num_logprobs=1)
        lp = lane.LogprobsLists(
            logprob_token_ids=np.array([[256, 257]], dtype=np.int32),
            logprobs=np.array([[-0.25, -0.5]]),
            sampled_token_ranks=np.array([1], dtype=np.int32),
        )
        proc.update_from_output(engine_core_output("r0", [256], logprobs=lp))
        entry = proc.logprobs[0]
        assert next(iter(entry.values())).decoded_token is None
        assert proc.cumulative_logprob == pytest.approx(-0.25)

    def test_sampled_token_dedup_via_dict_key(self):
        # sampled inside topk: dict insert-twice == insert-once, sampled first
        tk = byte_fallback_tokenizer()
        proc = processor(tk, num_logprobs=2)
        lp = lane.LogprobsLists(
            logprob_token_ids=np.array(
                [[256, 256, 257]], dtype=np.int32
            ),  # top1 == sampled
            logprobs=np.array([[-0.25, -0.25, -0.5]]),
            sampled_token_ranks=np.array([1], dtype=np.int32),
        )
        proc.update_from_output(engine_core_output("r0", [256], logprobs=lp))
        entry = proc.logprobs[0]
        assert list(entry) == [256, 257]
        assert entry[256].rank == 1  # sampled rank won the race


class TestUnicodeRepair:
    def test_multibyte_char_repaired_with_real_tokenizer(self):
        # 中 = E4 B8 AD across three byte tokens; each intermediate decodes
        # to U+FFFD, the completing token re-decodes with context and takes
        # the whole char, predecessors get ""
        tk = byte_fallback_tokenizer()
        proc = processor(tk, num_logprobs=1)
        rows = [([0xE4], [0.0]), ([0xB8], [0.0]), ([0xAD], [0.0])]
        for (ids, lps) in rows:
            lp = lane.LogprobsLists(
                logprob_token_ids=np.array([ids], dtype=np.int32),
                logprobs=np.array([lps]),
                sampled_token_ranks=np.array([1], dtype=np.int32),
            )
            proc.update_from_output(
                engine_core_output("r0", ids, logprobs=lp)
            )
        got = [
            next(iter(pos.values())).decoded_token for pos in proc.logprobs
        ]
        assert got == ["", "", "中"]

    def test_lateral_candidates_repaired_independently(self):
        # same position alternatives [sampled=AD, top1=E4] — the横向 axis:
        # each candidate independently re-decoded with the纵向 context
        tk = byte_fallback_tokenizer()
        proc = processor(tk, num_logprobs=1)
        # context: one full byte token E4 already landed
        lp0 = lane.LogprobsLists(
            logprob_token_ids=np.array([[0xE4, 0xB8]], dtype=np.int32),
            logprobs=np.array([[-0.1, -0.2]]),
            sampled_token_ranks=np.array([1], dtype=np.int32),
        )
        proc.update_from_output(engine_core_output("r0", [0xE4], logprobs=lp0))
        lp1 = lane.LogprobsLists(
            logprob_token_ids=np.array([[0xB8, 0xAD]], dtype=np.int32),
            logprobs=np.array([[-0.1, -0.2]]),
            sampled_token_ranks=np.array([1], dtype=np.int32),
        )
        proc.update_from_output(engine_core_output("r0", [0xB8], logprobs=lp1))
        lp2 = lane.LogprobsLists(
            logprob_token_ids=np.array([[0xAD, 0xE4]], dtype=np.int32),
            logprobs=np.array([[-0.1, -0.2]]),
            sampled_token_ranks=np.array([1], dtype=np.int32),
        )
        proc.update_from_output(engine_core_output("r0", [0xAD], logprobs=lp2))
        last = proc.logprobs[2]
        decoded = {tid: l.decoded_token for tid, l in last.items()}
        # sampled AD with context [E4, B8] -> 中; alternative E4 alone -> ""
        assert decoded[0xAD] == "中"
        assert decoded[0xE4] == ""

    def test_correct_decoded_token_clean_prefix_strip(self):
        # vLLM's own unit-test shape: decode([101, 102]) = "hello valid"
        proc = processor(Mock(), num_logprobs=1)
        proc.tokenizer.decode.side_effect = (
            lambda ids: {(101, 102): "hello valid", (101,): "hello "}.get(
                tuple(ids), "�"
            )
        )
        assert proc._correct_decoded_token(102, [101]) == "valid"

    def test_correct_decoded_token_genuinely_incomplete_empty(self):
        proc = processor(Mock(), num_logprobs=1)
        proc.tokenizer.decode.return_value = "�"
        assert proc._correct_decoded_token(100, []) == ""
        assert proc._correct_decoded_token(100, [50]) == ""

    def test_correct_decoded_token_prefix_mismatch_normalization(self):
        # tokenizer normalization breaks startswith -> longest common prefix
        proc = processor(Mock(), num_logprobs=1)
        proc.tokenizer.decode.side_effect = (
            lambda ids: {(60, 70): "  Token", (60,): "Tok"}.get(
                tuple(ids), "�"
            )
        )
        # clean_prefix "Tok" is not a prefix of "  Token" -> common len 0
        assert proc._correct_decoded_token(70, [60]) == "  Token"

    def test_verify_tokens_ignores_clean_tail(self):
        # U+FFFD mid-string is genuine; only the TAIL triggers repair
        proc = processor(Mock(), num_logprobs=1)
        out = proc._verify_tokens(
            ["ok", "a�b"], [1, 2], []
        )
        assert out == ["ok", "a�b"]

    def test_context_capped_at_four(self):
        # decode never resolves (always replacement char): the loop walks
        # num_ctx = 1..min(len(context), 4) and gives up — with 6 context
        # tokens available, the 5th/6th are never tried (4 bytes is the
        # UTF-8 multi-byte upper bound)
        proc = processor(Mock(), num_logprobs=1)
        calls = []

        def decode(ids):
            calls.append(list(ids))
            return "�"

        proc.tokenizer.decode.side_effect = decode
        out = proc._correct_decoded_token(9, [1, 2, 3, 4, 5, 6])
        assert out == ""  # genuinely incomplete -> empty
        assert sorted(len(c) for c in calls) == [2, 3, 4, 5]
        assert max(len(c) for c in calls) == 4 + 1  # 4 context + the token

    def test_get_sampled_context_ids_flat_and_nested(self):
        nested = [{10: lane.Logprob(-0.1, 1, "a")}, {11: lane.Logprob(-0.2, 1, "b")}]
        flat = lane.FlatLogprobs()
        flat.append(nested[0])
        flat.append(nested[1])
        assert lane.LogprobsProcessor._get_sampled_context_ids(nested) == [10, 11]
        assert lane.LogprobsProcessor._get_sampled_context_ids(flat) == [10, 11]
        assert lane.LogprobsProcessor._get_sampled_context_ids(None) == []
        # None placeholder positions (prompt first token) contribute nothing
        with_none = [None, {12: lane.Logprob(-0.3, 1, "c")}]
        assert lane.LogprobsProcessor._get_sampled_context_ids(with_none) == [12]


# ===========================================================================
# m7 helper — convert_ids_list_to_tokens (non-incremental detokenization)
# ===========================================================================


class TestNonIncrementalDetokenize:
    def test_plain_decode_path(self):
        tk = byte_fallback_tokenizer()
        assert lane.convert_ids_list_to_tokens(tk, [256, 257]) == [
            "hello",
            " world",
        ]

    def test_empty_ids(self):
        assert lane.convert_ids_list_to_tokens(byte_fallback_tokenizer(), []) == []

    def test_leading_space_restored_metaspace(self):
        tk = MetaspaceLikeTokenizer()
        out = lane.convert_ids_list_to_tokens(tk, [0, 1])  # ▁Hello ▁world
        assert out == [" Hello", " world"]

    def test_double_marker_two_spaces(self):
        tk = MetaspaceLikeTokenizer()
        out = lane.convert_ids_list_to_tokens(tk, [3])  # ▁▁Hi
        assert out == ["  Hi"]

    def test_no_marker_tokenzier_short_circuits(self):
        # backend-less tokenizer (e.g. byte-level) takes the plain path
        tk = byte_fallback_tokenizer()
        assert lane._get_leading_space_marker(tk) is None


# ===========================================================================
# m10/m11 — FlatLogprobs + rank chain
# ===========================================================================


class TestFlatLogprobs:
    def test_append_fast_parallel_lists(self):
        flat = lane.FlatLogprobs()
        flat.append_fast([5, 6], [-0.1, -0.2], iter([3, 1]), ["a", "b"])
        flat.append_fast([7], [-0.3], iter([2]), ["c"])
        assert flat.token_ids == [5, 6, 7]
        assert flat.logprobs == [-0.1, -0.2, -0.3]
        assert flat.ranks == [3, 1, 2]
        assert flat.decoded_tokens == ["a", "b", "c"]
        assert flat.start_indices == [0, 2]
        assert flat.end_indices == [2, 3]
        assert len(flat) == 2

    def test_getitem_rebuilds_dict(self):
        flat = lane.FlatLogprobs()
        flat.append_fast([5, 6], [-0.1, -0.2], iter([3, 1]), ["a", "b"])
        entry = flat[0]
        assert isinstance(entry, dict)
        assert entry[5] == lane.Logprob(-0.1, 3, "a")
        assert entry[6] == lane.Logprob(-0.2, 1, "b")

    def test_append_none_keeps_empty_position(self):
        # the prompt first-token placeholder
        flat = lane.create_prompt_logprobs(flat_logprobs=True)
        assert len(flat) == 1
        assert flat.start_indices == flat.end_indices == [0]
        assert flat[0] == {}

    def test_slice_returns_shifted_flatlogprobs(self):
        flat = lane.FlatLogprobs()
        flat.append_fast([5], [-0.1], iter([1]), ["a"])
        flat.append_fast([6], [-0.2], iter([2]), ["b"])
        sub = flat[1:]
        assert isinstance(sub, lane.FlatLogprobs)
        assert sub.start_indices == [0]
        assert sub.token_ids == [6]

    def test_object_count_is_constant(self):
        # m10: six primitive lists regardless of positions × ranks
        flat = lane.FlatLogprobs()
        for i in range(50):
            flat.append_fast([i, i + 1], [-0.1, -0.2], iter([1, 2]), ["x", "y"])
        inner_lists = [
            v
            for v in vars(flat).values()
            if isinstance(v, list)
        ]
        assert len(inner_lists) == 6  # start/end + 4 data lists, nothing else

    def test_iteration_via_sequence_base(self):
        flat = lane.FlatLogprobs()
        flat.append_fast([5], [-0.1], iter([1]), ["a"])
        entries = list(flat)
        assert entries == [{5: lane.Logprob(-0.1, 1, "a")}]

    def test_immutable_setitem_raises(self):
        flat = lane.FlatLogprobs()
        with pytest.raises(TypeError):
            flat[0] = {}

    def test_rank_chain_sampled_first(self):
        # chain((rank,), range(1, k+1)) — nested dict face
        container = lane.create_sample_logprobs(flat_logprobs=False)
        lane.append_logprobs_for_next_position(
            container, [9, 8, 7], [-0.1, -0.2, -0.3], ["x", "y", "z"], 5, 2
        )
        entry = container[0]
        assert list(entry) == [9, 8, 7]
        assert entry[9].rank == 5  # the sampled token's vocab rank
        assert entry[8].rank == 1  # top1
        assert entry[7].rank == 2  # top2

    def test_rank_chain_minus_one_uses_all_columns(self):
        container = lane.create_sample_logprobs(flat_logprobs=False)
        lane.append_logprobs_for_next_position(
            container, [9, 8], [-0.1, -0.2], ["x", "y"], 1, -1
        )
        assert len(container[0]) == 2

    def test_create_prompt_logprobs_nested_first_none(self):
        nested = lane.create_prompt_logprobs(flat_logprobs=False)
        assert nested == [None]

    def test_create_sample_logprobs_flat_vs_nested(self):
        assert isinstance(
            lane.create_sample_logprobs(flat_logprobs=True), lane.FlatLogprobs
        )
        assert lane.create_sample_logprobs(flat_logprobs=False) == []


# ===========================================================================
# m13/m12 — prompt assembly + DELTA pop
# ===========================================================================


class TestPromptAssembly:
    def test_shape_recovery_and_flat_detokenize(self):
        tk = byte_fallback_tokenizer()
        proc = processor(tk, num_logprobs=1, prompt_k=1)
        # [num_prompt_tokens, k+1]: prompt of 3 tokens -> 2 scored positions
        pt = lane.LogprobsTensors(
            torch.tensor([[256, 257], [257, 256]], dtype=torch.int32),
            torch.tensor([[-0.1, -0.3], [-0.2, -0.4]]),
            torch.tensor([1, 2], dtype=torch.int32),
        )
        proc.update_from_output(
            engine_core_output("r0", [1], prompt_tensors=pt)
        )
        # [None, pos0, pos1]
        assert len(proc.prompt_logprobs) == 3
        assert proc.prompt_logprobs[0] is None
        p0 = next(iter(proc.prompt_logprobs[1].values()))
        assert p0.decoded_token == "hello"
        assert p0.logprob == pytest.approx(-0.1)
        # no cumulative on the prompt path
        assert proc.cumulative_logprob == 0.0

    def test_pop_returns_all_once(self):
        proc = processor(None, num_logprobs=None, prompt_k=1)
        pt = lane.LogprobsTensors(
            torch.tensor([[256, 257]], dtype=torch.int32),
            torch.tensor([[-0.1, -0.3]]),
            torch.tensor([1], dtype=torch.int32),
        )
        proc.update_from_output(
            engine_core_output("r0", [1], prompt_tensors=pt)
        )
        first = proc.pop_prompt_logprobs()
        assert len(first) == 2
        # taken == forgotten: the second pop returns the empty container (the
        # real `if plp:` keeps the reset skipped but still returns the list)
        assert proc.pop_prompt_logprobs() == []
        assert proc.prompt_logprobs == []

    def test_disabled_prompt_returns_none(self):
        proc = processor(None, num_logprobs=1)  # no prompt_logprobs
        assert proc.pop_prompt_logprobs() is None


# ===========================================================================
# s13 — exit loading: DELTA tail slicing / cumulative into CompletionOutput
# ===========================================================================


class TestExitLoading:
    def _state(self, output_kind, detok=None):
        params = sp(
            logprobs=1, output_kind=output_kind
        )
        return make_state(
            "r0", params, detokenizer=detok or DetokFace()
        )

    def test_delta_slices_logprobs_tail(self):
        state = self._state(lane.RequestOutputKind.DELTA)
        tk = byte_fallback_tokenizer()
        for ids in ([256], [257]):
            lp = lane.LogprobsLists(
                logprob_token_ids=np.array([ids], dtype=np.int32),
                logprobs=np.array([[-0.1]]),
                sampled_token_ranks=np.array([1], dtype=np.int32),
            )
            state.logprobs_processor.update_from_output(
                engine_core_output("r0", ids, logprobs=lp)
            )
        # two positions accumulated; DELTA step for the LAST token only
        co = state._new_completion_output([257], None, None)
        assert len(co.logprobs) == 1
        assert next(iter(co.logprobs[0])) == 257
        assert co.cumulative_logprob == pytest.approx(-0.2)

    def test_final_returns_everything(self):
        state = self._state(lane.RequestOutputKind.FINAL_ONLY)
        lp = lane.LogprobsLists(
            logprob_token_ids=np.array([[256, 257]], dtype=np.int32),
            logprobs=np.array([[-0.1, -0.5]]),
            sampled_token_ranks=np.array([1], dtype=np.int32),
        )
        state.logprobs_processor.update_from_output(
            engine_core_output("r0", [256], logprobs=lp)
        )
        co = state._new_completion_output([256], "stop", None)
        assert len(co.logprobs) == 1

    def test_request_output_prompt_logprobs_pop_on_delta(self):
        params = sp(
            logprobs=1, prompt_logprobs=1,
            output_kind=lane.RequestOutputKind.DELTA,
        )
        state = make_state("r0", params, detokenizer=DetokFace())
        pt = lane.LogprobsTensors(
            torch.tensor([[256, 257]], dtype=torch.int32),
            torch.tensor([[-0.1, -0.3]]),
            torch.tensor([1], dtype=torch.int32),
        )
        state.logprobs_processor.update_from_output(
            engine_core_output("r0", [1], prompt_tensors=pt)
        )
        ro = state.make_request_output([1], None, None, None)
        assert len(ro.prompt_logprobs) == 2
        # DELTA semantics: all prompt logprobs AT ONCE at end of prefill,
        # then forgotten — next output carries none
        ro2 = state.make_request_output([2], None, None, None)
        assert ro2.prompt_logprobs == []

    def test_request_output_carries_completion(self):
        state = self._state(lane.RequestOutputKind.CUMULATIVE)
        lp = lane.LogprobsLists(
            logprob_token_ids=np.array([[256, 257]], dtype=np.int32),
            logprobs=np.array([[-0.1, -0.5]]),
            sampled_token_ranks=np.array([1], dtype=np.int32),
        )
        state.logprobs_processor.update_from_output(
            engine_core_output("r0", [256], logprobs=lp)
        )
        ro = state.make_request_output([256], None, None, None)
        assert ro.outputs[0].logprobs is not None
        assert ro.outputs[0].cumulative_logprob == pytest.approx(-0.1)


# ===========================================================================
# s14 — OpenAI three-field record (token/logprob/bytes)
# ===========================================================================


def serving(return_tokens_as_token_ids=False):
    return lane.OpenAIServingChat(
        engine_client=None,
        models=None,
        response_role="assistant",
        return_tokens_as_token_ids=return_tokens_as_token_ids,
    )


class TestOpenAIRecord:
    def test_bytes_field_gives_utf8_truth(self):
        # a multi-byte token's bytes are its raw UTF-8 sequence
        s = serving()
        lp = {0xAD: lane.Logprob(-0.1, 1, "中")}
        out = s._get_top_logprobs(lp, 1, None, should_return_as_token_id=False)
        rec = out[0]
        assert rec.token == "中"
        assert rec.bytes == list("中".encode("utf-8"))  # [228, 184, 173]
        assert rec.logprob == pytest.approx(-0.1)

    def test_logprob_clamped_at_minus_9999(self):
        s = serving()
        lp = {1: lane.Logprob(-30000.0, 1, "x")}
        rec = s._get_top_logprobs(lp, 1, None, False)[0]
        assert rec.logprob == -9999.0

    def test_top_logprobs_truncation(self):
        s = serving()
        lp = {
            i: lane.Logprob(-0.1 * i, i, f"t{i}") for i in (1, 2, 3, 4)
        }
        out = s._get_top_logprobs(lp, 2, None, False)
        assert len(out) == 2
        out_all = s._get_top_logprobs(
            lp, 2, None, False, return_all=True
        )
        assert len(out_all) == 4  # logprob_token_ids mode returns all

    def test_return_all_when_minus_one(self):
        s = serving()
        lp = {i: lane.Logprob(-0.1, i, "x") for i in (1, 2, 3)}
        assert len(s._get_top_logprobs(lp, -1, None, False)) == 3

    def test_create_chat_logprobs_step_content(self):
        s = serving()
        tk = byte_fallback_tokenizer()
        pos = {256: lane.Logprob(-0.2, 1, "hello"), 257: lane.Logprob(-0.9, 2, " world")}
        out = s._create_chat_logprobs(
            token_ids=[256],
            top_logprobs=[pos],
            tokenizer=tk,
            num_output_top_logprobs=1,
        )
        content = out.content[0]
        assert content.token == "hello"
        assert content.logprob == pytest.approx(-0.2)
        assert content.bytes == list(b"hello")
        assert len(content.top_logprobs) == 1
        assert content.top_logprobs[0].token == "hello"

    def test_create_chat_logprobs_missing_step_falls_back(self):
        # step has no logprobs for the token id -> decode + bytes, no clamp
        s = serving()
        tk = byte_fallback_tokenizer()
        out = s._create_chat_logprobs(
            token_ids=[256], top_logprobs=[None], tokenizer=tk
        )
        c = out.content[0]
        assert c.token == "hello"
        assert c.logprob == -9999.0  # model default
        assert c.bytes == list(b"hello")
        assert c.top_logprobs == []

    def test_return_as_token_id_placeholder(self):
        s = serving(return_tokens_as_token_ids=True)
        lp = {5: lane.Logprob(-0.1, 1, None)}
        rec = s._get_top_logprobs(lp, 1, None, True)[0]
        assert rec.token == "token_id:5"
        assert lane.format_token_id_placeholder(5) == "token_id:5"

    def test_decoded_token_none_bytes_none(self):
        # decoded_token None (tokenizer=None lane) -> bytes None, not []
        s = serving()
        pos = {5: lane.Logprob(-0.1, 1, None)}
        out = s._create_chat_logprobs(
            token_ids=[5], top_logprobs=[pos], tokenizer=None,
            return_as_token_id=True,
        )
        assert out.content[0].bytes is None


# ===========================================================================
# end-to-end — one request through the whole logprobs lane
# ===========================================================================


class TestLaneIntegration:
    def test_one_request_logprobs_journey(self):
        # entry -> batch register -> sampler (raw + gather) -> D2H ->
        # scheduler slice -> msgpack crossing -> process_outputs ->
        # completion output. Two requests share the bus: only one wants
        # logprobs (batch-uniform k accounts for the max).
        torch.manual_seed(7)
        vocab = 16
        ib = input_batch(vocab_size=vocab)
        ib.add_request("a", 0, sp(logprobs=2, output_kind=lane.RequestOutputKind.DELTA))
        ib.add_request("b", 1, sp())
        md = ib._make_sampling_metadata()
        assert md.max_num_logprobs == 2

        logits = torch.randn(2, vocab)
        sampler = lane.Sampler()
        out = sampler.forward(logits.clone(), md)
        lt = out.logprobs_tensors
        assert lt.logprobs.shape == (2, 3)  # batch-uniform k+1 = 3 for both

        mrr = lane.ModelRunnerOutput(
            req_ids=["a", "b"],
            req_id_to_index={"a": 0, "b": 1},
            sampled_token_ids=out.sampled_token_ids.tolist(),
        )
        async_out = lane.AsyncGPUModelRunnerOutput(
            model_runner_output=mrr,
            sampled_token_ids=out.sampled_token_ids,
            logprobs_tensors=lt,
            invalid_req_indices=[],
            async_output_copy_stream=COPY_STREAM,
            vocab_size=vocab,
        )
        mrr = async_out.get_output()
        assert isinstance(mrr.logprobs, lane.LogprobsLists)

        sched = lane.Scheduler(
            {
                "a": fake_sched_request("a", num_logprobs=2),
                "b": fake_sched_request("b"),
            }
        )
        ecoss = sched.update_from_output(
            SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1}), mrr
        )

        # crossing
        wire = lane.MsgpackEncoder().encode(ecoss[0])
        back = lane.MsgpackDecoder(lane.EngineCoreOutputs).decode(wire)

        # arrival + assembly
        tk = byte_fallback_tokenizer()
        op = lane.OutputProcessor(tokenizer=tk, log_stats=False)
        req = make_request("a", [1, 2], sp(logprobs=2, output_kind=lane.RequestOutputKind.DELTA))
        req.external_req_id = "ext-a"
        op.add_request(req, "", None, 0, None, detokenizer=DetokFace())
        res = op.process_outputs(back.outputs)
        ro = res.request_outputs[0]
        co = ro.outputs[0]

        # the sampled token IS argmax (greedy) and IS first in the container
        ref = ref_logprobs(logits)
        sampled = logits.argmax(-1)
        assert co.token_ids == [sampled[0].item()]  # request "a" only
        entry = co.logprobs[0]
        first_id = next(iter(entry))
        assert first_id == sampled[0].item()
        assert entry[first_id].logprob == pytest.approx(ref[0, sampled[0]].item())
        assert co.cumulative_logprob == pytest.approx(ref[0, sampled[0]].item())

        # cross-check the top-2 columns against a hand gather
        topv, topi = torch.topk(ref[0], 2)
        for j in range(2):
            tid = topi[j].item()
            assert entry[tid].logprob == pytest.approx(topv[j].item())
