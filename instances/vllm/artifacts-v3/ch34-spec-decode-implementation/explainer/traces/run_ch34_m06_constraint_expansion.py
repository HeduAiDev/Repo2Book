# ch34 m6 worked example driver — 逐草稿位约束扩展。
# 全程驱动精简版真码：
#   - expand_batch_to_tokens / apply_sampling_constraints（rejection_sampler.py:L510-L605）
#   - RejectionSampler._combine_outputs_with_spec_tokens（L376-L391 逐位前缀行）
#   - RejectionSampler.apply_logits_processors（L289-L346：penalties repeat +
#     bad_words_with_drafts + MinTokens.apply_with_spec_decode）
#   - MinTokensLogitsProcessor.apply_with_spec_decode（logits_processor/builtin.py:L235-L286，
#     docstring 例 num_draft=[2,3,1] 原样驱动，经 update_state(BatchUpdate) 真实装载）
import json
import os
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np
import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    RejectionSampler,
    apply_sampling_constraints,
    expand_batch_to_tokens,
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def r6(x):
    return round(float(x), 6)


def vec(t):
    return [r6(v) if isinstance(v, float) else int(v) for v in t.tolist()]


def main():
    trace = {
        "mechanism": "m6",
        "what": "逐草稿位约束扩展：expand_batch_to_tokens / 草稿前缀历史 / MinTokens 特化版",
        "env": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
    }

    # ══ Part A：expand_batch_to_tokens + apply_sampling_constraints ══════
    # batch 3 请求 num_draft=[2,0,1]（req1 零草稿——ngram 场景），温度
    # [0.0(greedy), 1.0, 0.7]：greedy 的 0 替 1 防除零（L523-L527 replace_from/to）
    num_draft_a = [2, 0, 1]
    cu_num_draft_a = np.cumsum(num_draft_a, dtype=np.int32)
    cu_num_draft_a_t = torch.from_numpy(cu_num_draft_a).to(DEVICE)
    temperature = torch.tensor([0.0, 1.0, 0.7], dtype=torch.float32, device=DEVICE)
    num_tokens_a = sum(num_draft_a)  # 3

    expanded_temperature = expand_batch_to_tokens(
        temperature, cu_num_draft_a_t, num_tokens_a,
        replace_from=0, replace_to=1,
    )
    V = 8
    logits_a = torch.tensor(
        [[2.0, 4.0, 6.0, 1.0, 0.5, 0.5, 0.5, 0.5],
         [1.0, 3.0, 5.0, 7.0, 0.5, 0.5, 0.5, 0.5],
         [0.7, 1.4, 2.1, 2.8, 0.7, 0.7, 0.7, 0.7]],
        dtype=torch.float32, device=DEVICE,
    )
    logits_a_before = logits_a.clone()

    import types

    SMetaA = types.SimpleNamespace(
        all_greedy=False, all_random=False,
        temperature=temperature, top_k=None, top_p=None,
    )

    logits_a_after = apply_sampling_constraints(logits_a, cu_num_draft_a_t, SMetaA)
    part_a = {
        "what": "expand_batch_to_tokens + apply_sampling_constraints（真跑）",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L510-L605",
        "params": {
            "num_draft_tokens": num_draft_a,
            "cu_num_draft_tokens": cu_num_draft_a.tolist(),
            "temperature_batch": vec(temperature),
            "note": "req0 greedy(temp 0) 两行、req1 零草稿零行、req2 一行",
        },
        "expanded_temperature": vec(expanded_temperature),
        "greedy_replacement": "req0 的 0 → 1（防除零，L523-L527）",
        "logits_row2_before_div": vec(logits_a_before[2]),
        "logits_row2_after_div_0_7": vec(logits_a_after[2]),
        "check_row2": vec(logits_a_before[2] / 0.7) == vec(logits_a_after[2]),
        "rows_unchanged_div_by_1": {
            "row0": vec(logits_a_after[0]) == vec(logits_a_before[0]),
            "row1": vec(logits_a_after[1]) == vec(logits_a_before[1]),
        },
    }

    # ══ Part B：_combine_outputs_with_spec_tokens（逐位前缀行）══════════
    rs = RejectionSampler(Sampler())
    output_token_ids = [[5, 6], [7], [8]]
    spec_token_ids = [[70, 71], [], [90]]
    combined = rs._combine_outputs_with_spec_tokens(output_token_ids, spec_token_ids)
    part_b = {
        "what": "RejectionSampler._combine_outputs_with_spec_tokens（真跑）",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L376-L391",
        "params": {
            "output_token_ids": output_token_ids,
            "spec_token_ids": spec_token_ids,
            "num_draft_tokens": num_draft_a,
        },
        "combined_rows": combined,
        "note": (
            "草稿位 i 的历史 = outputs + spec[:i]（只拼前缀、不含未『说出口』的后续草稿）；"
            "req1 零草稿整段跳过（对应 0 个 target 行）；与 Sampler 同名方法语义不同——"
            "那边拼整段 spec、逐请求一行（sampler.py:L358-L369，bonus 位用）"
        ),
    }

    # ══ Part C：apply_logits_processors（penalties repeat + bad_words）═══
    # 走真实 SamplingMetadata：presence=0.5 只挂 req0；bad_words 只挂 req0；
    # combined 历史（Part B）作为『草稿当已输出』的前缀历史喂进去。
    # 词表取 VC=100：草稿 token 70/71/90 必须是合法 token id
    VC = 100
    logits_c = torch.zeros((3, VC), dtype=torch.float32, device=DEVICE)
    logits_c[0, [2, 4, 6, 8]] = torch.tensor([2.0, 4.0, 6.0, 8.0], device=DEVICE)
    logits_c[1, [2, 4, 6, 8]] = torch.tensor([2.0, 4.0, 6.0, 8.0], device=DEVICE)
    logits_c[2, :] = 1.0
    logits_c_before = logits_c.clone()
    freq = torch.zeros(3, dtype=torch.float32, device=DEVICE)
    pres = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
    rep = torch.ones(3, dtype=torch.float32, device=DEVICE)  # 1.0 = no-op
    prompt_tok = torch.tensor([[3], [3], [3]], dtype=torch.long, device=DEVICE)
    smeta_c = SamplingMetadata(
        temperature=temperature,
        all_greedy=False, all_random=False,
        top_p=None, top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=False,
        prompt_token_ids=prompt_tok,
        frequency_penalties=freq,
        presence_penalties=pres,
        repetition_penalties=rep,
        output_token_ids=output_token_ids,
        allowed_token_ids_mask=None,
        bad_words_token_ids={0: [[70, 5]]},  # 禁词：70 之后不许出 5
        logitsprocs=LogitsProcessors(),
        logprob_token_ids=None,
        spec_token_ids=spec_token_ids,
    )
    md_c = SpecDecodeMetadata(
        draft_token_ids=torch.tensor([70, 71, 90], dtype=torch.int32, device=DEVICE),
        num_draft_tokens=num_draft_a,
        cu_num_draft_tokens=cu_num_draft_a_t,
        cu_num_sampled_tokens=torch.tensor([3, 4, 6], dtype=torch.int32, device=DEVICE),
        target_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE),
        bonus_logits_indices=torch.tensor([2, 3, 5], dtype=torch.int32, device=DEVICE),
        logits_indices=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device=DEVICE),
    )
    logits_c_after = rs.apply_logits_processors(logits_c, smeta_c, md_c)
    delta = (logits_c_after - logits_c_before).cpu()
    part_c = {
        "what": "apply_logits_processors：presence 惩罚 repeat 展开 + bad_words 草稿前缀（真跑）",
        "provenance": "vllm/v1/sample/rejection_sampler.py:L289-L346 + vllm/v1/sample/ops/bad_words.py:L39-L58",
        "params": {
            "presence_penalties_batch": vec(pres),
            "bad_words": {0: [[70, 5]]},
            "combined_history_rows": combined,
            "penalties_history_rows": combined,
        },
        "repeat_indices": torch.arange(3).repeat_interleave(
            torch.tensor(num_draft_a)).tolist(),
        "presence_effect_rows": {
            "row0(历史[5,6])": {"token5_delta": r6(delta[0, 5]), "token6_delta": r6(delta[0, 6]),
                                "token70_delta": r6(delta[0, 70]),
                                "note": "历史含 5,6 → 各 −0.5；70 不在历史 → 不动"},
            "row1(历史[5,6,70])": {"token6_delta": r6(delta[1, 6]), "token70_delta": r6(delta[1, 70]),
                                  "token5_final": "-inf（先 −0.5 惩罚、后 bad_words 封 −inf）"},
            "row2(历史[8])": "req2 无惩罚（presence=0）",
        },
        "bad_words_effect": {
            "bad_word": [70, 5],
            "prefix_required": [70],
            "row0_history_tail": [6],
            "row1_history_tail": [70],
            "row1_token5_after": "-inf",
            "row0_token5_after": r6(logits_c_after[0, 5].item()),
            "note": "只有 row1 的历史以 70 结尾 → 前缀匹配 → token5 封 -inf；row0 历史不含 70 → 不封",
        },
        "row1_token5_is_neg_inf": bool(logits_c_after[1, 5].item() == float("-inf")),
        "row0_token5_is_neg_inf": bool(logits_c_after[0, 5].item() == float("-inf")),
    }

    # ══ Part D：MinTokens.apply_with_spec_decode（docstring 例 [2,3,1]）══
    mt = MinTokensLogitsProcessor(None, DEVICE, False)
    # 经真实 update_state(BatchUpdate) 装载三个请求；req2 已达标（min=2 且已产 2）→ 立即摘除
    mt.update_state(BatchUpdate(
        batch_size=3,
        added=[
            (0, SamplingParams(min_tokens=4, all_stop_token_ids={2}), None, [10, 11]),
            (1, SamplingParams(min_tokens=7, all_stop_token_ids={2}), None, [10, 11]),
            (2, SamplingParams(min_tokens=2, all_stop_token_ids={2}), None, [10, 11]),
        ],
        removed=(),
        moved=(),
    ))
    tracked_after_update = sorted(mt.min_toks.keys())
    num_draft_d = [2, 3, 1]
    logits_d = torch.full((6, V), 1.0, dtype=torch.float32, device=DEVICE)
    logits_d_before = logits_d.clone()
    logits_d = mt.apply_with_spec_decode(logits_d, num_draft_d)
    col2_after = logits_d[:, 2].cpu().tolist()
    part_d = {
        "what": "MinTokens.apply_with_spec_decode（docstring 例 num_draft=[2,3,1] 真跑）",
        "provenance": "vllm/v1/sample/logits_processor/builtin.py:L235-L286（docstring 自带 worked example）",
        "params": {
            "num_draft_tokens": num_draft_d,
            "logits_shape": [6, V],
            "cumsum": np.concatenate([[0], np.cumsum(num_draft_d)]).tolist(),
            "requests": {
                "req0": {"min_tokens": 4, "current_len": 2, "remaining": 2,
                         "rows": [0, 1]},
                "req1": {"min_tokens": 7, "current_len": 2, "remaining": 5,
                         "n_mask_capped_by_draft": 3, "rows": [2, 3, 4]},
                "req2": {"min_tokens": 2, "current_len": 2,
                         "removed_by_update_state": True},
            },
            "stop_token_ids": [2],
        },
        "tracked_requests_after_update_state": tracked_after_update,
        "req1_cap": "remaining=5 > num_draft=3 → n_mask=min(5,3)=3（封位不超本拍草稿行数）",
        "token2_before": vec(logits_d_before[:, 2]),
        "token2_after": [("-inf" if v == float("-inf") else r6(v)) for v in col2_after],
        "masked_rows": [i for i, v in enumerate(col2_after) if v == float("-inf")],
        "unmasked_row5_reason": "req2 已达 min_tokens（update_state 摘除，L197-L208）",
    }

    trace.update({
        "part_a_expand_and_constraints": part_a,
        "part_b_combine_history": part_b,
        "part_c_penalties_badwords": part_c,
        "part_d_mintokens_spec": part_d,
    })

    out_path = pathlib.Path(__file__).resolve().parent / "ch34_m06_constraint_expansion.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print(json.dumps(trace, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
