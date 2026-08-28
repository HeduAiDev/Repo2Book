"""ch18-m06 token 扁平收集 —— 驱动脚本（host 纯 CPU）。

单拍剧本：三个新请求混合进批，num_scheduled_tokens = [2, 5, 3] ——
恰是 _prepare_inputs 源码注释自带的算例（gpu_model_runner.py:L1981-L1989），
这里把它真跑一遍：

  r1: prompt 6 token、前缀命中 4（num_computed=4）→ 本拍续 2（chunk 收尾）
  r2: prompt 5 token、num_computed=0                → 本拍全量 5（首 prefill）
  r3: prompt 10 token、num_computed=7               → 本拍续 3（chunked prefill 续块）

记录 _prepare_inputs 全套中间量：
  np.repeat 展开 req_indices / _get_cumsum_and_arange 的 cu_num_tokens 与
  请求内偏移 query_pos / positions / 二维坐标编一维 token_indices（按源码恒等式
  token_indices = positions + req_indices·M 从观测量导出）/ torch.index_select
  收出的 input_ids / query_start_loc 尾部 pad 非递减 / GPU 端 seq_lens /
  logits_indices（每请求末 token = 采样位）。

跑法：python explainer/traces/run_ch18_m06_gather.py
产物：explainer/traces/ch18_m06_gather.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation._host_seams import SamplingParams  # noqa: E402
from implementation.gpu_model_runner import GPUModelRunner  # noqa: E402
from implementation.output import CachedRequestData, NewRequestData, SchedulerOutput  # noqa: E402

VOCAB = 32
MAX_NUM_REQS = 4
MAX_MODEL_LEN = 16  # M = token_ids_cpu.shape[1]
MAX_NUM_BATCHED_TOKENS = 64


def _greedy():
    return SamplingParams(temperature=0.0)


def _vllm_config():
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        max_model_len=MAX_MODEL_LEN,
        runner_type="generate",
        enable_prompt_embeds=False,
        uses_mrope=False,
        uses_xdrope_dim=0,
        is_encoder_decoder=False,
        dtype=torch.float32,
        logits_processors=None,
        get_vocab_size=lambda: VOCAB,
    )
    cache_config = SimpleNamespace(
        block_size=16,
        calculate_kv_scales=False,
        use_replayssm=False,
        mamba_cache_mode="align",
        kv_sharing_fast_prefill=False,
    )
    scheduler_config = SimpleNamespace(
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=MAX_NUM_REQS,
        async_scheduling=False,
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        speculative_config=None,
        compilation_config=None,
        lora_config=None,
        load_config=None,
        offload_config=None,
        observability_config=None,
        reasoning_config=None,
    )


def _new_req(req_id, prompt, block_ids, num_computed):
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[],
        sampling_params=_greedy(),
        pooling_params=None,
        block_ids=(list(block_ids),),
        num_computed_tokens=num_computed,
        lora_request=None,
    )


def _logits_row(pick):
    row = [0.0] * VOCAB
    row[pick % VOCAB] = 1.0
    return row


def main():
    r = GPUModelRunner(_vllm_config(), torch.device("cpu"))
    from types import SimpleNamespace
    r.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])

    prompts = {
        "r1": [11, 12, 13, 14, 15, 16],                    # 6 tokens, 前缀命中 4
        "r2": [21, 22, 23, 24, 25],                        # 5 tokens, 全新
        "r3": [31, 30, 29, 28, 27, 26, 25, 24, 23, 22],    # 10 tokens, 已算 7
    }
    num_computed = {"r1": 4, "r2": 0, "r3": 7}
    num_scheduled = {"r1": 2, "r2": 5, "r3": 3}
    picks = {"r1": 1, "r2": 2, "r3": 3}

    r.enqueue_logits([{rid: _logits_row(p) for rid, p in picks.items()}])

    so = SchedulerOutput(
        scheduled_new_reqs=[
            _new_req("r1", prompts["r1"], [1], num_computed["r1"]),
            _new_req("r2", prompts["r2"], [2], num_computed["r2"]),
            _new_req("r3", prompts["r3"], [3], num_computed["r3"]),
        ],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[], resumed_req_ids=set(), new_token_ids=[],
            all_token_ids={}, new_block_ids=[], num_computed_tokens=[],
            num_output_tokens=[],
        ),
        num_scheduled_tokens=dict(num_scheduled),
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    assert r.execute_model(so) is None

    total = so.total_num_scheduled_tokens
    num_reqs = r.input_batch.num_reqs
    M = int(r.input_batch.token_ids_cpu.shape[1])

    # ---- 观测量（全部来自 runner 持久缓冲的真实内容） ----
    num_scheduled_np = np.array(
        [num_scheduled[rid] for rid in r.input_batch.req_ids], dtype=np.int32
    )
    req_indices_obs = r.req_indices.np[:total].tolist()
    query_pos_obs = r.query_pos.np[:total].tolist()
    # cu_num_tokens 恰被写进 query_start_loc.np[1:num_reqs+1]（L2073-L2078）
    cu_num_tokens_obs = r.query_start_loc.np[1 : num_reqs + 1].tolist()
    query_start_loc_full = r.query_start_loc.np[: r.max_num_reqs + 1].tolist()
    positions_obs = r.positions[:total].tolist()
    seq_lens_obs = r.seq_lens[:num_reqs].tolist()
    gathered = r.input_ids.cpu[:total].tolist()
    token_rows = {
        f"{rid}@{i}": r.input_batch.token_ids_cpu[i].tolist()
        for i, rid in enumerate(r.input_batch.req_ids)
    }
    token_flat_first48 = r.input_batch.token_ids_cpu.flatten()[: 3 * M].tolist()

    # ---- 导出量（按源码恒等式从观测量计算，供表格展示） ----
    # token_indices = positions_np + req_indices * M（gpu_model_runner.py:L2007-L2014）
    token_indices_derived = [
        p + ri * M for p, ri in zip(positions_obs, req_indices_obs)
    ]

    out = r.sample_tokens(None)
    sampled = {rid: out.sampled_token_ids[i][0] for i, rid in enumerate(out.req_ids)}

    trace = {
        "driver": "run_ch18_m06_gather.py",
        "mechanism": "ch18-m06 token 扁平收集（gpu_model_runner.py:L1743-L1767 _get_cumsum_and_arange + L1977-L2024 收集主段）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch18 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "trace_environment": "Windows host 纯 CPU（positions/seq_lens 在 CPU device 上由同一源码行算出；GPU 端数值路径归 ch19）",
        "config": {
            "max_num_reqs": MAX_NUM_REQS,
            "max_model_len": MAX_MODEL_LEN,
            "M_token_ids_cpu_cols": M,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "block_size": 16,
            "vocab": VOCAB,
        },
        "requests": {
            "r1": {"prompt": prompts["r1"], "num_computed": 4, "scheduled": 2,
                   "phase": "前缀命中后续算尾 chunk（prefill 收尾）"},
            "r2": {"prompt": prompts["r2"], "num_computed": 0, "scheduled": 5,
                   "phase": "全新首拍全量 prefill"},
            "r3": {"prompt": prompts["r3"], "num_computed": 7, "scheduled": 3,
                   "phase": "chunked prefill 续块"},
        },
        "num_scheduled_tokens_in_batch_order": num_scheduled_np.tolist(),
        "source_comment_example": "源码注释算例（L1981-L1989）：[2,5,3] -> req_indices [0,0,1,1,1,1,1,2,2,2]；cu [2,7,10]；query_pos [0,1,0,1,2,3,4,0,1,2]",
        "observed": {
            "req_indices": req_indices_obs,
            "query_pos": query_pos_obs,
            "cu_num_tokens_from_query_start_loc": cu_num_tokens_obs,
            "query_start_loc_full_with_pad": query_start_loc_full,
            "positions": positions_obs,
            "seq_lens": seq_lens_obs,
            "gathered_input_ids": gathered,
            "token_rows_full": token_rows,
            "token_ids_cpu_flat_first48": token_flat_first48,
            "logits_indices": (np.asarray(query_start_loc_full[1:num_reqs + 1]) - 1).tolist(),
        },
        "derived": {
            "token_indices": token_indices_derived,
            "identity": "token_indices = positions + req_indices * M（L2007-L2014；从上方观测量按恒等式导出，读者可用 flat 视图手验：flat[t] = row·M + col）",
        },
        "sampled_after_beat": sampled,
        "writeback_positions": {
            rid: int(r.input_batch.num_tokens_no_spec[i])
            for i, rid in enumerate(r.input_batch.req_ids)
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "ch18_m06_gather.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print(f"trace -> {out_path}")
    print("num_scheduled:", num_scheduled_np.tolist())
    print("req_indices :", req_indices_obs)
    print("query_pos   :", query_pos_obs)
    print("cu          :", cu_num_tokens_obs)
    print("qsl(padded) :", query_start_loc_full)
    print("positions   :", positions_obs)
    print("seq_lens    :", seq_lens_obs)
    print("tok_indices :", token_indices_derived)
    print("gathered    :", gathered)
    print("sampled     :", sampled)


if __name__ == "__main__":
    main()
