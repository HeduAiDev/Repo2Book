# ch19-m13 worked-example driver — the padding four-piece, one coherent beat.
#
# Scenario (reader-hand-checkable): 5 active requests, uniform decode
# (1 token each -> num_tokens=5), capture_sizes=[1,8], max_num_seqs=8.
# The real runner method _determine_batch_execution_and_padding dispatches the
# beat to FULL with padded key (8, 8, True) — 3 pad rows/3 pad tokens.
# Then all four padding spans run against the same beat:
#   1) query_start_loc tail non-decreasing   (pinned span, L2073-L2078)
#   2) block_table tail rows NULL_BLOCK_ID   (pinned closure span, L2325-L2341)
#   3) slot_mapping tail -1                  (real full method _get_slot_mappings)
#   4) positions tail zeroed                 (pinned span, L3662-L3664)
# Spans 1/2/4 are driven verbatim on test doubles (same as the chapter tests);
# spans 3 and the dispatch ruling run through the companion's real methods.
# All buffers start with STALE tails from an earlier beat, so the before/after
# contrast shows real overwrite, not a no-op.
import json
import sys
from pathlib import Path
from types import SimpleNamespace

CHAPTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHAPTER))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from implementation.config.compilation import (  # noqa: E402
    CUDAGraphMode,
    CompilationConfig,
    CompilationMode,
)
from implementation.v1.cudagraph_dispatcher import CudagraphDispatcher  # noqa: E402
from implementation.v1.worker.gpu_model_runner import (  # noqa: E402
    NULL_BLOCK_ID,
    GPUModelRunner,
)


# same mimic as tests/test_compile_capture.py::make_seam_vllm_config
def make_seam_vllm_config(compilation_config, max_num_seqs=8):
    cfg = SimpleNamespace(
        compilation_config=compilation_config,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            tensor_parallel_size=1,
            use_sequence_parallel_moe=False,
            is_moe_model=None,
            use_ubatching=False,
            all2all_backend="deepep_low_latency",
        ),
        speculative_config=None,
        num_speculative_tokens=0,
        lora_config=None,
        observability_config=SimpleNamespace(cudagraph_metrics=True),
    )
    if compilation_config.mode == CompilationMode.VLLM_COMPILE:
        compilation_config.set_splitting_ops_for_v1(
            all2all_backend=cfg.parallel_config.all2all_backend,
            data_parallel_size=cfg.parallel_config.data_parallel_size,
        )
    if compilation_config.mode is None:
        compilation_config.mode = CompilationMode.VLLM_COMPILE
    if all(s not in compilation_config.custom_ops for s in ("all", "none")):
        if (
            compilation_config.backend == "inductor"
            and compilation_config.mode != CompilationMode.NONE
        ):
            compilation_config.custom_ops.append("none")
        else:
            compilation_config.custom_ops.append("all")
    if compilation_config.cudagraph_mode is None:
        compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if compilation_config.cudagraph_capture_sizes:
        compilation_config.max_cudagraph_capture_size = (
            compilation_config.cudagraph_capture_sizes[-1]
        )
        compilation_config.post_init_cudagraph_sizes()
    return cfg


class CpuGpuDouble:
    """ch18-domain CpuGpuBuffer test double (np view + copy_to_gpu counter)."""

    def __init__(self, np_arr):
        self.np = np_arr
        self.gpu = torch.from_numpy(np_arr.copy()).to(torch.int32)
        self.copies = 0

    def copy_to_gpu(self):
        self.copies += 1
        self.gpu = torch.from_numpy(self.np.copy()).to(torch.int32)


def main():
    trace = {"mechanism": "ch19-m13", "driver": Path(__file__).name}
    # pinned spans this driver drives (anchors cited by the worked-example table)
    trace["source_anchors"] = {
        "ruling": "vllm/v1/worker/gpu_model_runner.py:L4265-L4278 "
                  "(call site) + L3932-L4044 (_determine_batch_execution_and_padding)",
        "padded_dims": "vllm/v1/worker/gpu_model_runner.py:L4280-L4292",
        "query_start_loc": "vllm/v1/worker/gpu_model_runner.py:L2073-L2078",
        "block_table": "vllm/v1/worker/gpu_model_runner.py:L2338-L2341",
        "slot_mapping": "vllm/v1/worker/gpu_model_runner.py:L4128-L4130",
        "positions": "vllm/v1/worker/gpu_model_runner.py:L3663-L3664",
    }

    cc = CompilationConfig(
        cudagraph_mode="FULL_AND_PIECEWISE",
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_capture_sizes=[1, 8],
    )
    cfg = make_seam_vllm_config(cc, max_num_seqs=8)

    # ---- runner assembled like tests' _make_runner + _runner_with_buffers
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = cfg
    runner.compilation_config = cc
    runner.parallel_config = cfg.parallel_config
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.input_batch = SimpleNamespace(lora_id_to_lora_request={})
    runner.uniform_decode_query_len = 1
    runner.cudagraph_dispatcher = CudagraphDispatcher(cfg)
    runner.cudagraph_dispatcher.initialize_cudagraph_keys(
        cudagraph_mode=cc.cudagraph_mode, uniform_decode_query_len=1
    )
    runner.observability_config = cfg.observability_config
    runner.device = torch.device("cpu")

    # scenario: 5 requests, uniform decode (1 token each)
    num_reqs = 5
    num_tokens = 5
    scheduled = np.ones(num_reqs, dtype=np.int32)

    # ---- beat ruling: the REAL _determine_batch_execution_and_padding
    mode, desc, ubatch, dp, stats = runner._determine_batch_execution_and_padding(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        num_scheduled_tokens_np=scheduled,
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
    )
    num_tokens_padded = desc.num_tokens
    num_reqs_padded = desc.num_reqs if desc.num_reqs is not None else num_reqs
    trace["ruling"] = {
        "input": {"num_tokens": num_tokens, "num_reqs": num_reqs,
                  "max_num_scheduled_tokens": 1},
        "mode": mode.name,
        "desc": [desc.num_tokens, desc.num_reqs, desc.uniform, desc.has_lora,
                 desc.num_active_loras],
        "num_tokens_padded": num_tokens_padded,
        "num_reqs_padded": num_reqs_padded,
        "pad_tokens": num_tokens_padded - num_tokens,
        "pad_rows": num_reqs_padded - num_reqs,
        "stat_num_paddings": stats.num_paddings if stats else None,
        "stat_runtime_mode": stats.runtime_mode if stats else None,
    }

    # ---- 1) query_start_loc (pinned span L2073-L2078, driven verbatim)
    # active inclusive cumsum for [1,1,1,1,1] -> [1,2,3,4,5]; buffer size 9
    cu = np.cumsum(scheduled).astype(np.int32)
    runner.query_start_loc = CpuGpuDouble(np.zeros(9, dtype=np.int32))
    qsl_before = runner.query_start_loc.np.tolist()
    runner.query_start_loc.np[0] = 0
    runner.query_start_loc.np[1 : num_reqs + 1] = cu
    runner.query_start_loc.np[num_reqs + 1 :].fill(cu[-1])
    runner.query_start_loc.copy_to_gpu()
    trace["qsl"] = {
        "buffer_len": 9,
        "before": qsl_before,
        "cu_active": cu.tolist(),
        "after": runner.query_start_loc.np.tolist(),
        "tail": runner.query_start_loc.np[num_reqs + 1 :].tolist(),
        "copies": runner.query_start_loc.copies,
    }

    # ---- 2) block_table (pinned closure span L2325-L2341, driven verbatim)
    # rows 0..4 = real block ids (non-zero, distinct); rows 5..7 = stale ids
    # from an earlier beat; the span overwrites rows [5:8) with NULL_BLOCK_ID
    real_rows = [[7, 9], [12, 15], [3, 6], [11, 2], [8, 10]]
    stale_rows = [[13, 4], [9, 14], [5, 6]]
    table = real_rows + stale_rows

    def get_device_tensor(padded):
        return torch.tensor(table, dtype=torch.int32).clone()

    blk = get_device_tensor(num_reqs_padded)
    blk_before = blk.tolist()
    blk[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)
    trace["block_table"] = {
        "num_reqs": num_reqs,
        "num_reqs_padded": num_reqs_padded,
        "before": blk_before,
        "after": blk.tolist(),
        "pad_rows_written": [num_reqs, num_reqs_padded],
        "null_block_id": NULL_BLOCK_ID,
    }

    # ---- 3) slot_mapping (REAL method _get_slot_mappings, full call)
    # slot buffer: 5 real distinct slots + stale tail, padded view to 8.
    # NOTE: _get_slot_mapping takes a VIEW of the persistent gpu buffer and
    # fills the pad span in place — record "before" BEFORE the call.
    slot_gpu = torch.tensor([10, 11, 12, 13, 14, 99, 98, 97],
                            dtype=torch.int64)
    slot_before = slot_gpu.tolist()
    runner.input_batch = SimpleNamespace(
        block_table=[
            SimpleNamespace(
                get_device_tensor=get_device_tensor,
                slot_mapping=SimpleNamespace(gpu=slot_gpu),
            )
        ]
    )
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=2),
                layer_names=["m.l0", "m.l1"],
            )
        ]
    )
    by_gid, by_layer = runner._get_slot_mappings(
        num_tokens_padded=num_tokens_padded,
        num_reqs_padded=num_reqs_padded,
        num_tokens_unpadded=num_tokens,
        ubatch_slices=None,
    )
    sm = by_gid[0]
    trace["slot_mapping"] = {
        "before": slot_before,
        "after": sm.tolist(),
        "pad_span": [num_tokens, num_tokens_padded],
        "pad_value": -1,
        "active_prefix": sm[:num_tokens].tolist(),
        "layers_sharing": sorted(by_layer.keys()),
        "n_layers": len(by_layer),
    }

    # ---- 4) positions (pinned span L3662-L3664, driven verbatim)
    # active positions distinct & non-zero; tail holds stale arange values
    runner.positions = torch.arange(16, dtype=torch.float32)
    runner.positions[:num_tokens] = torch.tensor(
        [7.0, 100.0, 3.0, 42.0, 55.0])
    pos_before = runner.positions[:num_tokens_padded].tolist()
    num_scheduled_tokens = num_tokens
    num_input_tokens = num_tokens_padded
    if num_input_tokens > num_scheduled_tokens:
        runner.positions[num_scheduled_tokens:num_input_tokens].zero_()
    trace["positions"] = {
        "active": [7, 100, 3, 42, 55],
        "before": pos_before,
        "after": runner.positions[:num_tokens_padded].tolist(),
        "pad_span": [num_scheduled_tokens, num_input_tokens],
    }

    out = Path(__file__).with_name("ch19_m13_padding.json")
    out.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    print("wrote", out.name, "| mode:", trace["ruling"]["mode"],
          "| desc:", trace["ruling"]["desc"])


if __name__ == "__main__":
    main()
