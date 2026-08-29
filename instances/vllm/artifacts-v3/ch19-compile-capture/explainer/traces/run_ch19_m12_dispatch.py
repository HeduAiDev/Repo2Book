# ch19-m12 worked-example driver — BatchDescriptor keys + table dispatch.
#
# Runs the companion's REAL CudagraphDispatcher end to end on host CPU:
#   - bs -> padded size table (_compute_bs_to_padded_graph_size)
#   - initialize_cudagraph_keys (FULL exact / PIECEWISE relaxed)
#   - dispatch beats (FULL hit -> PIECEWISE fallback -> NONE)
#   - get_capture_descs (largest-first ordering)
# Two scales:
#   toy scale   : capture_sizes=[1,2,4], max_num_seqs=8  (reader can hand-check)
#   default scale: the documented default size pattern
#                 [1,2,4]+range(8,256,8)+range(256,513,16)  (compilation.py:L698-L706),
#                 max_num_seqs=256 (arg_utils default) — how many graphs a
#                 default -O2 engine captures at startup.
import json
import sys
from pathlib import Path
from types import SimpleNamespace

CHAPTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHAPTER))

from implementation.config.compilation import (  # noqa: E402
    CUDAGraphMode,
    CompilationConfig,
    CompilationMode,
)
from implementation.forward_context import BatchDescriptor  # noqa: E402
from implementation.v1.cudagraph_dispatcher import CudagraphDispatcher  # noqa: E402


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
        observability_config=SimpleNamespace(cudagraph_metrics=False),
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


def key_tuple(d: BatchDescriptor):
    return [d.num_tokens, d.num_reqs, d.uniform, d.has_lora, d.num_active_loras]


def sorted_keys(dispatcher, mode):
    return sorted(
        (key_tuple(d) for d in dispatcher.cudagraph_keys[mode]),
        key=lambda t: t[0],
    )


def beat(dispatcher, label, num_tokens, **kw):
    mode, desc = dispatcher.dispatch(num_tokens=num_tokens, **kw)
    return {
        "label": label,
        "input": {"num_tokens": num_tokens, **{k: str(v) for k, v in kw.items()}},
        "mode": mode.name,
        "desc": key_tuple(desc),
        "waste": desc.num_tokens - num_tokens,
    }


def main():
    trace = {"mechanism": "ch19-m12", "driver": Path(__file__).name}

    # ------------------------------------------------------------------ toy
    cc = CompilationConfig(
        cudagraph_mode="FULL_AND_PIECEWISE",
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_capture_sizes=[1, 2, 4],
    )
    cfg = make_seam_vllm_config(cc, max_num_seqs=8)
    d = CudagraphDispatcher(cfg)
    d.initialize_cudagraph_keys(
        cudagraph_mode=cc.cudagraph_mode, uniform_decode_query_len=1
    )
    trace["toy"] = {
        "capture_sizes": list(cc.cudagraph_capture_sizes),
        "max_cudagraph_capture_size": cc.max_cudagraph_capture_size,
        "max_num_seqs": 8,
        "bs_to_padded": {str(bs): d._bs_to_padded_graph_size[bs]
                         for bs in range(1, cc.max_cudagraph_capture_size + 1)},
        "piecewise_keys_relaxed": sorted_keys(d, CUDAGraphMode.PIECEWISE),
        "full_keys_exact": sorted_keys(d, CUDAGraphMode.FULL),
        "beats": [
            beat(d, "uniform decode 3 -> padded FULL", 3, uniform_decode=True),
            beat(d, "mixed batch 3 (non-uniform) -> PIECEWISE", 3),
            beat(d, "cascade attn bans FULL (uniform 2) -> PIECEWISE", 2,
                 uniform_decode=True, invalid_modes={CUDAGraphMode.FULL}),
            beat(d, "over max_size 9 -> NONE", 9),
            beat(d, "force eager 4 -> NONE", 4,
                 valid_modes={CUDAGraphMode.NONE}),
        ],
        "capture_descs": [
            {"mode": mode.name,
             "descs": [key_tuple(x) for x in descs]}
            for mode, descs in d.get_capture_descs()
        ],
    }

    # FULL-only mode: exact num_reqs for every key (FA3 scheduler_metadata)
    cc_full = CompilationConfig(
        cudagraph_mode="FULL",
        mode=CompilationMode.NONE,
        cudagraph_capture_sizes=[1, 2, 4],
    )
    cfg_full = make_seam_vllm_config(cc_full, max_num_seqs=8)
    d_full = CudagraphDispatcher(cfg_full)
    d_full.initialize_cudagraph_keys(
        cudagraph_mode=cc_full.cudagraph_mode, uniform_decode_query_len=1
    )
    trace["full_only_mode"] = {
        "full_keys_exact": sorted_keys(d_full, CUDAGraphMode.FULL),
        "piecewise_keys": [],
        "beats": [
            beat(d_full, "bs 3 -> padded FULL (num_reqs exact 4)", 3),
            beat(d_full, "bs 4 exact hit, zero waste", 4),
        ],
    }

    # ------------------------------------------------------- default scale
    # documented default pattern (vllm/config/compilation.py:L698-L706):
    #   [1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16),
    #   max = min(max_num_seqs*2, 512); max_num_seqs default 256
    default_sizes = (
        [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))
    )
    cc_def = CompilationConfig(
        cudagraph_mode="FULL_AND_PIECEWISE",
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_capture_sizes=default_sizes,
    )
    cfg_def = make_seam_vllm_config(cc_def, max_num_seqs=256)
    d_def = CudagraphDispatcher(cfg_def)
    d_def.initialize_cudagraph_keys(
        cudagraph_mode=cc_def.cudagraph_mode, uniform_decode_query_len=1
    )
    n_pw = len(d_def.cudagraph_keys[CUDAGraphMode.PIECEWISE])
    n_full = len(d_def.cudagraph_keys[CUDAGraphMode.FULL])
    trace["default_scale"] = {
        "sizes_source": "vllm/config/compilation.py:L698-L706 docstring pattern",
        "max_num_seqs": 256,
        "n_capture_sizes": len(default_sizes),
        "max_cudagraph_capture_size": cc_def.max_cudagraph_capture_size,
        "max_num_tokens_for_decode_full": 1 * 256,
        "n_piecewise_keys": n_pw,
        "n_full_keys": n_full,
        "n_total_graphs": n_pw + n_full,
        "bs9_padded_to": d_def._bs_to_padded_graph_size[9],
        "bs9_waste": d_def._bs_to_padded_graph_size[9] - 9,
        "bs9_beat": beat(d_def, "default scale bs 9 -> padded 16", 9),
        "largest_first_piecewise": [
            key_tuple(x) for x in d_def.get_capture_descs()[0][1][:3]
        ],
    }

    out = Path(__file__).with_name("ch19_m12_dispatch.json")
    out.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    print("wrote", out.name, "| toy keys:",
          len(trace["toy"]["piecewise_keys_relaxed"]),
          len(trace["toy"]["full_keys_exact"]),
          "| default graphs:", trace["default_scale"]["n_total_graphs"])


if __name__ == "__main__":
    main()
