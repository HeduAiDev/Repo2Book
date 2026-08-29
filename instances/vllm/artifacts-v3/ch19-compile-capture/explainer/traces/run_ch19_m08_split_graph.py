# ch19-m08 worked-example driver — split_graph piecewise splitting.
#
# Runs the companion's REAL pipeline on a 2-layer toy whose per-layer op order
# mirrors Attention.forward:  qkv proj -> slices/views -> unified_kv_cache_update
# -> unified_attention_with_output -> o proj.  All on host CPU with real torch
# (real FX symbolic_trace, real torch.ops.vllm.* registered by the companion,
# real set_splitting_ops_for_v1 account, real split_graph/should_split).
#
# Three splitting tables are compared:
#   A) the real v1 account: set_splitting_ops_for_v1() = 13 attention ops + 2 kv-update ops
#   B) attention-only table (13 ops, kv update NOT a split point) — the
#      issue #33267 trap: kv update would land inside a *compiled* piece
#   C) empty table — no split at all
#
# Plus a numeric equivalence check: stitched split_gm(x) == original gm(x).
import json
import sys
from pathlib import Path
from types import SimpleNamespace

CHAPTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHAPTER))

import torch  # noqa: E402

from implementation.config import set_current_vllm_config  # noqa: E402
from implementation.config.compilation import (  # noqa: E402
    CUDAGraphMode,
    CompilationConfig,
    CompilationMode,
)
from implementation.compilation.backends import split_graph  # noqa: E402
from implementation.compilation.partition_rules import should_split  # noqa: E402
from implementation.forward_context import set_forward_context  # noqa: E402
from implementation.model_executor.layers.attention.attention import (  # noqa: E402
    Attention,
)


# ---------------------------------------------------------------------------
# helpers — same mimic as tests/test_compile_capture.py::make_seam_vllm_config
# (VllmConfig.__post_init__ accounting for the bare-CompilationConfig face)
# ---------------------------------------------------------------------------

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


class RecordingImpl:
    """ch21-domain AttentionImpl test double (same observable face as tests' _FakeImpl)."""

    supports_quant_query_input = False
    forward_includes_kv_cache_update = False

    def __init__(self):
        self.kv_updates = 0
        self.forwards = 0

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        self.kv_updates += 1

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, *, output,
                output_scale=None, output_block_scale=None):
        self.forwards += 1
        output.copy_(query.sum(dim=-1, keepdim=True).expand_as(output))


def make_attention(prefix, cc, cfg):
    with set_current_vllm_config(cfg):
        attn = Attention(
            num_heads=2,
            head_size=4,
            scale=0.25,
            num_kv_heads=2,
            prefix=prefix,
            vllm_config=cfg,
        )
    attn.impl = RecordingImpl()
    attn.attn_backend = SimpleNamespace(forward_includes_kv_cache_update=False)
    return attn


# ---------------------------------------------------------------------------
# the toy: 2 layers, each shaped like a real decoder layer's op sequence
# ---------------------------------------------------------------------------

class ToyLayer(torch.nn.Module):
    def __init__(self, layer_name: str):
        super().__init__()
        self.layer_name = layer_name
        self.in_proj = torch.nn.Linear(4, 24, bias=False)
        self.o_proj = torch.nn.Linear(8, 4, bias=False)

    def forward(self, x):
        qkv = self.in_proj(x)
        q = qkv[:, 0:8].view(-1, 2, 4)
        k = qkv[:, 8:16].view(-1, 2, 4)
        v = qkv[:, 16:24].view(-1, 2, 4)
        out = torch.empty_like(q)
        dep = torch.ops.vllm.unified_kv_cache_update(k, v, self.layer_name)
        torch.ops.vllm.unified_attention_with_output(
            q, k, v, out, self.layer_name, kv_cache_dummy_dep=dep
        )
        return self.o_proj(out.view(-1, 8))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = ToyLayer("toy.l0")
        self.l1 = ToyLayer("toy.l1")

    def forward(self, x):
        return self.l1(self.l0(x))


def describe_split(gm, splitting_ops):
    """Run the real split_graph and describe each piece: node census + split flags."""
    split_gm, items = split_graph(gm, list(splitting_ops))
    pieces = []
    for it in items:
        submod = getattr(split_gm, it.submod_name)
        node_names, split_nodes, n_comp = [], [], 0
        for n in submod.graph.nodes:
            if n.op in ("placeholder", "output"):
                continue
            n_comp += 1
            node_names.append(n.name)
            if n.op == "call_function" and should_split(n, splitting_ops):
                qn = (
                    n.target._qualified_op_name
                    if isinstance(n.target, torch._ops.OpOverloadPacket)
                    else f"{n.target.name()}"
                )
                split_nodes.append(qn.split("::")[-1])
        pieces.append({
            "submod": it.submod_name,
            "graph_id": it.graph_id,
            "is_splitting_graph": it.is_splitting_graph,
            "n_nodes": n_comp,
            "nodes": node_names,
            "split_op_nodes_inside": split_nodes,
        })
    return split_gm, items, pieces


def main():
    torch.manual_seed(19)
    trace = {"mechanism": "ch19-m08", "driver": Path(__file__).name}

    # -- real v1 account (set_splitting_ops_for_v1) + the 13-op attention list
    cc = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    cfg = make_seam_vllm_config(cc)
    attn_only = list(CompilationConfig._attention_ops)
    trace["splitting_tables"] = {
        "v1_account_len": len(cc.splitting_ops),
        "attention_ops_len": len(attn_only),
        "v1_account_tail": [op.split("::")[-1] for op in cc.splitting_ops[-2:]],
        "v1_account_head": [op.split("::")[-1] for op in cc.splitting_ops[:2]],
    }

    # -- trace + split under the three tables
    model = ToyModel()
    gm = torch.fx.symbolic_trace(model)
    total_nodes = sum(
        1 for n in gm.graph.nodes if n.op not in ("placeholder", "output")
    )
    trace["traced_model"] = {"total_compute_nodes": total_nodes}

    split_gm_a, items_a, pieces_a = describe_split(gm, cc.splitting_ops)
    split_gm_b, items_b, pieces_b = describe_split(gm, attn_only)
    split_gm_c, items_c, pieces_c = describe_split(gm, [])
    trace["split_A_v1_account_15_ops"] = {
        "n_pieces": len(pieces_a),
        "n_compiled_pieces": sum(1 for p in pieces_a if not p["is_splitting_graph"]),
        "n_seams": sum(1 for p in pieces_a if p["is_splitting_graph"]),
        "pieces": pieces_a,
    }
    trace["split_B_attention_only_13_ops"] = {
        "n_pieces": len(pieces_b),
        "pieces": pieces_b,
        "kv_update_lands_in": [
            p["submod"] for p in pieces_b
            if "unified_kv_cache_update" in p["split_op_nodes_inside"] or
            any("kv_cache_update" in nm for nm in p["split_op_nodes_inside"])
        ],
        # under table B the kv-update node is NOT a split point: find which
        # *compiled* piece now contains it
        "kv_update_inside_compiled_piece": [
            p["submod"] for p in pieces_b
            if not p["is_splitting_graph"]
            and any("kv_cache_update" in nm for nm in p["nodes"])
        ],
    }
    trace["split_C_empty_table"] = {
        "n_pieces": len(pieces_c),
        "pieces": pieces_c,
    }

    # -- consecutive-split-point merge evidence: kv update and attention share
    #    one seam submod under table A (never two separate one-node submods)
    seam_pair = [
        p for p in pieces_a if p["is_splitting_graph"]
    ]
    trace["consecutive_merge"] = {
        "seams": [
            {"submod": p["submod"], "split_ops": p["split_op_nodes_inside"],
             "n_nodes": p["n_nodes"]}
            for p in seam_pair
        ],
        "kv_and_attn_share_seam": all(
            len(p["split_op_nodes_inside"]) == 2 for p in seam_pair
        ),
    }

    # -- numeric equivalence: stitched pieces == original graph, in order
    attn0 = make_attention("toy.l0", cc, cfg)
    attn1 = make_attention("toy.l1", cc, cfg)
    x = torch.randn(6, 4)
    slot = torch.zeros(6, dtype=torch.int64)
    with set_forward_context(
        attn_metadata={"toy.l0": "MD", "toy.l1": "MD"},
        vllm_config=cfg,
        slot_mapping={"toy.l0": slot, "toy.l1": slot},
    ):
        out_split = split_gm_a(x)
    with set_forward_context(
        attn_metadata={"toy.l0": "MD", "toy.l1": "MD"},
        vllm_config=cfg,
        slot_mapping={"toy.l0": slot, "toy.l1": slot},
    ):
        out_ref = gm(x)
    trace["equivalence"] = {
        "out_shape": list(out_split.shape),
        "allclose": bool(torch.allclose(out_split, out_ref)),
        "max_abs_diff": float((out_split - out_ref).abs().max()),
        "impl_kv_updates_per_layer": [attn0.impl.kv_updates, attn1.impl.kv_updates],
        "impl_forwards_per_layer": [attn0.impl.forwards, attn1.impl.forwards],
    }

    # -- dummy data-dependency probe (feeds m05 figure): the kv-update op
    #    returns an EMPTY tensor of key's dtype — the ordering token
    with set_forward_context(
        attn_metadata={"toy.l0": "MD", "toy.l1": "MD"},
        vllm_config=cfg,
        slot_mapping={"toy.l0": slot, "toy.l1": slot},
    ):
        k = torch.randn(6, 2, 4)
        v = torch.randn(6, 2, 4)
        dep = torch.ops.vllm.unified_kv_cache_update(k, v, "toy.l0")
    trace["dummy_dep_probe"] = {
        "numel": dep.numel(),
        "shape": list(dep.shape),
        "dtype_matches_key": dep.dtype == k.dtype,
    }

    out = Path(__file__).with_name("ch19_m08_split_graph.json")
    out.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    print("wrote", out.name, "| pieces A/B/C:",
          len(pieces_a), len(pieces_b), len(pieces_c),
          "| allclose:", trace["equivalence"]["allclose"])


if __name__ == "__main__":
    main()
