"""ch28-m08 驱动脚本 —— DeepseekV4MoE 装配分岔重放：gate 三模式 + 双后端二岔 + 三条 NotImplemented。

跑法(host, 纯 CPU torch): python run_ch29_m08_moe_assembly_branch.py
输出: ch29_m08_moe_assembly_branch.json(与本脚本同目录)

素材对应 dossier 机制 ch28-m08(needs_worked_example + needs_figure)。

真源（判定条件逐字拷贝/复刻, 不改语义）:
- vllm/models/deepseek_v4/nvidia/model.py:L530-L561(use_mega_moe 三重限定:
  无 EP → NotImplementedError / scoring != sqrtsoftplus → NotImplementedError /
  expert_dtype != fp4 → NotImplementedError, 消息原文带修法提示)、
  L565-L568(gate=GateLinear out_dtype=fp32)、L569-L587(hash MoE: 前 num_hash_layers
  层 tid2eid=randint(0,E,(vocab,topk)); noaux_tc → e_score_correction_bias;
  hash_indices_dtype=int64 if mega else int32)、L595-L613(shared_experts:
  intermediate=moe_intermediate_size*n_shared_experts, reduce_results=use_mega_moe)、
  L615+(_init_mega_moe_experts / _init_fused_moe_experts 二岔)。
- vllm/config/kernel.py:L121-L131(moe_backend 合法值表)、L191-L193(默认 "auto")。
- vllm/models/deepseek_v4/nvidia/model.py:L202-L246(MegaMoE fp4 字节形状——本脚本
  只重放形状算术, 字节账细节归 ch29_m10)。

四件:
① 逐场景重放装配判定: S1 hash 层(mega on) / S2 noaux_tc 层(mega on) /
   S3 fused 后端 / S4-S6 三条 NotImplementedError(消息原文)。
② 每场景打印建出的参数形状(tid2eid / e_score_correction_bias / gate / shared)。
③ use_mega_moe 判定式与 _use_sequence_parallel 门(model.py:L805-L813)复刻。
④ 专家参数形状按 mega 路径打印(E=4, EP world=2, H=128, I=128——取自
   tests/models/test_deepseek_v4_mega_moe.py:L58-L67 的真实测试构造)。
"""
import json
from pathlib import Path

import torch

torch.manual_seed(0)

E, TOPK = 4, 2
H, I = 128, 128
VOCAB = 16  # 本例用 16 词的小词表(真实模型 vocab≈13 万, tid2eid 形状同构)
NUM_HASH_LAYERS = 1
NSHARED = 1
EP_WORLD = 2
NUM_LOCAL = E // EP_WORLD  # 2


def use_mega_moe(moe_backend: str) -> bool:
    # model.py:L533-L535 verbatim
    return moe_backend == "deep_gemm_mega_moe"


def use_sequence_parallel(pp, ep_enabled, tp, moe_backend, dp) -> bool:
    # model.py:L805-L813 verbatim
    return (
        pp == 1
        and ep_enabled
        and tp > 1
        and (use_mega_moe(moe_backend) or dp > 1)
    )


def assemble(layer_idx, *, moe_backend, ep_enabled, scoring="sqrtsoftplus",
             expert_dtype="fp4", topk_method="noaux_tc"):
    """重放 DeepseekV4MoE.__init__ 的装配判定(model.py:L519-L613)。"""
    events = []
    mega = use_mega_moe(moe_backend)
    if mega and not ep_enabled:
        # model.py:L536-L541 消息原文
        return {"use_mega_moe": mega, "raise": (
            "NotImplementedError: DeepSeek V4 MegaMoE currently requires "
            "expert parallel. Enable it with --enable-expert-parallel, or pick "
            "a different moe backend."
        )}
    if mega and scoring != "sqrtsoftplus":
        # model.py:L552-L555 消息原文
        return {"use_mega_moe": mega, "raise": (
            "NotImplementedError: DeepSeek V4 MegaMoE currently supports "
            "sqrtsoftplus routing only."
        )}
    if mega and expert_dtype != "fp4":
        # model.py:L556-L561 消息原文
        return {"use_mega_moe": mega, "raise": (
            "NotImplementedError: DeepSeek V4 MegaMoE only supports fp4 "
            "experts; got expert_dtype=" + repr(expert_dtype) + ". Drop "
            "--kernel-config moe_backend=deep_gemm_mega_moe for this "
            "checkpoint."
        )}
    built = {"use_mega_moe": mega, "gate_out": [H, E], "gate_out_dtype": "fp32"}
    # hash MoE 判定: model.py:L573 verbatim
    is_hash_moe = layer_idx < NUM_HASH_LAYERS
    hash_dtype = "int64" if mega else "int32"  # model.py:L571-L572
    if is_hash_moe:
        tid2eid = torch.randint(0, E, (VOCAB, TOPK),
                                dtype=getattr(torch, hash_dtype))
        built["gate.tid2eid"] = {"shape": list(tid2eid.shape), "dtype": hash_dtype,
                                 "sample_rows_7_3": [tid2eid[7].tolist(),
                                                     tid2eid[3].tolist()]}
    elif topk_method == "noaux_tc":
        bias = torch.randn(E, dtype=torch.float32)  # 形状重放(model.py:L589-L593)
        built["gate.e_score_correction_bias"] = {"shape": [E], "dtype": "fp32",
                                                 "values": [round(v, 4) for v in bias.tolist()]}
    if NSHARED is not None:
        built["shared_experts"] = {
            "intermediate_size": I * NSHARED,
            "reduce_results": mega,  # model.py:L606: EP 下免 all_reduce
        }
    if mega:
        built["experts"] = "DeepseekV4MegaMoEExperts (EP)"
        built["expert_param_shapes"] = {
            "w13_weight": {"shape": [NUM_LOCAL, 2 * I, H // 2], "dtype": "uint8"},
            "w13_weight_scale": {"shape": [NUM_LOCAL, 2 * I, H // 32], "dtype": "uint8"},
            "w2_weight": {"shape": [NUM_LOCAL, H, I // 2], "dtype": "uint8"},
            "w2_weight_scale": {"shape": [NUM_LOCAL, H, I // 32], "dtype": "uint8"},
        }
    else:
        built["experts"] = "FusedMoE 工厂 (TP)"
    return built


out = {
    "env": "host Miniconda python 3.11.11, torch 2.11.0+cu128 (纯 CPU), pin=vLLM v0.27.1 (6e448d0ea)",
    "constants": {
        "E": E, "topk": TOPK, "H": H, "I": I, "vocab_toy": VOCAB,
        "num_hash_layers": NUM_HASH_LAYERS, "n_shared_experts": NSHARED,
        "ep_world_size": EP_WORLD, "num_local_experts": NUM_LOCAL,
        "moe_backend_default": "auto",
        "moe_backend_legal_values": ["auto", "triton", "batched_triton",
                                     "deep_gemm", "deep_gemm_mega_moe", "cutlass",
                                     "flashinfer_trtllm", "flashinfer_cutlass",
                                     "flashinfer_cutedsl", "flashinfer_b12x",
                                     "marlin", "humming", "triton_unfused",
                                     "aiter", "flydsl", "hpc", "emulation"],
    },
    "scenarios": {
        "S1_layer0_hash_mega": assemble(0, moe_backend="deep_gemm_mega_moe", ep_enabled=True),
        "S2_layer1_noaux_tc_mega": assemble(1, moe_backend="deep_gemm_mega_moe", ep_enabled=True),
        "S3_layer1_fused": assemble(1, moe_backend="auto", ep_enabled=False),
        "S4_mega_no_ep": assemble(1, moe_backend="deep_gemm_mega_moe", ep_enabled=False),
        "S5_mega_softmax": assemble(1, moe_backend="deep_gemm_mega_moe", ep_enabled=True, scoring="softmax"),
        "S6_mega_fp8": assemble(1, moe_backend="deep_gemm_mega_moe", ep_enabled=True, expert_dtype="fp8"),
    },
    "sequence_parallel_gate": {
        "claim": "pp==1 and ep and tp>1 and (mega or dp>1)  (model.py:L805-L813)",
        "case_A_pp1_ep_tp2_mega": use_sequence_parallel(1, True, 2, "deep_gemm_mega_moe", 1),
        "case_B_tp1": use_sequence_parallel(1, True, 1, "deep_gemm_mega_moe", 1),
        "case_C_dp2_mega_off": use_sequence_parallel(1, True, 2, "auto", 2),
        "case_D_pp2": use_sequence_parallel(2, True, 2, "deep_gemm_mega_moe", 1),
    },
    "forward_call_signature_mega": {
        "experts_args": ["hidden_states", "topk_weights", "topk_ids",
                         "activation_clamp=swiglu_limit"],
        "then": "final_hidden_states += shared_output  (shared 外部相加, model.py:L735-L737)",
    },
    "forward_call_signature_fused": {
        "is_internal_router": "MoERunner 内置 router 时 hidden 直喂 experts (model.py:L749-L756)",
        "else": "外部 gate 先算 router_logits 再进 FusedMoE (shared 聚合在内部)",
    },
}

dst = Path(__file__).parent / "ch29_m08_moe_assembly_branch.json"
with open(dst, "w", newline="\n", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("wrote", dst)
print(json.dumps(out["scenarios"], ensure_ascii=True, indent=1)[:1500])
