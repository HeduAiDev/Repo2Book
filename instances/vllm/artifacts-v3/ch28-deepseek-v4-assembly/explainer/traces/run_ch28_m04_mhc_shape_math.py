"""ch28-m04 驱动脚本 —— mHC 多流残差：一个 token 走完「2D embed → hc_mult 条流 → hc_head 压回 2D」。

跑法(host, 纯 CPU torch): python run_ch28_m04_mhc_shape_math.py
输出: ch28_m04_mhc_shape_math.json(与本脚本同目录)

素材对应 dossier 机制 ch28-m04(needs_worked_example + needs_figure)。

数学真源（逐字拷贝，不改一行）:
- sinkhorn_normalize_ref / mhc_pre_ref / mhc_post_ref / hc_head_ref
  = vllm 树内 tests/kernels/test_mhc_kernels.py:L14-L90 —— 这四个纯 torch 参考实现
  正是 tilelang 生产核(torch.ops.vllm.mhc_*_tilelang, vllm/model_executor/kernels/mhc/)
  的对拍基准(测试 atol 5e-2)。教学口径：本脚本跑出的数值 = 生产核的参考语义，
  生产核本体是 tilelang 生成核、Python 侧无实现（dossier 诚实边界）。
- 参数形状/装配: vllm/models/deepseek_v4/nvidia/model.py:L849-L897
  (mix_hc=(2+hc)*hc, hc_dim=hc*H, hc_scale 形状 (3,), hc_post_alpha=2.0 硬编码)、
  L1076-L1094(hc_head_fn (hc_mult,hc_dim)/hc_head_base (hc_mult,)/hc_head_scale (1,))、
  L1371-L1380(finalize_mhc_broadcast_weights: fn.view(-1,hc,H).sum(1) 预折)、
  L1200-L1212(层尾顺序: mhc_post → _mtp_hidden_buffer.copy_(flatten(1)) → hc_head → norm)。
- 超参取值: hc_mult=2 与 H=2 是本例自选的小参数(读者可手查形状)；
  sinkhorn_repeat=20 / eps=1e-6 对齐 tests/kernels/test_mhc_kernels.py:L113-L115 的取值
  (真实模型读 config.hc_sinkhorn_iters / config.hc_eps，pin 树内无 DSV4 config.json)；
  hc_post_mult_value=2.0 用模型真值(model.py:L852)——测试里用 1.0。
- 注意力/FFN 两半的「子层输出」在本例中是固定占位值 x_attn/x_ffn——mHC 混合
  不关心 x 从哪来(生产里 = 注意力/MoE 输出)，注意力本体归 ch25/26。

三件:
① 一个 token 的完整旅程逐站形状+数值(embed 2D → 首层 broadcast 展开 →
   三件套 (residual, post_mix, res_mix) → ffn 半融合核 post+pre → 层尾 mhc_post
   → flatten 进 _mtp_hidden_buffer → hc_head 压回 2D)。
② Sinkhorn 收敛演示: repeat=1/2/3/5/20 时 res_mix 行和/列和偏离 1 的最大值
   (双随机化 = res_mix 作为「流间混合矩阵」的合法性来源)。
③ 首层 broadcast 可预折的数值验证: 2D 输入各流相同 ⇒ 用预折 fn_broadcast
   (fn 对流维求和) 直接从 2D 算 mixes ≡ 先 repeat 成多流再用 fn —— allclose。
"""
import json
from pathlib import Path

import torch

torch.manual_seed(0)

# ---- 以下四个函数逐字拷贝自 tests/kernels/test_mhc_kernels.py:L14-L90 (pin v0.27.1) ----


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """mHC pre reference kernel from tilelang repo."""
    hc_mult = residual.shape[-2]

    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = (
        residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    )

    hc_scale = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ],
    )
    mixes = mixes * hc_scale + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)

    res_mix = sinkhorn_normalize_ref(
        res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps
    )

    layer_input = (residual * pre_mix).sum(-2).bfloat16()

    return post_mix, res_mix, layer_input


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """mHC post reference kernel from tilelang repo."""
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def hc_head_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    residual_flat = residual.flatten(-2).float()
    residual_norm = residual_flat * torch.rsqrt(
        residual_flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre_mix = torch.nn.functional.linear(residual_norm, fn)
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    return torch.sum(pre_mix.unsqueeze(-1) * residual.float(), dim=-2).bfloat16()


# ---- 本例参数(小而具体, 读者可手查) ----
T, HC, H = 1, 2, 2
MIX_HC = (2 + HC) * HC  # = 8, model.py:L853
HC_DIM = HC * H  # = 4, model.py:L854/L1017
RMS_EPS = HC_EPS = 1e-6  # 测试取值
SINKHORN_REPEAT = 20  # 测试取值(tests/kernels/test_mhc_kernels.py:L115)
HC_POST_ALPHA = 2.0  # 模型硬编码真值(model.py:L852); 测试里用 1.0

x0 = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)  # embed 出口, 2D (T,H)
fn_attn = torch.tensor(
    [[0.10, 0.20, 0.30, 0.40], [0.50, 0.10, 0.20, 0.30],
     [0.30, 0.40, 0.50, 0.10], [0.20, 0.30, 0.40, 0.50],
     [0.40, 0.50, 0.10, 0.20], [0.10, 0.10, 0.20, 0.20],
     [0.30, 0.30, 0.40, 0.40], [0.50, 0.20, 0.30, 0.10]],
    dtype=torch.float32,
)  # (mix_hc=8, hc*H=4); 生产形状 (8,4) 是 (2+hc)*hc 行 × hc*H 列
fn_ffn = torch.tensor(
    [[0.20, 0.10, 0.40, 0.30], [0.30, 0.50, 0.10, 0.20],
     [0.10, 0.20, 0.30, 0.40], [0.40, 0.10, 0.50, 0.20],
     [0.20, 0.30, 0.10, 0.50], [0.50, 0.40, 0.20, 0.10],
     [0.10, 0.50, 0.40, 0.30], [0.30, 0.20, 0.50, 0.40]],
    dtype=torch.float32,
)
hc_scale = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)  # 形状 (3,), model.py:L884-L897
hc_base = torch.tensor([0.1, -0.1, 0.2, 0.0, -0.2, 0.05, -0.05, 0.15], dtype=torch.float32)
hc_head_fn = torch.tensor([[0.2, -0.1, 0.3, 0.05], [-0.15, 0.25, 0.1, 0.2]], dtype=torch.float32)  # (hc, hc*H)
hc_head_scale = torch.tensor([0.8], dtype=torch.float32)  # (1,)
hc_head_base = torch.tensor([0.1, -0.1], dtype=torch.float32)  # (hc,)
x_attn = torch.tensor([[0.6, -0.3]], dtype=torch.bfloat16)  # 注意力半层输出(占位)
x_ffn = torch.tensor([[0.4, 0.9]], dtype=torch.bfloat16)  # FFN(MoE)半层输出(占位)


def r(t, nd=4):
    if torch.is_tensor(t):
        return [r(v, nd) for v in t] if t.dim() else round(float(t), nd)
    return round(float(t), nd)


def shape(t):
    return list(t.shape)


out = {
    "env": "host Miniconda python 3.11.11, torch 2.11.0+cu128 (纯 CPU), pin=vLLM v0.27.1 (6e448d0ea)",
    "refs_copied_verbatim": "tests/kernels/test_mhc_kernels.py:L14-L90 (sinkhorn_normalize_ref/mhc_pre_ref/mhc_post_ref/hc_head_ref)",
    "params": {
        "T": T, "hc_mult": HC, "H": H, "mix_hc": MIX_HC, "hc_dim": HC_DIM,
        "rms_eps": RMS_EPS, "hc_eps": HC_EPS, "sinkhorn_repeat": SINKHORN_REPEAT,
        "hc_post_alpha(model.py:L852 真值, 测试用 1.0)": HC_POST_ALPHA,
        "hc_mult_real_test_baseline": 4,
        "hc_mult_real_test_baseline_provenance": "tests/kernels/test_mhc_kernels.py:L99/L143/L187 (parametrize hc_mult=[4]); 真实模型读 config.hc_mult",
    },
    "param_shapes": {
        "hc_attn_fn / hc_ffn_fn": [MIX_HC, HC_DIM],
        "hc_attn_base / hc_ffn_base": [MIX_HC],
        "hc_attn_scale / hc_ffn_scale": [3],
        "hc_head_fn": [HC, HC_DIM],
        "hc_head_base": [HC],
        "hc_head_scale": [1],
        "_mtp_hidden_buffer": ["max_num_batched_tokens", HC_DIM],
    },
    "inputs": {
        "x0_embed_2D": r(x0.float()),
        "fn_attn": r(fn_attn), "fn_ffn": r(fn_ffn),
        "hc_scale": r(hc_scale), "hc_base": r(hc_base),
        "hc_head_fn": r(hc_head_fn), "hc_head_scale": r(hc_head_scale),
        "hc_head_base": r(hc_head_base),
        "x_attn_placeholder": r(x_attn.float()),
        "x_ffn_placeholder": r(x_ffn.float()),
    },
    "journey": [],
    "sinkhorn_convergence": {},
    "broadcast_prefold_check": {},
}

# ① 首层 broadcast: 2D → 各流相同的多流, 再走 mhc_pre (DecoderLayer.forward L911-L929 的
#    mhc_pre_broadcast_tilelang 在核内做同一件事; 预折等价性见 ③)
residual0 = x0.unsqueeze(1).expand(T, HC, H).contiguous()  # 两流相同
post_mix1, res_mix1, x1 = mhc_pre_ref(
    residual0, fn_attn, hc_scale, hc_base, RMS_EPS, HC_EPS, HC_EPS,
    HC_POST_ALPHA, SINKHORN_REPEAT,
)
out["journey"].append({
    "step": 1, "what": "首层 mhc_pre(broadcast): 2D x0 按流展开后过 pre 门控",
    "shapes": {"x_in": shape(x0), "residual": shape(residual0),
               "post_mix": shape(post_mix1), "res_mix": shape(res_mix1),
               "layer_input": shape(x1)},
    "values": {"residual(两流相同)": r(residual0.float()),
               "post_mix": r(post_mix1), "res_mix": r(res_mix1),
               "layer_input(进注意力)": r(x1.float())},
})

# ② 注意力半层输出(占位值; 生产 = forward_mqa+_o_proj 的输出)
out["journey"].append({
    "step": 2, "what": "注意力半层(占位输出, 本体归 ch25/26)",
    "shapes": {"x_attn": shape(x_attn)},
    "values": {"x_attn": r(x_attn.float())},
})

# ③ ffn 半的融合核 = 上一个 post + 下一个 pre (mhc_fused_post_pre_tilelang,
#    test_mhc_fused_post_pre 的 run_ref 同序: 先 mhc_post_ref 再 mhc_pre_ref)
residual2 = mhc_post_ref(x_attn, residual0, post_mix1, res_mix1)
post_mix2, res_mix2, x2 = mhc_pre_ref(
    residual2, fn_ffn, hc_scale.clone(), hc_base, RMS_EPS, HC_EPS, HC_EPS,
    HC_POST_ALPHA, SINKHORN_REPEAT,
)
out["journey"].append({
    "step": 3, "what": "融合核 post+pre(ffn 半): mhc_post(x_attn) 再 mhc_pre(fn_ffn)",
    "shapes": {"residual_after_post": shape(residual2), "post_mix": shape(post_mix2),
               "res_mix": shape(res_mix2), "layer_input(进 FFN)": shape(x2)},
    "values": {"residual_after_post": r(residual2.float()), "post_mix": r(post_mix2),
               "res_mix": r(res_mix2), "layer_input(进 FFN)": r(x2.float())},
})

# ④ FFN(MoE) 半层输出(占位值)
out["journey"].append({
    "step": 4, "what": "FFN/MoE 半层(占位输出)",
    "shapes": {"x_ffn": shape(x_ffn)},
    "values": {"x_ffn": r(x_ffn.float())},
})

# ⑤ 层尾 mhc_post 塌回 3D → flatten 进 _mtp_hidden_buffer → hc_head 压回 2D
residual_out = mhc_post_ref(x_ffn, residual2, post_mix2, res_mix2)
flat = residual_out.flatten(1)  # _mtp_hidden_buffer.copy_(hidden.flatten(1)), model.py:L1200-L1202
hidden_final = hc_head_ref(residual_out, hc_head_fn, hc_head_scale, hc_head_base,
                           RMS_EPS, HC_EPS)
out["journey"].append({
    "step": 5, "what": "层尾 mhc_post: 塌回 (T,hc_mult,H) 多流残差",
    "shapes": {"residual_out": shape(residual_out)},
    "values": {"residual_out(pre-hc_head 残差)": r(residual_out.float())},
})
out["journey"].append({
    "step": 6, "what": "_mtp_hidden_buffer.copy_(flatten(1)): 先于 hc_head 暂存",
    "shapes": {"mtp_buffer_slice": shape(flat)},
    "values": {"flat": r(flat.float())},
})
out["journey"].append({
    "step": 7, "what": "hc_head: (T,hc_mult,H) 加权压回单流 (T,H)",
    "shapes": {"hidden_out": shape(hidden_final)},
    "values": {"hidden_out(进 norm→compute_logits)": r(hidden_final.float())},
})

# Sinkhorn 收敛演示: 不同 repeat 下 res_mix 行/列和偏离 1 的最大值
for rep in [1, 2, 3, 5, 20]:
    _, rm, _ = mhc_pre_ref(
        residual0, fn_attn, hc_scale.clone(), hc_base, RMS_EPS, HC_EPS, HC_EPS,
        HC_POST_ALPHA, rep,
    )
    col_dev = float((rm.sum(-2) - 1).abs().max())
    row_dev = float((rm.sum(-1) - 1).abs().max())
    out["sinkhorn_convergence"][f"repeat={rep}"] = {
        "res_mix": r(rm), "max|col_sum-1|": round(col_dev, 6),
        "max|row_sum-1|": round(row_dev, 6),
    }

# broadcast 可预折验证: fn_broadcast = fn.view(mix_hc, hc, H).sum(1) (model.py:L1371-L1380)
fn_broadcast = fn_attn.view(MIX_HC, HC, H).sum(1)  # (mix_hc, H)
x0_flat = x0.float()
sqrsum = (x0_flat.square().sum(-1)).unsqueeze(-1)
mixes_a = x0_flat @ fn_broadcast.T * (sqrsum / HC_DIM + RMS_EPS).rsqrt()
residual_flat = residual0.flatten(-2, -1).float()
mixes_b = residual_flat @ fn_attn.T * (sqrsum / fn_attn.shape[-1] + RMS_EPS).rsqrt()
out["broadcast_prefold_check"] = {
    "claim": "2D 输入各流相同 ⇒ 预折 fn对流维求和 后直接从 2D 算 mixes ≡ 先展开多流再乘 fn",
    "fn_broadcast_shape": shape(fn_broadcast),
    "max_abs_diff": round(float((mixes_a - mixes_b).abs().max()), 8),
    "allclose": bool(torch.allclose(mixes_a, mixes_b, atol=1e-6)),
}

dst = Path(__file__).parent / "ch28_m04_mhc_shape_math.json"
with open(dst, "w", newline="\n", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("wrote", dst)
print(json.dumps(out["broadcast_prefold_check"], ensure_ascii=True))
print(json.dumps(out["sinkhorn_convergence"], ensure_ascii=True))
