#!/usr/bin/env python3
"""ch20 weight-absorption worked example — 纯 PyTorch/host 可跑（与昇腾无关）。

复现两件事:
  (A) 形状代数: kv_lora_rank(L)=4, num_heads(N)=2, qk_nope_head_dim(P)=3, v_head_dim(V)=3 的小尺寸下,
      process_weights_after_loading 拆出的 W_UK_T=(N,P,L)、W_UV=(N,L,V),
      _q_proj_and_k_up_proj 的 torch.bmm 把 q_nope(B,N,P) 吸收成 ql_nope(B,N,L)。
  (B) 单头数值等价: 结合律 (q·W_UK^T)·kv_c == q·(W_UK^T·kv_c) —— 两端各算一遍, 逐位相等。
输出即 explainer 逐轮表的数字来源(数字不许编: lint_explainer 逐个核)。
"""
import json
import torch

torch.set_printoptions(precision=1)

# ---------- (A) 形状代数(与章节 §20.4 形状表同尺寸的小样例) ----------
B, N, P, L, V = 1, 2, 3, 4, 3   # batch tok, heads, qk_nope_head_dim, kv_lora_rank, v_head_dim

# kv_b_proj.weight 在加载期: (N*(P+V), L) -> .T -> (L, N*(P+V)) -> view(L,N,P+V) -> split
kv_b_weight = torch.arange(float(N * (P + V) * L)).reshape(N * (P + V), L)
w = kv_b_weight.T.reshape(L, N, P + V)
W_UK, W_UV = w.split([P, V], dim=-1)          # (L,N,P), (L,N,V)
W_UK_T = W_UK.permute(1, 2, 0).contiguous()   # (N,P,L)
W_UV_r = W_UV.transpose(0, 1).contiguous()    # (N,L,V)

# _q_proj_and_k_up_proj 运行期吸收
q_nope = torch.ones(B, N, P)                   # (B,N,P)
q_nope_t = q_nope.transpose(0, 1)              # (N,B,P)
ql_nope = torch.bmm(q_nope_t, W_UK_T)          # (N,B,L)
ql_nope = ql_nope.transpose(0, 1)              # (B,N,L)

print("=== (A) shape algebra (B,N,P,L,V)=(%d,%d,%d,%d,%d) ===" % (B, N, P, L, V))
print("W_UK_T.shape =", tuple(W_UK_T.shape))   # (2,3,4)
print("W_UV.shape   =", tuple(W_UV_r.shape))   # (2,4,3)
print("q_nope.shape =", tuple(q_nope.shape))   # (1,2,3)
print("ql_nope.shape=", tuple(ql_nope.shape))  # (1,2,4)  已投进 4 维 latent

# ---------- (B) 单头数值等价(章节 §20.4 的 a==b==19.0) ----------
q1 = torch.tensor([1., 2., 3.])                              # (P,)
W_UK_T1 = torch.tensor([[1., 0., 2., 1.],
                        [0., 1., 1., 0.],
                        [2., 1., 0., 1.]])                   # (P,L)
kv_c = torch.tensor([1., 0., 1., 2.])                       # (L,)

k_nope = W_UK_T1 @ kv_c        # 朴素: 先解压 K  -> (P,)
a = (q1 @ k_nope).item()       # query·k_nope   -> 标量

ql1 = q1 @ W_UK_T1             # 吸收: 先投 latent -> (L,)
b = (ql1 @ kv_c).item()        # ql·kv_c        -> 标量

print("\n=== (B) single-head numeric equivalence (P=3,L=4) ===")
print("naive  k_nope = W_UK_T @ kv_c =", k_nope.tolist())   # [5.0,1.0,4.0]
print("naive  a = q . k_nope         =", a)                 # 19.0
print("absorb ql_nope = q @ W_UK_T   =", ql1.tolist())      # [7.0,5.0,4.0,4.0]
print("absorb b = ql_nope . kv_c     =", b)                 # 19.0
print("allclose(a, b) =", bool(abs(a - b) < 1e-6))          # True

# 机读镜像(便于核对)
print("\n=== JSON ===")
print(json.dumps({
    "shape": {"W_UK_T": list(W_UK_T.shape), "W_UV": list(W_UV_r.shape),
              "q_nope": list(q_nope.shape), "ql_nope": list(ql_nope.shape)},
    "k_nope": k_nope.tolist(), "a": a,
    "ql_nope1": ql1.tolist(), "b": b,
}, ensure_ascii=False))
