# -*- coding: utf-8 -*-
"""训练目标三件套 —— arXiv:2607.05147 §3.3 Eq.(9)-(12)。

L_ce 交叉熵(Eq.9,预测正确下一 token)+ L_tv 分布匹配(Eq.10,罚 draft 与
target 的 L1 距离——TV 距离=接受率代理:逐位接受概率 = 1−½‖p^d−p^t‖₁,
最小化 L_tv 即直接最大化期望接受率)+ L_conf 二元交叉熵(Eq.11,置信头学
解析标签 c*)。三者按位置权重 w_k=exp(−(k−1)/γ) 加权(前位贡献期望接受长度
更多);总目标 Eq.(12) 默认 α=(0.1, 0.9, 1.0)——几乎全押分布对齐。

训练不在 vLLM 推理路径内(开源在 DeepSpec 训练仓);本文件是论文侧目标函数
的忠实参考实现 + 一个有限差分梯度下降的示教轨迹(「L_tv ↓ ⇒ 接受率 ↑」的
可运行版;优化器论文未指定,有限差分是对目标函数本身最少的额外假设)。
"""
import numpy as np

from spec_decode import softmax_lastdim, analytical_acceptance_rate
from acceptance_length import expected_accepted_length


# PAPER: §3.3 —— 位置权重 w_k = exp(−(k−1)/γ)(k=1..γ;w_1=1、单调不增)
def position_weights(gamma):
    k = np.arange(1, gamma + 1)
    return np.exp(-(k - 1) / gamma)


# PAPER: §3.3 Eq.(9) —— L_ce = −Σ_k w_k·log p_k^d(x_k*)(x*=ground-truth token);
#   输入 [N,γ,V]/[N,γ],对 N 平均、对 k 加权求和
def cross_entropy_loss(draft_probs, target_tokens):
    draft_probs = np.asarray(draft_probs, dtype=float)
    target_tokens = np.asarray(target_tokens, dtype=int)
    _, gamma, _ = draft_probs.shape
    w = position_weights(gamma)
    p_star = np.take_along_axis(draft_probs, target_tokens[..., None], axis=-1)[..., 0]
    return float(np.mean(-np.sum(w * np.log(p_star), axis=1)))


# PAPER: §3.3 Eq.(10) —— L_tv = Σ_k w_k·‖p_k^d − p_k^t‖₁(最小化 ⇔ 直接
#   最大化期望接受率:逐位接受概率 = 1−½‖p^d−p^t‖₁)
def tv_loss(draft_probs, target_probs):
    draft_probs = np.asarray(draft_probs, dtype=float)
    target_probs = np.asarray(target_probs, dtype=float)
    _, gamma, _ = draft_probs.shape
    w = position_weights(gamma)
    l1 = np.abs(draft_probs - target_probs).sum(axis=-1)  # [N,γ]
    return float(np.mean(np.sum(w * l1, axis=1)))


# PAPER: §3.3 Eq.(11) —— L_conf = −Σ_k w_k·[c*·log c_k + (1−c*)·log(1−c_k)]
#   (BCE,置信头学 Eq.(8) 的解析接受率标签 c*)
def confidence_loss(c, c_star):
    c = np.clip(np.asarray(c, dtype=float), 1e-12, 1.0 - 1e-12)
    c_star = np.asarray(c_star, dtype=float)
    # 逐位 BCE:c 与 c* 必须同形状(防回归:c*(γ,1) 会与 c(γ) 广播成外积、
    # 静默把 Eq.(11) 改写成全位置标签牵引的畸变目标——见 test_toy_trajectory_loss_follows_eq12)
    assert c.shape == c_star.shape, (c.shape, c_star.shape)
    gamma = c.shape[-1]
    w = position_weights(gamma)
    bce = -(c_star * np.log(c) + (1.0 - c_star) * np.log(1.0 - c))
    return float(np.mean(np.sum(w * bce, axis=1)))


# PAPER: §3.3 Eq.(12) —— L = α_ce·L_ce + α_tv·L_tv + α_conf·L_conf
#   (默认权重 α_ce=0.1、α_tv=0.9、α_conf=1.0)
def total_loss(draft_probs, target_tokens, target_probs, c, c_star,
               alpha_ce=0.1, alpha_tv=0.9, alpha_conf=1.0):
    return (
        alpha_ce * cross_entropy_loss(draft_probs, target_tokens)
        + alpha_tv * tv_loss(draft_probs, target_probs)
        + alpha_conf * confidence_loss(c, c_star)
    )


# PAPER: §3.3 —— 单步损失(给定参数与采到的 ground-truth 块):draft logits z
#   → p=softmax(z);置信 logit zc → c=σ(zc);c* 由当前 p 与冻结 target 解析算出
def _loss_of(z, zc, target_probs, target_tokens, alphas):
    gamma = target_probs.shape[0]
    p = softmax_lastdim(z)
    c = 1.0 / (1.0 + np.exp(-zc))
    c_star = np.array([analytical_acceptance_rate(target_probs[k], p[k]) for k in range(gamma)])  # 扁平 (γ,)
    return total_loss(
        p[None], target_tokens[None], target_probs[None], c[None], c_star[None], *alphas
    )


# PAPER: §3.3 —— 有限差分梯度下降示教:目标分布冻结(target 全程冻结、只有
#   drafter/置信头可学),每步从 target 采一个 γ-token 块作 L_ce 的 x*;
#   轨迹逐项记录 loss/L_tv/逐位接受率/E[τ]——「最小化 L_tv 直接最大化期望
#   接受率」(Eq.(10) 下文断言)的可运行版。优化器论文未指定(训练细节在
#   DeepSpec);有限差分是对目标函数最少的额外假设
# PAPER: §3.3 —— 有限差分梯度下降示教轨迹
def train_toy_drafter(target_probs, n_steps=300, lr=0.5, rng=None, alphas=(0.1, 0.9, 1.0)):
    target_probs = np.asarray(target_probs, dtype=float)  # [γ,V] 冻结
    gamma, vocab = target_probs.shape
    if rng is None:
        rng = np.random.default_rng(0)
    z = rng.normal(0.0, 1.0, size=(gamma, vocab))  # draft logits(可学)
    zc = rng.normal(0.0, 1.0, size=gamma)  # 置信 logit(可学)
    eps = 1e-5
    trace = []
    for step in range(n_steps):
        p = softmax_lastdim(z)
        target_tokens = np.array([int(rng.choice(vocab, p=target_probs[k])) for k in range(gamma)])
        loss = _loss_of(z, zc, target_probs, target_tokens, alphas)
        gz = np.zeros_like(z)
        for idx in np.ndindex(z.shape):
            zp = z.copy()
            zp[idx] += eps
            gz[idx] = (_loss_of(zp, zc, target_probs, target_tokens, alphas) - loss) / eps
        gzc = np.zeros_like(zc)
        for i in range(gamma):
            zcp = zc.copy()
            zcp[i] += eps
            gzc[i] = (_loss_of(z, zcp, target_probs, target_tokens, alphas) - loss) / eps
        z = z - lr * gz
        zc = zc - lr * gzc
        rates = np.array([analytical_acceptance_rate(target_probs[k], p[k]) for k in range(gamma)])
        trace.append({
            "step": step,
            "loss": loss,
            "tv": tv_loss(p[None], target_probs[None]),
            "accept_rates": rates.tolist(),
            "expected_tau": expected_accepted_length(rates),
        })
    return trace
