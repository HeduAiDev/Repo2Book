# -*- coding: utf-8 -*-
"""期望接受长度 τ 与前缀存活链式法则 —— arXiv:2607.05147 §2.1/§4.2 Table 1
(脚注 4:报告口径含 bonus token)/§3.2.2(a_{r,j}=∏c_{r,i})。

τ 是 Eq.(1) 的分母:加速=三杠杆(draft 更快 / draft 更准 / 验证更聪明)里的
「更准」。逐位条件接受率 α_k(=TV 恒等式 1−½‖p_d−p_t‖₁ 逐位的值)下,
位 k 的存活概率是累积乘积 a_k=∏_{i≤k}α_i;每周期期望发射
E[τ]=1+Σ_k a_k(1 = 恢复 token 或 bonus——首位账随口径展开,fn.4 含 bonus);
恒 α 时闭式 (1−α^{γ+1})/(1−α)。

条件率⇄无条件率互推与 vLLM 同构:vllm/v1/spec_decode/utils.py:L598-L601
unconditional_to_conditional_rates(c_i = p_i/p_{i−1},p_0=1;synthetic 接受率
模式的测试注入旋钮)。
"""
import numpy as np


# PAPER: §3.2.2 (Algorithm 1 line 2) —— a_{r,j} = ∏_{i≤j} c_{r,i}(条件置信→前缀存活,
#   链式法则正向;单调不增 ⇒ 全局按 a 排序天然尊重块内前缀依赖)
def prefix_survival_probs(confidences):
    return np.cumprod(np.asarray(confidences, dtype=float))


# PAPER: §2.1 + §4.2 脚注 4 —— E[τ] = 1 + Σ_{k=1..γ} ∏_{i≤k} α_i(产出含
#   恢复/bonus token 的口径;恒 α:1+Σα^k = (1−α^{γ+1})/(1−α))
def expected_accepted_length(cond_rates):
    a = prefix_survival_probs(cond_rates)
    return 1.0 + float(a.sum())


# PAPER: §2.1/§4.2 —— 恒 α 闭式 (1−α^{γ+1})/(1−α)(α=0.8、γ=3 → 2.952;
#   α→1 极限 = γ+1,每周期全收+bonus)
def expected_accepted_length_constant(alpha, gamma):
    if alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


# PAPER: §3.2.2 —— 条件率 → 无条件率:a = cumprod(c)(链式法则正向,
#   与 prefix_survival_probs 同一运算;独立成函数供对读)
def conditional_to_unconditional_rates(cond_rates):
    return prefix_survival_probs(cond_rates)


# PAPER: §3.2.2 —— 无条件率 → 条件率:c_i = p_i/p_{i−1}(p_0=1;链式法则逆向;
#   vLLM unconditional_to_conditional_rates 同构,前位为 0 时该位置 0)
def unconditional_to_conditional_rates(rates):
    out = []
    prev = 1.0
    for p in rates:
        out.append(p / prev if prev > 0.0 else 0.0)
        prev = p
    return out


# PAPER: §2.1 —— 抽象接受过程蒙特卡洛:位 k 以 α_k 接受、首拒即停,
#   每周期发射 = 接受数 + 1(拒绝位恢复 token 或全收 bonus)——E[τ] 公式的经验对拍
def simulate_accepted_length(cond_rates, n_cycles, rng):
    total = 0
    for _ in range(n_cycles):
        acc = 0
        for a in cond_rates:
            if rng.random() < a:
                acc += 1
            else:
                break
        total += acc + 1
    return total / n_cycles
