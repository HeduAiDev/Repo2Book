# -*- coding: utf-8 -*-
"""置信度调度 —— arXiv:2607.05147 §3.2.1(Post-hoc Calibration:STS)+
§3.2.2(Algorithm 1:硬件感知前缀调度器,Θ=τ·SPS(B))+ Appendix A
(回顾式选择偏差反例:0.85/0.15 ≠ 0.7/0.3)。

vLLM 边界(写作侧诚实账):置信度调度未进推理路径——confidence_head 权重
随 checkpoint 到货但被显式 skip(vllm/model_executor/models/qwen3_dspark.py:
L184-L188);最近的已落地近亲是 V1 调度器的 dynamic_sd_lookup 按批大小查表定 K
(vllm/config/speculative.py:L179-L186 num_speculative_tokens_per_batch_size)。
本文件是论文侧 Algorithm 1/STS 的忠实参考实现。
"""
import numpy as np

from acceptance_length import prefix_survival_probs
from spec_decode import accepted, sample_recovered, acceptance_rate_sum_min


# PAPER: §3.2.2 Algorithm 1 —— 硬件感知前缀调度器:输入每请求置信序列
#   c_{r,1..γ} 与 profiling 一次的吞吐曲线 SPS(B);逐行对应:
#   L1-3 a_{r,j}=∏c_{r,i} → L4 全局按 a 降序 → L5-6 初始化(B=R、τ*=R、
#   Θ_best=R·SPS(R))→ L7-15 贪心准入(B+1、τ*+a_{r,j}、Θ=τ*·SPS(B),
#   Θ 不再超过 Θ_best 即 break=早停/非前瞻)→ L16 返回 ℓ*
# PAPER: §3.2.2 Algorithm 1 —— 硬件感知前缀调度器
def hardware_aware_prefix_scheduler(confidences, sps):
    R = len(confidences)
    gamma = len(confidences[0])
    surv = [prefix_survival_probs(c) for c in confidences]  # lines 1-3
    candidates = sorted(  # line 4:a>0 的 (r,j) 全集,按 a 降序
        (
            (float(surv[r][j]), r, j)
            for r in range(R)
            for j in range(gamma)
            if float(surv[r][j]) > 0.0
        ),
        reverse=True,
    )
    ell = [0] * R
    B = R
    tau_star = float(R)  # line 5
    theta_best = R * float(sps(R))
    ell_star = list(ell)  # line 6
    trace = [{"event": "init", "B": B, "tau": tau_star, "theta": theta_best}]
    for a_val, r, j in candidates:  # line 7(已按 a 降序)
        ell[r] = j + 1
        B += 1
        tau_star += a_val  # line 8
        theta = tau_star * float(sps(B))  # line 9
        trace.append({"event": "admit", "req": r, "pos": j + 1, "a": a_val,
                      "B": B, "tau": tau_star, "theta": theta})
        if theta > theta_best:  # line 10
            theta_best = theta
            ell_star = list(ell)  # line 11
        else:
            trace.append({"event": "break"})  # lines 12-13(早停=非前瞻)
            break
    return {"ell_star": ell_star, "theta_best": theta_best, "trace": trace}


# PAPER: Appendix A —— 无早停的回顾式全局搜索(R=1、γ=2):Θ_ℓ 全评估后取
#   argmax。c_2 依赖已实现的 x_1(Markov 特征)⇒ ℓ 的选择泄露 x_1(选择偏差)。
#   Θ_0=1·SPS(1)、Θ_1=(1+a_1)·SPS(2)、Θ_2=(1+a_1+a_1c_2)·SPS(3)
def retrospective_global_search(a1, c2, sps):
    thetas = [1.0 * float(sps(1)), (1.0 + a1) * float(sps(2)), (1.0 + a1 + a1 * c2) * float(sps(3))]
    return int(np.argmax(thetas)), thetas


# PAPER: Appendix A —— 回顾式调度的输出分布偏差(经验模拟):x_1=A(高 c_2)→
#   ℓ=2(准入、按准则验证)、x_1=B(低 c_2)→ℓ=0(target 重采)⇒
#   P(Y=A)=P(x_1=A)·1+P(x_1=B)·p_t(A)=0.5+0.5×0.7=0.85 ≠ 0.7(不无损)。
#   a_1 由 TV 恒等式从 (p_t,p_d) 解析算出(论文:与假设的 a_1=0.8 一致)
# PAPER: Appendix A —— 回顾式调度的输出分布偏差(经验模拟)
def retrospective_output_distribution(p_t, p_d, c2_high, c2_low, sps, n, rng):
    p_t = np.asarray(p_t, dtype=float)
    p_d = np.asarray(p_d, dtype=float)
    a1 = acceptance_rate_sum_min(p_t, p_d)
    counts = np.zeros(p_t.shape[0])
    for _ in range(n):
        x1 = int(rng.choice(p_d.shape[0], p=p_d))
        c2 = c2_high if x1 == 0 else c2_low
        ell, _ = retrospective_global_search(a1, c2, sps)
        if ell >= 1:
            u = float(rng.random())
            if accepted(x1, p_t, p_d, u):
                y = x1
            else:
                y = sample_recovered(p_t, p_d, rng)
        else:
            y = int(rng.choice(p_t.shape[0], p=p_t))
        counts[y] += 1
    return counts / n


# PAPER: §3.2.1 (Post-hoc Calibration) —— ECE = Σ_b (n_b/N)·|acc_b − conf_b|
#   (Expected Calibration Error;调度要的是累积乘积的绝对幅度,不是排序)
def expected_calibration_error(predictions, outcomes, num_bins=10):
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=bool)
    n = predictions.shape[0]
    edges = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    idx = np.clip(np.digitize(predictions, edges), 0, num_bins - 1)
    ece = 0.0
    for b in range(num_bins):
        mask = idx == b
        if mask.any():
            ece += (mask.sum() / n) * abs(outcomes[mask].mean() - predictions[mask].mean())
    return ece


# PAPER: §3.2.1 —— 温度缩放 c = σ(z/T)(保序变换:T>0 时对 z 单调)
def temperature_scaled_confidence(logits, temperature):
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float) / temperature))


# PAPER: §3.2.1 (STS) —— Sequential Temperature Scaling:链式法则 ⇒ 前缀联合
#   接受概率 = 累积乘积 ∏c_i;从左到右逐位做 1D 网格搜索:位 k 冻结已校准的
#   前缀位,找让累积乘积 ECE 最小的温度标量;保序不扰排序
def sequential_temperature_scaling(conf_logits, prefix_outcomes, grid=None):
    conf_logits = np.asarray(conf_logits, dtype=float)  # [N,γ] σ 前的原始 logit
    prefix_outcomes = np.asarray(prefix_outcomes, dtype=bool)  # [N,γ] 前 k 位全被接受
    n, gamma = conf_logits.shape
    if grid is None:
        grid = np.arange(0.25, 5.01, 0.25)
    temperatures = np.ones(gamma)
    calibrated = np.zeros((n, gamma))
    cum = np.ones(n)
    for k in range(gamma):
        best_T, best_ece = 1.0, np.inf
        for T in grid:
            cand_cum = cum * temperature_scaled_confidence(conf_logits[:, k], T)
            ece = expected_calibration_error(cand_cum, prefix_outcomes[:, k])
            if ece < best_ece:
                best_ece, best_T = ece, float(T)
        temperatures[k] = best_T
        calibrated[:, k] = temperature_scaled_confidence(conf_logits[:, k], best_T)
        cum = cum * calibrated[:, k]
    return {
        "temperatures": temperatures,
        "calibrated": calibrated,
        "cumulative": np.cumprod(calibrated, axis=1),
    }
