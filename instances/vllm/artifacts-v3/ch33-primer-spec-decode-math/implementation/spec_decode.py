# -*- coding: utf-8 -*-
"""投机解码验证数学 —— arXiv:2607.05147 §1/§2.1(+§3.2.1 Eq.(8) 的 TV 恒等式)。

论文 §2.1 的验证规则:位 k 上 token x_k 以 min(1, p_t(x_k)/p_d(x_k)) 被接受;
验证左到右,首个拒绝位丢弃其后所有 token;被拒位从残差分布 norm(max(0, p_t−p_d))
恢复一个 token;全接受追加一个 bonus token(只从 target 分布采)。§1 断言该规则
精确保持 target 分布(无损)——本文件把断言做成可运行检验:单位置经验分布、
整流生成流的 bigram 条件分布,都对拍 target。

与 vLLM 落地的对读锚(写作侧素材,不在本文件重复实现):
- 概率空间判据(无 min):vllm/v1/sample/rejection_sampler.py:L805-L837
  (L829 `draft_prob > 0 and target_prob / draft_prob >= uniform_prob`,u∈[0,1));
- 对数空间形态:vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:L622-L625
  (`log_p(x) > log(u) + log_q(x)`);
- 残差 Gumbel-max 免归一化:rejection_sampler.py:L913-L952(score = prob·inv_q 取
  argmax,注释明说不需要先归一化)、对数稳定式 log r = a+log(max(1−exp(b−a),0)):
  rejection_sampler_utils.py:L748-L759;
- 术语表 accepted/recovered/bonus:rejection_sampler.py:L38-L59(实现严格遵循
  arXiv:2211.17192)。

约定:分布一律行向量(numpy float64);rng 为 np.random.default_rng(...)。
"""
import numpy as np


# PAPER: §3.1 Eq.(4)(重构口径)—— softmax 算子(数值稳定版:减行最大值;Eq.(4)
# 在论文包被 LaTeXML 截断,softmax 形式由 Eq.(5)+vLLM 落地双向重构,见 dspark.py)
def softmax_lastdim(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


# PAPER: §2.1 —— 接受概率 min(1, p_t(x)/p_d(x))(论文原式)
def acceptance_probability(p_t, p_d, x):
    return min(1.0, float(p_t[x]) / float(p_d[x]))


# PAPER: §2.1 —— 判据的 u 形态:u < p_t(x)/p_d(x);u∈[0,1) ⇒ min(1,·) 隐式
#   (vLLM V1 kernel L829 的形态:target_prob/draft_prob >= uniform_prob)
def accepted(x, p_t, p_d, u):
    return bool(p_d[x] > 0.0 and u * p_d[x] < p_t[x])


# PAPER: §2.1 —— 同一判据的对数空间形态 log p(x) > log u + log q(x)
#   (vLLM V2 kernel L622-L625 的姿态:免 softmax 物化、免下溢)
def accepted_logspace(x, log_p_t, log_p_d, u):
    return bool(log_p_t[x] > np.log(u) + log_p_d[x])


# PAPER: §2.1 —— 残差分布(未归一化)r(x) = max(0, p_t(x) − p_d(x));
#   拒绝总质量 ½‖p_t−p_d‖₁ = Σ_x r(x)(无损证明的两半恰好对账)
def residual_distribution(p_t, p_d):
    return np.maximum(np.asarray(p_t, dtype=float) - np.asarray(p_d, dtype=float), 0.0)


# PAPER: §2.1 —— Gumbel-max 免归一化采样:argmax_x prob(x)/E_x,E_x~Exp(1) 独立
#   (≡ 从 norm(prob) 抽样;argmax 对正缩放不变 ⇒ 归一化因子 Z 不用算。
#   vLLM V1 kernel L944 score = prob·inv_q 取 argmax 与此同构)
def gumbel_max_argmax(weights, exp_draws):
    return int(np.argmax(np.asarray(weights, dtype=float) / exp_draws))


# PAPER: §2.1 —— 被拒位从残差恢复一个 token:norm(max(0, p_t−p_d)) 的 Gumbel-max 采样
def sample_recovered(p_t, p_d, rng):
    residual = residual_distribution(p_t, p_d)
    total = float(residual.sum())
    assert total > 0.0, "残差质量为 0 ⇒ p_t=p_d ⇒ 恒接受,不可能走到恢复"
    exp_draws = rng.exponential(1.0, size=residual.shape[0])
    return gumbel_max_argmax(residual, exp_draws)


# PAPER: §3.2.1 Eq.(8) —— 逐位解析接受率的 Σmin 形态:Σ_x min(p_t(x), p_d(x))
#   (与 1−½‖p−q‖₁ 恒等:TV 恒等式,两实现互为对拍;Eq.(8) 的 c* 与 §2.1 的
#   接受概率期望是同一条式子——训练监督与验证准则共用)
def acceptance_rate_sum_min(p_t, p_d):
    return float(np.minimum(p_t, p_d).sum())


# PAPER: §3.2.1 Eq.(8) —— c* = 1 − ½‖p_d − p_t‖₁(总变差距离 TV 形态)
def analytical_acceptance_rate(p_t, p_d):
    return 1.0 - 0.5 * float(np.abs(np.asarray(p_t) - np.asarray(p_d)).sum())


# PAPER: §2.1 —— 一次验证周期:draft γ 个 → 左到右 min(1,p_t/p_d) 逐位接受;
#   首拒位从残差恢复并截断;全接受追加 bonus(只从 target 分布采,γ+1 行)。
#   术语 accepted/recovered/bonus = vLLM rejection_sampler.py:L38-L59
def verify_block(draft_tokens, draft_probs, target_probs, rng):
    draft_tokens = [int(t) for t in draft_tokens]
    draft_probs = np.asarray(draft_probs, dtype=float)
    target_probs = np.asarray(target_probs, dtype=float)
    gamma = len(draft_tokens)
    assert target_probs.shape[0] == gamma + 1, "target 需给 γ+1 行(γ 个验证位 + 1 个 bonus 位)"
    emitted = []
    info = {"steps": [], "n_accepted": 0, "rejected_at": None, "recovered": None, "bonus": None}
    for k in range(gamma):
        x = draft_tokens[k]
        q = draft_probs[k]
        assert q[x] > 0.0, "draft 概率为 0 的 token 不可能被采出"
        u = float(rng.random())  # float64(vLLM 口径:规避 u=0.0 的边界事故)
        ratio = float(target_probs[k][x]) / float(q[x])
        ok = accepted(x, target_probs[k], q, u)
        info["steps"].append(
            {"k": k, "token": x, "u": u, "ratio": ratio,
             "accept_prob": min(1.0, ratio), "accepted": ok}
        )
        if ok:
            emitted.append(x)
            info["n_accepted"] += 1
        else:
            rec = sample_recovered(target_probs[k], q, rng)
            emitted.append(rec)
            info["rejected_at"] = k
            info["recovered"] = rec
            return emitted, info["n_accepted"], info
    bonus = int(rng.choice(target_probs.shape[1], p=target_probs[gamma]))
    emitted.append(bonus)
    info["bonus"] = bonus
    return emitted, gamma, info


# PAPER: §2.1 —— greedy draft(one-hot q)退化:接受 ⟺ draft == target argmax;
#   拒绝位/bonus 直接取 target argmax(残差=挖掉 draft 位的 target 分布,被拒时
#   draft≠argmax ⇒ 残差 argmax 仍是 target argmax)⇒ 输出 ≡ target greedy rollout。
#   vLLM V2 greedy 分支 L564-L586/V1 L756-L757 的参考实现形态
# PAPER: §2.1 —— greedy draft(one-hot q)退化的验证
def verify_block_greedy(draft_tokens, target_probs):
    draft_tokens = [int(t) for t in draft_tokens]
    target_probs = np.asarray(target_probs, dtype=float)
    gamma = len(draft_tokens)
    emitted = []
    info = {"steps": [], "n_accepted": 0, "rejected_at": None, "recovered": None, "bonus": None}
    for k in range(gamma):
        t_star = int(np.argmax(target_probs[k]))
        ok = draft_tokens[k] == t_star
        info["steps"].append({"k": k, "token": draft_tokens[k], "target_argmax": t_star, "accepted": ok})
        emitted.append(t_star)
        if ok:
            info["n_accepted"] += 1
        else:
            info["rejected_at"] = k
            info["recovered"] = t_star
            return emitted, info["n_accepted"], info
    bonus = int(np.argmax(target_probs[gamma]))
    emitted.append(bonus)
    info["bonus"] = bonus
    return emitted, gamma, info


# PAPER: §1 —— 无损(单位置):draft x~p_d、按准则接受/从残差恢复,
#   经验输出分布 == p_t("acceptance rule preserves the target distribution exactly")
def empirical_output_distribution(p_t, p_d, n, rng):
    p_t = np.asarray(p_t, dtype=float)
    p_d = np.asarray(p_d, dtype=float)
    counts = np.zeros(p_t.shape[0])
    for _ in range(n):
        x = int(rng.choice(p_d.shape[0], p=p_d))
        u = float(rng.random())
        if accepted(x, p_t, p_d, u):
            y = x
        else:
            y = sample_recovered(p_t, p_d, rng)
        counts[y] += 1
    return counts / n


# PAPER: §2.1 —— 投机周期循环(draft γ 个 → target 一次前向的所有条件分布逐位验证
#   → 发射、以最后发射 token 为下一周期 anchor),生成 n_tokens 流。
#   target_cond_fn(prefix_last) → 下一 token 分布(单位置条件化的玩具 target);
#   drafter_fn(prev) → (draft_tokens, draft_probs 行;greedy 模式 probs 可为 None)。
#   经验 τ = 每周期平均发射 token 数(含 bonus 口径,fn.4)
# PAPER: §2.1 —— 投机周期循环(draft→验证→发射)
def generate_stream(target_cond_fn, drafter_fn, anchor, n_tokens, rng, greedy=False):
    stream = []
    cycles = 0
    total_emitted = 0
    total_accepted = 0
    prev = int(anchor)
    while len(stream) < n_tokens:
        draft_tokens, draft_probs = drafter_fn(prev)
        rows = []
        walk = prev
        for k in range(len(draft_tokens)):
            rows.append(np.asarray(target_cond_fn(walk), dtype=float))
            walk = int(draft_tokens[k])
        rows.append(np.asarray(target_cond_fn(walk), dtype=float))  # bonus 行
        target_probs = np.asarray(rows)
        if greedy:
            emitted, n_acc, _ = verify_block_greedy(draft_tokens, target_probs)
        else:
            emitted, n_acc, _ = verify_block(draft_tokens, draft_probs, target_probs, rng)
        stream.extend(emitted)
        total_emitted += len(emitted)
        total_accepted += n_acc
        cycles += 1
        prev = emitted[-1]
    stats = {
        "cycles": cycles,
        "mean_emitted_per_cycle": total_emitted / cycles,
        "mean_accepted_per_cycle": total_accepted / cycles,
    }
    return stream[:n_tokens], stats


# PAPER: §2.1 Eq.(1) —— 每生成 token 平均延迟 L = (T_draft + T_verify)/τ
def latency_per_token(t_draft, t_verify, tau):
    return (t_draft + t_verify) / tau


# PAPER: §2.1 Eq.(1) —— 对无投机基线的加速比(基线每 token 一次 target 前向,
#   延迟 = T_verify;Eq.(1) 的直接推论,§1 的加速账)
def speedup_vs_plain(t_draft, t_verify, tau):
    return t_verify / latency_per_token(t_draft, t_verify, tau)
