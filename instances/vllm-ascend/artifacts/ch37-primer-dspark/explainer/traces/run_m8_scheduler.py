#!/usr/bin/env python3
"""m8 — 硬件感知动态调度 Algorithm 1 的教学玩具轨迹（论文机制；本 PR 快照未实现）。

复现 paper §5：按累计前缀存活概率贪心录取，每步用 Theta = tau * SPS(B) 评估吞吐，
吞吐不再提升即早停（因果闸门）。c_k / SPS 为便于心算的玩具值，非厂方实测；
生产数字（V4-Flash +51% 等）见论文表，未独立复现。
"""

# 逐位置置信度 c_k（块内锚点往后，玩具值）——论文侧由 confidence_head 给出，本 PR 未接入
c = [0.9, 0.8, 0.5, 0.4]   # c_1..c_4

# SPS(k)：草稿长度 k 下 profile 好的每秒步数（越长每步越重 -> 步率越低，玩具查表）
SPS = {1: 100.0, 2: 90.0, 3: 75.0, 4: 55.0}


def prefix_survival(c, k):
    p = 1.0
    for i in range(k):
        p *= c[i]
    return p


admitted = 0
prev_theta = -1.0
print("k | c_k | P_k=prod c_i | tau(k)=sum P | SPS(k) | Theta=tau*SPS | decision")
tau = 0.0
for k in range(1, len(c) + 1):
    Pk = prefix_survival(c, k)          # 累计存活概率 ∏_{i<=k} c_i
    tau = tau + Pk                       # 期望被接受 token 数 = 前缀存活概率之和
    theta = tau * SPS[k]                 # 吞吐目标 Theta = tau * SPS(B)
    if theta > prev_theta:
        decision = "admit (Theta up)"
        admitted = k
        prev_theta = theta
    else:
        decision = "STOP (Theta down) -> early-stop"
        print(f"{k} | {c[k-1]} | {Pk:.4g} | {tau:.4g} | {SPS[k]} | {theta:.4g} | {decision}")
        break
    print(f"{k} | {c[k-1]} | {Pk:.4g} | {tau:.4g} | {SPS[k]} | {theta:.4g} | {decision}")

print(f"admitted = {admitted} draft tokens (peak Theta={prev_theta:.4g})")
# 对照：盲目取满 N=4 的吞吐
tau4 = sum(prefix_survival(c, k) for k in range(1, 5))
theta4 = tau4 * SPS[4]
print(f"fixed N=4: tau={tau4:.4g} Theta={theta4:.4g}  "
      f"dynamic/fixed = {prev_theta/theta4:.4g}x")
theta1 = prefix_survival(c, 1) * SPS[1]
print(f"conservative N=1: Theta={theta1:.4g}  dynamic/N1 = {prev_theta/theta1:.4g}x")
