# RemoveLayoutConversions 四阶段手工走查（trace_source=manual）

无法在 host 上跑 pin dump（host 无 CUDA，make_ttgir 需 cuda:NN target）；本走查按
`RemoveLayoutConversions.cpp:L43-L60`（算法总述）+ `L1126-L1185`（runOnOperation 四步）+
`L208-L230`（setEncoding 对 convert 令 dst=src）逐分支手工推演，convert 计数可由 IR 结构直接数出。

## 输入 IR（一个冗余 convert 往返 — 消到 0 的最干净示例）

accelerate_matmul 之外的 pass（coalesce/plan_cta）常留下这种「转过去又转回来」的 convert：

```
%0 = tt.load %p          : tensor<128x64xf16, #blocked>        // load 锚点 = #blocked
%1 = ttg.convert_layout %0 : ... -> tensor<128x64xf16, #mma>   // convert #1
%2 = ttg.convert_layout %1 : ... -> tensor<128x64xf16, #blocked> // convert #2
%3 = arith.addf %2, %2   : tensor<128x64xf16, #blocked>
tt.store %q, %3          : tensor<128x64xf16, #blocked>        // store 锚点 = #blocked
```

初始 convert_layout 计数 = **2**（%1, %2）。

## 四阶段逐步（convert 计数轨迹）

| 阶段 | 源码 | 动作 | 每值编码集合 | convert 数 |
|------|------|------|--------------|-----------|
| ①initAnchorLayout | L168-L206 | 锚点=load%0→#blocked、函数参数、store 操作数期望#blocked（dot/atomic 本例无） | %0:{blocked} | 2 |
| ②propagateLayout | L208-L230 | 前向定点迭代:遇 convert 令 dst:=src(消 convert 意图)→%1:=#blocked、%2:=#blocked;addf inferDst→%3:=#blocked | %1:{blocked} %2:{blocked} %3:{blocked} | 2 |
| ③resolveConflicts | L311-L332 | 每个值只剩单编码(全 #blocked)→无冲突,不插新 convert | 全 {blocked} | 2 |
| ④rewrite+canonicalize | L666-L717 / L1150 | 按选定编码重写:两个 convert 都成 #blocked→#blocked no-op → ConvertLayoutOp canonicalize 折叠死 convert | — | **0** |

轨迹:2 → 2 → 2 → **0**。消除 2 个 convert_layout。

## 对照:matmul 场景里哪些 convert 活到 OptimizeDotOperands

BlockedToMMA 重写后典型残留 4 个 convert（accm=acc→mma、ad=A→dot_operand、bd=B→dot_operand、
db=result mma→blocked）。四阶段 + backward remat（L971-L1010）对它们的处置：
- **accm（→mma）**：acc 若是 splat 常量 → backward remat 把常量直接以 #mma 造出 → 消除。
- **ad / bd（→dot_operand）**：backward remat/hoist **显式跳过**到 DotOperandEncoding 的 convert
  （L975-L977，给 fused-attention 让路）→ **存活**，交由 OptimizeDotOperands 处理。
- **db（mma→blocked）**：下游是 store（偏 blocked）→ 存活为必要 convert；下游若是链式 dot 则可继续消。

这说明 make_ttgir 里 RemoveLayoutConversions **跑三次**（compiler.py L225/L228/L243）的必要性：
单趟消不净，每次结构变换（accelerate_matmul / pipeline / prefetch）后都要再消一轮。
