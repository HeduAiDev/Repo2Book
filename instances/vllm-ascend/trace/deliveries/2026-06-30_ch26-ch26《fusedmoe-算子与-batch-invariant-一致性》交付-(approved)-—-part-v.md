# ch26《FusedMoE 算子与 batch-invariant 一致性》交付 (APPROVED) — Part VI 收官

- **Type**: delivery
- **Chapter**: 26
- **Date**: 2026-06-30
- **Timestamp**: 2026-06-30T14:46:32Z
- **Agents involved**: archivist, reviewer, writer
- **User present**: False
- **Tags**: ch26, fused-moe, moe-comm-method, token-dispatcher, batch-invariant, f3, f10, delivery, part-vi

## What happened

Part VI 收官章完工，reviewer APPROVED（24 issues 全 negotiable/non-blocking：6 维评审建议 + 18 个 reader-comprehension 名词补释/拆段，含 1 个 lint_formulas 内联密度非阻断项）。讲全书最大单体 OOT 算子 AscendFusedMoE(FusedMoE)(ops/fused_moe/fused_moe.py:L335)继承 vLLM FusedMoE、覆写 forward/forward_impl 走通信-计算二分；moe_comm_method.py 的 _MoECommMethods 注册表按 MoECommType(ALLGATHER/MC2/ALLTOALL/FUSED_MC2)三选一取 AllGather/MC2/AlltoAll/FusedMC2CommImpl，每种配一对 PrepareAndFinalizeWith*+TokenDispatcherWith*；token_dispatcher.py 按专家路由把 token 转置式重分发(all_to_all-v/MC2)；batch_invariant.py 的 override_envs_for_invariance(关 matmul_allreduce、设 HCCL_DETERMINISTIC=strict/LCCL_DETERMINISTIC=1)+enable_batch_invariant_mode 配 triton batch-invariant kernel,保证逐位可复现。host 无 NPU/CANN,精简版只验可读控制流(三选一注册表/重分发形状代数/forward_impl 二分/env 覆盖),真实通信与算子不真跑。

## Why it matters

Part VI(算子与编译层)收官:FusedMoE 是 ch23「换头不换身」机制压力最大的实证(最复杂算子也靠继承+forward_oot 顶替);MoE 难点在 token 按专家重分发的通信,昇腾按 soc/EP 选 MC2/all_to_all/all_gather;batch-invariant 是昇腾对可复现推理的额外保证。回收伏笔 f3(ch06 埋:NPUCommunicator.all_to_all 真正用武之地=MoE 按专家重分发)与 f10(ch15 埋:select_moe_comm_method 只选 moe_comm_type,本章是通信原语 MC2/all_to_all/all_gather 的真正算子落地),两者均已在 arc-map 标 resolved_in ch26。

## What to remember

ch26 delivered/APPROVED。5 个新接口已入 bible interfaces.json(AscendFusedMoE.forward/forward_impl 二分 / _MoECommMethods 三选一注册表+setup_moe_comm_method / MoECommMethod ABC prepare-finalize-fused_experts / MoETokenDispatcher token_dispatch-token_combine 重分发 / batch_invariant override_envs+enable_mode)。f3+f10 已回收(arc-map resolved_in ch26)。reviewer 24 issues 全非阻断,未退章。Part VI 收官。
