# ch26 实现说明 —— 只做减法的精简版（FusedMoE 算子与 batch-invariant）

活动实例 `vllm-ascend`。host 无 NPU/CANN：真实 MC2 / all_to_all / triton/AscendC kernel 不真跑；
精简版只验**可读控制流**——三选一注册表（f10）/ token 重分发形状代数（f3）/ forward_impl 通信-计算二分 /
batch_invariant 的 env 覆盖与算子替换（纯 Python，可在 host 跑测试）。

## 验收判据
把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本精简版（同名/同结构/同控制流，只删不增）。
`python3 scripts/lint_fidelity.py instances/vllm-ascend/artifacts/ch26-fusedmoe-batch-invariant` 无 BLOCKING。
`python3 -m pytest tests/` 15 passed。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版文件 | 真实源码 `vllm_ascend/...:Lxxx` | 改动 | 原因 |
|---|---|---|---|
| `fused_moe.py` `AscendFusedMoE.__init__/forward/forward_impl` | `vllm_ascend/ops/fused_moe/fused_moe.py:L335-L738` | 删 multistream/gate 并流、init_eplb/dynamic_eplb 负载表、shared experts 全套、量化 FUSED_MC2 list 包装、return_with_event、static_kernel 回绕、版本分叉 | 并流/EPLB/shared-expert/量化均为正交维度；保留『继承 FusedMoE→super().__init__ 复用身体→换 quant_method/base_quant_method/runner→覆写 forward_impl(prepare→apply→finalize)』换头主干 |
| `fused_moe.py` `AscendUnquantizedFusedMoEMethod.apply` | `vllm_ascend/ops/fused_moe/fused_moe.py:L129-L262` | 删 zero_experts/capturer/force_load_balance 外的量化与 FUSED_MC2 权重打包分支 | 保留 select_experts 选 topk → build_fused_experts_input → moe_comm_method.fused_experts 主线 |
| `fused_moe.py` `AscendMoERunner` | `vllm_ascend/ops/fused_moe/fused_moe.py:L265-L284` | 删 _maybe_reduce_shared/forward_impl 转调/SP 包裹，保留 `use_dp_chunking` + `_fused_output_is_reduced` | 『只覆写两处行为旋钮』的换头实证 |
| `moe_comm_method.py` `_MoECommMethods/setup/get` | `vllm_ascend/ops/fused_moe/moe_comm_method.py:L48-L62` | 无删减（注册表本体） | **回收 f10**：MoECommType 枚举 → 真正建好的 *CommImpl 实例 |
| `moe_comm_method.py` `MoECommMethod.fused_experts` | `vllm_ascend/ops/fused_moe/moe_comm_method.py:L87-L182` | 删 record_event 计时、dtype assert | 保留 token_dispatch→_apply_mlp→token_combine 三段流水线骨架 + 两个 @abstractmethod 策略接口 |
| `moe_comm_method.py` 4×`*CommImpl` | `vllm_ascend/ops/fused_moe/moe_comm_method.py:L185-L348` | FusedMC2.fused_experts 删 dispatch_ffn_combine/decode 完整入参 | 每种通信方式 = 一对 dispatcher+prepare_finalize；FusedMC2 覆写 fused_experts 把三步融一 |
| `token_dispatcher.py` MC2/AllGather/All2AllV | `vllm_ascend/ops/fused_moe/token_dispatcher.py:L101-L666` | 删 get_dispatch/combine_mc2_kwargs 拼装、A3/A5 extra_args、quant 分支、_preprocess histc 统计、多本地专家 postprocess | 保留三种 token 重分发的形状契约；**回收 f3**：All2AllV.token_dispatch 调 async_all_to_all |
| `comm_utils.py` `async_all_to_all` | `vllm_ascend/ops/fused_moe/comm_utils.py:L26-L62` | 删多流 COMM_STREAM 分支；device 改 input_.device | all2all-v 不等长 split 主线不变；**f3 底层落点** |
| `prepare_finalize.py` base + 3×`*With*` | `vllm_ascend/ops/fused_moe/prepare_finalize.py:L40-L519` | 删 SP/PCP/flashcomm 分支、multistream gate、pad_and_split_input_ids | 保留 DP/TP 基本路径：AllGather 的 DP all-gather/reduce-scatter；MC2/All2All 的 pad+TP 切片 |
| `experts_selector.py` `select_experts` | `vllm_ascend/ops/fused_moe/experts_selector.py:L30-L137` | 删 weight_prefetch、mix_placement、三个 helper 实现体（stub） | 保留『融合 gating 算子 vs 原子回退』分流骨架 |
| `ascend_forward_context.py` `MoECommType/select_moe_comm_method` | `vllm_ascend/ascend_forward_context.py:L26-L319` | 删 A2/A3/A5/310P 阈值细判，保留按 EP/soc/token 选枚举骨架 | **f10 起点**（埋于 ch15） |
| `batch_invariant.py` override/enable/init/reduce_sum | `vllm_ascend/batch_invariant.py:L50-L150` | 无删减（已是顶层开关） | 关漂移源 env + torch.library 替换 aten 算子为确定性实现 |
| `triton/batch_invariant/matmul.py` `matmul_persistent` + 包装 | `vllm_ascend/ops/triton/batch_invariant/matmul.py:L24-L437` | 删 kernel 指针运算体、linear_persistent、高维 reshape 分发分支；linear_batch_invariant 改走 matmul_persistent | host 无 triton/NPU 不真跑；保留固定 BLOCK_M/N/K + allow_tf32=False 说明批不变原理 |

## 注意
- `_EXTRA_CTX` 是 ch15 的前向上下文容器（本章已减法移出 `ascend_forward_context.py`）；测试经 conftest 注入占位。
- 量化路径（W8A8/W4A8/MXFP*）整体删除——本章只跑非量化 `QuantType.NONE`，呈现完整 dispatch→mlp→combine 骨架。
