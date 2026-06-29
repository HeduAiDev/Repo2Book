# ch15 精简版实现笔记 —— 一次 execute_model 的真实数据流

只做减法（subtract-only）：与 vllm-ascend v0.21.0rc1 (80610e44) / 对照基座 vllm v0.21.0
**同名、同结构、同控制流**。只删除 `dossier.json.subtraction_plan.delete` 批准项，`must_keep`
符号全部保留（`lint_fidelity` 已校验通过）。昇腾算子 / 真实 all_reduce / 真实前向不在 host 跑——
精简版只验三处纯 Python 控制流：**select_moe_comm_method 决策 / set_ascend_forward_context 注入 /
_sync_metadata_across_dp 打包**，外加 **execute_model 派发骨架**的调用顺序。

## 文件 ↔ 真实源码

| 精简版文件 | 真实源码（规范路径） | 角色 |
| --- | --- | --- |
| `implementation/forward_context.py` | `vllm/forward_context.py` | 被昇腾包住的**基座**：`set_forward_context` 建 ForwardContext/DPMetadata，经 `current_platform.set_additional_forward_context` 让平台注入 |
| `implementation/ascend_forward_context.py` | `vllm_ascend/ascend_forward_context.py` | **本章主角**：`set_ascend_forward_context` 包基座再注入昇腾字段；`select_moe_comm_method` 选通信方式；`MoECommType` 枚举 |
| `implementation/model_runner_v1.py` | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner` 主线方法 + 模块级 `_post_process_cudagraph_mode` |
| `implementation/utils.py` | `vllm_ascend/utils.py` | `should_skip_allreduce_across_dp_group` / `AscendDeviceType` / 谓词 helpers |
| `implementation/moe_comm_method.py` | `vllm_ascend/ops/fused_moe/moe_comm_method.py` | `get_moe_comm_method` 查表工厂（通信原语实现留 ch26） |

## 1:1 Source Map（精简版符号 ↔ Lxxx ↔ 改动 ↔ 原因）

| 精简版符号 | 真实源码 :Lxxx | 改动 | 原因 |
| --- | --- | --- | --- |
| `NPUModelRunner.execute_model` | `model_runner_v1.py:L1904` | 删 ngram replace / EC connector 早返回+assert / mamba+use_compress / PCP 分支 / 后处理 PCP·PP 罕见分支 | `subtraction_plan.delete` 批准的横切；主干 dense/MoE 单拍前向派发骨架完整保留 |
| `set_ascend_forward_context` | `ascend_forward_context.py:L56` | 逐字保留 `with set_forward_context(**kwargs)` 这层包裹 + 全部注入字段 | 本章关键接缝（must_keep），证据「包基座再注入」必须逐字 |
| `select_moe_comm_method` | `ascend_forward_context.py:L233` | 仅压缩 docstring，决策树逐分支保留（A2/A3/_310P/A5） | must_keep；writer 要讲 soc/EP/token 三因素 |
| `set_forward_context`（基座） | `vllm/forward_context.py:L251` | 删 track_batchsize 计时 + finally 统计日志；删收尾 ir_op_priority/enable_torch_wrap 编译包裹（dossier embed elide L310-330） | 性能日志（delete）；编译机制与本章注入演示无关，保留 `override_forward_context` 让上下文生效 |
| `NPUModelRunner._sync_metadata_across_dp` | `model_runner_v1.py:L627` | 仅减 FIXME 注释 | must_keep；DP 同步核心：`[2,dp]` 打包 + sum-allreduce 即广播 + max(tokens)/min(mode) |
| `_post_process_cudagraph_mode` | `model_runner_v1.py:L4867` | 逐字保留 | must_keep；取 `tensor[1,:].min()` → 任一 NONE 则全 NONE |
| `should_skip_allreduce_across_dp_group` | `utils.py:L1083` | 仅删 warning_once 日志 | must_keep；dense 直跳 / MoE 仅 KV consumer 且 MC2 条件跳 |
| `NPUModelRunner._determine_batch_execution_and_padding` | `model_runner_v1.py:L2846` | 删 CUDAGraphStat 指标填充 | must_keep；调 DP 同步并据 synced 结果重新 dispatch（valid_modes 收敛全 DP 一致） |
| `NPUModelRunner._prepare_inputs` | `model_runner_v1.py:L748` | 折叠整形主体 L760-L1272 + 折叠 spec 分支 | dossier 批准（_prepare_inputs 主体正交）；保留 dense logits_indices = query_start_loc-1 |
| `NPUModelRunner._build_attention_metadata` | `model_runner_v1.py:L2942` | 折叠逐后端 builder 主体 | 注意力后端实体留 ch18；保留「建好交给前向」接口语义 |
| `NPUModelRunner._model_forward` | `model_runner_v1.py:L2756` | 逐字保留 | must_keep；含 flash_comm_v1 的 all_gather 回收 |
| `NPUModelRunner._sample` | `model_runner_v1.py:L2553` | 逐字保留 | must_keep；sampler / rejection_sampler 派发（采样器内部留后续章） |
| `get_moe_comm_method` | `moe_comm_method.py:L51` | 保留查表；折叠各 *CommImpl 注册 | must_keep；MoE 通信原语实现留 ch26 |

## 借助测试桩的运行约束

- 昇腾 NPU/CANN 在 host 不可用：`_sync_metadata_across_dp` 在 `dp_allreduce_on_npu=True` 时
  会 `torch.zeros(..., device="npu")`，测试把该分配落到 cpu，仅验「选中 npu device group + `.cpu()`
  回拷」的路由决策；真实跨卡 `dist.all_reduce` 由测试以「各卡填己列后求和」模拟（即源码注释所述
  「零初始化 + 仅本 rank 填自己 + sum-allreduce = 汇齐广播」）。
- 重运行时依赖（`vllm.*` / `vllm_ascend.ascend_config` / 通信组）在 `tests/conftest.py` 以 sys.modules
  桩注入；五个精简版文件按**规范模块名**登记，彼此 import 解析到精简版自身。
