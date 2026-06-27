# v0.21.0 更新摘要 — Group C：custom ops & torch.compile（目标章 ch23）

基线 `f3fef1235` → 标签 `v0.21.0`。本组 4 文件：

| 文件 | diffstat | 性质 |
|---|---|---|
| `vllm/model_executor/layers/layernorm.py` | −252 / +29 | 大幅改写（**非纯移动**，见下） |
| `vllm/compilation/backends.py` | +23 / −2 | 新增 pre-grad pass + codegen 三元组 |
| `vllm/compilation/decorators.py` | 无改动 | SKIP |
| `vllm/model_executor/custom_op.py` | 无改动 | SKIP |

关联提交：`d58c42e19 [vLLM IR] 2/N fused_add_rms_norm and maybe_inplace overload (#36823)`、`213f10bfd [Bugfix] Fix codegen for unqualified names (#40726)`。

> **关于 layernorm.py 的 −252：先验证「搬去哪了」。** 删掉的 252 行（`forward_static`、`fused_add_rms_norm`、`dispatch_rocm_rmsnorm_func`、`forward_hip`、Oink 快路径）**不是简单移动到某个 `layers/layernorm/` 包**——`vllm/ir`（`vllm/ir/op.py`、`vllm/ir/ops/layernorm.py`）在基线就已存在，本次提交是把 RMSNorm 的「手写融合 kernel vs 纯 torch」选择权**整体下沉进 `vllm.ir` 这一算子抽象层**，由 `KernelConfig.ir_op_priority`（`vllm/config/kernel.py`）驱动。这是一次真实的职责迁移 + 行为变更，不是 SKIP。

---

## 教学性变更（teachable）：共 2 条

### 1. RMSNorm 的 cuda/native 二分被「IR 算子层」收编 —— ch23 核心例子已过时
- **class**：BEHAVIOR-CHANGE
- **v0.21.0 锚点**：`vllm/model_executor/layers/layernorm.py` 的 `RMSNorm.forward_cuda` 与 `RMSNorm.forward_native`
- **目标章**：ch23（直接命中 **23.3 节，正文 L294–L316**）
- **真实 diff 依据**：
  - 旧 `forward_cuda`（章节 L294–L310 逐字引用）有 `residual` 时调 `fused_add_rms_norm`（C++/CUDA 预编译融合 kernel）——该函数及其调用在 v0.21.0 **被整段删除**。
  - 新 `forward_cuda` 只剩一条特例：`VLLM_BATCH_INVARIANT and residual is None and variance_size_override is None` 时走 `rms_norm_batch_invariant`，**否则一律 `return self.forward_native(x, residual)`**。
  - 新 `forward_native`：`residual is None` → `ir.ops.rms_norm(...)`；否则 → **`ir.ops.fused_add_rms_norm.maybe_inplace(...)`**（`maybe_inplace` overload 定义于 `vllm/ir/op.py:L412+`）。
  - 同时删除：`forward_static`（章节 L270 引用的那串 fp32→平方→均值→rsqrt→乘权重的纯 torch 流程）、`forward_hip`、ROCm `dispatch_rocm_rmsnorm_func`、Oink SM100 快路径。
- **整合建议（书声线）**：23.3 的二元对立——「`forward_cuda` 调不透明的预编译融合 kernel，`forward_native` 摊成 Inductor 看得见的纯 torch 算子」——在 v0.21.0 已被改写：`RMSNorm.forward_cuda` 不再自己持有手写融合 kernel，除 `VLLM_BATCH_INVARIANT` 特例外**直接委托给 `forward_native`**，而 `forward_native` 把两路都交给 `vllm.ir` 的算子句柄（`ir.ops.rms_norm`、`ir.ops.fused_add_rms_norm.maybe_inplace`）。可在 23.3 末尾补一段「版本演进」：手写 kernel 与纯 torch 的取舍**从 `CustomOp` 的 `forward_*` 方法体下沉到了 `vllm.ir` 这一中间算子层**，由 `KernelConfig.ir_op_priority`（`vllm/config/kernel.py`，每个 IR 算子一份优先级列表，表头可为 `"native"`）在更靠后的阶段裁决——`forward_native` 里新增的 `pass_weight`/`pass_weight_add` 正是据这份优先级「预判会不会派发到 native」来决定是否把全 1 权重传下去（避免 TPU 上的 identity-weight 问题，见 issue 39370）。**保留 23.1/23.2 的两级 dispatch 主线不动**（`enabled()`/`default_on()`/`custom_ops` 配置在 `custom_op.py` 中**完全未变**），只把 23.3 这个落地实例标注为「v0.21.0 起经 `vllm.ir` 间接化」。
- **diagram impact**：23.3 若有「`forward_cuda`→融合 kernel / `forward_native`→torch 算子串」的对照图，箭头需改为两路**先汇入 `vllm.ir.ops`**，再由 `ir_op_priority` 决定底层落到融合 kernel 还是 native。若无专图则仅文字补注，无需重绘 roadmap。

### 2. VllmBackend 新增 pre-grad「就地函数化」pass —— 第 2 级编译管线多一道工序
- **class**：NEW-FEATURE
- **v0.21.0 锚点**：`vllm/compilation/backends.py` 的 `VllmBackend.configure_post_pass`，新依赖 `vllm/compilation/passes/ir/inplace_functionalization.py` 的 `VllmIRInplaceFunctionalizationPass`
- **目标章**：ch23（23.5–23.7 讲 `VllmBackend` 编译管线处）
- **真实 diff 依据**：
  - `configure_post_pass` 新增：把 `VllmIRInplaceFunctionalizationPass(self.vllm_config)` 注册为 inductor 的 `pre_grad_custom_pass`，并把该 key 加入 `_cache_config_ignore_prefix`（避免被 pickle 进 AOTAutograd 缓存键）。该 pass 文件在基线**不存在**，v0.21.0 新增。
  - `generate_execution_code(...)` 返回值从 2 元组扩成 **3 元组**（新增 `consts`），`compile_execution_fn`、`SerializableGraph`/缓存路径同步加 `consts=consts`（codegen 不合格名修复 #40726 配套）。
- **整合建议（书声线）**：上一条 `maybe_inplace` overload 的「下半场」就在这里——`vllm.ir` 把带 in-place 语义的算子（如 `fused_add_rms_norm.maybe_inplace`）先以函数式形态喂给 Inductor，再由 `VllmBackend.configure_post_pass` 注册的 `VllmIRInplaceFunctionalizationPass` 这道 **pre-grad pass** 在编译期把它还原成就地写。可在 23.5/23.7 讲 `VllmBackend` 时点一句：v0.21.0 起，`configure_post_pass` 除了配置 post-grad pass manager，还会注入这道 IR 就地函数化 pre-grad pass，并刻意将其排除出编译缓存键。属可选补注，不改主骨架。
- **diagram impact**：若 23.5–23.7 有 `VllmBackend` 管线图（split_graph → 逐段编译），可在「送 Inductor 前」加一个 pre-grad pass 节点；非必须。

---

## SKIP 清单（已核对，无 reader-facing 变更）
- `vllm/compilation/decorators.py`：v0.21.0 区间内**零改动**（`@support_torch_compile` 不变，23.4/23.5 引用安全）。
- `vllm/model_executor/custom_op.py`：v0.21.0 区间内**零改动**（`CustomOp` 基类、`register`、`enabled()`/`default_on()`、`dispatch_forward` 全部不变，23.1/23.2 整节安全）。
- layernorm.py 内的 ROCm/Oink/`forward_static`/`forward_hip` 删除：属实现搬迁，非新增可教概念，已并入变更 1 的「版本演进」注解，不单列。
