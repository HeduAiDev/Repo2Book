# C6 — 自定义算子与编译（ops + csrc + compilation）

子系统测绘 digest。对象 vllm-ascend v0.21.0rc1；前缀 `vllm_ascend/`。对照基座 vLLM v0.21.0（`vllm/`）。

## 一句话定位

vllm-ascend 作为 vLLM 的 OOT 平台插件，靠**三套相互独立的「顶替」机制**把 CUDA 路径换成昇腾路径：
(1) `CustomOp.register_oot` —— 在算子层用 `AscendXxx.forward_oot` 顶替 vLLM 的 `MyOp.forward_native`（RMSNorm/SiluAndMul/RoPE/Linear/FusedMoE…）；
(2) `direct_register_custom_op` + `torch.library`（C++ `TORCH_LIBRARY(_C_ascend)` + Python meta）—— 把 AscendC/aclnn kernel 注册成 `torch.ops` 可被图捕获的算子；
(3) `Platform.get_static_graph_wrapper_cls` / `get_compiler_cls` —— 用 `ACLGraphWrapper`（对位 cudagraph）和 `AscendCompiler`（对位 `VllmBackend`/inductor）替换 vLLM 编译栈。书里这是 vLLM 书 ch22（custom op dispatch）/ ch23（torch.compile + cudagraph）的昇腾对位篇。

## 三大机制的真实落点（已逐字核对源码）

### 机制1：算子层 OOT 顶替（CustomOp.register_oot）
- vLLM 基座：`vllm/model_executor/custom_op.py` —— `CustomOp.dispatch_forward`（L174+）在 `enabled()` 时返回 `self.forward_oot`（L205），否则 `forward_native`；`register_oot`（L332）把外部子类登记进 op registry。
- vllm-ascend：`vllm_ascend/utils.py:register_ascend_customop`（L638–765）有一张 **30 项映射表** `REGISTERED_ASCEND_OPS`（"RMSNorm"→`AscendRMSNorm`, "SiluAndMul"→`AscendSiluAndMul`, "FusedMoE"→`AscendFusedMoE`…），末尾 `for ... CustomOp.register_oot(...)`（L761）批量登记；310P 分支再覆盖一批 `*310`。每个 Ascend 类只重写 `forward_oot`（如 `ops/activation.py` `AscendSiluAndMul.forward_oot`→`torch_npu.npu_swiglu`；`ops/layernorm.py` `AscendRMSNorm.forward_oot`→`torch_npu.npu_add_rms_norm`/`_C_ascend.npu_add_rms_norm_bias`）。
- 关键张力：`forward_oot` 里又分两路——`enable_custom_op()` 真走自家 AscendC 融合算子（`_C_ascend.*`），否则回退 `torch_npu.*` 原子算子。这是「自定义/融合算子怎么替换 vLLM 算子」的核心二分。

### 机制2：C++/aclnn kernel 注册（torch.library + meta）
- `csrc/torch_binding.cpp`（2820 行，~65 个 `ops.def`/`ops.impl`，`TORCH_LIBRARY(_C_ascend)`，全部 `torch::kPrivateUse1`）注册真实 AscendC 算子：`npu_add_rms_norm_bias`、`mla_preprocess`、`batch_matmul_transpose`、`get_masked_input_and_mask`、`bgmv/sgmv_expand/shrink`（LoRA）、`moe_*`、`npu_recurrent_gated_delta_rule`…
- `csrc/torch_binding_meta.cpp`（1744 行）+ `vllm_ascend/meta_registration.py`（`Library("_C_ascend","IMPL")` + `register_meta_if_necessary`，L44–94）提供 **meta/fake 实现** —— 没有 meta 就无法 trace/`torch.compile`/aclgraph 捕获。这是「kernel 怎么注册」最该讲清的一环。
- 纯 Python 侧算子用 `vllm/utils/torch_utils.py:direct_register_custom_op`：`ops/register_custom_ops.py`（299 行）登记 11 个 `torch.ops.vllm.*`（`maybe_chunk_residual`、`maybe_all_gather_and_maybe_unpad`、`maybe_pad_and_reduce`、`matmul_and_reduce`、`npu_rotary_embedding`、`muls_add`…），每个都配 `_xxx_fake`，`dispatch_key="PrivateUse1"`。Triton kernel（`ops/triton/`）经此把 `muls_add_triton` 等包成可捕获 op。

### 机制3：图模式（torchair / aclgraph 对位 vLLM 编译栈）
- vLLM 基座对照：`vllm/compilation/`（`VllmBackend`、`PostGradPassManager`、cudagraph）、`vllm/platforms/interface.py:get_static_graph_wrapper_cls`(L886)/`simple_compile_backend`(L136)。
- vllm-ascend：`platform.py` 把两个钩子改向自家实现——`get_compiler_cls`→`compilation.compiler_interface.AscendCompiler`（L179）；`get_static_graph_wrapper_cls`→`compilation.acl_graph.ACLGraphWrapper`（L820）。
- `compiler_interface.py`（345 行）`AscendCompiler(CompilerInterface)`：`compile()` 二选一——`enable_npugraph_ex` 走 `npugraph_ex_compile`（新 `npugraph_ex`，失败回退 `torchair.get_npu_backend`，`mode="reduce-overhead"` 即 aclgraph）；否则 `fusion_pass_compile`（`compile_fx`+`aot_autograd`，跑自家 `GraphFusionPassManager`）。
- `acl_graph.py`（~430 行）`ACLGraphWrapper`：对位 vLLM `cuda_graph.py`，用 `torch.npu.NPUGraph()` capture/replay，按 `BatchDescriptor` 分桶缓存 `ACLGraphEntry`，含 stream-resource 耗尽（错误码 207008）的诊断兜底。
- `compilation/graph_fusion_pass_manager.py` + `passes/`（1629 行，9 个 pass）：`GraphFusionPassManager`（注释自陈"PostGradPassManager 的对位，因 torch_npu 暂不支持 triton 故自定义"），按 `ascend_compilation_config` 开关挂 `AddRMSNormQuantFusionPass`、`QKNormRopeFusionPass`、`MatmulAllReduceAddRMSNormPass`、`MulsAddFusionPass`、`SequenceParallelism(Moe)Pass`。
- 配套：`ops/__init__.py:register_dummy_fusion_op`（L43，worker 启动调用）把 `torch.ops._C_ascend.rms_norm` 等塞 `dummyFusionOp` 占位，让 vLLM 的 fusion pass 匹配机制不报缺算子。

## 建议章节（中等深度，5 章）

### 章 A：CustomOp 的 OOT 顶替机制 —— 昇腾算子如何替换 vLLM 算子 ★最值得
- focus：`register_ascend_customop` 30 项映射表 + `CustomOp.dispatch_forward`→`forward_oot` 的分发链；以 `AscendSiluAndMul`/`AscendRMSNorm` 为标本，讲 `enable_custom_op()` 真融合算子 vs `torch_npu` 回退的二分。
- key_source_paths：`vllm_ascend/utils.py` L638–765、`vllm_ascend/ops/activation.py`、`vllm_ascend/ops/layernorm.py`、`vllm_ascend/ops/__init__.py`。
- pairs_with：`vllm/model_executor/custom_op.py`（`dispatch_forward`/`register_oot`/`forward_native`）、`vllm/model_executor/layers/{activation,layernorm}.py`；vLLM 书 ch22。
- teach_value：这是整个 OOT 插件「换头不换身」的总开关，读懂它才懂 vllm-ascend 怎么 0 改 vLLM 模型代码就换硬件。
- est_size：中（~1.0x）。deps：platform/插件注册时机章。

### 章 B：torch.library 算子注册与 meta 实现 —— AscendC kernel 怎么进图 ★次值得
- focus：`TORCH_LIBRARY(_C_ascend)`（C++ def/impl on PrivateUse1）+ `direct_register_custom_op`（Python+fake）+ meta 注册三条注册路径；为何「无 meta 不能 torch.compile/aclgraph」。
- key_source_paths：`csrc/torch_binding.cpp`、`csrc/torch_binding_meta.cpp`、`vllm_ascend/meta_registration.py`、`vllm_ascend/ops/register_custom_ops.py`。
- pairs_with：`vllm/_custom_ops.py`、vLLM `csrc` 的 `TORCH_LIBRARY(_C)` + `torch/library` fake 注册；vLLM 书 ch22/23。
- teach_value：把「C++ kernel ↔ Python torch.ops ↔ 图捕获」打通，是后面图模式章的地基。
- est_size：中（~1.0x）。deps：章 A。

### 章 C：AscendCompiler —— torch.compile 后端对位 VllmBackend
- focus：`get_compiler_cls` 钩子替换；`AscendCompiler.compile` 的 npugraph_ex/torchair vs fusion_pass 两路；`compute_hash`/cache。
- key_source_paths：`vllm_ascend/compilation/compiler_interface.py`、`vllm_ascend/platform.py` L175–179。
- pairs_with：`vllm/compilation/compiler_interface.py`（`CompilerInterface`/inductor）、`vllm/compilation/backends.py`（`VllmBackend`）；vLLM 书 ch23。
- teach_value：讲清「昇腾没有 inductor，靠 torchair/npugraph_ex 顶替」的编译入口。
- est_size：中（~0.9x）。deps：章 B。

### 章 D：ACLGraph + 图融合 pass —— cudagraph 与 PostGradPass 的对位
- focus：`ACLGraphWrapper` 的 `NPUGraph` capture/replay 与 `BatchDescriptor` 分桶；`GraphFusionPassManager`+`passes/` 九个融合 pass（norm-quant/qknorm-rope/allreduce-rms/muls-add/SP）；`register_dummy_fusion_op` 占位的巧思。
- key_source_paths：`vllm_ascend/compilation/acl_graph.py`、`vllm_ascend/compilation/graph_fusion_pass_manager.py`、`vllm_ascend/compilation/passes/`、`vllm_ascend/ops/__init__.py` L36–51。
- pairs_with：`vllm/compilation/cuda_graph.py`、`vllm/compilation/passes/`(`PostGradPassManager`)、`get_static_graph_wrapper_cls`；vLLM 书 ch23。
- teach_value：图模式「捕获 + 融合」全貌，承接章 C 落到运行时。
- est_size：中（~1.0x）。deps：章 C。

### 章 E（可选/合并候选）：FusedMoE 算子与 batch-invariant 一致性
- focus：`ops/fused_moe/`（~4k 行：`fused_moe.py`/`token_dispatcher`/`moe_comm_method` MC2/AllToAll）作为最大单体 OOT 算子的标本；`batch_invariant.py`（149 行）+ `ops/triton/batch_invariant/` 如何覆盖 env/算子保证可复现。
- key_source_paths：`vllm_ascend/ops/fused_moe/`、`vllm_ascend/batch_invariant.py`、`vllm_ascend/ops/triton/batch_invariant/`。
- pairs_with：`vllm/model_executor/layers/fused_moe/`、vLLM 的 batch_invariant 路径；vLLM 书 ch22 + MoE 篇。
- teach_value：把机制 1 用在最复杂算子上的实证；batch-invariant 是昇腾特有的可复现保证。若篇幅紧，batch-invariant 可并入章 A 尾节。
- est_size：中偏大（~1.1x，可砍半）。deps：章 A、章 D。

注：`ops/` 下大量模型专用算子（gdn/dsa/mla/rope_dsv4/qwen2_decoder/cv_linear…）属各模型适配，不在本子系统主线，仅在章 A/E 当例子点到，避免铺开。xlite/（`xlite.py`/`xlite_model_runner`/`xlite_worker`）是独立轻量执行路径，建议归 worker/runner 子系统，本子系统不展开。

## 接进 vLLM 主线的 2–3 句总结

vllm-ascend 不改 vLLM 一行模型代码：算子层靠 `CustomOp.register_oot` 把 30 个 `forward_native` 静默替换成调用 `torch_npu`/AscendC 融合算子的 `forward_oot`；kernel 层靠 `TORCH_LIBRARY(_C_ascend)`+meta 把 AscendC/aclnn 算子注册成可被图捕获的 `torch.ops`；编译层靠 `Platform.get_compiler_cls`/`get_static_graph_wrapper_cls` 两个钩子，用 `AscendCompiler`（torchair/npugraph_ex 顶替 inductor）和 `ACLGraphWrapper`（NPUGraph 顶替 cudagraph）整体换掉 vLLM 的 `torch.compile`+cudagraph 栈，并以自家 `GraphFusionPassManager` 对位 `PostGradPassManager`。读者据此看清：vLLM 的「平台插件 + CustomOp dispatch + 编译后端可插拔」三个扩展点，正是昇腾顶替 CUDA 的全部入口。
