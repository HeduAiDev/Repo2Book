# ch25 实现笔记 — AscendCompiler 与 ACLGraph（只做减法）

精简版按 dossier 的 `subtraction_plan` 产出，与真实源码**同名/同结构/同控制流，只删不增**。
所有删除均为 `subtraction_plan.delete` 明确批准项，并就地标 `# SUBTRACTED:`；`must_keep` 的 30 个
符号全部保留（`lint_fidelity` 校验通过）。

## 1:1 Source Map

| 精简版文件 | 对照真实源码 | 改动 | 原因 |
|---|---|---|---|
| `implementation/platform_hooks.py` | `vllm_ascend/platform.py:L156-L179, L816-L820` | 截取 `pass_key` + 3 个编译钩子（`get_pass_manager_cls` / `get_compile_backend` / `get_static_graph_wrapper_cls`），NPUPlatform 其余成员 SUBTRACTED | 本章焦点是「两个（+一个）返回字符串的平台钩子整体顶替 vLLM 编译栈」；NPUPlatform 其余分散他章 |
| `implementation/compiler_interface.py` | `vllm_ascend/compilation/compiler_interface.py:L39-L277` | 保留 `compile_fx` / `fusion_pass_compile` / `_configure_backend` / `npugraph_ex_compile` / `AscendCompiler.{compute_hash,initialize_cache,compile}`；删 `patched_get_compiled_gm` 缓存巧思、`load()`、`_compute_decode_cudagraph_batch_sizes`、`enable_static_kernel` 分支 | 缓存只影响二次启动速度、static_kernel 是可选加速开关，均不改「二分编译 + tuple 适配 + aclgraph 模式」主线（delete 项 1、2） |
| `implementation/graph_fusion_pass_manager.py` | `vllm_ascend/compilation/graph_fusion_pass_manager.py:L25-L79` | 原样保留（无删减） | 对位 PostGradPassManager 的 `__call__`（串 pass + recompile）/ `add` / `configure`（按开关注册 pass）是本章主干 |
| `implementation/passes/norm_quant_fusion_pass.py` | `vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:L29-L86, L477-L513` | 保留代表性 `AddRMSNormQuantPattern` + `AddRMSNormQuantFusionPass`；删其余 7 个同构 Pattern 变体（delete 项 3），其在 `__init__` 中的 register 调用随删类一并 SUBTRACTED | 8 个 Pattern 结构同构（仅签名/bias/all_gather 不同），保留一个即可讲透 pattern→replacement 融合；引用已删类无法运行 |
| `implementation/passes/base_pattern.py` | `vllm_ascend/compilation/passes/base_pattern.py:L20-L63` | 原样保留（无删减） | `register()` 向 inductor + npugraph_ex 双注册替换规则是本章核心机制 |
| `implementation/ops_dummy_fusion.py` | `vllm_ascend/ops/__init__.py:L36-L51` | 截取 `dummyFusionOp` + `register_dummy_fusion_op`，原文件顶部其它 ops 子模块注册 SUBTRACTED | 本章焦点是「占位算子作 pattern 匹配锚点」巧思；其它 ops 注册属他章 |
| `implementation/acl_graph.py` | `vllm_ascend/compilation/acl_graph.py:L27-L272` | 保留 207008 兜底 + `ACLGraphEntry` + `ACLGraphWrapper.{__init__,__call__}`；删 `GraphParams`/`weak_ref_workspaces`/三套 set/get/update 全局函数（delete 项 4）+ 生命周期样板 `clear_all_graphs`/`clear_graphs`/`unwrap`/`cudagraph_wrapper`/`__getattr__`（delete 项 5）；`__call__` 内对已删 `weak_ref_workspaces` 的 3 处调用随之 SUBTRACTED | 这些是 full-graph 模式 workspace 簿记与属性透传样板，capture/replay/分桶/207008 主干不依赖 |

## 关于「连带删除」的说明（非自行判断、是 delete 的必然结果）

- `norm_quant_fusion_pass.py` 的 `AddRMSNormQuantFusionPass.__init__`：delete 项 3 批准删除 7 个 Pattern
  变体类。`__init__` 原本对它们逐个 `register`，引用已删类必然 `NameError`，故对应 register 调用一并
  SUBTRACTED，只保留对 `AddRMSNormQuantPattern` 的注册。
- `acl_graph.py` 的 `__call__`：delete 项 4 批准删除 `weak_ref_workspaces` 与三套 `_graph_params` 全局，
  `__call__` 捕获分支原本 `weak_ref_workspaces(_graph_params/...)`，引用已删符号必然报错，故就地 SUBTRACTED。

均为 delete 批准项的直接连带，不涉及对 `must_keep` 之外细节的自主删减。

## 测试（host 可跑，纯控制流）

`tests/conftest.py` 注入 stub（torch.npu / vllm.* / vllm_ascend.* / torch_npu）并合成 `torch.npu`
命名空间，把实现按规范点分模块名加载（让相对/绝对 import 解析）。被测的是**真实控制流**：

- `test_acl_graph_207008.py`：207008 / 两个标志串识别（含「裸 207008 不误判」），改写带指引报错 + 原异常 chain。
- `test_acl_graph_wrapper.py`：mode=NONE/不匹配直跑、首见 descriptor 用 NPUGraph 捕获建 entry、不同
  descriptor 各捕一张图（分桶）、同 descriptor 再见走 `replay()` 返回缓存 output、捕获中 207008 改写 / 非 207008 原样上抛。
- `test_compiler_dispatch.py`：`compile()` 按 `enable_npugraph_ex` 二选一、cache_dir/vllm_config 透传、`disable_cache` 强制 cache_dir=None。
- `test_pass_manager.py`：`__call__` 只跑 `is_applicable_for_range` 通过的 pass、按注册序、最后 `recompile()`；`add` 类型校验。
- `test_dummy_fusion_and_hooks.py`：`register_dummy_fusion_op` 挂 8 个锚点；三个平台钩子返回正确类路径字符串 + `pass_key`。

真实 npugraph_ex/torchair 编译与 NPUGraph capture 需 NPU/CANN，不在 host 真跑（用 stub 替边界对象）。
全 27 测试通过；`lint_fidelity` 无 BLOCKING。
