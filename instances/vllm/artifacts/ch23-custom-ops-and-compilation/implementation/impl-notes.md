# ch23 实现笔记 — CustomOp 两级 dispatch / @support_torch_compile / piecewise 切图

只做减法精简版。三个被解读的子系统各成一个文件，`_runtime.py` 提供脱离 CUDA/vLLM 在 host
跑起来的最小上下文（CompilationConfig / CompilationMode / current_platform / 全局访问点）。
源码 pin `f3fef123`。

## 验收判据
把真实 vLLM 删掉所有 `# SUBTRACTED:` 标注的分支（OOT 插件、Oink/aiter 加速、AOT 缓存、
哈希/缓存子系统、多平台细分、动态形状 UNBACKED 多分支），≈ 得到本精简版。控制流（构造期一次
性 dispatch、enabled/default_on 决定 native vs cuda、按 mode 决定 do_not_compile、在
splitting_ops 处切图、非切分段建 PiecewiseBackend+包 CUDA graph、attention 段 eager）逐处对齐。

## 1:1 Source Map

| 精简版符号 | 真实 vllm 源 | 改动 | 原因 |
|---|---|---|---|
| `custom_op.CustomOp` (`__new__`/`__init__`/`forward`/`dispatch_forward`/`maybe_compile`/`enabled`/`default_on`/`register`) | `vllm/model_executor/custom_op.py:L103-L320` | `__new__` 删 OOT 替换分支；`maybe_compile` 删 compile_options/逐参 mark_dynamic 包裹；删 `register_oot` | OOT 插件与动态形状细节非本章主线；构造期一次性 dispatch + enabled/default_on 逻辑原样保留 |
| `custom_op.RMSNorm` (`__init__`/`forward_static`/`forward_native`/`forward_cuda`) | `vllm/model_executor/layers/layernorm.py:L102-L318` | `forward_native` 无 residual 一路 `ir.ops.rms_norm`→`forward_static`；`forward_cuda` 删 Oink/batch_invariant 分支、`ir.ops.rms_norm`→`forward_static`；`__init__` 删 aiter/Oink 检测 | host 无 `ir.ops`/CUDA kernel；用 forward_static 复现纯 torch 数值，保留 native(可融合) vs cuda(不透明 kernel) 对照 |
| `custom_op.fused_add_rms_norm` | `vllm/model_executor/layers/layernorm.py:L56` | `ops.fused_add_rms_norm`(C++ kernel)→纯 torch in-place 复现 | host 无预编译融合 kernel；保留「forward_cuda 调对 Inductor 不透明的融合 kernel」语义 |
| `attention_op.unified_attention_with_output` + `_fake` + `direct_register_custom_op` | `vllm/model_executor/layers/attention/attention.py:L706-L754`、`vllm/utils/torch_utils.py:L899` | 真实 impl(取 metadata→self.impl.forward) 用占位写 output 复现；`infer_schema` 改显式 schema 串、dispatch_key→CPU | f17 回收重点是「注册成带 fake_impl 的不透明 torch.ops.vllm 算子」，attention forward 本身前章已讲 |
| `compilation.should_split` | `vllm/compilation/partition_rules.py:L14` | 无（原样） | 切点判定逻辑直接搬，对真实 fx 图工作 |
| `compilation.split_graph` / `SplitItem` | `vllm/compilation/backends.py:L405,L547` | 删 `_decompose_size_nodes`/`_merge_empty_only_subgraphs`/lazy-graph-module/tuple_return 兼容 | FX 正确性/版本兼容细节不改「在 splitting_ops 处切、切分段单独成图」主流程 |
| `compilation.PiecewiseCompileInterpreter` / `PiecewiseBackend` / `wrap_with_cudagraph_if_needed` | `vllm/compilation/backends.py:L627,L681,L724` + `piecewise_backend.py` | PiecewiseBackend 用轻量转发替身；CUDA graph wrap 记录决定后返回原 backend | host 无 Inductor/CUDA graph；保留「非切分子图建 backend+按需包 CUDA graph、attention 段不建」控制流 |
| `compilation.VllmBackend` (`__init__`/`__call__`) | `vllm/compilation/backends.py:L996` | 删缓存/哈希子系统、fake_args 提取、post_pass | 缓存与 FX 细节不改「切图→逐段编译→包 CUDA graph」；单次调用守卫保留 |
| `compilation.support_torch_compile` / `_support_torch_compile` / `TorchCompileWithNoGuardsWrapper` | `vllm/compilation/decorators.py:L118,L331` + `wrapper.py` | 删 FloatTensor/IntermediateTensors 注解分支、位置参数校验、AOT 路径、traced_files/config patch、UNBACKED 多分支；wrapper 用 torch.compile(self.forward, backend=VllmBackend) 复现 | 保留「推断动态维(Tensor→dim0)、注入 wrapper 到 __bases__、按 mode 定 do_not_compile、首编触发 VllmBackend/后续缓存」主控制流 |
| `_runtime.CompilationConfig` / `CompilationMode` / `set_splitting_ops_for_v1` / `current_platform` / `get_cached_compilation_config` | `vllm/config/compilation.py:L37,L469,L738,L1082` + `vllm/config/__init__.py` + `vllm/platforms` | 仅保留本章字段子集；platform 用可设 kind 的替身 | 让 dispatch/切图能在单进程被测，控制流（读 custom_ops、默认切点取 _attention_ops）一致 |

## 跑测试
host 纯 torch（不 import vllm）：`python3 -m pytest tests/ -q`（29 passed）。
