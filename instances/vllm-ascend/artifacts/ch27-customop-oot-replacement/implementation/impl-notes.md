# ch23 精简版实现笔记 —— CustomOp 的 OOT 顶替

「只做减法」的忠实精简版：与真仓同名同结构同控制流，只删不增。host 无 NPU/CANN，
真实 `torch_npu.*` / `torch.ops._C_ascend.*` 算子由测试的记录替身承接（不真算），只验可读控制流。

## 文件 ↔ 真实源码 映射

| 精简版文件 | 规范源码路径 | 角色 |
|---|---|---|
| `custom_op.py` | `vllm/model_executor/custom_op.py` | 基座 CustomOp：`__new__`（换身）/ `dispatch_forward`（换头）/ `register`+`register_oot` / `op_registry_oot` |
| `base_activation.py` | `vllm/model_executor/layers/activation.py` | 基座 `SiluAndMul` / `QuickGELU`（「身」，含 forward_native/forward_cuda 对照） |
| `base_layernorm.py` | `vllm/model_executor/layers/layernorm.py` | 基座 `RMSNorm`（「身」） |
| `ascend_activation.py` | `vllm_ascend/ops/activation.py` | 标本一：`AscendSiluAndMul`/`AscendQuickGELU` 只覆 forward_oot |
| `ascend_layernorm.py` | `vllm_ascend/ops/layernorm.py` | 标本二：`AscendRMSNorm.forward_oot` 的融合 vs 回退二分 |
| `ascend_fused_moe.py` | `vllm_ascend/ops/fused_moe/fused_moe.py` | 注册表第四代表项（完整 forward_oot 是 ch26 专题） |
| `utils.py` | `vllm_ascend/utils.py` | 注册表总开关 `register_ascend_customop` + 第二层开关 `enable_custom_op` |

## 1:1 Source Map（精简版 ↔ 源码行 ↔ 改动 ↔ 原因）

| 精简版符号 | 源码:行 | 改动 | 原因 |
|---|---|---|---|
| `register_ascend_customop` | `vllm_ascend/utils.py:L638` | import 仅留 activation/layernorm/fused_moe 3 组代表；dict 留 4 项代表；删 deepseek_mla/is_310p 支线 | subtraction_plan.delete 批准；机制与算子条数无关，保留代表项即演示「建表→遍历 register_oot」全流程 |
| `REGISTERED_ASCEND_OPS` | `vllm_ascend/utils.py:L52,L684` | 28 项→4 项代表 | 每项同构「类名字符串→Ascend 子类」，留 4 则不改控制流 |
| `_ASCEND_CUSTOMOP_IS_REIGISTERED` | `vllm_ascend/utils.py:L644,L765` | 原样保留 | 幂等闸，解释「只生效一次」 |
| `enable_custom_op` | `vllm_ascend/utils.py:L357` | 删 batch-invariant/A5 特例 + bootstrap + ImportError 二次重试 | 边缘/host 环境细节，不改「import 成败→融合 vs 回退」二分 |
| `register_oot` / `op_registry_oot` | `vllm/model_executor/custom_op.py:L332,L22` | 原样保留 | 遍历写入目标 + __new__ 据此换身的数据源 |
| `CustomOp.__new__` | `vllm/model_executor/custom_op.py:L109` | 原样保留 | 「换身」机制 |
| `CustomOp.dispatch_forward` | `vllm/model_executor/custom_op.py:L174` | 删 rocm/cpu/tpu/xpu 平台分支，留 oot/native + `is_out_of_tree` | subtraction_plan.delete 批准；本章只关心昇腾的 oot 与回退 native |
| `CustomOp.maybe_compile` | `vllm/model_executor/custom_op.py:L209` | 保留 `enable` 早退；删 torch.compile 编译机理 | torch.compile 插件管线与顶替主线无关，host 无后端 |
| `AscendSiluAndMul.forward_oot` | `vllm_ascend/ops/activation.py:L31` | 删 weight_prefetch 调用 | 性能优化，与「顶替为 npu_swiglu」语义无关 |
| `AscendRMSNorm` | `vllm_ascend/ops/layernorm.py:L28` | __init__ 删 quant m4 bias 探测 + _bias_weight_loader；forward_oot 删尾部 prefetch | 量化边缘场景；保留 self.bias 字段供二分引用即可 |
| `AscendRMSNorm.forward_oot` | `vllm_ascend/ops/layernorm.py:L63` | 原样保留二分主体 | 承载「融合 `npu_add_rms_norm_bias` vs 原子 `npu_add_rms_norm` 回退」 |
| `base RMSNorm` | `vllm/model_executor/layers/layernorm.py:L38` | 删 poly_norm/kernels 导入/其余 norm 类/forward_xpu | 仅作「身」对照，非本章主线 |

## 验证

- `python3 -m pytest tests/ -q` → 11 passed（host，纯 Python 控制流）。
- `python3 scripts/lint_fidelity.py <chapter_dir>` → 保真度全部通过（must_keep 16 项全在）。

## 给 writer 的提示

- 「换身」与「换头」是两个正交动作：`__new__`（实例化哪个类）先于 `__init__`（绑哪个 forward）。
- 两层二分正交：第一层 `dispatch_forward` 选 oot vs native；第二层 `forward_oot` 内 `enable_custom_op()` 选融合 vs 回退。
- `register_oot` 把 `.name` 覆盖成类名键（`'RMSNorm'`），区别于基座 `register` 的 lowercase（`'rms_norm'`）——测试 `test_register_oot_overrides_name_with_class_key` 钉死这一点。
- `ascend_fused_moe.py` 的 forward_oot 主体已整体标 SUBTRACTED 指向 ch26；本章它只作注册表第四代表项。
