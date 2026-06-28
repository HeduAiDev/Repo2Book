# ch03 精简版实现笔记 — 两段式 monkey-patch（subtract-only）

只做减法：与 vllm-ascend 真实源码**同名、同结构、同控制流**，只删不增。每个 `# SUBTRACTED:`
都标了删了什么 / 为何仍正确 / 原行号。昇腾 NPU/CANN 代码 host 不可跑；**patch 重绑定本身是
纯 Python**，故 `tests/` 用 stub 把 vLLM/vllm_ascend 目标命名空间放进 `sys.modules`，import
（被减法后的）patch 模块，断言重绑定如真仓库一样生效。

## 1:1 Source Map

| 精简版文件 | 源码（规范路径） | 招式 / 角色 | 主要减法 |
|---|---|---|---|
| `triggers.py` | `vllm_ascend/utils.py:L511-L515`、`vllm_ascend/platform.py:L181-L186`、`vllm_ascend/__init__.py:L20-L69`、`vllm_ascend/worker/worker.py:L82-L102` | 单一入口 `adapt_patch` + 4 个触发点（platform①② / worker） | 删各源文件中与 patch 触发无关的成员/语句（量化注入、worker 构造其余步骤等） |
| `patch_ledger.py` | `vllm_ascend/patch/__init__.py:L17-L27` + 三条样本台账 | 两段式总纲注释 + patch 台账（纯注释） | 删 800+ 行逐条业务 patch 台账，仅留三个样本对应条目 |
| `patch_platform_init.py` | `vllm_ascend/patch/platform/__init__.py:L17-L53` | platform 段清单（裸 import = 副作用） | 折叠中段 ~13 条业务 patch import，保留 `is_310p` / EPLB 环境变量两处条件骨架 |
| `patch_worker_init.py` | `vllm_ascend/patch/worker/__init__.py:L18-L70` | worker 段清单 | 折叠 ~10 条无条件 import，保留 `HAS_TRITON` / `vllm_version_is` / `is_310p` / try-ImportError 四处条件骨架 |
| `patch_multiproc_executor.py` | `vllm_ascend/patch/platform/patch_multiproc_executor.py:L8-L211` | 技法①整类替换 | 删 ~180 行从 vLLM 复制的执行器内部实现，留类头 + `daemon=False` 差异 + 末行重绑 |
| `patch_mamba_manager.py` | `vllm_ascend/patch/platform/patch_mamba_manager.py:L7-L53` | 技法②工厂(注册表)替换 | 删 `find_longest_cache_hit` 块命中扫描实现，留类名重绑 + `spec_manager_map[MambaSpec]` 派发表改写 |
| `patch_scheduler.py` | `vllm_ascend/patch/platform/patch_scheduler.py:L1-L45` | 技法③方法替换 | 仅删多行块对齐/eagle 注释，控制流完整保留 + 末行方法绑定 |
| `patch_distributed_platform.py` | `vllm_ascend/patch/platform/patch_distributed.py:L20-L89` | 技法④库函数 wrapper + 技法⑤from-import 缓存陷阱 | 折叠 all_reduce int64 归约体，留 broadcast wrapper 全控制流 + `broadcast`/`distributed_c10d.broadcast` 双绑 + 设备门控 |
| `patch_triton.py` | `vllm_ascend/patch/worker/patch_triton.py:L1-L20` | 技法④极简样本 | 删 `HAS_TRITON=False` 两个纯 PyTorch 回退算子，留 `triton.next_power_of_2 = next_power_of_2` |
| `patch_distributed_worker.py` | `vllm_ascend/patch/worker/patch_distributed.py:L16-L233` | 综合样本（技法①+④+⑤） | 删 HCCL pg 注册表 helper/复用细节，留 `_wrap_destroy_distributed_environment`（@wraps + 幂等标记）、`GroupCoordinatorPatch` 类头 + `all_to_all`、整类重绑 + 同名再导出双绑 |

## 测试映射（tests/test_two_stage_patch.py）

| 测试 | 验证的真实可观察行为 |
|---|---|
| `test_adapt_patch_dispatches_to_platform_or_worker` | `is_global_patch=True/False` → import platform/worker 包（二分入口） |
| `test_ensure_global_patch_is_idempotent` | `_GLOBAL_PATCH_APPLIED` 守卫保证 platform 段多入口触发只 apply 一次 |
| `test_register_connector_triggers_global_patch` | general_plugins 入口先调 `_ensure_global_patch`（ch02 f1 回收） |
| `test_scheduler_method_replacement` | 技法③：`Scheduler._mamba_block_aligned_split` 被换、不建子类 |
| `test_mamba_manager_factory_table_replacement` | 技法②：类名 + `spec_manager_map[MambaSpec]` 派发表同时改 |
| `test_multiproc_executor_whole_class_replacement` | 技法①：模块属性 `MultiprocExecutor` 被子类覆盖 |
| `test_distributed_wrapper_and_cache_trap` | 技法④wrapper 回落原 fn + 技法⑤`broadcast`/`distributed_c10d.broadcast` 双绑 |
| `test_triton_next_power_of_2_rebind` | 技法④：给 triton 模块补 `next_power_of_2` |
| `test_worker_distributed_comprehensive` | 综合：整类替换 + `destroy_distributed_environment` 同名双绑 + 幂等标记 + 新增 `all_to_all` |

运行：`python3 -m pytest tests/ -q`（host 即可，纯 Python 重绑定逻辑；9 passed）。
