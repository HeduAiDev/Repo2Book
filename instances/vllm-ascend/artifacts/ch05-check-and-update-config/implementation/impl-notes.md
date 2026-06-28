# ch05 精简版实现笔记 — check_and_update_config + AscendConfig 配置面

## 运行方式
host 纯 Python（无需 NPU/CANN）：
```
cd instances/vllm-ascend/artifacts/ch05-check-and-update-config
python3 -m pytest tests/ -q          # 30 passed
```
本章两条主线（配置改写 cascade reset / 三级取值 + 开放 dict 解析）都是纯 dict/getattr 逻辑，可在 host 跑通并打断点。

## 设计：忠实子集如何在无 vLLM 的 host 上跑
真源码 import 大量 vLLM/CANN 符号（logger、CompilationMode/CUDAGraphMode、AttentionBackendEnum、is_310p、refresh_block_size、init_ascend_config…）。精简版把**与本章控制流无关的外部符号**收进 `_support.py`，每个都标 `# SOURCE:` 指向真实定义、`# SUBTRACTED:` 说明删了什么。被解读的主线代码（`platform.py` / `ascend_config.py` / `envs.py`）逐行对应真源码。VllmConfig 用 `SimpleNamespace` 充当输入夹具（真源码全程 `getattr` 鸭子取值，故无需真实类型）——夹具在测试里，不是杜撰的源码逻辑。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）
| 精简版位置 | 真实源码 | 改动 | 原因 |
| --- | --- | --- | --- |
| `platform.py` `NPUPlatform.check_and_update_config` | `vllm_ascend/platform.py:L413-L714` | 保留编排骨架：守卫早退 / `_validate_*` / `_fix_incompatible_config` / `init_ascend_config` / enforce_eager 编译改写 / worker_cls 落定 / PYTORCH_NPU_ALLOC_CONF；删 kv_transfer 补丁、ascend_compilation/fusion 回写、cudagraph splitting_ops 分支、SP sizes 重算、PD/sparse 互斥 raise、custom_ops 等周边 | 删减计划批准：这些是别章子系统钩入点；本章只讲『平台=配置改写器』主骨架 |
| `platform.py` `NPUPlatform._fix_incompatible_config` | `vllm_ascend/platform.py:L979-L1180` | 保留段1(disable_cascade_attn)/段2(cpu_kvcache)/段4(nvtx)/段5(partial_prefills)/段8(force_false_flags+backend+splits)/段9(numa_bind 改写+numa_bind_nodes)；删段3/6/7 与段9 的 ray_nsight/numa_bind_cpus/enable_dbo/ubatch_size | 删项与保留段同构（getattr→warn→写回安全值）；保留 numa_bind 这唯一『改写非丢弃』特例 |
| `platform.py` `_validate_layer_sharding_config` / `_validate_draft_decode_context_parallel_config` | `vllm_ascend/platform.py` 同名方法 | 退化为 no-op，仅保留调用点 | 一致性校验非本章主线；保留调用点以忠实 check_and_update_config 控制流 |
| `ascend_config.py` `AscendConfig.__init__` | `vllm_ascend/ascend_config.py:L32-L283` | 保留 xlite/ascend_compilation/eplb 三个子配置解析 + 两个 `_get_config_value` 标量；删其余十余个同范式子配置与交叉校验 | 保留代表演示『开放 dict→强类型』；保留 eplb_config 因单例守卫探测它 |
| `ascend_config.py` `AscendConfig._get_config_value` | `vllm_ascend/ascend_config.py:L284-L296` | 逐字保留 | 三级取值核心（additional_config 是否压过已塌缩的 env_value），must_keep |
| `ascend_config.py` `AscendCompilationConfig.__init__` | `vllm_ascend/ascend_config.py:L498-L538` | 保留命名形参 + `**kwargs` 后门 + static_kernel assert；删长 docstring | 强类型子配置 + 无 schema 后门范例，must_keep |
| `ascend_config.py` `init_ascend_config`/`_is_ascend_config_initialized`/`get_/clear_ascend_config` | `vllm_ascend/ascend_config.py:L775-L819` | 逐字保留（clear 删 enable_sp 清理一行） | 进程级懒加载单例语义（refresh/同一 vllm_config/完整性守卫），must_keep |
| `envs.py` `env_variables` + `__getattr__`/`__dir__` | `vllm_ascend/envs.py:L30-L126` | 保留 4 项代表（含 must_keep 的两项被三级取值消费的开关）+ 懒求值；删其余约三十项 | env+default 在 lambda 内塌缩的实现处，must_keep；其余同范式重复 |
| `_support.py` | logger / CompilationMode / CUDAGraphMode / AttentionBackendEnum / AscendDeviceType / is_310p / get_ascend_device_type / refresh_block_size | 最小可运行替身，各标真实 `# SOURCE:` | 让上述主线代码在无 vLLM/CANN 的 host 上按同一控制流跑通 |

## 文件名冲突注意
精简版忠实保留 `platform.py` 文件名（= 真源码 `vllm_ascend/platform.py`），与 stdlib `platform` 同名。测试用 `importlib.util.spec_from_file_location` 按路径加载以避开 `sys.modules` 缓存冲突。
