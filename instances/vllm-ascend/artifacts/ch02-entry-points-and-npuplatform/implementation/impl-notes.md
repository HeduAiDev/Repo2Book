# ch02 精简版 impl-notes —— entry points 与 NPUPlatform

**只做减法**：每个 def/class 与规范源码同名/同结构/同控制流；删除处全标 `# SUBTRACTED:`。
昇腾真实运行依赖 torch_npu/CANN（host 无），故精简版只验**纯 Python 控制流**（平台
发现/选择、qualname 解析、懒加载单例、工厂钩子查表、设备分代分流）。真正的 import
那一刻（`resolve_obj_by_qualname` → `import vllm_ascend.platform` 连带 torch_npu）在真机才发生。

## 跨文件 import 适配（唯一结构性改写）
真实源码分布在 `vllm/*` 与 `vllm_ascend/*` 两个包，用 `from vllm.platforms import …` 形式互引。
精简版为独立可跑，落成本目录的**扁平模块**，import 改写为扁平名（如 `from vllm_plugins import
load_plugins_by_group`）。**符号名、控制流、字面量全部不变**；每处改写在文件头/行内注明真实写法。

## 1:1 Source Map

| 精简版（implementation/） | 规范源码 `vllm/…` · `vllm_ascend/…` | 改动 | 原因 |
|---|---|---|---|
| `vllm_import_utils.py` :: `resolve_obj_by_qualname` | vllm/utils/import_utils.py:L104-L110 | 逐字 | qualname→类对象解析点，无可删 |
| `vllm_plugins.py` :: `load_plugins_by_group` | vllm/plugins/__init__.py:L28-L66 | 删全部 logging；`envs.VLLM_PLUGINS`→`None` | 日志纯诊断；白名单为 None=全部加载，host 无 vllm.envs |
| `vllm_platforms.py` :: `cuda/rocm/tpu/xpu/cpu_platform_plugin` | vllm/platforms/__init__.py:L60-L200 | 探测体→`return None` 占位 | 删除计划批准项；host 无对应硬件，builtin 全不激活，让 OOT(ascend) 成唯一激活者 |
| `vllm_platforms.py` :: `resolve_current_platform_cls_qualname` | vllm/platforms/__init__.py:L212-L252 | 删 logger.info/debug | 诊断输出；chain→func()→集合交集→elif(OOT优先) 控制流不变 |
| `vllm_platforms.py` :: `__getattr__` / `_current_platform` | vllm/platforms/__init__.py:L257-L284 | 删 `_init_trace=format_stack()` | 仅调试回溯，与懒加载-缓存-单例语义无关；注释逐字保留（官方‘为何懒加载’解释） |
| `vllm_interface.py` :: `PlatformEnum` / `AttentionBackendEnum` | vllm/platforms/interface.py:L38-L47 · v1/attention/backends/registry.py:L34-L44 | AttentionBackendEnum 只留 FLASH_ATTN | 本章只用 FLASH_ATTN 做 get_attn_backend_cls 特例判定 |
| `vllm_interface.py` :: `Platform`（基类钩子默认值） | vllm/platforms/interface.py:L105-L890 | 只留 is_out_of_tree + 5 个 get_*_cls 默认 + 签名 | 作 NPUPlatform 覆盖的对照；需真硬件的方法体全 SUBTRACTED |
| `vllm_ascend_init.py` :: `register` | vllm_ascend/__init__.py:L40-L43 | 逐字 | **本章核心**：返回 qualname 字符串、不 import 的纯回调 |
| `vllm_ascend_init.py` :: `_ensure_global_patch` / `register_connector/…` | vllm_ascend/__init__.py:L20-L75 | 删 adapt_patch 与 4 个下游真注册 | 下游依赖 torch_npu（ch03 主题）；保留‘先打 patch 再注册’骨架与幂等标志 |
| `vllm_ascend_platform.py` :: `NPUPlatform`（类属性 + 工厂钩子） | vllm_ascend/platform.py:L134-L820 | 留身份属性 + 6 个 get_*_cls return + worker_cls 改写片段 | 钩子 return 纯字符串可跑；需真 NPU 的方法体 SUBTRACTED |
| `vllm_ascend_platform.py` :: `get_attn_backend_cls` / `_validate_fa3_backend` | vllm_ascend/platform.py:L738-L793 | `_validate_fa3_backend`→`return False` 占位 | FA3 依赖外部包 flash_attn_npu_v3（host 无）；保留 backend_map/310 查表主干 |
| `vllm_ascend_platform.py` :: `check_and_update_config` | vllm_ascend/platform.py:L602-L612 | 只留 worker_cls 'auto'→qualname 改写块 | 其余配置改写属 ch10/config；`init_ascend_config` 降为读 vllm_config.ascend_config |
| `vllm_ascend_utils.py` :: `AscendDeviceType`/`get_ascend_device_type`/`is_310p` | vllm_ascend/utils.py:L122-L123,L768-L818 | `check_ascend_device_type` 的 `torch_npu.get_soc_version()`→入参 | host 无 torch_npu；保留 soc_version→分代区间映射表（可单测） |

## 验收
- `python3 -m pytest tests/test_ch02.py -q` → 31 passed。
- `python3 scripts/lint_fidelity.py <chapter_dir>` → 保真度全部通过（must_keep 24 符号全在）。

## 给 writer 的提示
- `current_platform` 懒加载端到端：精简版用扁平 qualname `vllm_ascend_platform.NPUPlatform`
  让 `resolve_obj_by_qualname` 在 host 可解析（见 `test_current_platform_lazy_singleton_*`）；
  真机 `register()` 返回的真实字符串 `vllm_ascend.platform.NPUPlatform` 才触发 torch_npu import。
  叙事请以真实字符串为准，扁平名只是 host 跑通的替身。
- `_init_ascend_device_type` 的 `from vllm_ascend import _build_info` 为构建期烙印产物，
  host 无该模块；测试改为直接设 `_ascend_device_type`（模拟烙印）来驱动 is_310p / worker_cls 分流。
