# ch07 精简版实现笔记 —— sleep-mode 与 camem 分配器

只做减法的忠实精简版。主角 `camem.py` 与真实 `vllm_ascend/device_allocator/camem.py`
逐行同构（它本身又是 vLLM `vllm/device_allocator/cumem.py` 的「同构换符号」移植）。

本章解读的真实源码（规范路径）：
- `vllm_ascend/device_allocator/camem.py`（主角：CaMemAllocator + create_and_map/unmap_and_release）
- `vllm_ascend/patch/platform/patch_camem_allocator.py`（把 camem 挂进 vLLM 的 fallback 写法）
- `vllm_ascend/worker/worker.py`（调用方旁证：load_model / sleep / wake_up 的 tag 驱动）
- `vllm/device_allocator/cumem.py`（对照基座：逐行对位的 GPU 原版）

## 验收判据
把真实 `camem.py` 删掉所有 `# SUBTRACTED` 标注的行（许可证头 + 两个无法在 host 导入的
模块级符号），应当 ≈ 得到本精简版。控制流一字未改。

## host 可读可跑边界
- **可跑（纯 Python 状态机）**：单例、`pointer_to_data` 账本、`current_tag` 打标签、
  malloc/free 回调、`use_memory_pool` 的 tag set/restore、`sleep`/`wake_up` 的
  offload/discard 路由、`get_current_usage`。测试用 monkeypatch 把 NPU/CANN 原语换成记录器。
- **不跑（需 NPU/CANN）**：真正的虚拟内存映射 `python_create_and_map`/`python_unmap_and_release`
  与 `aclrtMemcpy`。host 无扩展 → `camem_available=False`（这正是真实「禁用 sleep mode」降级路径）。

## Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| `camem.py` 文件头注释 | `camem.py:L1-L18` | 删除 | Apache 许可证块 + 顶部注释，非代码逻辑（零脚手架/噪声） |
| `camem.py` `def memcpy(...)` 占位 | `camem.py:L27` `from acl.rt import memcpy` | 换符号占位 | host 无 CANN；保留 must_keep 符号 `memcpy`，sleep/wake 按原样调用，测试 monkeypatch 驱动 |
| `camem.py` `import logging; logger=...` | `camem.py:L28` `from vllm.logger import logger` | 换 stdlib | host 无 vllm，用标准库 logging 顶替（与 ch06 同约定） |
| `camem.py` `libcudart = None` | `camem.py:L72` | **原样保留** | 移植自 vLLM 的死变量（全文无引用），保留以如实展示「换符号」未清理的残留 |
| `camem.py` `find_loaded_library`…`get_current_usage` | `camem.py:L31-L273` | 逐字一致 | 主线全保留：try/except 条件导入、HandleType、AllocationData、create_and_map/unmap_and_release、pluggable allocator、CaMemAllocator 全方法 |
| `patch_camem_allocator.py` 可选 import | `patch_camem_allocator.py:L17` 硬 import | try/except 降级 | host 无 vllm；缺失时 `model_config_module=None`，hasattr 守护天然 no-op；逻辑保留 |
| `patch_camem_allocator.py` 守护行多 `is not None` | `patch_camem_allocator.py:L27-L28` | 加前置 None 守护 | 仅因上方硬 import 被降级，不改语义（hasattr 守护 fallback 不变） |
| `worker_excerpt.py` NPUWorker 三方法 | `worker.py:L200-L226, L544-L555, L762-L773` | 摘录 | 调用方旁证：load_model(tag='weights') / initialize_from_config(tag='kv_cache') / sleep(offload_tags 两档) / wake_up；方法体非 camem 的日志/权重格式细节按 dossier 计划 SUBTRACTED |

## 「移植=同构换符号」核对（camem vs cumem）
- C 扩展：`vllm.cumem_allocator` → `vllm_ascend.vllm_ascend_C`
- 拷贝原语：`libcudart.cudaMemcpy(dst,src,count)` → `acl.rt.memcpy(dst,destMax,src,count,kind)`
  （CANN 强制 destMax 上界 + 显式方向枚举 D2H=2/H2D=1，destMax 给 size*2 宽松上界）
- pluggable allocator：`CUDAPluggableAllocator` → `NPUPluggableAllocator`；`torch.cuda.*` → `torch.npu.*`
- env 名：`PYTORCH_CUDA_ALLOC_CONF` → `PYTORCH_NPU_ALLOC_CONF`
- 结构性差异（非换符号，移植时点早于上游演进）：`__init__` 期硬 assert expandable_segments
  （vLLM 后改成 use_memory_pool 内动态开关）；vLLM 有 strong-ref 回调 / snapshot 释放零分配，ascend 无。
