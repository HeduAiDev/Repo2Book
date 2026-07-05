# ch14 精简版 impl-notes —— NPUModelRunner 设备层猴补 + 图捕获接缝

只做减法的忠实子集。host 无 NPU/CANN，真实 torch.npu/ACLGraph 不真跑；测试在
`sys.modules` 注入 torch.npu / vllm / vllm_ascend 桩，验证纯 Python 的"装/卸"与决策控制流。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版符号 | 真实源码 (规范路径:行) | 改动 | 原因 |
|---|---|---|---|
| `GPUModelRunner`（被继承父类） | vllm/v1/worker/gpu_model_runner.py:L1056-L1064 / L6075-L6106 | 仅 import + 对照 | 244KB 父类；override 钩子标注 + 散落 torch.cuda.* / graph_capture / CUDAGraphWrapper 即猴补目标 |
| `GraphCaptureContext` | vllm_ascend/worker/model_runner_v1.py:L196-L198 | 原样 | 与父类同形数据类（stream 字段），热替换依据 |
| `graph_capture` (NPU版) | vllm_ascend/worker/model_runner_v1.py:L201-L230 | 删 docstring 大段（L204-L216） | 纯注释，控制流不变；torch.npu.Stream + wait_stream + with torch.npu.stream 全保留 |
| `NPUModelRunner.__init__` | vllm_ascend/worker/model_runner_v1.py:L256-L490 | 删 PCP/DCP/EPLB/sparse/multimodal/offloader/buffer 等正交字段初始化 | subtraction_plan.delete；保留 `with _torch_cuda_wrapper(): super().__init__()` 与 sampler/attn_state/use_aclgraph 三处替换 |
| `_init_device_properties` | vllm_ascend/worker/model_runner_v1.py:L580-L581 | 原样 | override 设备钩子①（num_sms=None） |
| `_sync_device` | vllm_ascend/worker/model_runner_v1.py:L583-L584 | 原样 | override 设备钩子②（torch.npu.synchronize） |
| `_use_aclgraph` | vllm_ascend/worker/model_runner_v1.py:L620-L625 | 原样 | 三条件决策 |
| `profile_cudagraph_memory` | vllm_ascend/worker/model_runner_v1.py:L4798-L4818 | 删 profiling 善后清 KV cache + gc（L4805-L4816） | subtraction_plan.delete；保留双 wrapper 包父方法 + reset_graph_params |
| `capture_model` | vllm_ascend/worker/model_runner_v1.py:L4820-L4824 | 原样 | 双 wrapper 包父方法范例 |
| `_get_gpu_model_runner_module_name` | vllm_ascend/worker/model_runner_v1.py:L4876-L4887 | 原样 | MRO 取父类模块名（为何改父模块而非本模块） |
| `_torch_cuda_wrapper` | vllm_ascend/worker/model_runner_v1.py:L4890-L4931 | 保留 Event/Stream/synchronize/mem_get_info 四个代表符号，删 default_stream/current_stream/stream | subtraction_plan.delete；try/except/finally 三分支与"成对装卸"骨架全保留 |
| `_replace_gpu_model_runner_function_wrapper` | vllm_ascend/worker/model_runner_v1.py:L4934-L4953 | 原样 | original_attrs 存值 + setattr 替换 + finally 还原 |
| `ACLGraphWrapper` (acl_graph.py) | vllm_ascend/compilation/acl_graph.py:L64-L153 | 删捕获/重放（__call__ 内部、__getattr__/unwrap 等） | 本章仅需同形可热替换（构造签名/_all_instances/clear_all_graphs/clear_graphs），捕获细节留图模式章节 |
| `reset_graph_params` (acl_graph.py) | vllm_ascend/compilation/acl_graph.py:L333-L337 | 原样 | profile 后被调用清图参数 |

## must_keep 核对（subtraction_plan.must_keep 全部在场）
NPUModelRunner / GPUModelRunner / _torch_cuda_wrapper / _replace_gpu_model_runner_function_wrapper /
_get_gpu_model_runner_module_name / graph_capture / GraphCaptureContext / ACLGraphWrapper /
_use_aclgraph / _init_device_properties / _sync_device / capture_model / profile_cudagraph_memory /
AscendSampler / AscendAttentionState / mem_get_info —— 均保留可检测。

## 验收
- `python3 -m pytest tests/` → 18 passed（纯 Python 装卸/决策/MRO/同形）。
- `python3 scripts/lint_fidelity.py <chapter_dir>` → 无 BLOCKING。
