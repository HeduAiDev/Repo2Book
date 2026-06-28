# ch06 精简版实现笔记（只做减法）

四条线 + 注入点①，逐文件对照真实源码。host 无 NPU/CANN：实际集合通信不跑；
ctypes 绑定范式 / disabled 降级 / 310P all_gather 模拟 / qualname 回调是纯 Python，host 可单测。

## 1:1 Source Map

| 精简版 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| `platform_injection.py::_BasePlatform.get_device_communicator_cls` | `vllm/platforms/interface.py:L769-L774` | SUBTRACTED 基类其余类体 | 只看「换通信器」一个 classmethod（基座默认返回 DeviceCommunicatorBase qualname） |
| `platform_injection.py::NPUPlatform.get_device_communicator_cls` | `vllm_ascend/platform.py:L803-L805` | SUBTRACTED NPUPlatform 其余覆写 | 注入点①：覆写返回 NPUCommunicator qualname，一个字符串换底座 |
| `platform_injection.py::_resolve_device_communicator` | `vllm/distributed/parallel_state.py:L370-L381` | SUBTRACTED GroupCoordinator.__init__ 其余初始化 + 顶部 vllm import | resolve_obj_by_qualname 字符串→类→统一签名实例化（4 kwargs，后两参 global_* 走默认） |
| `npu_communicator.py::NPUCommunicator` | `vllm_ascend/distributed/device_communicators/npu_communicator.py:L23-L68` | 仅删许可证头 | 注入点②：子类化基类，只动 __init__（device/ca_comm）+ 新增 all_to_all |
| `pyhccl_wrapper.py::hcclUniqueId / hcclDataTypeEnum / hcclRedOpTypeEnum` | `…/pyhccl_wrapper.py:L38-L103` | from_torch 删非演示 dtype/op 分支（plan 批准） | 4108B root info、枚举值按 hccl_types.h（与 NCCL 不同），常量表逐字保留 |
| `pyhccl_wrapper.py::HCCLLibrary` | `…/pyhccl_wrapper.py:L113-L253` | __init__ logger.error 多行→单行；vllm/vllm_ascend import→stdlib/占位 | exported_functions C 签名表 + CDLL 加载 + restype/argtypes 绑定，对位 NCCLLibrary |
| `pyhccl.py::PyHcclCommunicator` | `…/pyhccl.py:L37-L171` | 删 __init__ docstring；StatelessProcessGroup/logger/current_stream import→占位 | ③范式样本：unique_id 建组 / CommInitRank / warmup / disabled 降级，对位 pynccl |
| `patch_distributed.py::communication_adaptation_310p` | `vllm_ascend/patch/platform/patch_distributed.py:L33-L85` | 删 c10d 第二次镜像赋值 ×2、all_reduce MAX numpy 分支、import 期 310P 守卫（均 plan 批准 / 避免全局猴补） | ④仅 310P：broadcast/int64 all_reduce → all_gather 模拟，复用 ch03 两段式技法 |

## 保真要点（写作须知）
- **all_to_all 是「新增」不是「重写」**：基座 DeviceCommunicatorBase 与 CudaCommunicator 都无 all_to_all。「子类化=只改差异点」仍成立（差异点 = 新增 all_to_all + __init__ 微调，其余全继承）。
- **pyhccl 当前未接入 NPUCommunicator**（npu_communicator.py:L32 TODO，全仓仅 tests/ut 引用）。NPU 集合通信实际走基类 `dist.*(group=device_group)` + 进程组 HCCL backend，不经过 pyhccl。pyhccl 是 pynccl ctypes 范式的移植样本/预留件。
- **构造签名口径**：基类 __init__ 实有 6 参（含 global_ranks/global_world_size 默认 None）；调用点（parallel_state）与 NPUCommunicator.super().__init__ 只传前 4 个。勿说「基类构造器只有 4 个参数」。
- **310P 守卫**：仅 `get_ascend_device_type()==AscendDeviceType._310P` 触发，A2/A3/A5 不补。精简版把 import 期守卫 SUBTRACTED（避免非昇腾 host 全局猴补），改由测试显式调 `communication_adaptation_310p()`。

## host 可跑范围
- `pyhccl_wrapper`：枚举/结构体尺寸/函数签名表/HCCL_CHECK —— `test_pyhccl_wrapper.py`（8 例）
- `pyhccl` disabled 降级（单卡 / 缺库）—— `test_pyhccl_disabled.py`（4 例）
- `patch_distributed` all_gather 模拟 + 降级直通（gloo 单进程组）—— `test_patch_distributed.py`（5 例）
- `platform_injection` qualname 回调 —— `test_platform_injection.py`（3 例）
- `npu_communicator` 结构断言（AST，不可 import：依赖 vllm 基座 + torch.npu）—— `test_npu_communicator_structure.py`（4 例）

22 例全过；`lint_fidelity` 无 BLOCKING。
