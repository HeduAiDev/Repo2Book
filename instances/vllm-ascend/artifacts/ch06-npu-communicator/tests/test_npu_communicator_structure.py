"""验证注入点②的核心论点「子类化 = 只改差异点」（must_keep: NPUCommunicator /
DeviceCommunicatorBase / all_to_all / ca_comm）。

npu_communicator.py 依赖 vllm 基座与 torch.npu，host 无法 import 实例化；这里用 AST
对精简版源做结构断言（这正是 dossier 记录的真实事实，非精简版自洽）：
- NPUCommunicator 直接继承 DeviceCommunicatorBase
- 整个类只定义 __init__ 与 all_to_all（其余集合通信全靠继承，零重写）—— 差异点唯一
- __init__ 设 self.device=torch.npu.current_device() 且 self.ca_comm=None（graph capture 接口占位）
- all_to_all 控制流 = tensor_split / split → dist.all_to_all(group=device_group) → cat
- 基类无 all_to_all（严格说是「新增」而非「重写」），all_to_all 不是 override 同名方法
"""
import ast
from pathlib import Path

SRC = (Path(__file__).resolve().parent.parent
       / "implementation" / "npu_communicator.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def _classdef(name):
    return next(n for n in ast.walk(TREE)
               if isinstance(n, ast.ClassDef) and n.name == name)


def test_subclasses_base_device_communicator():
    cls = _classdef("NPUCommunicator")
    bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", None) for b in cls.bases]
    assert "DeviceCommunicatorBase" in bases


def test_only_two_methods_init_and_all_to_all():
    cls = _classdef("NPUCommunicator")
    methods = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
    # 「只改差异点」：唯有 __init__（微调）与 all_to_all（新增），其余集合通信全继承。
    assert methods == ["__init__", "all_to_all"]


def test_init_sets_npu_device_and_ca_comm_placeholder():
    src = SRC
    assert "super().__init__(cpu_group, device, device_group, unique_name)" in src
    assert "self.device = torch.npu.current_device()" in src
    assert "self.ca_comm = None" in src  # graph capture 接口对齐占位


def test_all_to_all_control_flow():
    src = SRC
    # 均分走 tensor_split，不均分走 split + scatter_sizes/gather_sizes 预分配。
    assert "torch.tensor_split(input_, self.world_size, scatter_dim)" in src
    assert "torch.split(input_, scatter_sizes, scatter_dim)" in src
    # 核心原语仍是 dist.all_to_all(group=self.device_group) —— 基类范式的延伸，不碰底层 HCCL。
    assert "dist.all_to_all(output_list, input_list, group=self.device_group)" in src
    assert "torch.cat(output_list, dim=gather_dim)" in src
