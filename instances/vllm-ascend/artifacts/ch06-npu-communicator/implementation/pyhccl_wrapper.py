"""换底座线③的底座：ctypes 绑定 libhccl.so —— pynccl_wrapper 的逐符号移植。

只做减法的忠实精简版。这一层是「不写 C++ 扩展也能直调厂商集合通信库」的关键：
C 类型 → ctypes 别名、torch.dtype/ReduceOp → HCCL 头文件整型常量、exported_functions
逐字照抄 C 函数签名、__init__ 用 CDLL 加载并给每个 C 函数挂 restype/argtypes。
与 vllm/distributed/device_communicators/pynccl_wrapper.py 结构完全同构，只换符号
（NCCL→HCCL、128B unique_id→4108B、枚举值按 hccl_types.h，与 NCCL 不同）。

host 可跑：本文件除被 SUBTRACTED 的 vllm/vllm_ascend 依赖外是纯 ctypes+torch，
类型别名 / 枚举映射 / 函数签名表 / 结构体尺寸都可直接单测（不需真的 libhccl.so）。
"""
# SUBTRACTED: 文件头 Apache-2.0 许可证注释块（原 pyhccl_wrapper.py:L1-L16）
import ctypes
import logging
import platform
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed import ReduceOp

# SUBTRACTED: from vllm.logger import logger —— 基座 logger；host 无 vllm，
#   用 stdlib logging 顶替，仅影响日志，不影响加载/绑定控制流。
logger = logging.getLogger(__name__)


# SUBTRACTED: from vllm_ascend.utils import find_hccl_library —— 定位 libhccl.so 的
#   平台相关 helper；host 无 CANN，给出忠实占位（被 __init__ 当默认实参引用）。
# SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L27 (import)
def find_hccl_library() -> str:
    raise RuntimeError("find_hccl_library: host 无 CANN/HCCL（精简版占位，原为 vllm_ascend.utils）")


# === export types and functions from hccl to Python ===
# for the original hccl definition, please check
# https://github.com/EternalLied/cann-hccl-new/.../hccl.h
# https://github.com/EternalLied/cann-hccl-new/.../hccl_types.h
# SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L34-L45
hcclResult_t = ctypes.c_int
hcclComm_t = ctypes.c_void_p


# SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L38-L39
class hcclUniqueId(ctypes.Structure):
    # 4108 字节 root info —— HCCL 与 NCCL(128B) 唯一的尺寸差异，必须照头文件抄死，不可臆造。
    _fields_ = [("internal", ctypes.c_byte * 4108)]


aclrtStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

hcclDataType_t = ctypes.c_int


# SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L48-L81
class hcclDataTypeEnum:
    # 枚举值序按 hccl_types.h，与 NCCL 不同（如 NCCL float16=6，HCCL float16=3）——
    # 「照搬范式只换符号」里『符号值本身也要换』的具象。常量表逐字保留。
    hcclInt8 = 0
    hcclInt16 = 1
    hcclInt32 = 2
    hcclFloat16 = 3
    hcclFloat32 = 4
    hcclInt64 = 5
    hcclUint64 = 6
    hcclUint8 = 7
    hcclUint16 = 8
    hcclUint32 = 9
    hcclFloat64 = 10
    hcclBfloat16 = 11
    hcclInt128 = 12

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L63-L81
        # SUBTRACTED: int8/uint8/int32/float64 分支（原 L65-L70, L77-L78）——
        #   平行 if 链，删非演示分支不改控制流骨架；保留代表分支（plan 批准）。
        if dtype == torch.int64:
            return cls.hcclInt64
        if dtype == torch.float16:
            return cls.hcclFloat16
        if dtype == torch.float32:
            return cls.hcclFloat32
        if dtype == torch.bfloat16:
            return cls.hcclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


hcclRedOp_t = ctypes.c_int


# SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L87-L103
class hcclRedOpTypeEnum:
    hcclSum = 0
    hcclProd = 1
    hcclMax = 2
    hcclMin = 3

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L93-L103
        # SUBTRACTED: PRODUCT/MIN 分支（原 L97-L98, L101-L102）—— 保留 SUM/MAX 代表分支（plan 批准）。
        if op == ReduceOp.SUM:
            return cls.hcclSum
        if op == ReduceOp.MAX:
            return cls.hcclMax
        raise ValueError(f"Unsupported op: {op}")


# SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L106-L110
@dataclass
class Function:
    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L106-L110
    name: str
    restype: Any
    argtypes: list[Any]


# SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L113-L253
class HCCLLibrary:
    # exported_functions = 「C 函数签名声明表」：每个 Function(name, restype, argtypes)
    # 逐字对应 hccl.h 里一个 C 原型。与 pynccl_wrapper.NCCLLibrary.exported_functions
    # 结构同构，只换函数名 Hccl* 与按 HCCL 头文件写的参数表。
    exported_functions = [
        # const char* HcclGetErrorString(HcclResult code);
        Function("HcclGetErrorString", ctypes.c_char_p, [hcclResult_t]),
        # HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo);
        Function("HcclGetRootInfo", hcclResult_t, [ctypes.POINTER(hcclUniqueId)]),
        # HcclResult HcclCommInitRootInfo(
        #   uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm);
        # note that HcclComm is a pointer type, so the last argument is a pointer to a pointer
        Function(
            "HcclCommInitRootInfo",
            hcclResult_t,
            [
                ctypes.c_int,
                ctypes.POINTER(hcclUniqueId),
                ctypes.c_int,
                ctypes.POINTER(hcclComm_t),
            ],
        ),
        # HcclResult HcclAllReduce(
        #   void *sendBuf, void *recvBuf, uint64_t count,
        #   HcclDataType dataType, HcclReduceOp op, HcclComm comm,
        #   aclrtStream stream);
        Function(
            "HcclAllReduce",
            hcclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                hcclDataType_t,
                hcclRedOp_t,
                hcclComm_t,
                aclrtStream_t,
            ],
        ),
        # HcclResult HcclBroadcast(
        #   void *buf, uint64_t count,
        #   HcclDataType dataType, uint32_t root,
        #   HcclComm comm, aclrtStream stream);
        Function(
            "HcclBroadcast",
            hcclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                hcclDataType_t,
                ctypes.c_int,
                hcclComm_t,
                aclrtStream_t,
            ],
        ),
        # HcclResult HcclCommDestroy(HcclComm comm);
        Function("HcclCommDestroy", hcclResult_t, [hcclComm_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: dict[str, Any] = {}

    # class attribute to store the mapping from library path
    # to the corresponding directory
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L177-L208
    def __init__(self, so_file: str | None = None):
        so_file = so_file or find_hccl_library()

        try:
            if so_file not in HCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                HCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = HCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            # SUBTRACTED: 原 logger.error 的多行排错提示文案（L186-L197）—— 纯日志，
            #   压成单行不影响加载控制流（plan 批准）。
            logger.error("Failed to load HCCL library. so_file=%s, error=%s, platform=%s", so_file, e, platform.platform())
            raise e

        if so_file not in HCCLLibrary.path_to_dict_mapping:
            _funcs: dict[str, Any] = {}
            for func in HCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            HCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = HCCLLibrary.path_to_dict_mapping[so_file]

    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L210-L211
    def hcclGetErrorString(self, result: hcclResult_t) -> str:
        return self._funcs["HcclGetErrorString"](result).decode("utf-8")

    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L213-L216
    def HCCL_CHECK(self, result: hcclResult_t) -> None:
        if result != 0:
            error_str = self.hcclGetErrorString(result)
            raise RuntimeError(f"HCCL error: {error_str}")

    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L218-L221
    def hcclGetUniqueId(self) -> hcclUniqueId:
        unique_id = hcclUniqueId()
        self.HCCL_CHECK(self._funcs["HcclGetRootInfo"](ctypes.byref(unique_id)))
        return unique_id

    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L223-L228
    def hcclCommInitRank(self, world_size: int, unique_id: hcclUniqueId, rank: int) -> hcclComm_t:
        comm = hcclComm_t()
        self.HCCL_CHECK(
            self._funcs["HcclCommInitRootInfo"](world_size, ctypes.byref(unique_id), rank, ctypes.byref(comm))
        )
        return comm

    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L230-L245
    def hcclAllReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: hcclComm_t,
        stream: aclrtStream_t,
    ) -> None:
        # `datatype` actually should be `hcclDataType_t`
        # and `op` should be `hcclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.HCCL_CHECK(self._funcs["HcclAllReduce"](sendbuff, recvbuff, count, datatype, op, comm, stream))

    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L247-L250
    def hcclBroadcast(
        self, buf: buffer_type, count: int, datatype: int, root: int, comm: hcclComm_t, stream: aclrtStream_t
    ) -> None:
        self.HCCL_CHECK(self._funcs["HcclBroadcast"](buf, count, datatype, root, comm, stream))

    # SOURCE: vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py:L252-L253
    def hcclCommDestroy(self, comm: hcclComm_t) -> None:
        self.HCCL_CHECK(self._funcs["HcclCommDestroy"](comm))


__all__ = [
    "HCCLLibrary",
    "hcclDataTypeEnum",
    "hcclRedOpTypeEnum",
    "hcclUniqueId",
    "hcclComm_t",
    "aclrtStream_t",
    "buffer_type",
]
