"""验证 ctypes 绑定层的可观察行为（对位 pynccl_wrapper，纯 Python，host 可跑）。

测的是「真实源码记录的行为」，不是精简版自洽：
- hcclUniqueId 是 4108 字节（HCCL root info，与 NCCL 128B 的关键差异）
- 枚举值按 hccl_types.h（float16=3、int64=5、SUM=0、MAX=2 等），from_torch 正确翻译
- exported_functions 逐字对应 hccl.h 的 C 原型（HcclCommInitRootInfo 末参是 HcclComm*
  即指针的指针 → POINTER(hcclComm_t)；HcclAllReduce 7 参）
- HCCL_CHECK 非零返回码抛错
"""
import ctypes

import pytest
import torch

import pyhccl_wrapper as w


def test_hccl_unique_id_is_4108_bytes():
    # HCCL root info = 4108 字节；NCCL 是 128 字节，这是两者唯一的尺寸差异，必须照头文件抄死。
    assert ctypes.sizeof(w.hcclUniqueId) == 4108
    uid = w.hcclUniqueId()
    assert len(bytes(uid.internal)) == 4108


def test_type_aliases_are_ctypes():
    assert w.hcclResult_t is ctypes.c_int
    assert w.hcclComm_t is ctypes.c_void_p
    assert w.aclrtStream_t is ctypes.c_void_p
    assert w.buffer_type is ctypes.c_void_p


def test_datatype_enum_values_follow_hccl_header():
    # 枚举值序按 HCCL 头文件，与 NCCL 不同（NCCL float16=6，HCCL float16=3）。
    assert w.hcclDataTypeEnum.hcclFloat16 == 3
    assert w.hcclDataTypeEnum.hcclFloat32 == 4
    assert w.hcclDataTypeEnum.hcclInt64 == 5
    assert w.hcclDataTypeEnum.hcclBfloat16 == 11
    assert w.hcclDataTypeEnum.hcclInt128 == 12


def test_datatype_from_torch_kept_branches():
    f = w.hcclDataTypeEnum.from_torch
    assert f(torch.int64) == 5
    assert f(torch.float16) == 3
    assert f(torch.float32) == 4
    assert f(torch.bfloat16) == 11
    with pytest.raises(ValueError):
        f(torch.complex64)


def test_redop_enum_and_from_torch():
    assert w.hcclRedOpTypeEnum.hcclSum == 0
    assert w.hcclRedOpTypeEnum.hcclMax == 2
    g = w.hcclRedOpTypeEnum.from_torch
    from torch.distributed import ReduceOp
    assert g(ReduceOp.SUM) == 0
    assert g(ReduceOp.MAX) == 2


def test_exported_functions_signature_table():
    by_name = {fn.name: fn for fn in w.HCCLLibrary.exported_functions}
    # 七个 C 原型逐字照抄（含错误串/建组/集合通信/销毁）。
    assert set(by_name) == {
        "HcclGetErrorString",
        "HcclGetRootInfo",
        "HcclCommInitRootInfo",
        "HcclAllReduce",
        "HcclBroadcast",
        "HcclCommDestroy",
    }
    # HcclCommInitRootInfo 末参 = HcclComm*（指针的指针），对应 hcclCommInitRank 里 byref(comm)。
    init = by_name["HcclCommInitRootInfo"]
    assert init.argtypes[-1] == ctypes.POINTER(w.hcclComm_t)
    assert init.argtypes[1] == ctypes.POINTER(w.hcclUniqueId)
    # HcclAllReduce: sendBuf, recvBuf, count, dtype, op, comm, stream = 7 参。
    assert len(by_name["HcclAllReduce"].argtypes) == 7
    # HcclGetRootInfo 取 root info：参数是 HcclRootInfo*。
    assert by_name["HcclGetRootInfo"].argtypes == [ctypes.POINTER(w.hcclUniqueId)]


def test_hccl_check_raises_on_nonzero():
    lib = w.HCCLLibrary.__new__(w.HCCLLibrary)  # 绕过 __init__（无 libhccl.so）
    lib._funcs = {"HcclGetErrorString": lambda code: b"boom"}
    lib.HCCL_CHECK(0)  # 0 = 成功，不抛
    with pytest.raises(RuntimeError, match="HCCL error: boom"):
        lib.HCCL_CHECK(1)


def test_init_without_library_raises():
    # host 无 CANN：find_hccl_library 占位会 raise，HCCLLibrary() 透传异常。
    with pytest.raises(Exception):
        w.HCCLLibrary()
