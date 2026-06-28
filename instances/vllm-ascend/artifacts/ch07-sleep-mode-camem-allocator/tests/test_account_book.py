"""验证账本（pointer_to_data）+ tag 流 + get_current_usage —— 纯 Python，host 可跑。

对位真实行为 vllm_ascend/device_allocator/camem.py：
  - python_malloc_callback 用设备虚拟地址 handle[2] 作 key 记账，并贴 current_tag；
  - python_free_callback 弹账本项、清 CPU 备份引用、把 handle 交还；
  - get_current_usage 累加每个 handle[1]（对齐字节数）。
"""
import camem
from camem import AllocationData, CaMemAllocator, HandleType


def _fresh():
    # 直接构造（非 get_instance）以隔离每个用例的账本；__init__ 仅校验 env 并初始化空账本。
    return CaMemAllocator()


def test_handle_type_is_four_tuple():
    # HandleType = (py_device, py_alignedSize, py_d_mem, py_p_memHandle)
    assert HandleType == tuple[int, int, int, int]


def test_malloc_callback_records_with_current_tag():
    a = _fresh()
    a.current_tag = "weights"
    handle = (0, 1024, 0x1000, 0xAAA)
    a.python_malloc_callback(handle)
    # key 是设备虚拟地址 handle[2]
    assert 0x1000 in a.pointer_to_data
    data = a.pointer_to_data[0x1000]
    assert isinstance(data, AllocationData)
    assert data.handle == handle
    assert data.tag == "weights"
    assert data.cpu_backup_tensor is None


def test_get_current_usage_sums_aligned_sizes():
    a = _fresh()
    a.current_tag = "weights"
    a.python_malloc_callback((0, 1024, 0x1000, 0xAAA))
    assert a.get_current_usage() == 1024
    a.current_tag = "kv_cache"
    a.python_malloc_callback((0, 2048, 0x2000, 0xBBB))
    assert a.get_current_usage() == 1024 + 2048


def test_free_callback_pops_and_returns_handle():
    a = _fresh()
    handle = (0, 1024, 0x1000, 0xAAA)
    a.python_malloc_callback(handle)
    a.pointer_to_data[0x1000].cpu_backup_tensor = object()  # 模拟有 CPU 备份
    returned = a.python_free_callback(0x1000)
    assert returned == handle
    assert 0x1000 not in a.pointer_to_data  # 已弹出
    assert a.get_current_usage() == 0


def test_get_instance_is_singleton():
    CaMemAllocator.instance = None  # 复位单例供本用例独立验证
    i1 = CaMemAllocator.get_instance()
    i2 = CaMemAllocator.get_instance()
    assert i1 is i2


def test_camem_disabled_on_host():
    # host 无 vllm_ascend_C 扩展 → 导入失败降级（这是真实「禁用 sleep mode」路径）。
    assert camem.camem_available is False
    assert camem.init_module is None
    assert camem.python_create_and_map is None
    assert camem.python_unmap_and_release is None


def test_find_loaded_library_returns_none_for_absent():
    assert camem.find_loaded_library("definitely_not_loaded_lib_xyz") is None
