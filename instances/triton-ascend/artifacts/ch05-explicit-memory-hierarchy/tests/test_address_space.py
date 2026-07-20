"""M1 —— 地址空间可见:ascend_address_space 与内存层级到语言层的落地。

SOURCE: third_party/ascend/language/cann/extension/core.py:L143-163
        (ascend_address_space_base / ascend_address_space_group)
"""
from conftest import FakeBuilder


def test_ascend_address_space_exposes_the_five_pybind_exported_levels(env):
    """语言层能写出的地址空间恰是 pybind 导出的五级：UB/L1/L0A/L0B/L0C。"""
    al = env.al
    for name in ("UB", "L1", "L0A", "L0B", "L0C"):
        assert hasattr(al.ascend_address_space, name), f"missing address space member {name}"


def test_zero_and_gm_stop_at_pybind_and_never_reach_python(env):
    """反向断言：Zero 与 GM 不在 al.ascend_address_space 上。

    HIVMAttrs.td:L188-194 一共定义了 7 级（Zero=0/GM=1/L1=2/L0A=3/L0B=4/L0C=5/UB=6），
    但 ascend_ir.cc:L412-417 的 py::enum_<hivm::AddressSpace> 只 .value() 了 5 个——
    Zero 与 GM 止步于 C++ 侧。ascend_address_space_group 反射的是 Python 侧的
    ascend_ir.AddressSpace.__dict__，所以这两个名字根本不会出现，kernel 里也就写不出
    `space=GM` 的 buffer：全局内存不是一种"可分配的 buffer 空间"，它只以入参指针的形式
    出现，进出片上必须走显式 copy（这正是本章与 ch06 索引搬运的前提）。
    """
    al = env.al
    for name in ("Zero", "GM"):
        assert not hasattr(al.ascend_address_space, name), (
            f"{name} 不该出现在 Python 层——它没有被 ascend_ir.cc 的 py::enum_ 导出")


def test_address_space_member_wraps_real_address_space_value(env):
    al = env.al
    ub = al.ascend_address_space.UB
    assert ub.real_address_space.name == "UB"


def test_address_space_to_ir_forwards_to_builder_get_target_attribute(env):
    al = env.al
    builder = FakeBuilder()
    ub = al.ascend_address_space.UB

    attr = ub.to_ir(builder)

    assert attr == "attr(UB)"
    assert ("get_target_attribute", ub.real_address_space) in builder.calls


def test_address_space_group_is_reflected_from_ascend_ir_enum(env):
    """ascend_address_space 的成员集合由 C++ 侧 ascend_ir.AddressSpace 反射生成——
    新增一级地址空间不需要改这段 Python（dossier design_decisions
    「地址空间集合由 C++ 侧 ascend_ir.AddressSpace 反射生成,而非 Python 硬编码」）。

    边界：反射的输入是 **Python 侧** 的 ascend_ir.AddressSpace.__dict__，所以"自动"
    只覆盖 pybind 导出过的成员——在 .td 里加一级而不在 ascend_ir.cc 里 .value()，
    语言层依然看不见（Zero/GM 就是现成的例子，见
    test_zero_and_gm_stop_at_pybind_and_never_reach_python）。
    """
    al = env.al
    names = {
        k for k, v in vars(al.ascend_address_space).items()
        if type(v).__name__ == "ascend_address_space_base"
    }
    assert names == {"L1", "UB", "L0A", "L0B", "L0C"}
