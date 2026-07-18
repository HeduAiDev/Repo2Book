"""m4/m6 —— setup_unified_builder 把 ascend_builder 的 emit 方法作为『插入点同步
wrapper』挂到主 builder 上；调用时把主 builder 当前的插入点/loc 搬到 delegate
（ascend_builder）身上执行，执行完再搬回来。

对照真实源码 third_party/ascend/language/cann/extension/builder.py:
    create_builder_method_wrapper —— 单个方法的委托 + 插入点同步
    attach_builder_methods       —— 批量 setattr
    setup_unified_builder        —— 反挂 _ascend_builder + 调 attach_builder_methods
"""
from conftest import FakeAscendBuilder, FakeBuilder


def test_attach_builder_methods_delegates_and_returns_result(env):
    main = FakeBuilder(context="ctx", compile_mode="simd")
    ascend = FakeAscendBuilder(context="ctx", arch="")

    env.ext_builder.attach_builder_methods(main, ascend, ["create_copy_buffer"])

    assert hasattr(main, "create_copy_buffer")
    main.create_copy_buffer(1, 2, kw="v")
    assert ("create_copy_buffer", (1, 2), {"kw": "v"}) in ascend.calls


def test_setup_unified_builder_reverse_attaches_ascend_builder(env):
    main = FakeBuilder(context="ctx", compile_mode="simd")
    ascend = FakeAscendBuilder(context="ctx", arch="")

    env.ext_builder.setup_unified_builder(main, ascend)

    assert main._ascend_builder is ascend
    for name in ("create_scope_op", "scope_return", "get_t_core_type_attr_name",
                 "get_t_core_type_cube_attr", "get_t_core_type_vector_attr",
                 "create_copy_buffer"):
        assert hasattr(main, name), f"{name} 应该被挂到主 builder 上"


def test_wrapper_synchronizes_insertion_point_before_and_after_delegate_call(env):
    """create_builder_method_wrapper 的关键行为（m6）：
    调用 main.create_scope_op(...) 时，应该：
      1. 先读 main 当前的插入点/loc；
      2. 把它们搬到 ascend_builder 上（restore_insertion_point/set_loc）；
      3. 在 ascend_builder 上真正执行委托方法；
      4. 再把 main 的插入点/loc 搬回去（对调用方而言，main 的状态没有被破坏）。
    """
    main = FakeBuilder(context="ctx", compile_mode="simd")
    ascend = FakeAscendBuilder(context="ctx", arch="")
    main.set_loc("kernel.py", 10, 0)
    main._ip = "main-ip-before-call"

    env.ext_builder.attach_builder_methods(main, ascend, ["create_scope_op"])
    main.calls.clear()
    ascend.calls.clear()

    op = main.create_scope_op({"noinline": "unit-attr"}, [])

    assert op.attrs == {"noinline": "unit-attr"}
    # ascend_builder 的插入点被同步成了 main 调用前的插入点。
    assert ("restore_insertion_point", "main-ip-before-call") in ascend.calls
    # main.set_loc(*args) 把 args 存成一个元组 saved_loc；wrapper 把它整个当**单个
    # 位置参数**转发给 delegate.set_loc(saved_loc)（源码 builder.py:L44 `delegate_
    # builder.set_loc(saved_loc)`，不是 `*saved_loc`），故 ascend 侧记录到的是
    # "外面套了一层" 的 (saved_loc,)。
    assert ("set_loc", (("kernel.py", 10, 0),)) in ascend.calls
    # 真正的委托方法确实在 ascend_builder 上执行了。
    assert any(c[0] == "create_scope_op" for c in ascend.calls)
    # 调用完成后，main 的插入点被恢复回同一个值——对调用方看起来插入点没有跑偏。
    assert main._ip == "main-ip-before-call"


def test_scope_return_wrapper_also_delegates(env):
    main = FakeBuilder(context="ctx", compile_mode="simd")
    ascend = FakeAscendBuilder(context="ctx", arch="")
    env.ext_builder.setup_unified_builder(main, ascend)

    main.scope_return(["v0", "v1"])

    assert ("scope_return", ["v0", "v1"]) in ascend.calls
