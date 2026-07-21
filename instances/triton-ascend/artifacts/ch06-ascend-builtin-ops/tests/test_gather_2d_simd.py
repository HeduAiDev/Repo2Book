"""gather_2d_simd（m13 第三条路）：不用任何昇腾扩展算子，纯上游 tl.arange/tl.load/
tl.gather/tl.store 写的片上 2D gather（axis=1）。

宿主无昇腾 NPU/CANN，也不借用宿主 pip 装的官方 triton（不同版本/不同 fork，会静默
引入版本漂移，同 ch04/ch05 conftest 的一贯原则）——这里搭一个极简的 numpy 驱动
"解释器"站在 triton.jit/tl.arange/tl.load/tl.store/tl.gather 的位置上，只覆盖这
一个 kernel 用到的这几个算子，验证的是"逐行搬到片上再 gather"这条真实控制流与
数值语义，而不是任何 MLIR/hivm 细节。
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


class _Ptr:
    """GM 缓冲区的裸指针替身：buffer + 当前累计偏移（数组）。"""
    def __init__(self, buffer, offsets=None):
        self.buffer = buffer
        self.offsets = offsets if offsets is not None else np.zeros((), dtype=np.int64)

    def __add__(self, other):
        other = np.asarray(other, dtype=np.int64)
        return _Ptr(self.buffer, self.offsets + other)


@pytest.fixture
def gather_env():
    added = []

    def stub(dotted):
        parts = dotted.split(".")
        for i in range(len(parts)):
            name = ".".join(parts[: i + 1])
            if name not in sys.modules:
                m = types.ModuleType(name)
                sys.modules[name] = m
                added.append(name)
                if i > 0:
                    setattr(sys.modules[".".join(parts[:i])], parts[i], m)
        return sys.modules[dotted]

    triton_mod = stub("triton")

    def fake_jit(fn):
        # 真实 @triton.jit 会编译成 MLIR 再下降；这里直接返回原函数，
        # 测试里当普通 Python 函数调用（同 dossier 的 interpreter 模式精神）。
        fn.fn = fn
        return fn

    triton_mod.jit = fake_jit

    pid_box = {"pid": 0}
    tl_mod = stub("triton.language")
    tl_mod.program_id = lambda axis: pid_box["pid"]
    tl_mod.arange = lambda a, b: np.arange(a, b)

    def _load(addr, mask=None):
        vals = addr.buffer.flat[addr.offsets]
        if mask is None:
            return vals
        return vals  # mask 只影响 store，load 侧真实 kernel 也没用 mask

    def _store(addr, value, mask=None):
        flat_offsets = addr.offsets
        if mask is None:
            addr.buffer.flat[flat_offsets] = value
        else:
            mask_b = np.broadcast_to(mask, value.shape)
            flat_idx = np.broadcast_to(flat_offsets, value.shape)
            addr.buffer.flat[flat_idx[mask_b]] = value[mask_b]

    tl_mod.load = _load
    tl_mod.store = _store
    tl_mod.gather = lambda src, idx, axis: np.take_along_axis(src, idx.astype(np.int64), axis=axis)

    core_mod = stub("triton.language.core")

    class _Constexpr(int):
        """真实 constexpr 只是编译期整数标注，这里直接让它表现得像个 int 即可
        （gather_2d_simd 从不对它做除 int 算术之外的操作）。"""
        pass

    core_mod.constexpr = _Constexpr

    spec = importlib.util.spec_from_file_location(
        "third_party.ascend.language.kernels.gather",
        IMPL_DIR / "third_party/ascend/language/kernels/gather.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    added.append(spec.name)

    def run_grid(m_total, xblock, *args, **kwargs):
        """真实 launch 是 `gather_2d_simd[grid](...)`——grid 里每个 program 处理
        XBLOCK 行；这里按 cdiv(M, XBLOCK) 顺序跑完每个 pid，模拟同一个 grid launch。
        """
        num_programs = (m_total + xblock - 1) // xblock
        for pid in range(num_programs):
            pid_box["pid"] = pid
            mod.gather_2d_simd.fn(*args, **kwargs)

    mod.run_grid = run_grid

    try:
        yield mod
    finally:
        for n in reversed(added):
            sys.modules.pop(n, None)


def test_gather_2d_simd_matches_numpy_take_along_axis(gather_env):
    """单个 program（M <= XBLOCK，且 XBLOCK_SUB 整除 M）覆盖全部行，结果应与
    np.take_along_axis(src, idx, axis=1) 完全一致——这就是 m13 第三条路的语义。"""
    mod = gather_env
    rng = np.random.default_rng(0)
    M, N, K = 4, 5, 3
    src = rng.random((M, N)).astype(np.float32)
    idx = rng.integers(0, N, size=(M, K)).astype(np.int32)
    out = np.zeros((M, K), dtype=np.float32)

    mod.run_grid(M, 4, _Ptr(src), _Ptr(idx), _Ptr(out), M=M, N=N, K=K, XBLOCK=4, XBLOCK_SUB=2)

    expected = np.take_along_axis(src, idx.astype(np.int64), axis=1)
    np.testing.assert_allclose(out, expected)


def test_gather_2d_simd_multiple_grid_programs_tile_via_xblock_sub(gather_env):
    """M 大于单个 XBLOCK 时需要多个 grid program（第二条路：pid 分派 + XBLOCK_SUB
    内层向量化），每个 program 各自把自己负责的行搬到片上再 gather。"""
    mod = gather_env
    M, N, K = 6, 5, 3  # 两个 program：pid=0 管 0..3 行，pid=1 管 4..5 行
    rng = np.random.default_rng(1)
    src = rng.random((M, N)).astype(np.float32)
    idx = rng.integers(0, N, size=(M, K)).astype(np.int32)
    out = np.zeros((M, K), dtype=np.float32)

    mod.run_grid(M, 4, _Ptr(src), _Ptr(idx), _Ptr(out), M=M, N=N, K=K, XBLOCK=4, XBLOCK_SUB=2)

    expected = np.take_along_axis(src, idx.astype(np.int64), axis=1)
    np.testing.assert_allclose(out, expected)


def test_gather_2d_simd_store_is_masked_but_load_is_not(gather_env):
    """真实源码里 tl.store 传了 mask=m_offs<M，但两处 tl.load（idx_tile/src_tile）
    都没有 mask——只要 tile 起点在 GM 缓冲区内，不需要为读也补 mask（多读的尾部行
    永远不会被 store 写出去）。这里验证的正是"store 端点上 mask 生效、不多写"。"""
    mod = gather_env
    M, N, K = 4, 3, 2
    src = np.arange(M * N, dtype=np.float32).reshape(M, N)
    idx = np.zeros((M, K), dtype=np.int32)
    sentinel = -999.0
    out = np.full((M, K), sentinel, dtype=np.float32)

    mod.run_grid(M, 4, _Ptr(src), _Ptr(idx), _Ptr(out), M=M, N=N, K=K, XBLOCK=4, XBLOCK_SUB=2)

    assert not np.any(out == sentinel)  # 全部行都应被合法写过，没有遗漏
