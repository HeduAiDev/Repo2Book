"""m6 —— extern_elementwise：(实参 dtype 元组) → __hmf_ 符号名的查表 dispatch。

对照 python/triton/language/core.py:L2691-2730（基座能力）与
third_party/ascend/language/cann/libdevice.py:L28-34(reciprocal)/L81-93(tanh)。

要点：extern 这条路是"点菜"——菜单(arg_type_symbol_dict)在函数定义处写死，
实参 dtype 不在菜单里就点不到菜；菜单本身还随 libdevice_simt 开关/910_95 架构而换。
逐轮记录：输入 dtype → 命中的 __hmf_ 符号 → 返回 dtype。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import build_env, dump  # noqa: E402


def _tensor(tl, name):
    return tl.tensor(handle=f"h-{name}", type=tl.dtype(name))


def _call(fn, tl, dtype_name, builder):
    """调一个 @core.extern 函数，回报它选中的符号与返回 dtype（或被拒的原因）。"""
    x = _tensor(tl, dtype_name)
    try:
        res = fn(x, _builder=builder)
        return {"ok": True, "symbol": res.handle.symbol, "ret_dtype": res.dtype.name, "error": ""}
    except KeyError as e:
        return {"ok": False, "symbol": None, "ret_dtype": None, "error": str(e)}


def main():
    rounds = []

    # --- 环境 A：默认（triton_enable_libdevice_simt() = False）---
    env, cleanup = build_env(simt_enabled=False)
    try:
        tl = env.tl_core
        ld = env.libdevice
        b = env.FakeBuilder()
        menu_recip = ["__hmf_recipf (fp32)", "__hmf_recipDh (fp16)"]
        for dt in ("fp32", "fp16", "bf16"):
            r = _call(ld.reciprocal, tl, dt, b)
            rounds.append({"round": len(rounds) + 1, "fn": "libdevice.reciprocal",
                           "env": "simt=False", "in_dtype": dt, "menu": menu_recip, **r})
        for dt in ("fp32", "fp16"):
            r = _call(ld.tanh, tl, dt, b)
            rounds.append({"round": len(rounds) + 1, "fn": "libdevice.tanh",
                           "env": "simt=False, 910_95=False",
                           "menu": ["__hmf_tanhf (fp32)", "__hmf_tanhDh (fp16)"],
                           "in_dtype": dt, **r})
        extern_calls_a = len([c for c in b.calls if c[0] == "create_extern_elementwise"])
    finally:
        cleanup()

    # --- 环境 B：libdevice_simt 开 + 编译目标 910_95 —— tanh 换一份菜单 ---
    env, cleanup = build_env(simt_enabled=True)
    try:
        tl = env.tl_core
        ld = env.libdevice
        ld.is_compile_on_910_95 = True  # libdevice.py 顶部 from ... import 进来的模块级名字
        b = env.FakeBuilder()
        for dt in ("fp32", "fp16"):
            r = _call(ld.tanh, tl, dt, b)
            rounds.append({"round": len(rounds) + 1, "fn": "libdevice.tanh",
                           "env": "simt=True, 910_95=True",
                           "menu": ["__hmf_tanh_fp32 (fp32)"],
                           "in_dtype": dt, **r})
        extern_calls_b = len([c for c in b.calls if c[0] == "create_extern_elementwise"])
    finally:
        cleanup()

    trace = {
        "mechanism": "m6",
        "source": ["python/triton/language/core.py:L2691-2730",
                   "third_party/ascend/language/cann/libdevice.py:L28-34",
                   "third_party/ascend/language/cann/libdevice.py:L81-93"],
        "note": ("符号只是被'引用'——FakeBuilder 只记录选中了哪个 __hmf_ 符号，"
                 "不计算它的数值（真值需昇腾 NPU/CANN，host 无此环境）。"),
        "extern_calls_env_a": extern_calls_a,
        "extern_calls_env_b": extern_calls_b,
        "rounds": rounds,
    }
    dump(trace, "m6.json")


if __name__ == "__main__":
    main()
