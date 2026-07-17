#!/usr/bin/env python3
"""m3 — 参数三桶切分 + cubin 十六进制内嵌 C 模板（compile.py L107-L155）。

compile.py 的 __main__ 里 triton.compile 走 driver.active(需 GPU/torch)；本脚本改为
显式 target=sm_90 拿真 cubin，再**逐字复刻** L116-L155 的参数分桶 + hexlify + 读真
compile.c/compile.h 模板 .format(**params)，落一份真的 compile.c/compile.h。
模板文件取自 pin 源码树 python/triton/tools/compile.{c,h}。

另附：纯 Python 跑参数分桶(L116-L129)在含一个 equal_to_1(:1) 参数时的行为——
展示『签名里一个 :1 让运行期 C 原型少一个形参』(full_signature 保留、signature 省略)。
"""
import binascii
import json
import os
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget, AttrsDescriptor
from triton.compiler.code_generator import kernel_suffix
from triton.backends.nvidia.driver import ty_to_cpp

TOOLS = Path("/mnt/e/Laboratory/Repo2Book/instances/triton/source/python/triton/tools")


@triton.jit
def add_kernel(X, Y, Out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(Out + offs, tl.load(X + offs, mask=mask) + tl.load(Y + offs, mask=mask), mask=mask)


def partition_args(arg_names, constants, signature, attrs):
    """逐字复刻 compile.py:L116-L129 的参数分桶。"""
    an, at, an1, at1 = [], [], [], []
    for i, arg_name in enumerate(arg_names):
        if arg_name not in constants:
            an.append(arg_name); at.append(signature[arg_name])
            an1.append(arg_name); at1.append(signature[arg_name])
        elif i in attrs.equal_to_1:
            an.append(arg_name); at.append(signature[arg_name])
    return an, at, an1, at1


def main():
    out = {"template_src": str(TOOLS)}

    # === 编译真 cubin（3 个指针带 :16，无 :1）===
    arg_names = ["X", "Y", "Out", "N"]
    signature = {"X": "*fp32", "Y": "*fp32", "Out": "*fp32", "N": "i32"}
    constants = {"BLOCK": 64}
    attrs = AttrsDescriptor.from_hints({0: 16, 1: 16, 2: 16})
    target = GPUTarget("cuda", 90, 32)
    ccinfo = triton.compile(
        triton.compiler.ASTSource(
            fn=add_kernel,
            signature={**signature, "BLOCK": "constexpr"},
            constants=constants,
            attrs=attrs,
        ),
        target=target,
        options={"num_warps": 4, "num_stages": 3},
    )
    cubin = ccinfo.asm["cubin"]

    # === 复刻 L116-L155：分桶 + hexlify + 灌模板 ===
    an, at, an1, at1 = partition_args(arg_names, constants, signature, attrs)
    num_warps, num_stages = 4, 3
    out_name = "add_kernel"
    sig_hash = "deadbeef"                 # 占位（真 hash 来自签名字符串，与本机制无关）
    meta_sig = f"warps{num_warps}xstages{num_stages}"
    const_sig = "x".join([str(v) for v in constants.values()])
    doc_string = [f"{k}={v}" for k, v in constants.items()]
    doc_string += [f"num_warps={num_warps}", f"num_stages={num_stages}"]
    suffix = kernel_suffix(signature.values(), attrs)     # 0d1d2d3
    func_name = "_".join([out_name, sig_hash, suffix])
    hex_ = str(binascii.hexlify(cubin))[2:-1]
    grid = (1, 1, 1)
    params = {
        "kernel_name": func_name,
        "triton_kernel_name": "add_kernel",
        "bin_size": len(hex_),
        "bin_data": ", ".join([f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]),
        "signature": ", ".join([f"{ty_to_cpp(t)} {n}" for n, t in zip(an1, at1)]),
        "full_signature": ", ".join([f"{ty_to_cpp(t)} {n}" for n, t in zip(an, at)]),
        "arg_pointers": ", ".join([f"&{a}" for a in an1]),
        "num_args": len(an1),
        "kernel_docstring": doc_string,
        "shared": ccinfo.metadata.shared,
        "num_warps": num_warps,
        "algo_info": "_".join([const_sig, meta_sig]),
        "gridX": grid[0], "gridY": grid[1], "gridZ": grid[2],
        "_placeholder": "",
    }
    filled = {}
    for ext in ["h", "c"]:
        tmpl = (TOOLS / f"compile.{ext}").read_text()
        filled[ext] = tmpl.format(**params)

    # compile.c 里真正内嵌的 cubin 数组头部（前 6 字节 = ELF magic 0x7f 'E' 'L' 'F'）
    cubin_head = params["bin_data"].split(", ")[:6]

    out.update({
        "arg_partition": {
            "arg_names_all": arg_names,
            "runtime_signature_args(arg_names_not_1)": an1,
            "full_signature_args(arg_names)": an,
            "suffix": suffix,
            "func_name": func_name,
            "shared_bytes": ccinfo.metadata.shared,
        },
        "cubin_bytes": len(cubin),
        "bin_size_hexchars": len(hex_),
        "cubin_array_head_6": cubin_head,
        "compile_h": filled["h"],
        "compile_c_head": filled["c"][:1400],   # 头部（含 CUBIN 数组开头 + load）
        "compile_c_launch_tail": filled["c"][-700:],  # 入口 cuLaunchKernel 段
    })

    # === :1 fold-out 演示（纯 Python，无需编译）===
    # 构造：X,Y,Out 指针(:16) + stride(:1, equal_to_1) —— 4 个运行期参数，stride 恒 1
    an2 = ["X", "Y", "Out", "stride"]
    sig2 = {"X": "*fp32", "Y": "*fp32", "Out": "*fp32", "stride": "i32"}
    attrs2 = AttrsDescriptor.from_hints({0: 16, 1: 16, 2: 16, 3: 1})
    const2 = {}
    for p, v in attrs2.get_constants().items():
        const2[an2[p]] = v                          # stride:1 并入常量
    a_all, _, a_rt, _ = partition_args(an2, const2, sig2, attrs2)
    out["fold_out_demo"] = {
        "hints": {"0": 16, "1": 16, "2": 16, "3": 1},
        "equal_to_1_params": attrs2.equal_to_1,
        "get_constants": {str(k): v for k, v in attrs2.get_constants().items()},
        "constants_after_merge": const2,
        "full_signature_args": a_all,            # 含 stride
        "runtime_signature_args": a_rt,          # 不含 stride —— 少一个形参
        "num_full": len(a_all),
        "num_runtime": len(a_rt),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
