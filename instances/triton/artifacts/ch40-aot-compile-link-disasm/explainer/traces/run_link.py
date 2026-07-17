#!/usr/bin/env python3
"""m5/m6/m7 — link.py 端到端：解析两份 tt-linker 头 → 生成分派器 .c/.h。
用 pin(3.2.0) 的真 link.py（纯 stdlib，无 GPU 依赖）跑真链接。

构造两份同名核 add 的特化头（同 algo_info=1024_warps4xstages3 → 归同组）：
  - add_deadbeef_012d : 第 2 位参数 N 带 :16 提示 (num_specs=1)
  - add_cafef00d_012  : 无特化            (num_specs=0)
运行期签名相同 (X,Y,N)。观测：
  m5 HeaderParser 拆出的 KernelLinkerMeta；
  m6 make_kernel_hints_dispatcher 的降序整除性分派链；
  m7 make_func_pointers / make_kernel_meta_const_dispatcher 的 algo_id 表。
"""
import os
import sys
import json
import subprocess
import tempfile

from triton.tools.link import (
    HeaderParser, make_kernel_hints_dispatcher, make_func_pointers,
    make_kernel_meta_const_dispatcher, make_default_algo_kernel,
)
import triton.tools.link as linkmod

HDR_012D = ("// tt-linker: add_deadbeef_012d:"
            "CUdeviceptr X, CUdeviceptr Y, int32_t N:1024_warps4xstages3\n")
HDR_012 = ("// tt-linker: add_cafef00d_012:"
           "CUdeviceptr X, CUdeviceptr Y, int32_t N:1024_warps4xstages3\n")


def main():
    # --- m5: 解析两份头 ---
    parser = HeaderParser()
    parser.extract_linker_meta(HDR_012D)
    parser.extract_linker_meta(HDR_012)

    groups = {}
    for name, metas in parser.kernels.items():
        groups[name] = [
            {
                "orig_kernel_name": m.orig_kernel_name,
                "arg_names": list(m.arg_names),
                "arg_ctypes": list(m.arg_ctypes),
                "sizes": list(m.sizes),
                "sig_hash": m.sig_hash,
                "suffix": m.suffix,
                "num_specs": m.num_specs,
            }
            for m in metas
        ]

    # --- m6: 整除性分派链 ---
    (name, metas), = parser.kernels.items()
    hints_dispatcher = make_kernel_hints_dispatcher(name, metas)

    # --- m7: algo_id 函数指针表 + meta-const 分派 ---
    names = list(parser.kernels.keys())
    meta = metas[0]
    func_pointers = make_func_pointers(names, meta)
    meta_const = make_kernel_meta_const_dispatcher(meta)
    default_algo = make_default_algo_kernel(meta)

    # --- 端到端：真跑 link.py __main__（subprocess），落 add_linked.c/.h ---
    tmpd = tempfile.mkdtemp(prefix="ch40_link_")
    p1 = os.path.join(tmpd, "add_012d.h")
    p2 = os.path.join(tmpd, "add_012.h")
    with open(p1, "w") as f:
        f.write(HDR_012D)
    with open(p2, "w") as f:
        f.write(HDR_012)
    out_base = os.path.join(tmpd, "add_linked")
    subprocess.check_call(
        [sys.executable, linkmod.__file__, p1, p2, "-o", out_base])
    with open(out_base + ".c") as f:
        linked_c = f.read()
    with open(out_base + ".h") as f:
        linked_h = f.read()

    out = {
        "group_key": name,
        "num_groups": len(parser.kernels),
        "num_metas_in_group": len(metas),
        "parsed_metas": groups,
        "hints_dispatcher_src": hints_dispatcher,
        "func_pointers_src": func_pointers,
        "meta_const_dispatcher_src": meta_const,
        "default_algo_src": default_algo,
        "linked_c_full": linked_c,
        "linked_h_full": linked_h,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
