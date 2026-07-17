#!/usr/bin/env python3
"""m8/m10/m11 — cuobjdump -sass 两行一指令解析 + BRA→LBB 重标（一份自足 trace）。
用 pin(3.2.0) 真编译一个含 data-dependent 循环(→BRA 回边)的 kernel 拿 cubin，
再走 disasm.get_sass/extract（内部调 cuobjdump -sass）。需真机 ptxas + cuobjdump。
目标显式 sm_90(Hopper)：绕开 driver.active(无 torch/GPU)，且落在 parseCtrl 控制字布局适用域。

产出三件：
  raw_pairs        —— m8：cuobjdump 原始两行一指令(FLINE 汇编体+首半编码 / SLINE 次半编码)
  bra_relabel_map  —— m10：每条 BRA 的原始十六进制目标 → 目标 offset(dec) → LBB 标签
  final_sass       —— extract 折叠+重标后的可读列表(ctrl 左列 = parseCtrl 结果)
"""
import json
import os
import subprocess
import tempfile

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.tools.disasm import (
    get_sass, path_to_cuobjdump, parseCtrl, processSassLines,
    FLINE_RE, SLINE_RE, BRA_RE,
)


@triton.jit
def loopy(X, N, BLOCK: tl.constexpr):
    acc = 0.0
    i = 0
    while i < N:
        acc += tl.load(X + i)
        i += BLOCK
    tl.store(X, acc)


def compile_cubin():
    target = GPUTarget("cuda", 90, 32)
    compiled = triton.compile(
        triton.compiler.ASTSource(
            fn=loopy,
            signature={"X": "*fp32", "N": "i32", "BLOCK": "constexpr"},
            constants={"BLOCK": 64},
            attrs=triton.backends.compiler.AttrsDescriptor.from_hints({0: 16}),
        ),
        target=target,
        options={"num_warps": 4, "num_stages": 3},
    )
    return compiled.asm["cubin"]


def main():
    cubin = compile_cubin()
    cuobj, cuobj_ver = path_to_cuobjdump()

    # --- 原始 cuobjdump -sass（extract 折叠前的两行格式） ---
    fd, path = tempfile.mkstemp(suffix=".cubin")
    try:
        with open(fd, "wb") as f:
            f.write(cubin)
        raw = subprocess.check_output([cuobj, "-sass", path]).decode()
    finally:
        os.remove(path)
    raw_lines = raw.splitlines()

    # 复刻 extract 的第一趟：配对 FLINE/SLINE，用真 processSassLines 归一并登记 BRA 目标。
    # processSassLines 内部会把结尾 " ;"→";"(BRA_RE 才匹配)并写 labels[target]=len(labels)。
    labels = {}              # raw_target_int -> label_idx（第一趟发现序）
    asm_buffer = []          # (ctrl, asm)  —— 与 extract 同序
    raw_pairs = []
    offsets = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if FLINE_RE.match(line):
            fline = line
            sline = raw_lines[i + 1]
            offset_hex = line.strip().split("*/")[0].strip("/* ")
            ctrl, asm = processSassLines(fline, sline, labels)
            offsets.append(offset_hex)
            asm_buffer.append((ctrl, asm))
            if len(raw_pairs) < 4:
                raw_pairs.append({
                    "offset": offset_hex,
                    "fline_raw": line.strip(),
                    "sline_raw": sline.strip(),
                    "asm_body": FLINE_RE.match(fline).group(1),
                    "ctrl_decoded": ctrl,
                })
            i += 2
        else:
            i += 1

    # --- m10：BRA 原始目标 → LBB 重标（第二趟，逐字复刻 extract L130-L143） ---
    bra_relabel_map = []
    for pos, (offh, (ctrl, asm)) in enumerate(zip(offsets, asm_buffer)):
        if BRA_RE.match(asm):
            target = int(BRA_RE.match(asm).group(2), 16)
            bra_relabel_map.append({
                "bra_at_offset": offh,
                "raw_target_hex": f"0x{target:x}",
                "target_offset_dec": target,
                "relabeled": BRA_RE.sub(rf"\1LBB{labels[target]};", asm),
            })

    # --- extract 折叠+重标后的最终可读 SASS（get_sass，带 lru_cache=m11） ---
    final_sass = get_sass(cubin)

    out = {
        "target": "cuda:sm_90",
        "cuobjdump_version": cuobj_ver,
        "cubin_bytes": len(cubin),
        "num_instructions": len(asm_buffer),
        "num_labels": len(labels),
        "raw_pairs": raw_pairs,
        "bra_relabel_map": bra_relabel_map,
        "labels_by_discovery_order": {f"0x{t:x}": lbl for t, lbl in labels.items()},
        "final_sass": final_sass,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
