#!/usr/bin/env python3
"""m03 headline worked example: 取本章自撰 num_blocks_kernel 的两阶段 TTIR。

复现配方来自 dossier.open_gaps_for_lead[3]（headless，无 GPU，只跑到 TTIR）。
在 triton==3.2.0 的 venv 里跑（其 Python 前端与 pin v3.2.0 逐字节相同）。
阶段一 = ASTSource.make_ir 输出（追踪期，任何 pass 之前）。
阶段二 = stages["ttir"]（make_ttir 之后，add_inliner + canonicalizer 已跑）。

用法：<venv>/bin/python run_m03_cdiv_ir.py
"""
import re
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir

print(f"# triton {triton.__version__}  ({triton.__file__})")


# ★ 本章自撰的最小载体 kernel（不是 tutorials 原文）——体内真调 tl.cdiv。
@triton.jit
def num_blocks_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    n_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    tl.store(out_ptr, n_blocks)


target = GPUTarget("cuda", 90, 32)      # 编译目标；机器上无需真有此卡
backend = make_backend(target)
options = backend.parse_options({})

ctx = ir.context()
ir.load_dialects(ctx)
backend.load_dialects(ctx)

src = ASTSource(
    num_blocks_kernel,
    signature={"out_ptr": "*i32", "n_elements": "i32"},
    constants={"BLOCK_SIZE": 1024},
)

# ---- 阶段一：追踪期 IR（make_ir 输出，dump 循环之外，从不落盘）----
mod = src.make_ir(options, backend.get_codegen_implementation(),
                  backend.get_module_map(), ctx)
stage1 = str(mod)
# 删掉行尾 loc(#locN) 位置注记，纯位置元数据、与机制无关（便于阅读，其余逐字）
stage1_clean = re.sub(r"\s*loc\(#loc\d*\)", "", stage1)
stage1_clean = re.sub(r"^#loc.*$", "", stage1_clean, flags=re.M)
stage1_clean = re.sub(r"\n{3,}", "\n\n", stage1_clean).strip()

print("\n================ STAGE 1: 追踪期 TTIR (make_ir 输出) ================")
print(stage1_clean)

# 追踪期 op 计数（cdiv 被调 tt.func 体内的 arith.*）
def count(mod_text, op):
    return len(re.findall(r"\b" + re.escape(op) + r"\b", mod_text))

print("\n---- 阶段一 op 计数（全模块）----")
for op in ["tt.call", "tt.func", "arith.addi", "arith.subi", "arith.divsi",
           "arith.extsi", "arith.cmpi", "arith.andi", "arith.constant"]:
    print(f"  {op:18s} = {count(stage1, op)}")

# mangled 被调函数名
m = re.search(r"@(cdiv__\w+)", stage1)
print(f"\n---- mangled 被调函数名 ----\n  {m.group(1) if m else 'NOT FOUND'}")

# ---- 阶段二：make_ttir 之后 ----
stages = {}
backend.add_stages(stages, options)
metadata = {"hash": "x", "target": target}
metadata.update(options.__dict__)
mod2 = stages["ttir"](mod, metadata)
stage2 = str(mod2)
stage2_clean = re.sub(r"\s*loc\(#loc\d*\)", "", stage2)
stage2_clean = re.sub(r"^#loc.*$", "", stage2_clean, flags=re.M)
stage2_clean = re.sub(r"\n{3,}", "\n\n", stage2_clean).strip()

print("\n================ STAGE 2: make_ttir 之后的 TTIR ================")
print(stage2_clean)

print("\n---- 阶段二 op 计数（全模块，逐 op 穷举）----")
for op in ["tt.call", "tt.func", "arith.constant", "arith.addi", "arith.divsi",
           "arith.subi", "arith.extsi", "arith.cmpi", "arith.andi",
           "tt.store", "tt.return"]:
    print(f"  {op:18s} = {count(stage2, op)}")

# 常量值确认：1024 与 1023
consts = re.findall(r"arith\.constant (-?\d+) : i32", stage2)
print(f"\n---- 阶段二 i32 常量 ----\n  {consts}")
