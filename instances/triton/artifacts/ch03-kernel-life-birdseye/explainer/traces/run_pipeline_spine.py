#!/usr/bin/env python3
"""ch03 主线/无GPU断裂 worked example：把 tutorials 的 add_kernel 走完整条降级管线。

复现配方 = INSTANCE.md 的 pin 精确 IR 验证配方（triton==3.2.0 venv，其 Python 前端与
pin v3.2.0 逐字节相同）。全程 headless：用显式 GPUTarget("cuda", 90, 32) 编译，
**不查询本机真实设备、不建 CUDA context**——所以每一级产物的存在本身就证明
make_ir→ttir→ttgir→llir→ptx→cubin 都是设备无关的纯编译（m11 无 GPU 断裂处地图）。
断裂线在其后的 CompiledKernel._init_handles→load_binary（灌进 GPU），本脚本不跨越它。

用法：<venv>/bin/python run_pipeline_spine.py
"""
import json
import re
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir

print(f"# triton {triton.__version__}  ({triton.__file__})")


# ★ tutorials/01-vector-add.py:L27 的 add_kernel（本章主线的那一个核，逐字照搬）。
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


# 显式编译目标：sm_90。注意——本机 GPU 不是 sm_90（是 Blackwell），
# 却能编译成功，正证明整条管线不碰真实设备。
target = GPUTarget("cuda", 90, 32)
backend = make_backend(target)
options = backend.parse_options({"num_warps": 4})

ctx = ir.context()
ir.load_dialects(ctx)
backend.load_dialects(ctx)

src = ASTSource(
    add_kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32",
               "n_elements": "i32", "BLOCK_SIZE": "constexpr"},
    constants={"BLOCK_SIZE": 1024},
)


def lines(s):
    return len(s.splitlines())


def count(text, tok):
    return len(re.findall(r"(?<![\w.])" + re.escape(tok) + r"(?![\w])", text))


report = {"triton": triton.__version__, "target": "cuda sm_90 warps=4",
          "constants": {"BLOCK_SIZE": 1024}, "stages": []}

# ---------- STAGE 0：追踪期 TTIR（make_ir 输出，任何 pass 之前）----------
mod = src.make_ir(options, backend.get_codegen_implementation(),
                  backend.get_module_map(), ctx)
s0 = str(mod)
print("\n================ STAGE 0: 追踪期 TTIR (make_ir) ================")
print(f"  lines            = {lines(s0)}")
print(f"  tt.func          = {count(s0, 'tt.func')}")
print(f"  tt.call          = {count(s0, 'tt.call')}")
print(f"  tt.load          = {count(s0, 'tt.load')}")
print(f"  tt.store         = {count(s0, 'tt.store')}")
print(f"  has #blocked     = {'#blocked' in s0}")
print(f"  has ttg./#shared = {'ttg.' in s0 or '#shared' in s0}")
report["stages"].append({
    "id": "trace-ttir", "stage": "make_ir (追踪期)", "product": "TTIR (内存, 不落盘)",
    "lines": lines(s0), "tt_call": count(s0, "tt.call"), "tt_load": count(s0, "tt.load"),
    "has_layout": "#blocked" in s0, "headless_ok": True})

# ---------- STAGE 1..5：五级降级（compile 的 for 循环干的事）----------
stages = {}
backend.add_stages(stages, options)          # nvidia compiler.py:L384-L389
print("\n  add_stages 注册的五级 keys =", list(stages.keys()))
report["registered_stages"] = list(stages.keys())

metadata = {"hash": "x", "target": target}
metadata.update(options.__dict__)

module = mod
STAGE_META = {
    "ttir":  ("make_ttir",  "TTIR (已内联/优化)"),
    "ttgir": ("make_ttgir", "TTGIR (贴布局)"),
    "llir":  ("make_llir",  "LLVM-IR (文本)"),
    "ptx":   ("make_ptx",   "PTX (虚拟汇编文本)"),
    "cubin": ("make_cubin", "cubin (sm_90 机器码, 二进制)"),
}
for ext, compile_ir in stages.items():
    module = compile_ir(module, metadata)
    fn, product = STAGE_META[ext]
    if ext == "cubin":
        blob = bytes(module)
        size = len(blob)
        magic = blob[:4].hex()
        print(f"\n================ STAGE .{ext}: {fn} ================")
        print(f"  bytes            = {size}")
        print(f"  ELF magic (7f454c46=ELF) = {magic}")
        report["stages"].append({
            "id": ext, "stage": fn, "product": product, "bytes": size,
            "elf_magic": magic, "headless_ok": True})
        continue
    text = str(module)
    entry = {"id": ext, "stage": fn, "product": product,
             "lines": lines(text), "headless_ok": True}
    print(f"\n================ STAGE .{ext}: {fn} ================")
    print(f"  lines            = {lines(text)}")
    if ext == "ttir":
        entry["tt_call"] = count(text, "tt.call")
        entry["has_layout"] = "#blocked" in text
        print(f"  tt.call          = {entry['tt_call']}   (追踪期若有 tt.call 到这里已被 add_inliner 抹平)")
        print(f"  has #blocked     = {entry['has_layout']}   (布局还没贴, 见 ttgir)")
    if ext == "ttgir":
        entry["has_blocked_layout"] = "#blocked" in text
        nw = re.search(r'"ttg.num-warps"\s*=\s*(\d+)', text) or re.search(r'num-warps.*?(\d+)', text)
        entry["num_warps_in_ir"] = nw.group(1) if nw else None
        print(f"  has #blocked     = {entry['has_blocked_layout']}   (convert_to_ttgpuir 首次贴布局)")
        print(f"  num-warps in IR  = {entry['num_warps_in_ir']}")
    if ext == "llir":
        entry["has_nvptx_triple"] = "nvptx64-nvidia-cuda" in text
        entry["has_define"] = "define" in text
        print(f"  target triple nvptx64-nvidia-cuda = {entry['has_nvptx_triple']}   (to_module 已跨到 LLVM 世界)")
        print(f"  has 'define'     = {entry['has_define']}")
    if ext == "ptx":
        ver = re.search(r'\.version\s+([\d.]+)', text)
        tgt = re.search(r'\.target\s+(\S+)', text)
        entry["ptx_version"] = ver.group(1) if ver else None
        entry["ptx_target"] = tgt.group(1) if tgt else None
        print(f"  .version         = {entry['ptx_version']}")
        print(f"  .target          = {entry['ptx_target']}   (仍是文本, 还不是机器码)")
    report["stages"].append(entry)

print("\n================ 断裂线证明 ================")
print("  以上 6 级产物全部在 headless（未建 CUDA context、编译目标 sm_90≠本机卡）下产出。")
print("  真正的 GPU 门槛在其后的 CompiledKernel._init_handles → load_binary")
print("  (compiler.py:L390) —— 本脚本刻意不跨越它。")
report["fracture_line"] = "CompiledKernel._init_handles -> load_binary (compiler.py:L390)"
report["headless_stages_count"] = len(report["stages"])

out = __file__.replace("run_pipeline_spine.py", "pipeline_spine.json")
with open(out, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n# wrote {out}")
