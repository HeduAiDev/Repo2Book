#!/usr/bin/env python3
"""ch24 教学素材取证驱动 —— 从 bishengir lit 夹具抽出「pass 断言输出」作为地面真值。

为什么不直接跑 pass:-hivm-inject-sync / -hivm-inject-block-sync / -hivm-cross-core-gss
都活在 bishengir-opt 里,需要从头构建整套 BiShengIR/MLIR 工具链(host 无 CUDA/无预
构建二进制),无法运行。但每个 lit 夹具的 `// CHECK: ...` 行本身就是**该 pass 的期望
输出**(上游 CI 每次都对着它 FileCheck 校验),因此这些 CHECK 行 = pass 真实产出的
可复现地面真值。本脚本把 `// CHECK:` 行按函数(fixture)归组抽出,喂给 explainer 的
逐轮表。裸 IR(同步前)= 非 CHECK 的 hivm.hir.* 计算 op 行。

用法:python3 extract_fixture_checks.py > fixture_checks.json
"""
import json
import re
from pathlib import Path

SRC = Path("/mnt/e/Laboratory/Repo2Book/instances/triton-ascend/source"
           "/third_party/ascend/AscendNPU-IR/bishengir/test/Dialect/HIVM")

FIXTURES = {
    "inject-sync.mlir": [
        "test_mem_injcet_sync_basic", "test_injcet_sync_loop",
        "test_injcet_sync_if", "test_injcet_sync_if_else",
        "test_injcet_sync_two_event_id", "test_widen_sync",
    ],
    "inject-block-sync.mlir": ["matmul_add_mul", "test_block_sync_normal"],
    "sync-solver-cross-core.mlir": ["test_block_sync_loop", "_attn_fwd"],
}

FUNC_RE = re.compile(r"func\.func @(\w+)\(")
CHECK_RE = re.compile(r"//\s*CHECK(?:-[A-Z]+)?:\s*(.*)")
# 计算/搬运 op(裸 IR 里同步前就存在的),用于还原"同步前"骨架
COMPUTE_RE = re.compile(r"^\s*(?:%[\w:]+\s*=\s*)?(hivm\.hir\.(load|store|vadd|vmul|"
                        r"vsub|matmul|mmadL1|fixpipe|nd2nz|set_ffts_base_addr)\b)")


def blocks(text):
    """按 // ----- 切分,返回每段 (funcname, lines)。"""
    for seg in text.split("// -----"):
        m = FUNC_RE.search(seg)
        if not m:
            continue
        yield m.group(1), seg.splitlines()


def main():
    out = {}
    for fname, wanted in FIXTURES.items():
        text = (SRC / fname).read_text(encoding="utf-8")
        for func, lines in blocks(text):
            if func not in wanted:
                continue
            checks, computes = [], []
            for ln in lines:
                cm = CHECK_RE.search(ln)
                if cm:
                    body = cm.group(1).strip()
                    # 归一化 FileCheck 转义 {{\[}} -> [
                    body = body.replace("{{\\[}}", "[")
                    if "hivm.hir." in body or "sync" in body:
                        checks.append(body)
                    continue
                cc = COMPUTE_RE.search(ln)
                if cc:
                    computes.append(cc.group(1))
            out.setdefault(fname, {})[func] = {
                "bare_ir_compute_ops": computes,
                "injected_sync_checks": checks,
            }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
