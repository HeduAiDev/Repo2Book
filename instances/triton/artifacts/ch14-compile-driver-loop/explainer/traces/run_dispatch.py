#!/usr/bin/env python3
"""m4 / m5 verification trace — 纯控制流,复现 compile() 的两处整数判定,跑在 host。

m4  起步级 first_stage(compiler.py:L263 + L265-266):
    stages 键序 = CUDABackend.add_stages 的插入序(third_party/nvidia/backend/compiler.py:L385-389)
                = ['ttir','ttgir','llir','ptx','cubin']
    first_stage = list(stages).index(src.ext)         # src.ext:ASTSource='ttir'(L71),IRSource=文件后缀(L112)
    if ir_source: first_stage += 1                     # IR 入口跳过起点这一级
    实际遍历 = stages[first_stage:]

m5  唯一后端 make_backend(compiler.py:L306-311):
    actives = [后端 for 后端 in 已发现后端 if 后端.supports_target(target)]
    len(actives) 必须 == 1,否则 RuntimeError
    CUDABackend.supports_target:target.backend=='cuda'(third_party/nvidia/backend/compiler.py:L135-138 语义)

输出存 dispatch.json。
"""
import json
import os

STAGES = ["ttir", "ttgir", "llir", "ptx", "cubin"]  # CUDABackend.add_stages 插入序,L385-389

def resolve_stages(ext, ir_source):
    first_stage = STAGES.index(ext)
    if ir_source:
        first_stage += 1
    ran = STAGES[first_stage:]
    skipped = STAGES[:first_stage]
    return {
        "src.ext": ext,
        "ir_source": ir_source,
        "first_stage_index": first_stage,
        "stages_run": ran,
        "num_stages_run": len(ran),
        "stages_skipped": skipped,
        "num_stages_skipped": len(skipped),
    }

def make_backend(discovered, target_backend):
    """discovered: {name: set(它 supports 的 target.backend)}。返回 actives 数与结果。"""
    actives = [name for name, supp in discovered.items() if target_backend in supp]
    ok = (len(actives) == 1)
    return {
        "target.backend": target_backend,
        "discovered_backends": list(discovered.keys()),
        "num_discovered": len(discovered),
        "actives": actives,
        "num_actives": len(actives),
        "result": "选中 " + actives[0] if ok else "RuntimeError(需恰好一个)",
    }

def main():
    # m4: 三种入口的起步级
    m4 = {
        "AST_ttir": resolve_stages("ttir", ir_source=False),   # @jit 源码入口,从头降
        "IR_ttgir": resolve_stages("ttgir", ir_source=True),   # 一份 .ttgir 直接进
        "IR_llir": resolve_stages("llir", ir_source=True),     # 一份 .llir 直接进
    }
    # m5: 后端选择(cuda + amd 两后端已发现)
    discovered = {"cuda": {"cuda"}, "amd": {"hip"}}
    m5 = {
        "target_cuda": make_backend(discovered, "cuda"),   # 恰好 cuda 命中 → 1
        "target_hip": make_backend(discovered, "hip"),     # 恰好 amd 命中 → 1
        "target_xpu_none": make_backend(discovered, "xpu"),# 无人命中 → 0 → 抛错
    }
    out = {"stages_order": STAGES, "m4_first_stage": m4, "m5_make_backend": m5}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dispatch.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
