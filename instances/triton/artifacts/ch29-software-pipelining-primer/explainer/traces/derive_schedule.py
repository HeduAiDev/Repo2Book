#!/usr/bin/env python3
"""ch29 primer 手工推演的算术后盾(非 Triton 运行,纯复算源码公式)。

本章无精简版(primer/skip_impl):真正的模调度跑在 Triton MLIR 的 TritonGPUPipeline
pass 里,需完整 LLVM/MLIR 构建 + GPU 才能跑通。此脚本只是把源码里那几行 stage/buffer
分配「算式」逐字搬成 Python,复算出 explainer 数值表里的每个数——每个公式都标了源码行号。

复算的源码算式(pin Triton v3.2.0):
  - stagesBetweenLoads = ceil(numStages-2, maxIndirectionLevel+1)
      MatmulLoopPipeline.cpp:565-566
  - dot(root use) 放最后一个 stage: numStages-1
      MatmulLoopPipeline.cpp:576
  - load 放 stage = (maxIndirectionLevel - indLevel) * stagesBetweenLoads
      MatmulLoopPipeline.cpp:590
  - distToUse = dot.stage - load.stage
      MatmulLoopPipeline.cpp:596-597
  - numBuffers = max(distToUse) (+1 若 MMAv3)
      MatmulLoopPipeline.cpp:940-955
  - maxStage = 最大 stage 下标 = numStages-1;prologue/epilogue 各 maxStage 段(loop
      `for i in [0, maxStage)`,PipelineExpander.cpp:287),源码注释称之为「maxStage-1 part」
      其中注释的 maxStage 指 stage 总数(=numStages) → 段数 = numStages-1
      PipelineExpander.cpp:90-92 / 107-110 / 287
"""
import json
import math


def ceil_div(a, b):
    # llvm::ceil<unsigned>(a, b) —— 向上取整除
    return (a + b - 1) // b


def derive(num_stages, max_indirection=0, is_mmav3=False,
           block_m=128, block_k=32, dtype_bytes=2):
    """对『单条直取 load 喂 dot』(indirection level 0)复算调度。"""
    # scheduleLoads (MatmulLoopPipeline.cpp:558-597)
    stages_between_loads = ceil_div(num_stages - 2, max_indirection + 1)
    dot_stage = num_stages - 1                       # root use 放最后 stage
    ind_level = 0                                    # 直取,无间接寻址
    load_stage = (max_indirection - ind_level) * stages_between_loads
    dist_to_use = dot_stage - load_stage
    # createAsyncOps (MatmulLoopPipeline.cpp:940-955)
    num_buffers = dist_to_use                        # = max(distToUse),单 load 即其本身
    if is_mmav3:
        num_buffers += 1
    # PipelineExpander: maxStage = 最大 stage 下标
    max_stage = dot_stage                            # = num_stages-1
    prologue_parts = max_stage                       # loop for i in [0, maxStage)
    epilogue_parts = max_stage
    # 共享内存代价:每 buffer 一份 A tile + 一份 B tile
    a_tile_bytes = block_m * block_k * dtype_bytes
    b_tile_bytes = block_k * block_m * dtype_bytes   # 取 BLOCK_N=BLOCK_M 便于心算
    per_buffer_bytes = a_tile_bytes + b_tile_bytes
    smem_bytes = num_buffers * per_buffer_bytes
    return {
        "num_stages": num_stages,
        "stages_between_loads": stages_between_loads,
        "load_stage": load_stage,
        "dot_stage": dot_stage,
        "dist_to_use": dist_to_use,
        "num_buffers_ampere": dist_to_use,
        "num_buffers_mmav3": dist_to_use + 1,
        "num_buffers": num_buffers,
        "max_stage": max_stage,
        "prologue_parts": prologue_parts,
        "epilogue_parts": epilogue_parts,
        "per_buffer_KB": per_buffer_bytes / 1024,
        "smem_KB_ampere": (dist_to_use * per_buffer_bytes) / 1024,
        "smem_KB_mmav3": ((dist_to_use + 1) * per_buffer_bytes) / 1024,
    }


def spacetime(num_stages=3, num_iters=5):
    """稳态时空图:每个时间片里,哪个迭代在哪个 stage。
    3 个 stage: 0=load, 1=wait, 2=dot。迭代 k 的 stage s 落在时间片 t = k + s。
    """
    stage_names = {0: "load", 1: "wait", 2: "dot"}
    max_t = num_iters + num_stages - 1
    slots = []
    for t in range(max_t):
        active = []
        for k in range(num_iters):
            for s in range(num_stages):
                if k + s == t:
                    active.append({"iter": k, "stage": s, "op": stage_names[s]})
        slots.append({"time_slice": t, "active": active,
                      "num_concurrent": len(active)})
    # 稳态定义:满 num_stages 个并发的时间片
    steady = [s["time_slice"] for s in slots if s["num_concurrent"] == num_stages]
    return {"num_stages": num_stages, "num_iters": num_iters,
            "slots": slots, "steady_slices": steady}


def latency(num_stages, t_load=4, t_dot=1):
    """延迟隐藏模型(教学用单位,非硬件 cycle):load 延迟 = t_load、dot = t_dot。

    深度 s 能预取 s-1 个迭代 → 某迭代的 load 在它的 dot 之前有 (s-1)*t_dot 的计算窗口
    可以偷偷跑完。窗口盖不住 load 延迟时,dot 每迭代 stall (t_load-(s-1)*t_dot)。
    这套「窗口 vs 延迟」是软件流水线藏延迟的教科书直觉(II/稳态严格式待核·回指
    Lam1988 DOI:10.1145/53990.54022);此处 stage 数 s 与 buffer 深度的对应是源码坐实的
    (distToUse=s-1,MatmulLoopPipeline.cpp:596-597)。
    """
    prefetch_window = (num_stages - 1) * t_dot          # 预取迭代数 * 每迭代计算时长
    if num_stages <= 1:                                  # 关流水线:每迭代全串
        per_iter = t_load + t_dot
        stall = t_load
    else:
        stall = max(0, t_load - prefetch_window)         # 窗口盖不住的残余延迟
        per_iter = t_dot + stall                         # 稳态每迭代耗时
    fully_hidden_at = math.ceil(t_load / t_dot) + 1      # 恰好盖住所需的 s*
    return {"num_stages": num_stages, "t_load": t_load, "t_dot": t_dot,
            "prefetch_window": prefetch_window, "stall_per_iter": stall,
            "steady_per_iter": per_iter, "fully_hidden_at_stages": fully_hidden_at}


if __name__ == "__main__":
    out = {
        "schedule_by_num_stages": [derive(n) for n in (2, 3, 4, 5)],
        "schedule_mmav3": [derive(n, is_mmav3=True) for n in (2, 3, 4)],
        "spacetime_ns3_it5": spacetime(3, 5),
        "latency_scan": [latency(n) for n in (1, 2, 3, 4, 5, 6)],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
