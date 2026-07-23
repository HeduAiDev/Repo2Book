#!/usr/bin/env python3
"""ch18 DAGSSBuffer double-buffer 教学素材驱动脚本 —— **manual/演示用**。

本章 skip_impl=true:DAGSSBuffer.cpp 单文件 5534 行纯 C++ MLIR pass,宿主无 CANN、编不动,
无 .py 精简版。此脚本**不是**运行编译器 pass,而是把 pin @2badfc89e 的两个纯函数
buildNBufferProducer / buildNBufferConsumer(N==2 特化)里的 arith.select 逻辑
逐行忠实地用 Python 重写一遍,用来**交叉核对手工推演的逐轮表**(数字自洽)。
select 语义严格对应源码:
  producer(third_party/ascend/lib/TritonAffinityOpt/DAGSSBuffer.cpp:L4602-L4636)
  consumer(third_party/ascend/lib/TritonAffinityOpt/DAGSSBuffer.cpp:L4732-L4758)
  constants[i]=i,故 N==2 时模数 constants[N]=constants[2]=2(L5186-5195)
输出 double_buffer.json(原始逐轮记录),给 explainer 表格逐格对账。
"""
import json

N = 2          # runOnOperation 固定传 bufferNum=2(DAGSSBuffer.cpp:L5527)
NUM_TILES = 4  # 演示用小循环:4 个 tile,读者可心算
T_LOAD = 3     # 演示用延迟单位(每块搬运耗时);非实测,仅用于「值不值」的时间线对比
T_COMPUTE = 2  # 演示用延迟单位(每块计算耗时)


def producer(front_cnt, new_dep_val, buff0, buff1):
    """忠实重写 buildNBufferProducer N==2 分支(L4626-L4636)。
    new_dep_val = 本轮新搬进来的 tile 编号(用整数代表张量内容)。"""
    buffer_index = front_cnt % constants(N)   # arith.remsi frontCnt, constants[N]
    is_buffer0 = (buffer_index == constants(0))  # arith.cmpi eq
    # newBuff0 = select(mask, newDepVal, buffs[0])
    new_buff0 = new_dep_val if is_buffer0 else buff0
    # newBuff1 = select(mask, buffs[1], newDepVal)
    new_buff1 = buff1 if is_buffer0 else new_dep_val
    next_cnt = front_cnt + constants(1)         # arith.addi frontCnt, 1
    return new_buff0, new_buff1, next_cnt, buffer_index


def consumer(post_cnt, old_buff0, old_buff1):
    """忠实重写 buildNBufferConsumer N==2 分支(L4750-L4758)。"""
    buffer_index = post_cnt % constants(N)
    is_buffer0 = (buffer_index == constants(0))
    # selected = select(mask, oldBuffs[0], oldBuffs[1])
    selected = old_buff0 if is_buffer0 else old_buff1
    next_cnt = post_cnt + constants(1)
    return selected, next_cnt, buffer_index


def constants(i):
    """constants[i] = i(DAGSSBuffer.cpp:L5191)。"""
    return i


def trace_producer_consumer():
    """m2:frontCnt / postCnt 各取 0..3,记录写侧/读侧各命中哪份 buffer。"""
    prod = []
    for cnt in range(4):
        _, _, _, idx = producer(cnt, new_dep_val=100 + cnt, buff0=0, buff1=1)
        prod.append({"front_cnt": cnt, "front_cnt_mod_N": idx,
                     "writes_buffer": idx})  # N==2:写侧命中 buffer==bufferIndex
    cons = []
    for cnt in range(4):
        _, _, idx = consumer(cnt, old_buff0=0, old_buff1=1)
        cons.append({"post_cnt": cnt, "post_cnt_mod_N": idx,
                     "reads_buffer": idx})
    return prod, cons


def trace_overlap():
    """m3:双缓冲稳态调度 —— frontCnt 领先 postCnt 一个身位,写读永远落在不同 buffer。
    prologue 先搬 tile0(frontCnt 0->1);其后每轮同时『搬下一块 + 算当前块』。"""
    rows = []
    buff = [None, None]      # buffer0, buffer1 当前装的 tile 编号
    front_cnt = 0
    post_cnt = 0

    # prologue:搬 tile0 进 buffer0
    b0, b1, front_cnt, w = producer(front_cnt, new_dep_val=0, buff0=buff[0], buff1=buff[1])
    buff = [b0, b1]
    rows.append({"phase": "prologue", "action": "load tile0",
                 "front_cnt_before": 0, "writes_buffer": w, "front_cnt_after": front_cnt,
                 "post_cnt": post_cnt, "reads_buffer": None,
                 "buffer0_holds": buff[0], "buffer1_holds": buff[1],
                 "overlap": False})

    # 稳态:第 i 轮 = 搬 tile(i+1) 同时算 tile(i)
    for i in range(NUM_TILES - 1):
        next_tile = i + 1
        wf = front_cnt % N
        rf = post_cnt % N
        # 写侧:搬 next_tile 进 buffer(front_cnt%2)
        b0, b1, front_cnt2, w = producer(front_cnt, new_dep_val=next_tile,
                                         buff0=buff[0], buff1=buff[1])
        # 读侧:算 tile i,读 buffer(post_cnt%2)
        selected, post_cnt2, r = consumer(post_cnt, old_buff0=buff[0], old_buff1=buff[1])
        rows.append({"phase": f"iter{i}",
                     "action": f"load tile{next_tile} + compute tile{i}",
                     "front_cnt_before": front_cnt, "writes_buffer": w,
                     "post_cnt_before": post_cnt, "reads_buffer": r,
                     "compute_reads_tile": selected,
                     "same_buffer": (w == r),
                     "overlap": (w != r),
                     "front_cnt_after": front_cnt2, "post_cnt_after": post_cnt2})
        buff = [b0, b1]
        front_cnt = front_cnt2
        post_cnt = post_cnt2

    # epilogue:算最后一块 tile(NUM_TILES-1)
    last = NUM_TILES - 1
    selected, post_cnt2, r = consumer(post_cnt, old_buff0=buff[0], old_buff1=buff[1])
    rows.append({"phase": "epilogue", "action": f"compute tile{last}",
                 "post_cnt_before": post_cnt, "reads_buffer": r,
                 "compute_reads_tile": selected, "post_cnt_after": post_cnt2,
                 "overlap": False})
    return rows


def timeline_totals():
    """m3 量化:单缓冲串行 vs 双缓冲重叠 的总耗时(演示延迟单位)。"""
    serial = NUM_TILES * (T_LOAD + T_COMPUTE)
    # 双缓冲(DMA 受限,T_LOAD>T_COMPUTE):prologue 一次 load + N 次背靠背 load + 尾计算
    double = T_LOAD + (NUM_TILES - 1) * T_LOAD + T_COMPUTE
    return {"serial_total": serial, "double_total": double,
            "speedup_x": round(serial / double, 2),
            "asymptotic_serial_per_tile": T_LOAD + T_COMPUTE,
            "asymptotic_double_per_tile": max(T_LOAD, T_COMPUTE),
            "asymptotic_speedup_x": round((T_LOAD + T_COMPUTE) / max(T_LOAD, T_COMPUTE), 2)}


def iterarg_expand():
    """m1:一份 buffer dep 扩容后新增的 iterArg 数(DAGSSBuffer.cpp:L4528-L4547)。"""
    buffer_copies_added = N - 1   # for i in [0, bufferNum-1): push buffs 副本
    counters_added = 2            # for i in [0,2): push counterInit
    return {"buffer_copies_added": buffer_copies_added,
            "counters_added": counters_added,
            "total_new_iterargs_per_dep": buffer_copies_added + counters_added,
            "extra_arg_base_idx_stride": 2 + N - 1}  # L5177/L5182 反算用


def main():
    prod, cons = trace_producer_consumer()
    out = {
        "meta": {"N": N, "num_tiles": NUM_TILES, "T_load": T_LOAD, "T_compute": T_COMPUTE,
                 "note": "manual/演示用:pin 无法编译,此为 select 逻辑的忠实 Python 重写"},
        "m1_iterarg_expand": iterarg_expand(),
        "m2_producer": prod,
        "m2_consumer": cons,
        "m3_overlap_rows": trace_overlap(),
        "m3_timeline_totals": timeline_totals(),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open("double_buffer.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
