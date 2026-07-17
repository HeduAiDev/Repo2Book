#!/usr/bin/env python3
"""ch30 环形缓冲游标演化 — 用真实 ttgir 里提取的常量复算 insert/extract 游标。

常量全部取自 pin 真编译产物 matmul_sm90_ns3.ttgir.mlir：
  - numBuffers = 3            <- local_alloc !tt.memdesc<3x128x64xf16,...>（第 0 维）
  - insertIdx 初值 = 1        <- iter_args %arg16 = %c1_i32
  - extractIdx 初值 = -1      <- iter_args %arg17 = %c-1_i32
  - 游标递推 idx=(idx+1)<3 ? idx+1 : 0   <- 循环体 addi/cmpi slt c3/select
  - 稳态发射谓词 iv < ub-2    <- %78=subi ub,c2 ; %79=cmpi slt iv,%78（maxStage=2）
  - prologue 预填槽位 0,1     <- memdesc_subview %42[0]/%42[1]
复算 = 把真实 IR 的标量递推按迭代展开，非另造模型。
"""
import json

NUM_BUFFERS = 3          # matmul_sm90_ns3.ttgir.mlir:L61  memdesc<3x128x64...>
INSERT_INIT = 1          # matmul_sm90_ns3.ttgir.mlir:L83  %arg16 = %c1_i32
EXTRACT_INIT = -1        # matmul_sm90_ns3.ttgir.mlir:L83  %arg17 = %c-1_i32
MAXSTAGE = 2             # num_stages-1
TRIP = 6                 # 演示用 K/BLOCK_K 迭代数（心算跟得上）


def wrap(idx):
    nxt = idx + 1
    return nxt if nxt < NUM_BUFFERS else 0


def main():
    # Prologue：填 maxStage(=2) 段，把迭代 0、1 的数据 async_copy 进槽 0、1。
    filled = {0: "iter0", 1: "iter1"}
    prologue = [
        {"segment": 0, "guard": "trip>0", "write_slot": 0, "data": "iter0"},
        {"segment": 1, "guard": "trip>1", "write_slot": 1, "data": "iter1"},
    ]

    # 稳态循环：每迭代先推进 extractIdx 读、再推进 insertIdx 写。
    insert = INSERT_INIT
    extract = EXTRACT_INIT
    rows = []
    for it in range(TRIP):
        extract = wrap(extract)      # %82 = select(...)  读游标
        read_slot = extract
        read_data = filled.get(read_slot, "?")
        insert = wrap(insert)        # %92 = select(...)  写游标
        write_slot = insert
        # 稳态发射谓词：仅当 iv < trip-2 才真正 async_copy（谓词化收尾）
        emit = it < (TRIP - MAXSTAGE)
        prefetch_iter = it + MAXSTAGE if emit else None
        if emit:
            filled[write_slot] = f"iter{prefetch_iter}"
        rows.append({
            "iter": it,
            "extractIdx": extract,
            "read_slot": read_slot,
            "consumes": read_data,
            "insertIdx": insert,
            "write_slot": write_slot,
            "predicate_iv_lt_trip_minus_2": emit,
            "prefetches": f"iter{prefetch_iter}" if emit else "closed",
        })

    out = {
        "constants": {
            "numBuffers": NUM_BUFFERS, "insert_init": INSERT_INIT,
            "extract_init": EXTRACT_INIT, "maxStage": MAXSTAGE, "trip": TRIP,
        },
        "prologue": prologue,
        "steady": rows,
    }
    with open("ringbuffer_trace.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
