# ch24 跨核同步夹具 —— 手抄地面真值(补 extract 脚本的去重盲区)

`extract_fixture_checks.py` 用函数名做 key,同名函数(如两个 `test_block_sync_loop`)会
互相覆盖。这里手抄补齐跨核相关夹具的关键 CHECK 行 + 行号,供 m9/m10 逐轮表溯源。

## m9 主例 — inject-block-sync.mlir @test_block_sync_normal (MIX_CV)
源:test/Dialect/HIVM/inject-block-sync.mlir:L32-L70,fusion_kind=MIX_CV,func_core_type=MIX
裸 IR 关键计算 op(同步前):
- nd2nz(gm→cbuf) ×2, mmadL1(cbuf→cc, Cube M 引擎), fixpipe(cc→gm arg2, Cube FIX)  ← Cube 段
- load(gm arg2→ub, Vector MTE2), vadd(ub, Vector V), store(ub→gm arg3, Vector MTE3)  ← Vector 段
跨核依赖:fixpipe 写 arg2(gm) → load 读 arg2(gm),Cube→Vector 经 gm。
注入(L58-L59):
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0     // 挂 fixpipe 后
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0  // 挂 load 前

## m9 对照 — inject-block-sync.mlir @matmul_add_mul (SHALLOW_CV)
源:test/Dialect/HIVM/inject-block-sync.mlir:L13-L28,fusion_kind=SHALLOW_CV,func_core_type=MIX
先 set_ffts_base_addr %arg4(L15)。matmul(Cube)→ call @add_mul_0(Vector 的 vadd/vmul)。
注入(L18-L20):
    hivm.hir.sync_block[<ALL_CUBE>, 0 : i64] tcube_pipe = <PIPE_FIX>          // 整核粗粒度 barrier
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1        // 细粒度置位
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1     // 细粒度等待

## m10 主例 — sync-solver-cross-core.mlir @test_block_sync_loop (CrossCoreGSS)
源:test/Dialect/HIVM/sync-solver-cross-core.mlir:L60-L90
RUN pipeline:hivm-cross-core-gss{always-use-pipe-s=true use-different-multibuffer-flag-ids=true}
先 set_ffts_base_addr %arg5(L61)。循环体内 Cube(mmadL1+fixpipe 写 arg2)与 Vector
(load 读 arg2 + vadd + store)交替;arg2(gm)是跨核交换缓冲,且跨迭代复用。
两个**全局** flag id 并存(scope 恒 0,由 flagIdCnt 递增分配):
    L69  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1   // 提到循环前(首次)
    L77  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1    // 循环体:Cube 等 Vector 上轮读完 arg2 再 fixpipe 覆写(back-edge, WAR)
    L79  hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0      // Cube fixpipe 写完 arg2 置位
    L82  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0   // Vector load 前等 arg2 就绪(RAW)
    L84  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1   // Vector 读完 arg2,置位供下轮 Cube
    L88  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1    // 沉到循环后(末次)
flag=0 = Cube→Vector 的 RAW(fixpipe 产 arg2、load 消费);flag=1 = Vector→Cube 的 WAR
back-edge(Vector 读完 arg2、Cube 下轮才能覆写)。flag=1 被外提循环(set 前 / wait 后)。

## 源码常量(file:Lxxx)
- 核内 event id 每池上限 kTotalEventIdNum = 8
  → include/bishengir/Dialect/HIVM/Transforms/InjectSync/SyncEventIdAllocation.h:L29
- 跨核 block sync flag 上限 kBlockSyncSetWaitEventIdNum = 16
  → 同上 SyncEventIdAllocation.h:L35;分配处 0x0f & flagIdCnt++
  → lib/Dialect/HIVM/Transforms/InjectBlockSync.cpp:L129
- 涉 PIPE_S 的核内同步保留位、"currently auto sync only 6 can be used"
  → lib/Dialect/HIVM/Transforms/InjectSync/SyncEventIdAllocation.cpp:L224-L226
- 跨核 block sync 保留 event id reservedBlockSyncEventIdNum = 2
  → SyncEventIdAllocation.cpp:L170
- op→pipe:load=PIPE_MTE2 / store=PIPE_MTE3 / fixpipe=PIPE_FIX(OpPipeTrait 静态标注)
  → include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td:L64 / L146 / L272
- CopyOp 动态 pipe:UB→UB=V, L0C(cc)→GM=FIX, GM→L1(cbuf)=MTE2, UB→L1=MTE3
  → lib/Dialect/HIVM/IR/OpPipeInterface/GetPipe.cpp:L41-L45
- memref<16x16x16xf16> 字节数 = 4096 elem × 2 B = 8192 B(解释夹具里 0/8192/16384/24576 偏移)
