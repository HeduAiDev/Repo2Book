# ch17 手工推演台账（trace_source=manual）

**为何 manual**：本章两文件（`third_party/ascend/lib/TritonAffinityOpt/DAGSync.cpp` 1333 行 +
`DAGScope.cpp` 1139 行）均为纯 C++ MLIR pass，仓内无 `.py` 精简版，宿主无 CANN 环境编不动；
且 `dag-sync`/`dag-scope` 两 pass 本身无 lit 夹具（只经 `compiler.py` 在昇腾硬件上跑），
不得伪造前后 IR dump。故所有逐轮表由**逐字对 pin @2badfc89e 源码手工推演**得到，凡引用源码
常量的数字标 `file:Lxxx`。其中纯整数算术部分（m7 的 nz shape、m15 的 flag%14）额外用
`verify_arith.py` 在 host 上按源码同式复算做交叉验证（输出见 `verify_arith.out`）——该脚本只
replay 源码里的整数公式（确定性、与硬件无关），**不是** pass dump。

贯穿实例：`out = a @ b + bias`（matmul+bias）。`a:[16,32]`、`b:[32,16]` fp16 → `dot` 结果
`[16,16]` 在 cube；`bias:[16,16]` 的加法在 vector。这条 cube→vector 边贯穿 m3→m4→m5→
m11→m12→m13→m14。m6/m7 的 VECTOR→CUBE 方向另取一例（vector 产出的张量喂回后续 dot）。

## m3 LegalizeDot（DAGSync.cpp:L793-843）
`funcOp.walk(DotOp)`：取三操作数 a/b/c(=累加器)。判 `c` 是否 dense splat 0（L805-812）。
`dot(a,b,bias)` 的 bias≠0 → `isZeroAccumulator=false` → 建 splat-0 常量 `zeroConstant`、
`newDot=dot(a,b,zeroConstant)`、`addOp=arith.addf(newDot, bias)`、
`originalResult.replaceAllUsesWith(addOp)`、原 dot `use_empty()` 则 `erase`（L836-840）。
产物：1 条 dot(cube)→addf(vector) 数据边。

## m4 runOnOperation 主 walk（DAGSync.cpp:L1258-1273 判据 + L243-247 needVectorCubeSync）
对每个 op 的每个输入边取 `inputType`，`needVectorCubeSync(inputType,currentType)` 仅当
一端 `CUBE_ONLY`、另一端 `VECTOR_ONLY` 为真。命中则 `processedPairs.insert({inputOp,op})`
去重，首见才插（同 block 走 `insertSyncAndMovement`，flag=`syncFlag%14`，L1281，之后 `syncFlag++`）。
本例：addf 的输入 newDot(cube) → (CUBE,VECTOR)=真 → 插 1 组；addf 的输入 bias(vector)、
store 的输入 addf(vector) → 同核 → 不插。

## m7 CBUF 32B nz 对齐（DAGSync.cpp:L386-419）
`blk = 32 / elem_bytes`（`getBlockElemsFor32BAlign` L386-396，`kAlignBytes=32` L387）；
`newCbubAllocShape` 要求 2D 静态且 `M%16==0`（否则 nullopt L414-415），新 shape=
`(N/blk, M/16, 16, blk)`（L418）。取 memref `[M=32,N=64]`：
| 类型 | elem_bytes | blk=32/eb | nz shape | 最内维字节 |
|---|---|---|---|---|
| fp16 | 2 | 16 | (4,2,16,16) | 16*2=32 |
| fp32 | 4 | 8  | (8,2,16,8)  | 8*4=32  |
| int8 | 1 | 32 | (2,2,16,32) | 32*1=32 |
总元素守恒 (N/blk)*(M/16)*16*blk = N*M = 2048。源码注释 L417 写死 (N/16,M/16,16,16) 是
fp16 特例，非 fp16 以代码 blk 为准（dossier embed L398-420 已标注）。verify_arith.out 复算一致。

## m8 processScfForSync（DAGSync.cpp:L106-212）
遍历 `initArgs`（L122）：迭代参数 `iterArg=loopBody->getArgument(i+1)`；`firstUser`=循环体内
首个用它的非 yield op（L125-146），`iterType`=firstUser 结果核；`yieldOperand=yield->getOperand(i)`，
`yieldType`=其核，`yieldDefiningOp`=其定义 op（L154-161）。
- `yieldType==CUBE_ONLY && iterType==VECTOR_ONLY`（L164）→ set(CUBE,PIPE_FIX/PIPE_V) 在
  yieldDefiningOp 后、`insertCubeToVectorDataMovement`、wait(VECTOR) 在 firstUser 前（L166-187）。
- `yieldType==VECTOR_ONLY && iterType==CUBE_ONLY`（L190）→ set(VECTOR,PIPE_MTE3/PIPE_MTE1)、
  wait(CUBE)；**注意数据搬运调用被注释掉**（源码 L202 为注释）——如实标注。
- 其余组合不插。

## m9 落点微调（DAGSync.cpp:L547-602）
`FindEarliestPosition(dstOp)`（CUBE→VECTOR 的 wait 用，L588-602）：从 dstOp 向上 `getPrevNode`，
`insertPos=prevOp` 逐步上移，遇 `SyncBlockSetOp(VECTOR)` 立即 `return insertPos`（停在既有屏障后），
跨 block 的 prevOp 跳过。
`FindLastestPosition(srcOp)`（VECTOR→CUBE 的 set 用，L547-586）：从 srcOp 向下 `getNextNode`
找首个「非 CUBE_ONLY 且操作数里有 CUBE_ONLY 定义」的 op；命中后向上回溯到首个
`BroadcastOp`（若其前是 `ExpandDimsOp` 则取 ExpandDims），把 set 插到 shape 最小处以防
UB 溢出（L560-568 注释）；途中遇 `SyncBlockWaitOp(VECTOR)` 立即返回。

## m10 addMemEffectsSync（DAGSync.cpp:L1085-1105 判据 + L1116 pipe/flag）
`SharedMemoryAliasAnalysis` 收 Read/Write 效应；对 `(prevEffect, effect)` 若
`(prevWrite || isWrite) && mayAlias && prevNode->isOn()!=currNode->isOn()`（L1087-1091）→
`srcCoreType = isWrite ? !currNode->isOn() : prevNode->isOn()`（L1092），
`findAncestorCommonBlock` 求公共祖先 block（L1093），`setAfter==waitBefore` 则跳过（L1100-1102），
否则 push `SyncCandidate`。后按 dominance/postDominance 排序去重，成对插
set(srcPipe=CUBE?PIPE_FIX:PIPE_MTE2)/wait(PIPE_S)，flag=`syncFlag%14`（L1114-1117）。

## m11 encapsulateWithScope（DAGScope.cpp:L69-151）
收 lastBlock 内非 constant/programid/alloc 的 op 按序（L69-138）→ 建第一个 `scope.scope`
`scopeOp` 搬入全部、补 `scope.return` → `scopeOp.setAttr(tcore_type=VECTOR)`（L146-148）→
建第二个空 `scope.scope` `newScopeOp` 打 `tcore_type=CUBE` → `return {scopeOp(aiv), newScopeOp(aic)}`
（L150）。恰 2 个 scope，顺序固定 VECTOR 先、CUBE 后。

## m12 collectOpsToMove（DAGScope.cpp:L158-310）
按结果 `valueTypes` 交 VECTOR_ONLY/CUBE_ONLY 置位（L164-185）；专规：`CopyOp`→仅 aiv（L186-189）、
`FixpipeOp`→仅 cube（L191-194）、`scf::YieldOp||scope::ScopeOp||scf::ForOp`→两边（L196-200）、
store 按操作数核、assert 按 condition 核、`SyncBlockSet/WaitOp` 按 `tcore_type` 属性（L251-261：
CUBE→cube 否则→aiv）。

## m13 SplitScope+processOperationToMove（DAGScope.cpp:L327-690）
两遍（L650-670）：aiv 遍 `MoveType=CUBE_ONLY`（丢 cube 结果）、aic 遍 `MoveType=VECTOR_ONLY`
（丢 vector 结果）；`aivUsedOp`/`aicUsedOp` 防重（L652-657, L666-670）。
`processOperationToMove` 对 scf.for/if/yield 按 MoveType 过滤迭代参数/结果重建（L378-462）。
收尾把原 op 汇入 deleteOp 逆序 erase（L672-688）。

## m14 addSyncOpsForBufferWait（DAGScope.cpp:L693-1074）
按 `tcore_type` 找 aicRegion(CUBE)/aivRegion(VECTOR)（L1054-1064）；`processFixpipeOpsInAIC`
（L885-903）对每个 fixpipe：① fixpipe 前插 wait(CUBE,PIPE_V/PIPE_FIX)，flag=
`findFixPipeFlagSafe(fixpipe)`（取 fixpipe 下一行 sync_block_set 的 static_flag_id，L848-866，
找不到 -1）；② aic 尾 return 前插 wait（`insertWaitBeforeFinalReturn` L725-753）；
③ aiv 头插 set(VECTOR,PIPE_V/PIPE_FIX)（`insertSetAtRegionStart` L756-782）；④ 回程（L905-939）：
fixpipe 后取下个 set 的 flag → aiv 内找同 flag 的 wait → 其后插 set(newflag)。
`static_flag_id` 跨 region 唯一配对 → 每个 wait 有配对 set。

## m15 flag 池与死锁（DAGSync.cpp:L1116/L1241/L1281）
三处 flag 均 `syncFlag % 14`：旗池 14 面（0..13），第 k 条跨核边用 `k%14`。
| syncFlag | flag=%14 |
|---|---|
| 0 | 0 |
| 13 | 13 |
| 14 | 0（回绕复用） |
配对不成（wait 等的 flag 无 set）或跨核互等成环 → 永久阻塞（死锁）。verify_arith.out 复算一致。
