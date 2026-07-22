# ch12 手工推演台账（trace_source=manual）

**为何 manual**：本章 `skip_impl=true`——triton-to-linalg 是纯 C++ MLIR pass，
无精简版可跑；宿主机无 CANN、无法产真实编译器 dump（对齐姊妹篇 ch25/28/30/32/33 惯例）。
交叉验证走两条：① pin 精确源码 `BlockPtrAnalysis.cpp/.h @2badfc89e` + `LoadStoreConverter.cpp`
的字面语义（每个引用源码常量的数字标 `file:Lxxx`）；② `unittest/Conversion/**/*.mlir`
lit 夹具的 RUN+CHECK 前后对照作 IR 素材（真实存在于 pin 内，非伪造 dump）。

下表每个数字要么是**explainer 自选的小而具体参数**（明确标「chosen」），
要么可溯源到源码常量（标 `file:Lxxx`）或 lit 夹具 CHECK 行。

---

## m3 createCastOp：三元组 → memref.reinterpret_cast

源码：`BlockPtrAnalysis.cpp` createCastOp `L322-L343`，inferBlockOffset `L144-L151`，
getResultMemrefType `L153-L165`。

选定输入 BlockData（chosen，2 维，非退化）：
- offsets = [8, 3]（chosen，各维元素偏移贡献）
- sizes   = [4, 2]（chosen）
- strides = [2, 1]（chosen）
- source  = `memref<?xf32>`

推演：
1. inferBlockOffset（L146-L150）：retOffset = 0；+8 → 8；+3 → 11。**总 offset = 11**。
   （各维 offset 累加成单一线性偏移——reinterpret_cast 只收一个 offset 参数。）
2. getResultMemrefType（L153-L165）：offset 是常量 11 → 传 11（否则 `ShapedType::kDynamic`）；
   StridedLayoutAttr(offset=11, strides=[2,1]) → 结果类型 `memref<4x2xf32, strided<[2,1], offset:11>>`。
3. size==1 维 stride 抬升（L325-L339）：sizes=[4,2] 无 size==1 维 → **不触发** MaxSIOp。
   （抬升仅对 `resultShape[i]==1` 且 stride 是动态 Value 的维，抬成 `max(stride,1)`，L331-L336。）
4. 发射（L341-L342）：`memref.reinterpret_cast %src to offset: [11], sizes: [4, 2], strides: [2, 1]`。

IR 形态交叉参考 `legal_stride.mlir`：`memref<?xf32> to memref<4x1xf32, strided<[?,?], offset: ?>>`
（该夹具 offset 是动态 loop iv %arg13，故 offset:? 、strided 动态；此处我们选常量演示塌缩）。

---

## m5 rewriteAddPtr：零 stride 修复（inferedSize 逆序扫）+ 落地驱动

源码：rewriteAddPtr `L1125-L1214`，Unstructured 分岔判定 `L1135-L1142`，
inferedSize 逆序循环 `L1162-L1178`（shouldReplaceStride `L1173`，替换 `L1174-L1176`，
`inferedSize *= sizeConst L1177`），known 存未修改态 `L1160`。

参数直接取 lit 夹具 `legal_stride.mlir`（可 CHECK 对照）：
- sizes   = [4, 1]（夹具 `sizes: [4, 1]`）
- strides 入 = [%c4, %c0]（夹具输入 `strides: [%c4, %c0]`，即 [4, 0]）
- MemAccType = StrucMemAcc（无间接 load）

inferedSize 逆序扫（L1170-L1178，初值 inferedSize=1）：
| i | sizeConst | strideConst | shouldReplace | strides[i] ← | inferedSize 累积 |
|---|-----------|-------------|---------------|--------------|------------------|
| 1 | 1         | 0           | true(size==1) | 1 (= inferedSize) | 1×1 = 1 |
| 0 | 4         | 4           | false         | 4 (不变)     | 1×4 = 4 |

结果 strides = [4, 1]。**与夹具输出 CHECK `strides: [%c4, %c1]` 逐位一致**（0→1 的抬升）。
known[result] 里存的仍是原始 [4, 0]（L1160+注释 L1163-L1165，分析态/物化态解耦）。
最后 createCastOp 发射 `memref.reinterpret_cast ... sizes:[4,1] strides:[%c4,%c1]`（L1195, replaceOp L1200）。

---

## m6 MemAccType 决策 + gather 回退

源码：MemAccVal 枚举 `H:L46`（Undefined=0, StrucMemAcc=1, UnstrucMemAcc=2），
merge 取大 `H:L66-L68`；parseIndirectLoad UnstrucMemAcc 落点 `L989`（scalar<1> 降级
StrucMemAcc `L982`，标量 offset `L992`）；rewriteAddPtr 分岔 `L1135-L1142`；
gather 回退 rewriteAddPtrToUnstrucMemAcc `L2158-L2232`（forUBs←blockSizes `L2185-L2189`，
combinedOffset=base+scalarOffset `L2216-L2217`，单元素 sizes[1]`L2222`/strides[1]`L2224`，
createCastOp {1} `L2225`，IndirectLoad 标 `L2229`）。

### merge 传播（lattice 全序取上确界）
| 地址链节点 | 本节点 MemAccVal | merge 后累积 |
|-----------|------------------|--------------|
| base splat ptr | StrucMemAcc = 1 | 1 |
| idx = tt.load(...) （非 <1> 张量）| UnstrucMemAcc = 2 (L989) | max(1,2) = 2 |
| addptr(base, idx) | merge → 2 | 2 → 走 gather (L1135) |

### gather 循环逐元素（chosen idx 张量 = [10, 3, 7, 1]，base = 100；N=4）
forUB = blockSizes = 4（L2185-L2189，chosen N=4）。每 iter：
extract 散乱 offset（L2210）→ IndexCast → combinedOffset = base + scalarOffset（L2216-L2217）
→ 单元素 reinterpret_cast sizes:[1] strides:[1]（L2222-L2225）。
| 循环 iv | extract offset | + base | combinedOffset | 发射 |
|---------|----------------|--------|----------------|------|
| 0 | 10 | 100 | 110 | reinterpret_cast offset:[110] sizes:[1] strides:[1] |
| 1 | 3  | 100 | 103 | reinterpret_cast offset:[103] sizes:[1] strides:[1] |
| 2 | 7  | 100 | 107 | reinterpret_cast offset:[107] sizes:[1] strides:[1] |
| 3 | 1  | 100 | 101 | reinterpret_cast offset:[101] sizes:[1] strides:[1] |
（idx 值与 base 为 chosen；combinedOffset 为算术；sizes/strides=1 源自 L2222/L2224。）

### 复杂度量化（theory 第 4 条）
- 结构化：1 条 reinterpret_cast + 1 条 memref.copy 描述整块 N=4 数据 → **2 条 op（O(1)）**。
- gather：4 次循环 × (1 extract + 1 reinterpret_cast + 1 memref.load) = **12 条 op（O(N)）** + 循环开销。

---

## m10 rewriteMakeTensorPtrOp：block_ptr 物化 + 转置维序（回收 f1）

源码：rewriteMakeTensorPtrOp `L1277-L1361`，newOffsets=offset×stride `L1311-L1313`，
accumulatePotentialOffsetOnBase `L1216-L1228`（front 累加 base recast offset `L1327`），
createRedundantOp 全 shape + tensor_ptr_full_shape 标 `L1232-L1275`/`L1354-L1355`，
目标 createCastOp `L1360`；转置 boundary_check 置换公式 `LoadStoreConverter.cpp:L350-L353`。

选定参数（chosen，row-major parent，非退化）：
- parent shape = [128, 64]，parent strides = [64, 1]
- block sizes  = [16, 32]
- block offsets（make_tensor_ptr 的 offset 操作数）= [2, 1]
- base recast offset = 0（源自原生 ptr，无前置 addptr 偏移）

推演：
1. parse base（L1287-L1297）→ source `memref<?xf32>`；offsets 各维过 `max(v,0)`（L1301-L1302）
   → [2, 1]；strides → [64, 1]。
2. newOffsets[i] = offset[i] × stride[i]（L1311-L1313）：[2×64, 1×1] = **[128, 1]**。
3. accumulate base 于 front（L1327）：newOffsets[0] = 128 + 0 = **128**。
4. createRedundantOp（L1232-L1275）：sizes ← parent 全 shape [128, 64]，各维 offset=[0, 0]
   （front 累加 base=0），发射一条冗余 reinterpret_cast 并打 `tensor_ptr_full_shape` 标
   （L1355），供 load/store 的 boundary_check 读全 shape。
5. 目标块 createCastOp（L1360）：inferBlockOffset = 128 + 1 = **129**；
   → `memref.reinterpret_cast %redundant to offset: [129], sizes: [16, 32], strides: [64, 1]`。
   （每个 block_ptr 物化出 **2** 条 reinterpret_cast：冗余全 shape + 目标块。）
6. 转置维序（`LoadStoreConverter.cpp:L350-L353` 源码注释自带示例，逐字引用）：
   `original_order = (0, 1)`, `boundary_check = (1,)` → `new_boundary_check[0] = (rank-1)-pos = 2-1-1 = 0`
   （pos=1 因 original_order[1]==boundary_check[0]==1；rank=2）。
   即 make_tensor_ptr 被 permute 后，boundary_check 轴号按此公式回改，保持边界检查落在正确轴。
