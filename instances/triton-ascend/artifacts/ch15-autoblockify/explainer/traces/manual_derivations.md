# ch15 AutoBlockify — 手工推演原始输出（trace_source=manual）

> **为何 manual**：本章 skip_impl（纯 C++ MLIR pass，无 .py 精简版，宿主无 CANN 编不动 `triton-opt`，
> 无法在 host 上跑出 dump）。交叉验证走 pin `2badfc89e70a9b7a5e88463a116c2feddce4b101` 内的 lit 夹具
> `third_party/ascend/unittest/Conversion/General/AutoBlockify/auto_blockify.mlir`（RUN+CHECK 前后对照）作 IR 素材。
> 下面每个标量数字均为**按源码常量手算**，出处标 `file:Lxxx` 或夹具 `auto_blockify.mlir:Lxxx`。
> **不伪造编译器 dump**——张量形状/算子链一律取自夹具 CHECK 断言。

## 共用的 worked-example 参数（chosen，非源码常量）

- 逻辑网格 grid = 3 × 2 × 1（numX=3, numY=2, numZ=1）→ 逻辑实例总数 G = 3·2·1 = **6**（chosen）。
- `autoBlockifySize` = **5**（源码常量：夹具 RUN 行 `auto_blockify.mlir:L1` `auto-blockify-size=5`）。
- 张量形状（源码常量）：夹具输入前导张量 `tensor<8x…>`（`auto_blockify.mlir:L48/L53`），折叠后 `tensor<5x8x…>`（`auto_blockify.mlir:L7/L35`）。

---

## m2 网格拍平 + blockifiedId 载体构造（preProcess）

源码：`AutoBlockify.cpp:L197-L249`（拍平），`L251-L271`（反解）。

### 拍平（logicalBlockNum / logicalBlockId）
- `logicalBlockNum = numX·numY·numZ = 3·2·1 = 6`（`AutoBlockify.cpp:L204-L205`；夹具 `L13-L14` VAL_7/VAL_8）。
- 取第 0 号物理块 idX=0, idY=0, idZ=0：
  `logicalBlockId = idX·(numY·numZ) + idY·numZ + idZ = 0·2 + 0·1 + 0 = 0`
  （`AutoBlockify.cpp:L217-L220`；夹具 `L18-L21` VAL_12..VAL_15）。

### blockifiedId 与 mask
- `blockifiedId = splat(logicalBlockId) + makeRange(0,size) = splat(0) + [0,1,2,3,4] = [0,1,2,3,4]`
  （`AutoBlockify.cpp:L222-L231`；夹具 `L16-L18,L22-L24` VAL_16/17/18）。
- `upperboundMask = (blockifiedId slt logicalBlockNum=6) = ([0,1,2,3,4] < 6) = [T,T,T,T,T]`
  （`AutoBlockify.cpp:L235-L236`；夹具 `L25-L26`）。
- `lowerboundMask = (blockifiedId sge 0) = ([0,1,2,3,4] >= 0) = [T,T,T,T,T]`
  （`AutoBlockify.cpp:L240-L241`；夹具 `L27`）。
- **`blockifiedIdMask = upperboundMask ORI lowerboundMask`**（`arith.ori`，非 `and`！
  `AutoBlockify.cpp:L242-L243`；夹具 `L28` `arith.ori`）：`= [T,T,T,T,T]`。

> **⚠️ ori 事实（dossier 决策 d7）**：`blockifiedId = logicalBlockId + range ≥ 0` **恒成立**（logicalBlockId≥0、range≥0），
> 故 `lowerboundMask` **永远全 True** → `ori` 结果**永远全 True**。即：载体里携带的 mask 对合法折叠不做任何屏蔽。
> 尾块「不足 size」的真正截断在 **blockify 循环上界**（m6：`min(max(blockNum-blockId,0),size)`）与 driver 的 blockNum clamp，
> **不在这条 mask**。正文描述 mask 合成必须逐字照写 `ori`，不得脑补成「越界 lane 被 and 掉」。

### 反解 program-id（第 0 号物理块，`AutoBlockify.cpp:L256-L271`；夹具 `L29-L32`）
- X 维：`(blockifiedId divsi yzNum=2) remsi numX=3` = `([0,0,1,1,2]) % 3` = `[0,0,1,1,2]`
- Y 维：`(blockifiedId divsi zNum=1) remsi numY=2` = `([0,1,2,3,4]) % 2` = `[0,1,0,1,0]`
- Z 维：`blockifiedId remsi zNum=1` = `[0,0,0,0,0]`
- 校验（k → 网格坐标，k = x·2 + y·1 + z）：
  0→(0,0,0)  1→(0,1,0)  2→(1,0,0)  3→(1,1,0)  4→(2,0,0) ✓ 与上一致 ✓

---

## m4 checkBlockifiable（use-def 递归守门）

源码：`AutoBlockify.cpp:L137-L191`；调用点 `L298-L311`。夹具两 kernel 对照。

- **kernel（`auto_blockify.mlir:L47-L59`）**：`program_id x` → muli → splat → addi → addptr → store。
  链上无硬拒绝 op、无 region op、store 无结果 → 每跳 `checkBlockifiable` 回 true → **整函数可整体批处理**（无循环）。
- **kernel2（`auto_blockify.mlir:L117-L134`）**：`program_id y (%a)` → `arith.cmpi (%b)` → `scf.if (%b)`。
  遇 `scf::IfOp`（`AutoBlockify.cpp:L154-L156`）：给 scf.if 打 `autoBlockifyRegionOpAttr` 标签、return true
  → 该 if **降级为 blockify 循环**（夹具 `L108-L114` 的 `scf.for … {auto_blockify_loop}`）。
  `program_id x` 链同 kernel，可批处理。
- **硬拒绝分支（源码 `AutoBlockify.cpp:L149-L150`，本夹具未触发）**：user 命中
  `cf::CondBranchOp / triton::IntToPtrOp / scf::WhileOp / triton::DotOp` 之一，或任一操作数是 tensor-ptr 类型
  → return false → `runOnOperation` 对整函数 `emitWarning("Cannot apply auto blockify")` 并 skip（`L306-L309`）。
- 终止性：`checkedValues.insert(v).second` 去重（`L138-L139`），每个 Value 至多深入一次；IR 中 Value 有限 → 有限步停。

---

## m6 blockify 循环上界（createBlockifyLoop）

源码：`Utils.cpp:L137-L145`：`upperBound = minsi( indexcast( maxsi( subi(blockNum,blockId), 0 ) ), size )`。
夹具 `auto_blockify.mlir:L104-L107`（VAL_40 subi / VAL_41 maxsi / VAL_42 index_cast / VAL_43 minsi）。
参数：blockNum=6（grid 3×2×1，chosen），size=5（`L1`）。

| blockId | subi=6-blockId | maxsi(.,0) | index_cast | minsi(.,5) | 迭代次数 | 覆盖逻辑实例 |
|--------:|---------------:|-----------:|-----------:|-----------:|---------:|:-------------|
| 0 | 6 | 6 | 6 | 5 | 5 | 0,1,2,3,4 |
| 5 (尾块) | 1 | 1 | 1 | 1 | 1 | 5 |
| 6 (越界，理论) | 0 | 0 | 0 | 0 | 0 | 无（maxsi 防负→空循环）|

循环体（`Utils.cpp:L165-L196`；夹具 `L109-L112`）：每 iv 从 `tensor<5xi1>` `tensor.extract` 出谓词、
`scf.if` 判定、`tensor.extract_slice %[iv,0][1,8][1,1]` 取第 iv 行 `tensor<8x!ptr>`、单点 `tt.store`。

---

## m9 收益量化（size=5 夹具前后对照）

参数：G=6（grid 3×2×1，chosen），size=5（`L1`）。

| 指标 | 未折叠 | 折叠 size=5 | 出处 |
|:-----|:------:|:-----------:|:-----|
| 调度块数 | G = 6 | ⌈6/5⌉ = 2 | G=grid 3×2×1 (chosen)；⌈G/size⌉ per dossier.theory |
| 前导张量形状 | tensor<8> | tensor<5x8> | `auto_blockify.mlir:L48` vs `L7` |
| 一条 tt.store 覆盖实例数 | 1 | 5 | `L57` vs `L44` |
| 覆盖这 5 实例的 store 指令数 | 5 | 1 | 派生自上一行 |

运行期 driver 再把 `blockNum` 截到物理核数：`blockNum = std::min(blockNum, num_physical_blocks)`
（`third_party/ascend/backend/driver.py:L788`）。启动/调度开销 O(G) → O(⌈G/size⌉)。
