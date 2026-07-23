# m5 — TritonToHIVM：sync_block_* 分派 + GetCoreAndPipes 落核翻转（手工推演，manual）

**为何 manual**：skip_impl，无 .py；不伪造 dump。逐值代入
`lib/TritonToHIVM/TritonToHIVM.cpp` 的 GetCoreAndPipes（L71-99）与 matchAndRewrite（L115-164）。

## ascend.custom 的载荷解析（L119-123）
- `args = op.getStrArgs()`；`arg = args[0]`（sender 核名："cube"/"vector"，或 all 路径的 all_cube…）
- `id = args[1].getInt()`（事件号）；`opName = op.getOpName()`（sync_block_all/set/wait）

## GetCoreAndPipes 两步（L71-99）——只被 set/wait 走
- Step1 pipe（L74-82）：consumer 恒 PIPE_MTE2（L76）；producer：sender=="cube"→PIPE_FIX（L79），否则 PIPE_MTE3（L81）
- Step2 core（L84-96）：
  - sender=="cube"：opName=="sync_block_set"→CUBE（L88），否则(wait)→VECTOR（L90）
  - sender!="cube"（vector）：set→VECTOR（L93），否则(wait)→CUBE（L95）

## 2×2 全枚举（id=3）
| # | op_name         | arg(sender) | producer(step1) | consumer(step1) | core(step2)        | 产物 hivm op                              |
|---|-----------------|-------------|-----------------|-----------------|--------------------|-------------------------------------------|
| 1 | sync_block_set  | cube        | PIPE_FIX (L79)  | PIPE_MTE2 (L76) | CUBE (L88)         | hivm.sync_block_set(CUBE,FIX,MTE2,id=3)   |
| 2 | sync_block_wait | cube        | PIPE_FIX (L79)  | PIPE_MTE2 (L76) | VECTOR (L90)       | hivm.sync_block_wait(VECTOR,FIX,MTE2,id=3)|
| 3 | sync_block_set  | vector      | PIPE_MTE3 (L81) | PIPE_MTE2 (L76) | VECTOR (L93)       | hivm.sync_block_set(VECTOR,MTE3,MTE2,id=3)|
| 4 | sync_block_wait | vector      | PIPE_MTE3 (L81) | PIPE_MTE2 (L76) | CUBE (L95)         | hivm.sync_block_wait(CUBE,MTE3,MTE2,id=3) |

**读法**：固定 sender，set 落 sender 自身核、wait 落对端核（#1↔#2：CUBE↔VECTOR；#3↔#4：VECTOR↔CUBE）
——set 是「我在自己核上发信号」、wait 是「对端核阻塞等信号」。producer pipe 只由 sender 决定
（cube→FIX、vector→MTE3），consumer 恒 MTE2，与 op_name 无关。

## all 路径（不经 GetCoreAndPipes，走 CreateSyncBlock，L125-145）
op_name=="sync_block_all" 时按 arg 三分派（硬编码 mode+pipe）：
- all_cube → SyncBlockMode::ALL_CUBE，pipe=(FIX, {})（L126-130）
- all_vector → ALL_VECTOR，pipe=({}, MTE3)（L131-135）
- all → ALL，pipe=(FIX, MTE3)（L136-140）
- 其它 → EmitUnknownOpError（L141-142）

## 计数
HIVM 舱 = **1** pattern（TritonCustomOpToHIVMSyncOpConversion，runOnOperation L174）；
op_name 三分派（all/set/wait），all 再分 3 子模式；未知 op_name/arg 一律 EmitUnknownOpError（L54-58）。
