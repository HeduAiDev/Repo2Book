# LinalgToHFusion 上抬素材(pin 内 lit 夹具 CHECK,非 host 运行 dump)

来源(逐字对 pin,行号为 pin 基线):
- `third_party/ascend/AscendNPU-IR/bishengir/test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir`
- `third_party/ascend/AscendNPU-IR/bishengir/test/Conversion/LinalgToHFusion/arange.mlir`

RUN 行:`bishengir-opt -convert-linalg-to-hfusion %s -split-input-file -verify-diagnostics | FileCheck %s`。
下列「上抬后」= 夹具里作者写死的 `// CHECK:` 期望 IR(即 pass 的合约输出)。host 无 bishengir-opt,
未实跑,故 trace_source=manual;CHECK 行是权威期望值。

## m7 — 四 pattern 各一例(match 什么 linalg → produce 什么 hfusion)

### ① LinalgMapToHFusionPattern(一元)  fixture L4–L10
IN : `linalg.map { func.call {callee = @__hmf_reluDh} } ins(%arg0 : tensor<6x6xf16>) outs(%arg0 ...)`
OUT: `hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}`

### ①' LinalgMapToHFusionPattern(二元)  fixture L151–L157
IN : `linalg.map { func.call {callee = @__hmf_ldexpDh} } ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) ...`
OUT: `hfusion.elemwise_binary {fun = #hfusion.binary_fn<ldexp>}`

### ② LinalgGenericToHFusionArangePattern  arange.mlir L10–L15 → L9
IN : `linalg.generic {…iterator_types=["parallel"]} outs(%arg0 : tensor<6xi32>) { ^bb0(%out): %0=linalg.index 0; %1=arith.index_cast %0; linalg.yield %1 }`
OUT: `hfusion.arange offset[%c0] strides[%c1] outs(%arg0 : tensor<6xi32>) -> tensor<6xi32>`  (%c0=0, %c1=1)

### ③ AtomicLinalgGenericToHFusionStorePattern  fixture L162–L176
IN : `linalg.generic {…} attrs = {GenericAtomicRMW = "fadd", …} { ^bb0: arith.addf; linalg.yield }`
OUT: `hfusion.atomic_rmw ins(...) outs(...) atomic_kind = <add>`
     (同 pattern:cas→hfusion.atomic_cas L180–196;exch→hfusion.atomic_xchg L200–214;umax→atomic_rmw<umax> L333;umin→atomic_rmw<umin> L351)

### ④ LinalgToHFusionReduceWithIndex  fixture L218–L243
IN : `linalg.reduce ins(...) outs(...) dimensions=[1] {reduce_mode = "max_with_index", tie_break_left = "true"} {...}`
OUT: `hfusion.reduce_with_index {tie_break_left = true} <max> ins(...) outs(...) dimensions = [1]`
     (ui8 输入 → `<maxui>` L369–L387;`annotation.mark %argN {UseIndexInput}` → 带 index 输入变体 L247–L273)

合法性框架(LinalgToHFusion.cpp:L477–L499):
- `addIllegalOp<linalg::MapOp>()` + `addIllegalOp<linalg::GenericOp>()`  → map/generic 必须被消费
- `addDynamicallyLegalOp<linalg::ReduceOp>`:仅当**无** `reduce_mode` 属性时合法(带标记才 illegal)
- legal 方言:memref / linalg / bufferization / tensor / hfusion / arith / math
- 注册 4 pattern(L465–L466),`applyPartialConversion`

## m8 — elementwise 上抬的 hfusion↔linalg 边界(同一夹具逐 case CHECK)

上抬到 **hfusion**(NPU 扩展词汇):
- relu   fixture L7   → `hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}`
- sqrt   L17          → `hfusion.elemwise_unary <sqrt>`
- rsqrt  L58          → `hfusion.elemwise_unary <rsqrt>`
- log1p  L94          → `hfusion.elemwise_unary <log1p>`
- tan    L114/L326    → `hfusion.elemwise_unary <tan>`
- tanh   L124/L316    → `hfusion.elemwise_unary <tanh>`
- atan   L134         → `hfusion.elemwise_unary <atan>`
- ilogb  L144         → `hfusion.elemwise_unary <ilogb>`
- ldexp  L154         → `hfusion.elemwise_binary <ldexp>`
- powf   L298         → `hfusion.elemwise_binary <powf>`
- powi   L306         → `hfusion.elemwise_binary <powi>`

保持 **linalg**(上游 Linalg 原生词汇,不重复造轮子):
- fabs   L27          → `linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}`
- exp    L37          → `linalg.elemwise_unary <exp>`
- log    L47          → `linalg.elemwise_unary <log>`
- recip  L68–L71      → `arith.constant 1.000000e+00` + `linalg.fill` + `linalg.elemwise_binary {fun = #linalg.binary_fn<div>}`  (recip = 1.0 / x,降解成 linalg fill+div)

边界判据(LinalgToHFusion.cpp 分支,dossier code_spine L62–L146):按 `func.call` 的 callee 名 `__hmf_<fn>` 分派;
落在上游 linalg 词汇表内 → 发 linalg 具名 op;NPU 扩展词汇 → 上抬 hfusion。
