# m6 — TritonToLLVM：tt.elementwise_inline_asm 直落 + 32 位打包（手工推演，manual）

**为何 manual**：skip_impl，无 .py；不伪造 dump。逐值代入
`lib/TritonToLLVM/TritonToLLVM.cpp` 的 packOperands（L52-77）与顶层分派（L238-243）。

## 顶层分派（ElementwiseInlineAsmOpConversion，L238-243）
`matchAndRewrite`：`op.getOperands().empty()` ? processScalarInlineAsm（L171） : processVectorInlineAsm（L187）
- 无操作数 → 标量路径（直接 createDestOps 发一次 inline asm）
- 有操作数 → 向量路径（拆元素→按 packedElement 分块→逐块发→重排回填）

## packOperands 打包算术（L52-77），固定 packedElement = 4
`numPackedElements = op.getPackedElement()`（L56）= 4
对每个操作数：`bitWidth = elemTy.getIntOrFloatBitWidth()`（L59）；
`numElementPerReg = std::max(32/bitWidth, 1u)`（L60），再 `std::min(_, numPackedElements)`（L61）。
每寄存器打 numElementPerReg 个元素（<32 位才向量打包；==1 则 L63-65 逐元素不打包）。
产出的 packed 操作数个数 = numPackedElements / numElementPerReg。

| 元素类型 | bitWidth | 32/bitWidth | numElementPerReg=min(max(·,1),4) | packed 操作数个数=4/该值 | 每寄存器打包       |
|----------|----------|-------------|----------------------------------|--------------------------|--------------------|
| f32      | 32       | 1           | 1                                | 4                        | 1 元素（标量，不打包 L63-65）|
| f16      | 16       | 2           | 2                                | 2                        | vector<2xf16>（L67-72）|
| i8       | 8        | 4           | 4                                | 1                        | vector<4xi8>（L67-72）|

**读法**：硬件寄存器以 32 位为打包单元 —— f32 正好占满 1 个寄存器（1 元素），
f16 两个拼满 32 位（2 元素/reg），i8 四个拼满（4 元素/reg）。三种类型给出三种不同的 reg 数，非退化。
不变量：numElementPerReg × bitWidth ≤ 32（因 numElementPerReg ≤ 32/bitWidth），故任一寄存器不超 32 位。

## 计数
LLVM 舱 = **1** pattern（ElementwiseInlineAsmOpConversion，runOnOperation L253）；
顶层 2 路径（标量/向量）。target 把 tensor/LLVM/arith 三方言设 legal（L250），applyPartialConversion（L254）。
