# 手工推演 trace — 判核规则 canRunOn 逐 case(m4)

> trace_source = manual(理由同 kernelG_fixpoint.md:纯 C++ pass,宿主编不动、无 lit 夹具)。
> 每行是照 `OpNode::canRunOn()`(DAG.cpp:L139-L182)+ `valueIsScalar`(DAG.cpp:L109-L125)手算的静态能力。

`canRunOn` 是**纯函数**(只看 op 类型/操作数类型,无迭代、无状态):结构 = 1 条 scf 早返回(L140-142)
+ TypeSwitch 四臂(DotOp 臂 L144 / 4 类常量流臂 L147 / SelectOp 臂 L150 / Default 臂 L156)。
OpAbility 取值有 3 种:PREFER_VECTOR / CUBE_ONLY / CUBE_AND_VECTOR(DAG.h:L27-L32)。

| # | op(IR 名) | 命中分支(源码行) | 关键判定 | OpAbility 返回 |
|---|-----------|------------------|----------|----------------|
| 1 | `scf.for` | scf 早返回 L140-142 | `opIsScf(op)`==true | CUBE_AND_VECTOR |
| 2 | `tt.dot`  | DotOp 臂 L144-146 | 是 triton::DotOp | **CUBE_ONLY** |
| 3 | `arith.constant` | 常量流臂 L147-149 | Case<arith::ConstantOp,...> | CUBE_AND_VECTOR |
| 4 | `tt.trans` | 常量流臂 L147-149 | Case<...,triton::TransOp,...> | CUBE_AND_VECTOR |
| 5 | `arith.select`(cond=`tensor<...xi1>`) | SelectOp 臂 L150-155 | `valueIsScalar(cond)`==false | **PREFER_VECTOR** |
| 6 | `arith.select`(cond=`i1` 标量) | SelectOp 臂 L150-155 | `valueIsScalar(cond)`==true | CUBE_AND_VECTOR |
| 7 | `tt.load` → `tensor<128x128xf16>` | Default 臂 L156-181 | result 是 rank-2 张量 → `valueIsScalar`==false → isVector=true | PREFER_VECTOR |
| 8 | `arith.addi`(全 `i32` 标量) | Default 臂 L156-181 | 所有 operand/result `isIntOrIndexOrFloat` → isVector 保持 false | CUBE_AND_VECTOR |

## valueIsScalar 的判定链(DAG.cpp:L109-L125),case 5/7/8 用到

- `i32`/`index`/`f32`:`type.isIntOrIndexOrFloat()`==true → 标量(L112-114)。
- `tensor<...>` rank==0:算标量(L116-118)。
- `!tt.ptr<...>`(标量指针):算标量(L120-122)。
- `tensor<128x128xf16>`(rank≥1):以上都不命中 → `return false`,即非标量=向量候选(L124)。

## 关于 Default 臂里两处被注释的 tensor-of-ptr 分支

源码 L160-162 / L169-171 有两处 `// if (valueIsTensorOfPtr(...)) return SCALAR;` **被注释掉**,
故 tensor-of-ptr(如 `tensor<128x!tt.ptr>`)在当前 Default 臂**不特殊处理**,按普通张量走 → PREFER_VECTOR。
写作若引这段须声明"去掉两处注释掉的 tensor-of-ptr 分支"。

## 全覆盖(而非枚举穷举)

任意 op 若前三臂(DotOp / 4 类常量流 / SelectOp)均不命中,一律落 **Default 臂**兜底返回 →
`canRunOn` 对**任意** op 都有返回值。这是"靠 Default 兜底的全覆盖",不是"把所有 op 类型枚举穷举"。
