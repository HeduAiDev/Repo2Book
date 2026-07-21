# ch01:L147 的 `(如 `linalg.add`)` 举例无据 —— 待回修(2026-07-21 立)

来源:ch09 评审的章外发现(reviewer 提出,Lead 已逐条追源码复核**属实**)。
ch01 已定稿提交,正文只有 writer 能改 → 记在此处,择机派 writer 定点小修。

## 现状

`instances/triton-ascend/artifacts/ch01-birdseye-ascend-backend/narrative/chapter.md:L147`:

> - `ttadapter` → `ttir_to_linalg`:把 TTIR 降成结构化 Linalg(MLIR 的线性代数方言)。
>   `named_ops=True` 要求产出带名字的 Linalg 算子(如 `linalg.add`),便于后段识别。

## 核实结论:**「带名字的 Linalg 算子」是 `.td` 的自述,`linalg.add` 这个例子在仓里无据**

`namedOps` 在实现里**只做两件事**(全仓 `lib/`+`include/` 搜 `namedOps` 仅 7 处命中,其余是构造/透传):

1. `third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp:L524`
   `return this->namedOps || !operateOnTensors;`
   → `namedOps` 为真时,**张量上的 `arith` 算子保持合法**(不被判为待转换)。
2. 同文件 `L651`
   `if (!this->namedOps) { linalg::populateElementwiseToLinalgConversionPatterns(patterns); }`
   → `namedOps` 为真时,**不**加载「逐元素 → `linalg.generic`」的转换模式。

也就是说这个开关的**实现语义是「别把逐元素 `arith` 转成 `linalg.generic`」(保持 `arith` 原样)**,
而**不是**「发射 `linalg` 具名算子」。全仓 `lib/`+`include/` 搜 **`linalg::AddOp` / 产出 `linalg.add` 的代码:零命中**。

`.td` 的 doc string(`third_party/ascend/include/TritonToLinalg/Passes.td:L13-L15`)写的是
`"use linalg named ops instead of linalg.generic"` —— ch01 忠实转述了这句**声明**,
但补的 `(如 `linalg.add`)` 是**声明之外自行具体化**的例子,实现不产出它。

## 这正是本书 ch09 已经点透的那个病种(exp-2026-07-21-10)

ch09 的 named op 一节讲的就是「**声明的默认值 ≠ 路径上的实际取值**」(`.td` 默认 `false`,
装配点传 `True`)。这里是同一病灶的**另一面**:「**声明的语义描述 ≠ 实现语义**」。
ch09 因为老实把实现语义留给 ch10(见 ch09 正文「实现语义留待 ch10 展开」一句),**没有被带偏**。

## 处置建议

1. **ch01 定点小修**(派 writer,一句话):去掉 `(如 `linalg.add`)`,改成不举具体算子的表述,
   例如「`named_ops=True` 让逐元素算子保持 `arith` 原样、不被摊成 `linalg.generic`(实现语义见第 10 章)」。
   ⚠️ 改前请 writer 自行复核上述两处 C++ 行号,**别只信本文件**。
2. **ch10 的必答项**:ch10 是 `triton_adapter` 分水岭章,必须正面交代 `namedOps` 的**实现语义**
   (以及 `.td` doc string 与实现不符这一事实)。已在此立此存照,发车 ch10 时注入 brief。

## 一并已处置(不必再管)

`outline-final.json` 的陈旧 `chNNb` 依赖 id:reviewer 只报了 ch10 的 `ch09b`,
Lead 顺手做了全书悬空 deps 体检,**实际有 6 处**(ch10→ch09b、ch27→ch26b、ch28→ch27b、
ch29→ch28b、ch30→ch29b、ch31→ch26b),已全部去掉 `b` 后缀;
复检:悬空 deps 0、自依赖/前向依赖 0、全书他处无 `chNNb` 残留。
