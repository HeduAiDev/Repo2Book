# 手工推演 trace — Kernel G(matmul + bias epilogue)核亲和不动点

> **trace_source = manual**。本章 skip_impl:纯 C++ MLIR pass(`third_party/ascend/lib/TritonAffinityOpt/DAG.cpp`
> ~534 行),宿主无 CANN 工具链编不动,triton-ascend 树内亦无针对 AffinityDAG 的 lit 夹具
> (grep RUN dag-* / TritonAffinity 全空)。故这里的每个数值是**照 pin 源码 @2badfc89e 的控制流逐行手算**
> 得到的,不是编译器 dump。凡引用源码常量的数字在正文里标 `file:Lxxx`。行号基线 = book-baseline(=v3.2.1 钉版)。

## 被追踪的 kernel

一个最小 matmul + bias epilogue(形状 128×128 仅为示意,与核决策无关):

```mlir
tt.func @mm(%pa: !tt.ptr<...>, %pb: !tt.ptr<...>, %po: !tt.ptr<...>, %bias: tensor<128x128xf32>) {
  %a  = tt.load %pa            : tensor<128x128xf16>   // READ effect
  %b  = tt.load %pb            : tensor<128x128xf16>   // READ effect
  %c  = tt.dot %a, %b          : tensor<128x128xf32>   // triton::DotOp
  %d  = arith.addf %c, %bias   : tensor<128x128xf32>   // elementwise
  tt.store %po, %d                                     // WRITE effect
  %z  = arith.constant 0 : i32                         // 未被使用的标量(演示两遍 diffuse 的兜底)
  tt.return
}
```

## 每节点的静态事实(建图后一次算定)

`ability = OpNode::canRunOn()`(DAG.cpp:L139-L182);memPolicy 由 op 的 MemoryEffect 决定
(DAG.cpp:L345-L355);outputs = 该节点的下游(OpNode 的 outputs 是 result 值,ValueNode 的 outputs 是消费它的 op)。

| node    | kind        | source op | ability(canRunOn)     | memPolicy | outputs        |
|---------|-------------|-----------|-----------------------|-----------|----------------|
| %pa     | ValueNode(blockarg) | 无(source=null) | —          | —      | [load_a]       |
| %pb     | ValueNode(blockarg) | 无      | —                     | —         | [load_b]       |
| %po     | ValueNode(blockarg) | 无      | —                     | —         | [store]        |
| %bias   | ValueNode(blockarg) | 无      | —                     | —         | [addf]         |
| load_a  | OpNode      | load_a    | PREFER_VECTOR(Default)| READ      | [%a]           |
| load_b  | OpNode      | load_b    | PREFER_VECTOR(Default)| READ      | [%b]           |
| dot     | OpNode      | dot       | CUBE_ONLY(DotOp臂)     | —(早返回) | [%c]           |
| addf    | OpNode      | addf      | PREFER_VECTOR(Default)| NONE      | [%d]           |
| store   | OpNode      | store     | PREFER_VECTOR(Default)| WRITE     | []             |
| const   | OpNode      | const     | CUBE_AND_VECTOR(常量臂)| NONE      | [%z]           |
| return  | OpNode      | return    | CUBE_AND_VECTOR(Default,无 operand/result)| NONE | [] |
| %a      | ValueNode   | load_a    | (取 load_a) PREFER_VECTOR | READ  | [dot]          |
| %b      | ValueNode   | load_b    | PREFER_VECTOR         | READ      | [dot]          |
| %c      | ValueNode   | dot       | CUBE_ONLY             | —         | [addf]         |
| %d      | ValueNode   | addf      | PREFER_VECTOR         | NONE      | [store]        |
| %z      | ValueNode   | const     | CUBE_AND_VECTOR       | NONE      | []             |

**节点总数 = 16**(4 block-arg 值 + 7 op + 5 result 值)。
→ `threshold = worklist.size() * 5 = 16 * 5 = 80`(DAG.cpp:L456 的 ×5)。

编码:U=UNDETERMINED(0) / V=VECTOR_ONLY(1) / C=CUBE_ONLY(2) / CV=CUBE_AND_VECTOR(3)
(位值出自 DAG.h:L34-L39)。`*` = isUpstreamOfCubeMem(传染标记)已置位。

## Pass 1:diffuse 迭代到不动点(以同步轮次呈现;真实代码是 LIFO worklist,不动点相同)

> 说明:源码用 `SmallSetVector` worklist、`pop_back_val()`(LIFO)逐个 absorb,pop 顺序由 map 迭代序
> 决定(代码注释 "Not sure if determinism is required", DAG.cpp:L453),**不保证确定**。此处按"每轮把所有节点各
> 求值一次(用上一轮的值)"的同步方式呈现,便于逐步核对;两种调度到达的**不动点唯一**(见 Pass 1 末尾)。

初值:所有节点 isOn=U,taint=F。

### 每轮 isOn(只列有变化/关键节点;空白=沿用上一轮)

| 轮 | dot | %c | store | %a | %b | %d | %po | load_a | load_b | addf | %bias | %pa | %pb | const/%z/return |
|----|-----|----|-------|----|----|----|----|--------|--------|------|-------|-----|-----|-----------------|
| R1 | C   | C  | V     | U  | U  | U  | U  | U      | U      | U    | U     | U   | U   | U               |
| R2 | C   | C  | V     | C* | C* | V  | V  | U      | U      | U    | U     | U   | U   | U               |
| R3 | C   | C  | V     | C* | C* | V  | V  | C*     | C*     | V    | U     | U   | U   | U               |
| R4 | C   | C  | V     | C* | C* | V  | V  | C*     | C*     | V    | V     | C*  | C*  | U               |
| R5 | (无变化 → 不动点)                                                                          |

### 关键单步核对(照 absorbCommon 分支 DAG.cpp:L323-L399)

- **dot(R1)**:`ability == CUBE_ONLY` → 直接 `return CUBE_ONLY`(L341-343)。硬钉,永不再变。
- **%c(R1)**:sourceNode=dot,`dot.canRunOn()==CUBE_ONLY` → `return CUBE_ONLY`(L341-343)。
- **store(R1)**:memPolicy=WRITE(L357)。`getWriteDataSource` 取到 %d(L358);R0 时 %d=U,
  `exactlyOneType(U)`=false(L360)→ 落到 `return VECTOR_ONLY`(L369)。store=V。
- **%a(R2)**:sourceNode=load_a,ability=PREFER_VECTOR,memPolicy=READ。遍历 outputs=[dot];
  R1 时 dot=C(CUBE_ONLY)→ switch 命中 `case CUBE_ONLY`(L377):条件
  `ability!=PREFER_VECTOR`(假)`|| output->isUpstreamOfCubeMem`(dot 未置位,假)`|| memPolicy==READ`(**真**)
  → 进入,`isUpstreamOfCubeMem = (F||F||true)=true`(L383-387),`newCoreType |= CUBE_ONLY`(L388)。
  → **%a = C,taint 置位**。这就是"喂给 cube 的读把上游拉向 cube"的第一跳(READ 触发,dot 本身没被 taint)。
- **load_a(R3)**:遍历 outputs=[%a];R2 时 %a=C 且 taint=T → `case CUBE_ONLY` 条件里
  `output->isUpstreamOfCubeMem`(真)命中 → load_a 也 = C 且 taint 置位。
- **%pa(R4)**:block arg,sourceNode=null → 走无 source 分支(L328-335):`newCoreType |= load_a.isOn()`,
  `isUpstreamOfCubeMem |= load_a.isUpstreamOfCubeMem`。R3 时 load_a=C*(taint)→ %pa=C,taint 置位。
- **%d(R2)**:sourceNode=addf,ability=PREFER_VECTOR,memPolicy=NONE。outputs=[store];R1 store=V
  → `case VECTOR_ONLY`(L391)`newCoreType |= VECTOR_ONLY` → %d=V。
- **store(R3)**:WRITE,数据源 %d;R2 时 %d=V,`exactlyOneType(V)`=true 且 `V != CUBE_ONLY`
  → `return currCt = VECTOR_ONLY`(L360-365)。store 稳定在 V,**不置 taint**(taint 只在数据源==CUBE_ONLY 时置,L361-363)。

### Pass 1 不动点(R5,无节点再变)

| 节点 | isOn | taint | 归属 |
|------|------|-------|------|
| %pa  | C    | ✓     | Cube |
| %pb  | C    | ✓     | Cube |
| load_a | C  | ✓     | Cube |
| load_b | C  | ✓     | Cube |
| %a   | C    | ✓     | Cube |
| %b   | C    | ✓     | Cube |
| dot  | C    | —     | Cube |
| %c   | C    | —     | Cube |
| addf | V    | —     | Vector |
| %d   | V    | —     | Vector |
| store| V    | —     | Vector |
| %po  | V    | —     | Vector |
| %bias| V    | —     | Vector |
| const| **U**| —     | 未定 |
| %z   | **U**| —     | 未定 |
| return | **U**| —   | 未定 |

**读法**:matmul 的两路操作数 load 及其地址(block arg 指针)被拉到 **Cube**(数据算在 cube 旁);
epilogue 的 addf、store、输出指针、bias 落 **Vector**。这正是异构双核该有的切法:
矩阵乘链留 cube,逐元素后处理去 vector。而孤立的标量 const/%z(无下游需求)与 return 仍 **UNDETERMINED**。

## 兜底(两遍 diffuse 之间)DAG.cpp:L474-L478

```
for(auto node : nodes)
  if (node->isOn() == UNDETERMINED)
    node->isOnPrivate = VECTOR_ONLY;
```

→ const、%z、return 三个残留 U 的节点被兜底成 **V(VECTOR_ONLY)**("拿不准就放最省的向量核")。

## Pass 2:再 diffuse 一遍(DAG.cpp:L480)

从上面兜底后的状态再迭代。逐节点求值:
- const(ability=CUBE_AND_VECTOR,memPolicy=NONE):outputs=[%z],%z=V(VECTOR_ONLY)→ `case VECTOR_ONLY`
  `|= VECTOR` → const=V。稳定。
- %z(source=const):outputs=[](无消费者)→ newCoreType=const.isOn()=V。稳定。
- return:outputs=[] → 保持 V。稳定。
- **其余 13 个节点均已是确定值**,重新 absorb 一遍返回同值、`changed=false`,不再唤醒邻居。

Pass 2 无进一步变化 → 整图不动点。**本 kernel 里 pass 2 未改动任何"有下游"的节点**——诚实说明:
这里兜底出的 V 只落在 %z/const/return(它们没有下游消费者,V 传不出去)。pass 2 的存在是为了让
**确实有下游的**兜底节点能把默认 V 继续沿数据流传染到不动点(源码设计意图,DAG.cpp design_decisions),
本例恰好不触发那条路径。

## 最终核标注(经 toHivm 落 hivm::TCoreType,DAG.h:L76-L90)

Cube:%pa %pb load_a load_b %a %b dot %c(8 个)→ hivm::TCoreType::CUBE
Vector:%po %bias addf %d store const %z return(8 个)→ hivm::TCoreType::VECTOR
(UNDETERMINED 若残留会落 CUBE_OR_VECTOR,本例已无残留。)
