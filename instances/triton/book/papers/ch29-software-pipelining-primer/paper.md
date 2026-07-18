# ch29 论文包 —《软件流水线与模调度：num_stages 到底调度了什么》

> 本章定位：**Pipeliner 群的前置原理章（primer）**。第 4 章讲过 `tl.range(..., num_stages=N)` 只是把一个提示挂到循环上；第 17 章讲过它在下降时被写成 `scf.for` 的 `tt.num_stages` 属性；第 25 / 26 章建立了「合并访存」「共享内存预算 / occupancy」这几把判据尺。到本章要回答那个一直悬着的问题：**`num_stages` 到底调度了什么？** 答案是一套经典编译理论——**软件流水线（software pipelining）+ 模调度（modulo scheduling）**：把循环体的算子拆到不同 **stage**，让「第 $`i{+}1`$ 次迭代的 load」和「第 $`i`$ 次迭代的 compute」在同一个物理循环体里同时在飞，用异步 load 把长访存延迟藏到计算背后。看懂这张「迭代 × stage」的时空图，就懂了为什么 `num_stages` 调大能藏延迟、调多了反而爆共享内存 / 寄存器。
>
> **本章的主真相源是 pin 的 Triton v3.2.0 源码（A 档）**，`lib/Dialect/TritonGPU/Transforms/Pipeliner/` 下五个文件逐字可核：`SoftwarePipeliner.cpp` 的**头注释**把整套架构一句话点透——软件流水线器「usually separated into two pieces, one that create a modulo schedule and an expander that rewrites the loop and emits a prologue and epilogue」（`:22–26`，A 档逐字）；`Schedule.cpp` + `Schedule.h` 给出 `CoarseSchedule` 这张「排期表」（每个 op 一个 `(stage, cluster)`）；`MatmulLoopPipeline.cpp` 给出「把喂 dot 的 load 换成多 buffer 异步预取」的建模（`preProcessLoopAndGetSchedule`）；`PipelineExpander.cpp` 给出「拿排期表展开成 prologue / 稳态 / epilogue」的落地。经典模调度文献（Lam 1988、Allan 1995）只作**学术出处（C 档）**。
>
> **红线：只写已核实内容。** A 档一切引文都能在标注的 `.cpp` / `.h` 行号处逐字核对。C 档模调度理论**只取被源码注释坐实的概念**：源码头注释里**逐字点名**了 `modulo schedule`、`prologue`、`epilogue` 三个词，这三者以源码为可核抓手 + 论文 DOI 作学术出处；而经典理论里的 **initiation interval（II，发起间隔）**、**steady state（稳态）** 这两个词**在 pin 的 Triton 源码里并未出现**（`grep` 全 `Pipeliner/` 目录，只 `modulo`/`prologue`/`epilogue` 命中）——本包**无联网 / WebFetch 能力**，无法逐字核实 Lam 1988 / Allan 1995 的定理与公式，故凡涉及 II / 稳态的**具体定义、下界公式、调度算法细节**，一律**标「待核·回指 DOI」**，只写教科书级共识直觉，**绝不编造论文里的具体定理 / 公式**。能从源码逐字或逐算式核到的（stage / cluster 分配、多 buffer 数 = `max(distToUse)`、prologue/epilogue 各 `maxStage-1` 段），照写。

---

## 0. 来源层级表（防越档编造）

| 档 | 含义 | 本章用到的具体来源 | 用法 |
|---|---|---|---|
| **A** | pin 源码逐字 / 源码注释（最高权威，本章主真相源） | `lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp`：**架构头注释**（`:19–27`，逐字点名 modulo schedule + prologue/epilogue）、`preCondition`（`:37`，跳过 distance>1 / 外层循环）、`pipelineLoop`（`:73`，先 `preProcessLoopAndGetSchedule` 建模、再 `pipelineForLoop` 展开）、`getNumStagesOrDefault`（`:100`，读循环上的 `kNumStagesAttrName` 属性、否则用全局 `numStages`）、`runOnOperation` 对 `num_stage <= 1` 直接 bail（`:113–114`）；`lib/.../Pipeliner/Schedule.cpp`：`CoarseSchedule::getOpsInOrder`（`:45`，按 cluster 排序）、`createFinalSchedule`（`:72`，把排期表落成 `(op, stage)` 序列）、`insertDepsOfOp`（`:18`）；`include/triton/Dialect/TritonGPU/Transforms/Schedule.h`：`CoarseSchedule` 结构体（`:68–95`，`numStages` / `clusters` / `opToStageAndCluster` 即 `op → (stage, cluster)`）；`lib/.../Pipeliner/MatmulLoopPipeline.cpp`：`createAsyncCopy` 把 `tt.load` 换成 `AsyncCopyGlobalToLocalOp`+`AsyncCommitGroupOp`+`AsyncWaitOp`（`:54–154`）、`scheduleLoads` 的 stage 分配（`:558–597`，`stagesBetweenLoads = ceil(numStages-2, ...)`、root use 放 `numStages-1`、`distToUse`）、`createAlloc` 把 buffer 首维扩成 `distance`（`:775–790`）、`createAsyncOps` 的 `numBuffers = max(distToUse)(+1 MMAv3)`（`:936–955`）、`preProcessLoopAndGetSchedule` 建模总装（`:1067–1134`）；`lib/.../Pipeliner/PipelineExpander.cpp`：文件头（`:9–21`，loop software pipelining）、`emitPrologue` 注释「creates `maxStage - 1` part」（`:90–92`）、`emitEpilogue` 注释同构（`:107–110`）、`emitPrologue` 实现（`:278–347`）、`pipelineForLoop` 五步总装（`:789–840`） | 所有核心论断——「流水线 = 跨迭代重叠」「stage/cluster 是排期表」「num_stages 决定预取深度 / buffer 份数」「prologue 填流水、epilogue 排空」——**逐字引 `.cpp` / `.h`**；这是本章基石 |
| **C** | 经典编译理论（模调度 / 软件流水线，学术出处） | **Lam, M. (1988).** *Software pipelining: An effective scheduling technique for VLIW machines.* PLDI '88. **DOI:10.1145/53990.54022**（模调度奠基）。**Allan, V. H., Jones, R. B., Lee, R. M., Allan, S. J. (1995).** *Software pipelining.* ACM Computing Surveys 27(3). **DOI:10.1145/192724.192731**（软件流水线综述） | 为「modulo schedule / prologue / epilogue」提供**学术出处**——这三个词**逐字印在 `SoftwarePipeliner.cpp:22–26` 注释里**，可据源码坐实。**II（initiation interval）/ steady state 的具体定义、下界公式、模调度算法细节本包未能联网逐字核实，一律标「待核·回指 DOI」**，只写教科书级共识直觉 |

> 红线复述：本包只登记**已核实**内容。A 档所有引文可在标注行号处逐字核对；C 档中「modulo schedule / prologue / epilogue」由 `SoftwarePipeliner.cpp` 头注释坐实（源码比任何二手叙述都权威），其余理论细节（II / 稳态公式）标「待核」。

---

## 1. 动机：长延迟的 load 串行挡在 compute 前面，算力空转

先看一段最朴素的 matmul 主循环（Triton 层面就是 `tl.range` 里一个 `tl.load` 喂一个 `tl.dot`）。下降成 `scf.for` 后，循环体的数据依赖是一条直线：

```
每次迭代 i：
  a = load(A_ptr + i)      ← 从 global memory 搬，延迟几百个 cycle
  b = load(B_ptr + i)      ← 同上
  acc = dot(a, b, acc)     ← Tensor Core 计算，必须等 a、b 到齐才能开始
```

问题在**依赖**：`dot` 用到 `a`、`b`，所以它**必须等**两个 `load` 完成。而 GPU 的 global memory load 延迟极长（几百个 cycle），Tensor Core 却很快。于是每一轮的时间线是「**长长的等 load → 短短的 compute → 又长长的等 load**」——计算单元大部分时间在**空转等数据**。这就是访存受限（memory-bound）循环性能拉胯的根因。

一个自然的念头：**第 $`i{+}1`$ 次迭代的 `load` 和第 $`i`$ 次迭代的 `dot` 之间没有依赖**（下一轮的输入数据地址是已知的，不依赖这一轮的计算结果）。既然无依赖，为什么要等？能不能**在做第 $`i`$ 次 `dot` 的同时，就把第 $`i{+}1`$ 次要用的数据异步搬过来**？这样 `dot` 的计算时间就把 `load` 的延迟**盖住**了。

这正是**软件流水线（software pipelining）** 要做的事，也是 `num_stages` 存在的全部理由。而它之所以叫「软件」流水线，是因为它不像硬件流水线那样靠电路自动重叠——而是**编译器在编译期，把循环体重排、跨迭代重叠、生成一个新的循环**来实现。Triton 的这个 pass 名字就叫 `TritonGPUPipeline`，入口在 `SoftwarePipeliner.cpp`。

---

## 2. 核心：软件流水线 = 跨迭代重叠；模调度 = 给每个 op 分配 stage

### 2.1 一句话架构：modulo schedule + expander（源码头注释逐字坐实）

`SoftwarePipeliner.cpp` 的文件头注释，把整套机制一句话讲清了（`:19–27`，A 档逐字）：

> ```
> // This file will create a schedule that will be handed over to the pipeline
> // expander.
> // Software pipeliners are usually separated into two pieces, one that create a
> // modulo schedule and an expander that rewrites the loop and emits a prologue
> // and epilogue. This pass first calls a helper that will pre-process the IR
> // to create async operations and create a modulo schedule. Then we call the
> // expander to generate the prologue and new loop.
> ```

这段注释就是本章的**理论骨架**，也把「哪些词有源码坐实」钉死了：软件流水线器分**两半**——

1. **建模（create a modulo schedule）**：一个 helper 预处理 IR，把 load 换成**异步算子**，并产出一张**模调度表**（modulo schedule）——给循环体里每个 op 分配它属于**哪一个 stage**。这一半在 `MatmulLoopPipeline.cpp` 的 `preProcessLoopAndGetSchedule`（§3）。
2. **展开（expander … emits a prologue and epilogue）**：拿到排期表，把循环**重写**成三段——**prologue（序幕，填流水线）+ 稳态循环体 + epilogue（尾声，排空流水线）**。这一半在 `PipelineExpander.cpp` 的 `pipelineForLoop`（§3.3）。

`pipelineLoop` 把这两半串起来（`SoftwarePipeliner.cpp:73–94`，A 档逐字，精简掉失败分支）：

```cpp
static bool pipelineLoop(scf::ForOp forOp, int numStages) {
  mlir::triton::PipeliningOption options;
  if (!preCondition(forOp))
    return false;
  bool foundSchedule = false;
  foundSchedule = preProcessLoopAndGetSchedule(forOp, numStages, options);  // ← 建模
  ...
  FailureOr<scf::ForOp> newForOp =
      mlir::triton::pipelineForLoop(rewriter, forOp, options);              // ← 展开
  ...
  mlir::triton::asyncLaunchDots(newForOp.value());
  return true;
}
```

> **术语坐实清单**：`modulo schedule`、`prologue`、`epilogue`——三词**逐字**取自上面这段头注释，C 档 Lam 1988 / Allan 1995 提供学术出处。而经典模调度里的 **initiation interval（II，发起间隔）** 与 **steady state（稳态）** ——`grep` 整个 `Pipeliner/` 目录**未命中**，本包无法联网核实其论文定义，故本章只用其**教科书级共识直觉**（下文标「待核」），不给具体公式。

### 2.2 「模调度」的直觉：把循环体切成 stage，让不同迭代的不同 stage 同时在飞

用一张「迭代 × 时间片」的时空图理解（这是本章要交给 illustrator 的核心图，见文末 key_figures）。设把循环体切成 3 个 stage（`num_stages=3`）：

- **stage 0**：发起 load（异步搬 A、B 到共享内存）
- **stage 1**：等 load 完成 + 从共享内存取数
- **stage 2**：dot 计算

**朴素串行**：每次迭代顺序做完 stage 0→1→2 才进下一次，时间线是锯齿状的空转。

**软件流水线后的稳态**：编译器错开各迭代的起点，让同一个「时间片」里同时跑着——

```
时间片  →   t0      t1      t2      t3      t4
迭代 i+2:                  load    wait    dot
迭代 i+1:          load    wait    dot
迭代 i  :  load    wait    dot
                    ↑ 同一时刻(t2)：迭代i在dot、迭代i+1在wait、迭代i+2在load
```

**关键洞察**：在稳态的任意一个时间片，硬件的三种资源（异步拷贝引擎、共享内存、Tensor Core）**同时被三个不同迭代的三个不同 stage 占用**——异步引擎在给 $`i{+}2`$ 搬数、Tensor Core 在给 $`i`$ 算。load 的长延迟被**别的迭代的 dot** 盖住了。这就是「用 async load 隐藏访存延迟」的全部含义。（「稳态 / steady state」的严格定义与 II 下界待核·回指 DOI:10.1145/53990.54022；此处只用其重叠直觉。）

### 2.3 排期表落地：CoarseSchedule 给每个 op 一个 (stage, cluster)

「模调度」在 Triton 里的数据结构就是 `CoarseSchedule`——一张 `op → (stage, cluster)` 的表。定义在 `Schedule.h`（`:68–95`，A 档逐字，精简）：

```cpp
class CoarseSchedule {
public:
  ...
  CoarseSchedule(int numStages) : numStages(numStages) {}
  int numStages;
  ClusterList clusters;
  using Cluster = decltype(clusters)::iterator;
  DenseMap<Operation *, std::pair<int, Cluster>> opToStageAndCluster;

  void insert(Operation *op, int stage, Cluster cluster) {
    opToStageAndCluster[op] = {stage, cluster};
  }
  ...
};
```

两个维度各司其职：

- **stage**（0 到 `numStages-1`）：这个 op 属于流水线的第几级——**决定它比它的消费者提前几个迭代执行**。load 放前面的 stage（提前预取），dot 放最后的 stage。
- **cluster**：**同一 stage 内部**的执行顺序分组（一个有序的 `std::list<int>`，可以 `newAtFront` / `newAtBack` 往两头插）。stage 决定「差几个迭代」，cluster 决定「同一迭代内谁先谁后」。

排期完成后，`createFinalSchedule` 把这张二维表**拍平**成一个 `(op, stage)` 的有序序列，交给展开器（`Schedule.cpp:72–80`，A 档逐字）：

```cpp
std::vector<std::pair<Operation *, unsigned>>
tt::CoarseSchedule::createFinalSchedule(scf::ForOp forOp) {
  SmallVector<std::tuple<Operation *, int, tt::CoarseSchedule::Cluster>>
      opsInOrder = getOpsInOrder(forOp);          // 先按 cluster 排好序
  std::vector<std::pair<Operation *, unsigned>> schedule;
  for (auto [op, stage, cluster] : opsInOrder)
    schedule.push_back({op, stage});               // 只保留 (op, stage) 交给 expander
  return schedule;
}
```

这个 `(op, stage)` 序列，就是头注释里说的「modulo schedule」——展开器唯一需要的输入。

---

## 3. 展开：num_stages 就是流水线深度

### 3.1 num_stages 从哪来、什么时候不流水

`num_stages` 的来源在 `getNumStagesOrDefault`——**优先用挂在循环上的属性，否则用全局默认**（`SoftwarePipeliner.cpp:100–108`，A 档逐字）：

```cpp
int getNumStagesOrDefault(scf::ForOp forOp) {
  // Use the attribute attached to the loop if it exists otherwise use the
  // global control.
  if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
    return numStages;
  return mlir::cast<IntegerAttr>(
             forOp->getAttr(mlir::triton::kNumStagesAttrName))
      .getInt();
}
```

那个 `kNumStagesAttrName` 属性，正是第 4 章 / 第 17 章埋下的线：你在 `tl.range(..., num_stages=N)` 写的 `N`，一路下降成 `scf.for` 上的 `tt.num_stages` 属性，到这里被读出来。而 `runOnOperation` 对 `num_stage <= 1` 的循环**直接跳过不流水**（`:113–114`，A 档逐字）：

```cpp
getOperation()->walk([&](scf::ForOp forOp) {
  // Bail out for loops with num_stage <= 1.
  if (getNumStagesOrDefault(forOp) > 1)
    loops.push_back(forOp);
});
```

**这就解释了 `num_stages=1` 为什么等于「关掉流水线」**：深度为 1 意味着没有「提前的迭代」，无从重叠。

### 3.2 深度 = 预取多少次未来迭代 = 共享内存里几份 buffer

`num_stages` 作为「流水线深度」，落到 matmul 上的物理含义是：**提前几个迭代把 load 搬进共享内存**，因此**需要共享内存里几份 buffer 轮转**。

第一步，`createAsyncCopy` 把每个喂 dot 的 `tt.load` **换成异步三件套**——发起拷贝、提交、等待（`MatmulLoopPipeline.cpp:87–110`，A 档逐字，精简掉布局转换分支）：

```cpp
tt::MemDescType allocTy = cast<tt::MemDescType>(alloc.getType());
SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
copyOffsets[0] = insertIdx;                                   // 写进 buffer 的第 insertIdx 格
...
auto view =
    builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, copyOffsets);
Operation *copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(   // cp.async：global→shared
    loc, src, view, mask, other, loadOp.getCache(), loadOp.getEvict(),
    loadOp.getIsVolatile());
Operation *commmit =
    builder.create<ttg::AsyncCommitGroupOp>(loc, copy->getResult(0));
Operation *wait =
    builder.create<ttg::AsyncWaitOp>(loc, commmit->getResult(0), 0);
```

`AsyncCopyGlobalToLocalOp` 就是 NVIDIA 的 `cp.async`——**发起后不阻塞**，线程可以继续往下做（去算上一轮的 dot），等真正要用数据时再 `AsyncWaitOp`。这就是「异步 load 藏延迟」的硬件落点。

第二步，那块 `alloc` **不是一份，而是 `distance` 份**——`createAlloc` 把 buffer 的形状**在最前面插一维 `distance`**（`MatmulLoopPipeline.cpp:775–790`，A 档逐字）：

```cpp
// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, Operation *loadOp,
                         ttg::SharedEncodingAttr sharedEnc, unsigned distance) {
  ...
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);          // ← 多开 distance 份！
  Type memdescType = mlir::triton::MemDescType::get(
      bufferShape, ty.getElementType(), sharedEnc, sharedMemorySpace,
      /*mutableMemory*/ true);
  Value alloc = builder.create<mlir::triton::gpu::LocalAllocOp>(
      loadOp->getLoc(), memdescType, Value());
  return alloc;
}
```

一个 `[BLOCK_M, BLOCK_K]` 的 tile，被扩成 `[distance, BLOCK_M, BLOCK_K]`——**`distance` 份 buffer 组成一个环形缓冲**，第 $`i`$ 次迭代写第 `insertIdx` 格、读第 `extractIdx` 格，轮流使用。这份 `distance` 就是 key_figures 里「N 份 buffer 轮转」要画的东西。

而 `distance`（即 buffer 份数 `numBuffers`）怎么定？`createAsyncOps` 取**所有 load 里「从发起到被 dot 使用跨了几个 stage」的最大值**（`MatmulLoopPipeline.cpp:940–955`，A 档逐字）：

```cpp
// Calculate the number of buffers needed for each load.
...
// Instead, we allocate the maximum number of buffers needed by any load.
int numBuffers =
    llvm::max_element(llvm::make_second_range(loadToInfo), [](auto &lhs,
                                                              auto &rhs) {
      return lhs.distToUse < rhs.distToUse;
    })->distToUse;
bool hasMMAV3 =
    llvm::any_of(loadToInfo, [](auto &kv) { return kv.second.loadIsMMAV3; });
if (hasMMAV3) {
  // For MMAv3, we need an extra buffer ...
  numBuffers++;
}
```

而那个 `distToUse` 正是「load 的 stage 到 dot 的 stage 的差」——`scheduleLoads` 里把 root use（dot）放最后一个 stage `numStages - 1`、把 load 放前面的 stage，两者相减（`MatmulLoopPipeline.cpp:576, 593–597`，A 档逐字，精简）：

```cpp
// Put the root uses of the loads in the last stage.
schedule.insert(use, numStages - 1, rootUsersCluster);
...
// Distance from the load to the use.
loadToInfo[loadOp].distToUse = schedule[use].first - schedule[loadOp].first;
```

**这条链就是本章的落地闭环**：`num_stages` ↑ → load 与 dot 的 stage 差 `distToUse` ↑ → `numBuffers` ↑ → 共享内存里 buffer 份数 ↑。**多一个 stage，就多一份 tile 大小的共享内存**。

### 3.3 展开成三段：prologue 填流水、稳态、epilogue 排空

拿到排期表后，`PipelineExpander.cpp` 的 `pipelineForLoop` 分五步把循环重写成三段式（`:789–840`，A 档逐字，精简掉 IR 细节，保留步骤骨架）：

```cpp
FailureOr<ForOp>
mlir::triton::pipelineForLoop(RewriterBase &rewriter, ForOp forOp, ...) {
  LoopPipelinerInternal pipeliner;
  if (!pipeliner.initializeLoopInfo(forOp, options))
    return failure();
  // 1. Emit prologue.
  if (failed(pipeliner.emitPrologue(rewriter)))                    // ← 序幕：填流水线
    return failure();
  // 2. Track values used across stages. ...
  ... crossStageValues = pipeliner.analyzeCrossStageValues();
  // 3. Create the new kernel loop ...
  ForOp newForOp = pipeliner.createKernelLoop(...);                // ← 稳态循环体
  if (failed(pipeliner.createKernel(newForOp, ...)))
    return failure();
  ...
  if (options.peelEpilogue) {
    // 4. Emit the epilogue after the new forOp.
    ... pipeliner.emitEpilogue(rewriter, returnValues);            // ← 尾声：排空流水线
  }
  // 5. Erase the original loop ...
  return newForOp;
}
```

- **Prologue（序幕）**：`emitPrologue` 的注释说得很准——「creates `maxStage - 1` part which will contain operations from stages `[0; i]`」（`:90–92`，A 档逐字）。也就是**在进正式循环前，先手动跑 `maxStage-1` 个「不完整」的迭代**，把前几个迭代的 load 提前发起——**把流水线灌满**。实现里就是一个 `for i in [0, maxStage)` 的循环，对每个 op 只在 `stages[op] <= i` 时才发射（`:305–307`，A 档逐字）：

  ```cpp
  for (Operation *op : opOrder) {
    if (stages[op] > i)
      continue;           // 还没轮到这个 stage 出场
    ...
  }
  ```

- **稳态（steady-state kernel loop）**：`createKernelLoop` + `createKernel` 生成新的循环体——**一次迭代里同时包含所有 stage 的 op**（第 $`i`$ 次迭代的 dot、第 $`i{+}1`$ 次的 wait、第 $`i{+}2`$ 次的 load 都在同一个循环体里）。跨 stage 存活的值靠额外的 `iter_args` 在迭代间传递（这正是第 17 章 loop-carried 变量的用武之地，`analyzeCrossStageValues` 负责识别）。

- **Epilogue（尾声）**：`emitEpilogue` 注释同构——「creates `maxStage - 1` part which will contain operations from stages `[i; maxStage]`」（`:107–110`，A 档逐字）。即循环跑完后，**再手动跑 `maxStage-1` 个「不完整」的迭代**，把还在飞的最后几个迭代的 dot 算完——**把流水线排空**。（注意 Triton 的 matmul 建模里 `options.peelEpilogue = false`，`MatmulLoopPipeline.cpp:1129`——它选择不剥离 epilogue、而用谓词化在稳态循环内收尾，这是实现选择，展开器两条路都支持。）

三段式合起来就是那句头注释：`prologue` + 新循环（稳态）+ `epilogue`。prologue 和 epilogue 各 `maxStage-1` 段——**深度越大，填 / 排的迭代越多**。

---

## 4. 落地：为什么 num_stages 调大藏延迟、调多了反而爆共享内存 / 寄存器

把前三节的链接起来，就得到 `num_stages` 调优的**代价—收益权衡**（这是本章的 perf_payoff）：

**收益（调大）**：`num_stages` ↑ → 预取深度 ↑ → 在飞的迭代更多 → **更长的访存延迟能被更多的 dot 计算盖住**。对访存受限的循环，调大 `num_stages` 直接提吞吐——这是隐藏延迟的正道。

**代价（调多）**：从 §3.2 那条闭环——`num_stages` ↑ → `numBuffers = max(distToUse)` ↑ → **共享内存里多一份 tile 大小的 buffer**（`createAlloc` 把首维扩成 `distance`）。共享内存是每个 SM 上被所有活跃 block 瓜分的稀缺资源（第 26 章的预算尺）：

- 每多一个 stage，A、B 两个操作数各多一份 `[BLOCK_M, BLOCK_K]` / `[BLOCK_K, BLOCK_N]` 的共享内存 buffer；
- 共享内存用量 ↑ → 每个 SM 能同时驻留的 block 数（occupancy）↓（第 26 章）；occupancy 太低，反而没有足够的 warp 来互相隐藏延迟，得不偿失；
- 极端情况共享内存**超过 SM 上限**，编译直接失败 / 回退；
- 稳态循环体里同时在飞的迭代多了，**跨 stage 存活的值要占更多寄存器**（`iter_args` 变多，§3.3），可能触发寄存器溢出（spill 到 local memory，第 26 章的另一把尺），把省下的延迟又赔回去。

所以 `num_stages` 不是越大越好，而是**在「藏住延迟」和「共享内存 / 寄存器预算」之间找平衡**——最优值取决于 tile 大小、数据类型位宽、目标 SM 的共享内存容量。这也是为什么它是 autotuner 要搜的关键 config 之一。

**本章到此为止**：我们建立了模调度的时空图直觉、stage/cluster/distance/prologue/epilogue 的词汇、以及「深度 → buffer 份数 → 共享内存」这条闭环。至于**建模的全部细节**（`scheduleLoads` 怎么按 indirection level 分 stage、`scheduleDependencies` / distance-1 依赖怎么补、MMAv3 的特殊 async、`asyncLaunchDots` 怎么插 `WarpGroupDot` 的 wait）和**展开的全部细节**（`createKernel` 怎么做模变量扩展、动态循环谓词化），留给**第 30 章《软件流水线落地》** 逐行拆——本章只把「num_stages 到底调度了什么」这张图交给你。

---

## 附 A：A 档源码锚点清单（逐条可核，pin v3.2.0）

| # | 规范路径（file:行号） | 内容 | 用在 |
|---|---|---|---|
| A1 | `lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp:19–27` | 架构头注释：modulo schedule + expander emits prologue/epilogue | §2.1（理论骨架、术语坐实） |
| A2 | `.../SoftwarePipeliner.cpp:37–58` | `preCondition`：跳过 distance>1、不流水外层循环 | §2.1（何时可流水） |
| A3 | `.../SoftwarePipeliner.cpp:73–94` | `pipelineLoop`：建模(`preProcessLoopAndGetSchedule`)→展开(`pipelineForLoop`)→`asyncLaunchDots` | §2.1 |
| A4 | `.../SoftwarePipeliner.cpp:100–108` | `getNumStagesOrDefault`：读 `kNumStagesAttrName` 属性，否则全局 `numStages` | §3.1 |
| A5 | `.../SoftwarePipeliner.cpp:110–119` | `runOnOperation`：`num_stage <= 1` 直接 bail | §3.1（=1 即关流水线） |
| A6 | `include/triton/Dialect/TritonGPU/Transforms/Schedule.h:68–95` | `CoarseSchedule` 结构体：`numStages`/`clusters`/`opToStageAndCluster`（op→(stage,cluster)） | §2.3（排期表数据结构） |
| A7 | `lib/.../Pipeliner/Schedule.cpp:45–70` | `getOpsInOrder`：按 cluster 排序 | §2.3 |
| A8 | `.../Schedule.cpp:72–80` | `createFinalSchedule`：拍平成 (op, stage) 序列交 expander | §2.3 |
| A9 | `lib/.../Pipeliner/MatmulLoopPipeline.cpp:54–154` | `createAsyncCopy`：load→`AsyncCopyGlobalToLocalOp`+commit+wait，subview 写入 `insertIdx` 格 | §3.2（异步预取三件套） |
| A10 | `.../MatmulLoopPipeline.cpp:558–597` | `scheduleLoads`：`stagesBetweenLoads=ceil(numStages-2,...)`、root use 放 `numStages-1`、`distToUse` | §3.2（stage 分配与距离） |
| A11 | `.../MatmulLoopPipeline.cpp:775–790` | `createAlloc`：buffer 首维扩成 `distance`（多 buffer 环形缓冲） | §3.2（N 份 buffer） |
| A12 | `.../MatmulLoopPipeline.cpp:936–955` | `createAsyncOps`：`numBuffers = max(distToUse)`（+1 MMAv3） | §3.2（buffer 份数怎么定） |
| A13 | `.../MatmulLoopPipeline.cpp:1067–1134` | `preProcessLoopAndGetSchedule`：建模总装 + `peelEpilogue=false` | §2.1 / §3.3 |
| A14 | `lib/.../Pipeliner/PipelineExpander.cpp:9–21` | 文件头：loop software pipelining（upstream fork） | §3.3 |
| A15 | `.../PipelineExpander.cpp:90–92` | `emitPrologue` 注释：creates `maxStage - 1` part，stages `[0; i]` | §3.3（prologue 填流水） |
| A16 | `.../PipelineExpander.cpp:107–110` | `emitEpilogue` 注释：creates `maxStage - 1` part，stages `[i; maxStage]` | §3.3（epilogue 排空） |
| A17 | `.../PipelineExpander.cpp:278–347` | `emitPrologue` 实现：`for i in [0,maxStage)`，`stages[op] > i` 跳过 | §3.3 |
| A18 | `.../PipelineExpander.cpp:789–840` | `pipelineForLoop` 五步总装：prologue→cross-stage→kernel loop→epilogue→erase | §3.3 |

---

## 附 B：C 档理论核实记录

| 概念 | 源码坐实？ | 学术出处 | 核实状态 |
|---|---|---|---|
| **modulo schedule（模调度）** | ✅ 逐字印在 `SoftwarePipeliner.cpp:22–26` | Lam 1988, DOI:10.1145/53990.54022（模调度 for VLIW 奠基） | 词由源码坐实；论文奠基地位 web-verified（meta.json）。**论文内定理 / II 下界公式未联网逐字核实，标待核** |
| **prologue（序幕）/ epilogue（尾声）** | ✅ 逐字印在 `SoftwarePipeliner.cpp:23–26`；`PipelineExpander.cpp:90–92 / 107–110` 给出「各 `maxStage-1` 段」 | Lam 1988；Allan 1995 综述 DOI:10.1145/192724.192731 | 词与「maxStage-1 段」由源码坐实；论文出处 web-verified |
| **stage / cluster / distance** | ✅ Triton 自有抽象，`Schedule.h:68–95`、`MatmulLoopPipeline.cpp:558–597` 逐字 | （Triton 特有命名，非论文术语） | 全部 A 档源码坐实 |
| **initiation interval（II，发起间隔）** | ❌ `grep` 整个 `Pipeliner/` 目录未命中 | Lam 1988 / Allan 1995 的核心量 | **待核·回指 DOI:10.1145/53990.54022**。本包无联网，不给 II 定义 / 下界公式；正文只用「稳态里各迭代错开固定间隔重叠」的教科书直觉 |
| **steady state（稳态）** | ❌ 源码未出现该词 | Allan 1995 综述 | **待核·回指 DOI:10.1145/192724.192731**。正文只用重叠直觉，不给严格定义 |

> 核实边界声明：本包组装环境**无网络 / WebFetch**。Lam 1988、Allan 1995 的 DOI 与「模调度奠基 / 软件流水线综述」的定位取自 meta.json（web-verified）；两篇论文的**具体定理、II 下界公式、模调度算法伪码未能逐字核实**，凡涉及一律标「待核·回指 DOI」。本章正文的**每一条机制论断都能落到 A 档源码行号**，理论概念只取被源码注释坐实的部分 + 教科书级共识直觉——**绝不编造论文内文**。

---

## 附 C：key_figures（交 illustrator 重绘）

1. **「迭代 × stage」时空图：朴素串行 vs 软件流水线**（§2.2 target）——上半画朴素串行的锯齿空转（每迭代 load→wait→dot 顺序做完才进下一次，Tensor Core 大段空转）；下半画流水线稳态：同一时间片 t2 里，迭代 $`i`$ 在 dot、迭代 $`i{+}1`$ 在 wait、迭代 $`i{+}2`$ 在 load 三色并行。**这是全章降低阅读难度的核心图**——「num_stages 调度了什么」一图看懂。锚点：`SoftwarePipeliner.cpp:19–27` 头注释 + Lam 1988。
2. **num_stages=N 时共享内存里 N 份 buffer 轮转示意**（§3.2 target）——画一个 `[distance=N, BLOCK_M, BLOCK_K]` 的环形缓冲：第 $`i`$ 次迭代 `AsyncCopy` 写入第 `insertIdx=i%N` 格、dot 从第 `extractIdx` 格读，N 份 buffer 首尾相接轮转；旁注「多一个 stage = 多一份 tile 大小的共享内存」。锚点：`MatmulLoopPipeline.cpp:775–790`（createAlloc 扩 distance 维）+ `:936–955`（numBuffers）。
