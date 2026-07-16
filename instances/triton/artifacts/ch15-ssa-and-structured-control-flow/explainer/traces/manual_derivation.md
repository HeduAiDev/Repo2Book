# ch15 手工推演台账（trace_source=manual 的原始底稿）

> 本章是 primer 原理章，**无可运行精简版**：讲的是 SSA/φ 的语义、MLIR 块参数取代 φ、
> Triton 前端用 AST dry-run + scope 差集直接构造 SSA 的做法——不下降真实 kernel 到 IR 跑数。
> 故所有数值轨迹按「SSA/φ 语义 + 源码常量」**手工推演**；引用源码常量处标 `code_generator.py:Lxxx`。
> 本文件是 explainer.json 里每张表的底稿，供 illustrator/writer 溯源。
>
> 源码基线：`python/triton/compiler/code_generator.py`（triton 实例 source，已本地核行号）。

---

## 运行例 A —— 单赋值版本链（ssa-single-assignment）

Python 片段（同名多次赋值）：

```python
acc = 0          # 语句 1
acc = acc + 1    # 语句 2
acc = acc * 2    # 语句 3
```

下降到 IR 的 SSA 版本链（每次赋值造新版本，下标 +1，绝不复用）：

| 语句 | SSA | 求值 | use-def |
|---|---|---|---|
| `acc = 0`      | acc_0 = 0            | 0 | 常量 |
| `acc = acc+1`  | acc_1 = acc_0 + 1   | 0+1 = 1 | 用 acc_0 |
| `acc = acc*2`  | acc_2 = acc_1 * 2   | 1*2 = 2 | 用 acc_1 |

- 3 条 Python 赋值 → 3 个 SSA 版本（acc_0, acc_1, acc_2）。
- 每个版本恰被 1 条语句赋值 → SSA 单赋值不变量成立。
- 版本下标即赋值序号；set_value 每次把名字写进 `local_defs`（`code_generator.py:L315-L322`），
  是「同名多版本」在前端的落点。

---

## 运行例 B —— if/else 汇合的 φ（phi-merge-semantics / block-args-replace-phi）

Python 片段：

```python
if cond:
    x = a        # x_1 = a
else:
    x = b        # x_2 = b
use(x)           # x_3 = φ(x_1, x_2)
```

具体取值：**a = 5, b = 7**（刻意取 5 ≠ 7，避开两支相等的巧合分支）。

φ 语义（按运行时前驱选值）：

| 运行路径 | 进入汇合块的前驱 | φ 选中实参 | x_3 值 |
|---|---|---|---|
| cond=True  | then 支 P1 | 第 1 实参 x_1 | 5 |
| cond=False | else 支 P2 | 第 2 实参 x_2 | 7 |

- 前驱数 n = 2 → φ 有 2 个实参；runtime 只走 1 条路径 → 选出 1 个值 x_3。
- φ 产出唯一新版本 x_3 → SSA 单赋值恢复。

φ ↔ 块参数等价（同一语义，方向相反）：

| | φ 写法（拉） | 块参数写法（推） |
|---|---|---|
| then 支落地 | 实参 x_1=5 占 φ 第 1 槽 | then terminator：`br ^merge(5)` |
| else 支落地 | 实参 x_2=7 占 φ 第 2 槽 | else terminator：`br ^merge(7)` |
| 汇合取值 | `x_3 = φ(x_1, x_2)` | `^merge(%x3: T)`：%x3 已由前驱推入 |

MLIR/Triton 落地：scf.if 两支各 `create_yield_op`（`code_generator.py:L670` then / `L677` else），
汇合后 `if_op.get_result(0)`（`code_generator.py:L680`）就是 φ 选出的 x_3。
φ（记号）→ yield/块参数（MLIR）→ `get_result`（Triton），三层同一件事。

---

## 运行例 C —— 累加循环的 loop-carried（scf-for-arg-layout / scf-yield / loop-carried-scope-diff）

Python 片段（两个 loop-carried + 一个纯临时 + 一个只读参数）：

```python
acc = 0.0            # 循环前定义
m = 0                # 循环前定义
N = ...              # 只读参数（循环前定义，循环内不再赋值）
for k in range(0, 3):     # 归纳变量 k
    acc = acc + k         # 循环内赋值
    m = max(m, k)         # 循环内赋值
    tmp = k * 2           # 循环内赋值，但循环前不存在
# 出循环：use(acc), use(m)
```

### C-1 scope 差集识别 loop-carried（local_defs ∩ liveins）

- `liveins` = 进入循环子区域前的父作用域快照（`enter_sub_region.__enter__`，`code_generator.py:L89`）
  = {acc, m, N, k, …}。注意 k 在 `L957-L959` 已 create_poison + set_value，**先于** enter_sub_region，故 k ∈ liveins。
- `local_defs` = dry-run 循环体后收集到的本区域赋值名（`L970` dry visit → `L972` block.erase）
  = {acc, m, tmp}。
- 交集判定（`L978` `for name in self.local_defs:` / `L980` `if name in liveins:`）：

| 变量 | ∈ liveins? | ∈ local_defs? | 交集 | 角色 |
|---|---|---|---|---|
| acc | ✓ | ✓ | ✓ loop-carried | iter_arg 0 |
| m   | ✓ | ✓ | ✓ loop-carried | iter_arg 1 |
| tmp | ✗ | ✓ | ✗ | 循环内临时（每轮重算，不 carry） |
| N   | ✓ | ✗ | ✗ | 只读参数（循环内未再赋值） |
| k   | ✓ | ✗ | ✗ | 归纳变量（占位在 liveins，体内未再赋值） |

- 交集 = {acc, m} = loop-carried 集合，长度 2 → `create_for_op` 的 `init_args` 长度 2（`L990`）。
- 红线：这是 dry-run + 集合交（**零支配边界计算**），不是 Cytron 的 φ 放置算法。

### C-2 scf.for 块参数布局（归纳变量 arg(0) + iter_arg arg(i+1)）

names 列表登记序：`[acc, m]`（i=0 是 acc，i=1 是 m）。块参数槽：

| 参数槽 | 绑定 | 源码 | i → arg(i+1) |
|---|---|---|---|
| arg(0) | 归纳变量 k | `get_induction_var`（占位 `L957` → 替换 `L1023`） | 归纳变量，不经 i+1 |
| arg(1) | acc（loop-carried i=0） | `for_op.get_body(0).arg(i+1)`, i=0（`L1002`） | 0 → 1 |
| arg(2) | m（loop-carried i=1） | `for_op.get_body(0).arg(i+1)`, i=1（`L1002`） | 1 → 2 |

- 块参数总数 = 1（归纳）+ 2（loop-carried）= 3；槽位 arg(1..2)。
- `+1` 正是跳过 arg(0) 的归纳变量，与官方 SCF 文档
  「an argument for the induction variable, followed by one argument for each loop-carried variable」逐字对上。

### C-3 scf.yield 值链（只追 acc，range(0,3) → k=0,1,2）

init：acc 初值 = 0（init_args = live_val，`L988` 附近）。逐轮 `acc = acc + k`：

| 轮次 | k | iter_arg 入值 arg(1) | acc = acc + k | yield 出值 | 去向 |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 + 0 = 0 | yield 0 | → 下轮 arg(1) |
| 2 | 1 | 0 | 0 + 1 = 1 | yield 1 | → 下轮 arg(1) |
| 3 | 2 | 1 | 1 + 2 = 3 | yield 3 | → `for_op.get_result(0)`（`L1027`） |

- 每轮 `create_yield_op`（`L1012`）把本轮 acc 新值推给下一轮 arg(1)；末轮推给 `get_result(0)`。
- acc 终值 = 0+1+2 = 3 = `for_op.get_result(0)`。
- 终止性：归纳变量每轮 +step(=1) 逼近 ub(=3)，有界 → 3 轮必停。

---

## 源码常量核对（file:Lxxx，本地已核）

| 常量/锚点 | 行 |
|---|---|
| create_poison 占位归纳变量 iv | `code_generator.py:L957` |
| set_value（写 lscope + local_defs） | `code_generator.py:L315-L322` |
| enter_sub_region.__enter__（liveins = lscope.copy） | `code_generator.py:L89` |
| dry visit 循环体 + block.erase | `code_generator.py:L970,L972` |
| 注释「If a variable ... loop-carried variable」 | `code_generator.py:L973` |
| 交集 `for name in local_defs / if name in liveins` | `code_generator.py:L978,L980` |
| create_for_op(lb,ub,step,init_args) | `code_generator.py:L990` |
| iter_arg 绑块参数 `arg(i + 1)` | `code_generator.py:L1002` |
| create_yield_op（loop terminator 推值） | `code_generator.py:L1012` |
| for_op.get_result(i)（结果外传） | `code_generator.py:L1027` |
| visit_if_scf / then yield / else yield / if_op.get_result | `code_generator.py:L656,L670,L677,L680` |
