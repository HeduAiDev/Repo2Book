# m1 trace record — `_builder.create_make_range(0, 16)` 穿过双语接缝

本章 `trace_source="manual"`：绑定的真身活在**编译好的 C++ 扩展 `libtriton.so`** 里，其
pin=v3.2.0 的构建产物不在本环境（本机安装的是 triton 3.6.0，版本不匹配，不能冒充 pin）。
故权威数值一律取自 pin 源码常量（`file:Lxxx`），下方另附一段**非权威**的活体佐证（3.6.0）以
证明这条绑定链真实存在、跨版本稳定。

## 权威：pin v3.2.0 源码常量（每个数字标 file:Lxxx）

选参数 `start=0, end=16`（小到可心算；BLOCK=16 是前几章 `tl.arange(0, BLOCK)` 的常见取值）。

| 层 | 源码锚 | 关键动作 | 关键标量 | 产物 |
|----|--------|----------|----------|------|
| ① Python 前端 | — | `_builder.create_make_range(0, 16)` | start=0, end=16 | 调 `ir.builder` 上名为 `create_make_range` 的方法 |
| ② pybind11 派发 | ir.cc:L883 | 按 `.def("create_make_range", …)` 的方法名找到对应 C++ lambda，把 `(0,16)` 转成 `int start, int end` | start=0, end=16 | 进入 lambda 体 |
| ③ lambda 算返回类型 | ir.cc:L885-886 | `RankedTensorType::get({end-start}, getI32Type())` → `{16-0}={16}`，元素 i32(32 位) | 形状=[16]，dtype=i32 | `retType = tensor<16xi32>` |
| ④ 落到公共底座 | ir.cc:L887 → L96-99 | `self.create<MakeRangeOp>(retType, 0, 16)`；`create<OpTy>` 先 `getLastLoc()` 取 loc，再 `builder->create<MakeRangeOp>(loc, retType, 0, 16)` | loc=lastLoc（1 个） | 建出 1 个 MLIR op |
| ⑤ MLIR op 产出 | — | MakeRangeOp 落成 | start=0, end=16 | `tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>` |
| ⑥ 返回 Python | ir.cc:L884(`-> Value`) | 结果 `mlir::Value` 按 return_value_policy 包成 `ir.value` | — | Python 侧拿到一个 `ir.value` 继续追踪 |

常量出处：
- `.def("create_make_range", …)` 方法名与 lambda 签名 `(int start, int end) -> Value`：`python/src/ir.cc:L883-884`
- `RankedTensorType::get({end - start}, self.getBuilder().getI32Type())`：`python/src/ir.cc:L885-886`
- `self.create<MakeRangeOp>(retType, start, end)`：`python/src/ir.cc:L887`
- `create<OpTy>` 底座：`auto loc = getLastLoc(); return builder->create<OpTy>(loc, …)`：`python/src/ir.cc:L96-99`
- create_* 方法总数（同走此底座）：`grep -c '.def("create_' python/src/ir.cc` = **129**（pin v3.2.0）

## 非权威活体佐证（本机 triton 3.6.0，仅证链路存在，勿引其数字为 pin）

```
create_make_range in dir(ir.builder): True
create_get_program_id in dir(ir.builder): True
num create_* on ir.builder (installed 3.6.0): 137        # 3.6.0 已增至 137，pin 3.2.0 为 129
passes subgroups present: ['common','convert','ttir','ttgpuir','llvmir','analysis']
ttir has add_combine: True
interpreter has load/store: True True
```

结论：`create_make_range` 确实作为 `ir.builder` 的一个 Python 方法存在，且 `passes` 六个子模块、
`interpreter.load/store` 均可从 `triton._C.libtriton` 反查到——与 `.cc` 里的 `.def`/
`ADD_PASS_WRAPPER`/`m.def` 名字一一对上。数量随版本漂移（137 vs 129）正说明：应以 pin 源码为准。
