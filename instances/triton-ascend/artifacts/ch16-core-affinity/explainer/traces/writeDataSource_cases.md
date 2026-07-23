# 手工推演 trace — getWriteDataSource 跳 i1 mask(m8)

> trace_source = manual(纯 C++ pass,宿主编不动、无 lit 夹具)。照 `getWriteDataSource`(DAG.cpp:L305-L315)手算。

规则:`op->getInputs().drop_front()`(跳过第一个 operand=写入目标/地址),在其余 input 里找**第一个
element-type 非 i1** 的值返回(i1=bool=mask,不是数据);全是 i1 或没有则 `return nullptr`(L314)。
判 i1 用 `getElementTypeOrSelf(node->value).isInteger(1)`(L308-309)。

| # | store 形态(operands 顺序) | drop_front 后逐个看 | 命中(返回) | 后续 absorbCommon WRITE 分支 |
|---|---------------------------|---------------------|-------------|------------------------------|
| 1 | `tt.store %po, %d`;inputs=[%po(ptr), %d(f32 张量)] | [%d]:f32≠i1 ✓ | 返回 **%d** | 若 %d 恰单核则 store 跟 %d;否则 VECTOR_ONLY |
| 2 | `tt.store %po, %d, %mask`;inputs=[%po(ptr), %d(f32 张量), %mask(i1 张量)] | [%d, %mask]:%d f32≠i1 ✓(mask 未看到就已返回) | 返回 **%d**(mask 被跳过) | 同上,mask 不参与定核 |
| 3 | `tt.store %po, %pred, %mask`;inputs=[%po(ptr), %pred(i1 张量), %mask(i1 张量)] | [%pred, %mask]:%pred i1 → 跳;%mask i1 → 跳;循环耗尽 | 返回 **nullptr**(L314) | `getWriteDataSource` 为空 → 落 `return VECTOR_ONLY`(L369) |

## 单步核对

- **case 1**:`inputRange.drop_front()` = [%d]。`getElementTypeOrSelf(%d)` = f32,`isInteger(1)`==false
  → 立即 `return %d`(L309-311)。
- **case 2**:drop_front = [%d, %mask]。第一个 %d 就非 i1 → 返回 %d,**循环根本没走到 %mask**。
  这就是"跳过 mask"的实现:mask 排在数据之后,遇到数据先返回。
- **case 3**(边界):被存的是 bool 谓词张量(element type i1),与 mask 同为 i1。两个都被 `isInteger(1)`
  判真而跳过,循环走完返回 nullptr → absorbCommon 的 WRITE 分支拿不到数据源,退回 `return VECTOR_ONLY`。
  即"存 bool 数据"这种少见情形下,store 一律判 Vector(诚实点出:i1 判定会把 i1 数据也当 mask 跳掉)。

## 终止性

`drop_front()` 后的 input 列表有限;循环每次前进一个 operand,要么中途 `return` 命中的数据、
要么走完 `return nullptr`。**有限步必返回**——终止来自 operand 列表有限,无迭代不动点问题。
