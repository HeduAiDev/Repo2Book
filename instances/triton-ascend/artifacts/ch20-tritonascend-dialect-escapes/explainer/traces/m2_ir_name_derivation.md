# m2 — IR 算子名推导（手工推演，trace_source=manual）

**为何 manual**：本章 skip_impl（纯 .td/.cpp 方言 + pass，无 .py 精简版）。IR 名不由运行产生，
而是 tablegen 从 `.td` 的两处字面字符串**静态**拼出：方言 `let name` + op 的 ODS 助记符。
下表逐 op 从 pin 源码字面推导，非任何 dump。

## 两处字面来源
- 方言前缀：`TritonAscendDialect.td:L15` → `let name = "ascend";`
- op 助记符：`TritonAscendOps.td:L34-L35` 基类
  `class TT_Ascend_Op<string mnemonic, ...> : Op<TritonAscend_Dialect, mnemonic, ...>;`
  → 每个 op `def XxxOp : TT_Ascend_Op<"助记符", ...>` 的**第一个模板实参**即打印时点号后的助记符。

MLIR AsmPrinter 打印规则：`<dialect.name>.<op.mnemonic>` → 前缀恒为 `ascend`，后缀恒为 td 里那个字面串。

## 逐 op 推导（全部取自 TritonAscendOps.td 字面）
| C++ 类（带命名空间）              | ODS 定义行 | 第一个模板实参（助记符字面）| 拼出的 IR 名          | 从类名倒推会错成            |
|----------------------------------|-----------|----------------------------|-----------------------|----------------------------|
| triton::ascend::ModOp            | L68       | "mod"                      | ascend.mod            | tt.mod / triton.ascend.mod |
| triton::ascend::CustomOp         | L388      | "custom"                   | ascend.custom         | tt.custom                  |
| triton::ascend::IndexPutOp       | L84       | "index_put"                | ascend.index_put      | ascend.indexput（丢下划线）|
| triton::ascend::GatherOutToUbOp  | L136      | "gather_out_to_ub"         | ascend.gather_out_to_ub | ascend.gatherouttoub（丢全部下划线）|

**关键非平凡分支**：IndexPutOp→"index_put"、GatherOutToUbOp→"gather_out_to_ub" 证明助记符是
snake_case，**无法**由 CamelCase 类名机械还原（下划线位置信息在类名里已丢失）——必须读 td 字面。
ModOp→"mod" 恰好巧合可小写还原，故不能只拿这种退化例，须并列非平凡例。

## 全方言一致性
`grep -c 'TT_Ascend_Op<"' TritonAscendOps.td = 11` → 全 11 个 op 都经此基类，故 11 个 IR 名
一律 `ascend.<助记符>`，无一例外。
