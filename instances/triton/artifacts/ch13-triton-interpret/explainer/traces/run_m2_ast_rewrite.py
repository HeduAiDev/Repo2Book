#!/usr/bin/env python3
"""m2 — AST 改写：把每个赋值包成 to_tensor（为何重写而非原样跑 Python）。

现场取证：直接加载 pin triton v3.2.0 源码里**真实的** ASTTransformer 类
（interpreter.py:L1109-L1126，visit_Assign 是纯 ast 变换，不依赖任何已编译的
libtriton），用它改写一个小核体的 AST，逐行对照「改写前 → 改写后」。

要点：ASTTransformer.visit_Assign 把每条赋值 `x = value` 改成
`x = triton.language.semantic.to_tensor(value, interpreter_builder, False)`——
右值被包一层 to_tensor（semantic.py:L111-L126），把裸 python 标量提升成 tl.tensor。
这样核体内每个中间量都带张量语义（dtype + numpy 后端），tl.* 运算才有统一语义可依；
若原样跑 Python，`c = 2` 会得到裸 int，后续张量运算/属性无处依附。

取证方式说明：本 trace 不需要 GPU、也不需要已编译 triton——它执行的是 pin v3.2.0
interpreter.py 里 ASTTransformer 类**逐字**源码（从源文件切片 exec，仅依赖标准库 ast）。
因此这段改写结果就是 v3.2.0 解释器对该核体产生的真实 AST。
"""
import ast
import inspect
import json
import os
import textwrap

PIN_SRC = "/mnt/e/Laboratory/Repo2Book/instances/triton/source/python/triton/runtime/interpreter.py"


def load_pinned_ast_transformer():
    """从 pin v3.2.0 源文件里切出 ASTTransformer 类逐字源码并 exec（仅依赖 ast）。"""
    src_lines = open(PIN_SRC, encoding="utf-8").read().splitlines()
    # L1109-L1126（1-based）：class ASTTransformer ... return node
    start = None
    for i, line in enumerate(src_lines):
        if line.startswith("class ASTTransformer(ast.NodeTransformer):"):
            start = i
            break
    assert start is not None, "找不到 ASTTransformer 类定义"
    # 收集到该类结束（下一处顶格非空行）
    body = [src_lines[start]]
    for line in src_lines[start + 1:]:
        if line and not line[0].isspace() and not line.startswith("class ASTTransformer"):
            break
        body.append(line)
    class_src = "\n".join(body)
    ns = {"ast": ast}
    exec(class_src, ns)
    return ns["ASTTransformer"], class_src


# 一个代表性的小核体（不实际执行，只取其源码做 AST 改写演示）
def sample_kernel(x_ptr, y_ptr, n, BLOCK):
    pid = tl.program_id(0)          # 赋值 1：program_id
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # 赋值 2：偏移向量
    c = 2                           # 赋值 3：裸标量常量
    y = tl.load(x_ptr + offs) * c   # 赋值 4：读入并乘常量
    tl.store(y_ptr + offs, y)       # 非赋值语句：不被改写


def main():
    ASTTransformer, class_src = load_pinned_ast_transformer()

    src = textwrap.dedent(inspect.getsource(sample_kernel))
    tree = ast.parse(src)

    # 收集改写前每条赋值语句
    before = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            before.append(ast.unparse(node).strip())

    # 用 pin v3.2.0 真实 ASTTransformer 改写
    new_tree = ASTTransformer().visit(tree)
    ast.fix_missing_locations(new_tree)

    after = []
    for node in ast.walk(new_tree):
        if isinstance(node, ast.Assign):
            after.append(ast.unparse(node).strip())

    n_assign = len(before)
    n_store = sum(
        1 for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    )

    print(f"[probe] 核体内赋值语句数 = {n_assign}（每条都被包一层 to_tensor）")
    print(f"[probe] 非赋值表达式语句（如 tl.store）数 = {n_store}（不被 visit_Assign 触碰）")
    print("[probe] 逐条改写对照：")
    for i, (b, a) in enumerate(zip(before, after)):
        print(f"  [{i}] BEFORE: {b}")
        print(f"  [{i}] AFTER : {a}")

    record = {
        "pinned_class_source_hash_head": class_src.splitlines()[0],
        "n_assign": n_assign,
        "n_store_stmt": n_store,
        "rewrites": [{"before": b, "after": a} for b, a in zip(before, after)],
    }
    out = os.path.join(os.path.dirname(__file__), "m2_ast_rewrite.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
