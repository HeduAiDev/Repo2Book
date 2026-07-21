#!/usr/bin/env python3
"""IR 算子名体例校验 —— 禁止「方言前缀 + C++ 类名」的混合写法(exp-2026-07-21-04)。

判据:**IR 算子名 = 方言 `let name` + ODS 助记符**,两处都要回 .td 查,**不能从 C++ 类名推**。
- MLIR 助记符惯例小写起头(triton-ascend 全仓 119 个助记符无一以大写开头);
- C++ 类名一律 CamelCase 且以 `Op` 结尾。
⇒ `<小写方言>.<大写开头…Op>` 这个形态,几乎必然是把 C++ 类名当成了 IR 算子名。

同一病种已连发三章:
  ch06 `tt.indirect_load`(方言前缀错,真名 `ascend.indirect_load`)
  ch08 `ascend.CustomOp`(真名 `ascend.custom`)
  ch07 `hivm.CustomOp` + Book Bible glossary(真名 `hivm.custom`)

规矩的 C++ 类名引用(`triton::ascend::CustomOp`、`hivm::SyncBlockSetOp`)带 `::` 作用域,不在本检查范围。

用法:
  python3 scripts/lint_ir_opname.py <chapter_dir>      # 扫正文 + 图 + 素材
  python3 scripts/lint_ir_opname.py --all              # 扫活动实例全书 + Book Bible
退出码 1 = 有问题。
"""
import glob
import json
import os
import re
import sys

import instance

# <小写方言>.<大写开头的 CamelCase…Op>;前面不能紧跟 ':'(排除 C++ 的 a::B 形式)
_BAD = re.compile(r'(?<![:\w])([a-z][a-z0-9_]{1,15})\.([A-Z][A-Za-z0-9]*Op)\b')

# 只对**真实存在的 MLIR 方言**报警。否则会误伤 Python 的 `模块.类名`——
# 实测误报:`ast.BinOp`(Python ast 模块)、`distributed.ReduceOp`(torch.distributed)。
# 方言名从实例源码的 .td 里 `let name = "..."` 抽,外加 MLIR 上游常见内建方言。
_BUILTIN_DIALECTS = {"func", "arith", "scf", "cf", "memref", "tensor", "linalg", "affine",
                     "vector", "gpu", "llvm", "math", "bufferization", "index", "annotation", "scope"}
_dialects_cache = {}


def known_dialects(source_root=None):
    key = str(source_root)
    if key in _dialects_cache:
        return _dialects_cache[key]
    names = set(_BUILTIN_DIALECTS)
    root = source_root or instance.source_dir()
    try:
        for td in glob.iglob(os.path.join(str(root), "**", "*.td"), recursive=True):
            try:
                for m in re.finditer(r'let\s+name\s*=\s*"([a-z][\w]*)"',
                                     open(td, encoding="utf-8", errors="replace").read()):
                    names.add(m.group(1))
            except OSError:
                continue
    except Exception:
        pass
    _dialects_cache[key] = names
    return names


def scan_text(text: str, dialects=None):
    """返回 [(命中写法, 建议), ...]。dialects=None 时用「本仓 .td 声明的方言 + MLIR 内建」。"""
    if dialects is None:
        dialects = known_dialects()
    out = []
    for m in _BAD.finditer(text or ""):
        dialect, cls = m.group(1), m.group(2)
        if dialect not in dialects:
            continue
        mnemonic = re.sub(r'Op$', '', cls)
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', mnemonic).lower()
        out.append((f"{dialect}.{cls}",
                    f"疑似把 C++ 类名当 IR 算子名。IR 名应为 方言名 + ODS 助记符"
                    f"(回 .td 查 `def {cls} : ...<\"<助记符>\">`),很可能是 `{dialect}.{snake}`;"
                    f"若确要引 C++ 类,请写带 :: 的完整形式(如 `{dialect}::{cls}`)。"))
    return out


def _targets(chapter_dir: str):
    pats = ["narrative/*.md", "diagrams/*.svg", "diagrams/*.py",
            "explainer/explainer.json", "dossier/dossier.json", "diagrams/figure-manifest.json"]
    for p in pats:
        for f in glob.glob(os.path.join(chapter_dir, p)):
            yield f


def _strip_review_notes(path: str, text: str) -> str:
    """把 figure-manifest.json 里 `blind_review.notes` 的内容剔掉再扫。

    评审记录必须能原样引用「被判错的写法」来说明问题(『页脚写着 X,应为 Y』)。
    若不豁免,门禁对这类章会**永远红**——而永远红不掉的门禁等于没有门禁
    (与 warn 噪音同一种失效模式)。豁免只限 notes;claim/figure_spec 等断言字段照常严查。
    """
    if os.path.basename(path) != "figure-manifest.json":
        return text
    try:
        d = json.loads(text)
    except ValueError:
        return text

    def scrub(o):
        if isinstance(o, dict):
            return {k: ("" if k == "notes" else scrub(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [scrub(x) for x in o]
        return o

    return json.dumps(scrub(d), ensure_ascii=False)


def lint(chapter_dir: str, dialects=None):
    issues = []
    for f in _targets(chapter_dir):
        try:
            text = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        text = _strip_review_notes(f, text)
        for bad, tip in scan_text(text, dialects=dialects):
            issues.append(f"  {os.path.relpath(f, chapter_dir)}: `{bad}` —— {tip}")
    return issues


def main():
    args = sys.argv[1:]
    if args and args[0] == "--all":
        dirs = sorted({os.path.dirname(os.path.dirname(p))
                       for p in glob.glob(instance.chapters_glob())})
        extra = glob.glob(os.path.join(str(instance.book_dir()), "bible", "*.json"))
    elif args:
        dirs, extra = [args[0]], []
    else:
        print("Usage: python3 lint_ir_opname.py <chapter_dir> | --all")
        return 1
    total = 0
    for d in dirs:
        iss = lint(d)
        if iss:
            total += len(iss)
            print(f"\n❌ {os.path.basename(d)} ({len(iss)}):")
            print("\n".join(iss))
    for f in extra:
        for bad, tip in scan_text(open(f, encoding="utf-8", errors="replace").read()):
            total += 1
            print(f"\n❌ {os.path.relpath(f)}: `{bad}` —— {tip}")
    if total == 0:
        print("✓ IR 算子名体例检查通过(无「方言前缀 + C++ 类名」混写)")
        return 0
    print(f"\n🔴 {total} 处 —— IR 算子名 = 方言 let name + ODS 助记符,不能从 C++ 类名推")
    return 1


if __name__ == "__main__":
    sys.exit(main())
