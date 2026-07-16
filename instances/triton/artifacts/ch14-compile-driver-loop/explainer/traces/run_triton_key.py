#!/usr/bin/env python3
"""m2 verification trace — triton_key() 的『枚举 + sha256 拼接』逻辑,跑在 pin v3.2.0 源码树上。

triton_key() 源码:python/triton/compiler/compiler.py:L134-L166
  contents = [sha256(compiler.py)]                       # frontend(compiler.py 自身)
           + [sha256(每个模块) for 模块 in walk(compiler 包)]
           + [sha256(每个模块) for 模块 in walk(backends 包)]
           + [sha256(libtriton.so)]                        # 编译好的 C++ 后端二进制
           + [sha256(每个模块) for 模块 in walk(language 包)]
  return __version__ + '-'.join(contents)

注意:真实 triton_key 用 pkgutil.walk_packages 枚举『已安装』的包,其 language.extra 子包会
按安装后端(cuda/hip)另拉入 libdevice 等文件——枚举总数因此依安装而异。本脚本刻意只枚举 pin
源码树自身的 .py(os.walk,不越出 TRITON_PATH),给出与安装无关的『源码树静态计数』作教学基线,
并做与安装无关、稳健的核心论证:改任一文件一个字节 → 整条 key 的 sha256 翻转。

本机无 libtriton.so 构建(源码-only clone),.so 项用固定占位符表示『它是 key 的一项』;
这不影响本脚本要证的两件事:桶结构 + 一处改动即整键翻转。
输出 JSON 存 triton_key.json。
"""
import hashlib
import json
import os

TRITON_PATH = os.path.abspath(
    "/mnt/e/Laboratory/Repo2Book/instances/triton/source/python/triton")
VERSION = "3.2.0"  # python/triton/__init__.py:L2

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def pkg_py_files(pkg):
    """pin 源码树里某包下的全部 .py(含 __init__ 与子包),排序稳定;不越出 TRITON_PATH。"""
    root = os.path.join(TRITON_PATH, pkg)
    out = []
    for dirpath, _dn, fnames in os.walk(root):
        for fn in sorted(fnames):
            if fn.endswith(".py"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)

def enumerate_inputs():
    """按 triton_key 顺序列出被 sha256 的输入(源码树静态基线)。"""
    inputs = []  # (bucket, path_or_None) ; path=None 表示 libtriton.so 占位
    inputs.append(("frontend", os.path.join(TRITON_PATH, "compiler", "compiler.py")))
    for p in pkg_py_files("compiler"):
        inputs.append(("compiler", p))
    for p in pkg_py_files("backends"):
        inputs.append(("backends", p))
    inputs.append(("libtriton.so", None))
    for p in pkg_py_files("language"):
        inputs.append(("language", p))
    return inputs

def build_key(inputs, mutate_path=None, extra=b""):
    contents = []
    for _bucket, p in inputs:
        if p is None:
            contents.append("<libtriton.so-placeholder>")
            continue
        with open(p, "rb") as f:
            data = f.read()
        if mutate_path is not None and os.path.samefile(p, mutate_path):
            data = data + extra
        contents.append(sha(data))
    return VERSION + "-".join(contents)

def main():
    inputs = enumerate_inputs()
    buckets = {}
    for b, _ in inputs:
        buckets[b] = buckets.get(b, 0) + 1

    # 桶计数(源码树静态基线)
    n_compiler = buckets.get("compiler", 0)
    n_backends = buckets.get("backends", 0)
    n_language = buckets.get("language", 0)
    n_frontend = buckets.get("frontend", 0)       # =1,compiler.py 被额外单列一次
    n_total_py = n_frontend + n_compiler + n_backends + n_language
    n_total_incl_so = n_total_py + 1               # + libtriton.so

    key0 = build_key(inputs)
    fp0 = sha(key0.encode())

    # 只改『一个文件的一个字节』——选 compiler 包里的 code_generator.py
    mut = os.path.join(TRITON_PATH, "compiler", "code_generator.py")
    key1 = build_key(inputs, mutate_path=mut, extra=b"\n# +1 line\n")
    fp1 = sha(key1.encode())

    out = {
        "version": VERSION,
        "note": "counts = pin 源码树静态基线(与安装无关);真实运行时 language.extra 后端扩展会另增,不改变结论",
        "bucket_counts": {
            "frontend(compiler.py 单列)": n_frontend,
            "compiler_pkg_py": n_compiler,
            "backends_pkg_py": n_backends,
            "language_pkg_py": n_language,
            "libtriton_so": 1,
        },
        "total_py_hashed_static": n_total_py,
        "total_inputs_incl_libtriton_so": n_total_incl_so,
        "files_edited": 1,
        "fingerprint_before_edit": fp0,
        "fingerprint_after_edit": fp1,
        "fingerprint_changed": fp0 != fp1,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triton_key.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
