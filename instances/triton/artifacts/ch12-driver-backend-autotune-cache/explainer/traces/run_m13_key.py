#!/usr/bin/env python3
"""m13 — 编译磁盘缓存键构造的忠实复刻（host，真实 sha256/base64）。

复刻 python/triton/compiler/compiler.py:L231-L232：
    key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
外加 python/triton/runtime/cache.py:L259-L261 的 _base64 得到目录名：
    base64.urlsafe_b64encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")

四段 hash（triton_key / src / backend / options）用短 stand-in 值占位以便肉眼比对；
sha256 与 base64 是真实计算。演示：同键 -> 同目录名（命中）；仅 src.hash 变 -> 目录名变（未命中）。
"""
import base64
import hashlib


def build_dirname(triton_key, src_hash, backend_hash, options_hash, env_vars):
    key = f"{triton_key}-{src_hash}-{backend_hash}-{options_hash}-{str(sorted(env_vars.items()))}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    dirname = base64.urlsafe_b64encode(bytes.fromhex(h)).decode("utf-8").rstrip("=")
    return key, h, dirname


TK = "TK1"          # triton_key()：库自身版本指纹
BK = "cuda90"       # backend.hash()
OPT = "opt_w4s3"    # options.hash()：num_warps/num_stages 等
ENV = {"TRITON_F32": "0"}

rounds = [
    ("首次编译       src=srcA", TK, "srcA", BK, OPT, ENV),
    ("重复同源       src=srcA", TK, "srcA", BK, OPT, ENV),
    ("改一行 kernel  src=srcB", TK, "srcB", BK, OPT, ENV),
]

print("m13 编译磁盘缓存键\n" + "=" * 60)
seen = {}
for label, tk, src, bk, opt, env in rounds:
    key, h, dirname = build_dirname(tk, src, bk, opt, env)
    hit = dirname in seen
    print(f"[{label}]")
    print(f"    key 串   = {key}")
    print(f"    sha256   = {h}")
    print(f"    目录名   = ~/.triton/cache/{dirname}/")
    print(f"    {'HIT  命中既有目录' if hit else 'MISS 新目录，逐产物落盘'}")
    print()
    seen[dirname] = label
print(f"不同目录名共 {len(seen)} 个（同键复用、改源即另起一目录）。")
