# -*- coding: utf-8 -*-
"""独立逐字核验：dossier.embed_excerpts[].code 与 v0.27.1 源文件切片逐字符比对。临时脚本。"""
import json
from pathlib import Path

SRC = Path("E:/Laboratory/Repo2Book/instances/vllm/source")
D = json.loads(Path("E:/Laboratory/Repo2Book/instances/vllm/artifacts-v3/ch30-grammar-compilation/dossier/dossier.json").read_text(encoding="utf-8"))

bad = 0
for e in D["embed_excerpts"]:
    rel = e["path"]  # vllm/... 规范路径 → 源树相对路径
    lo, hi = (int(x[1:]) for x in e["lines"].split("-"))
    src_lines = (SRC / rel).read_text(encoding="utf-8").splitlines()
    expect = "\n".join(src_lines[lo - 1:hi])
    if expect != e["code"]:
        bad += 1
        print(f"MISMATCH {rel} {e['lines']}")
        for i, (a, b) in enumerate(zip(expect.splitlines(), e["code"].splitlines())):
            if a != b:
                print(f"  line {lo + i}: src={a!r}\n           dos={b!r}")
        ea, eb = expect.splitlines(), e["code"].splitlines()
        if len(ea) != len(eb):
            print(f"  line count: src={len(ea)} dossier={len(eb)}")
    else:
        print(f"OK  {rel} {e['lines']} ({hi - lo + 1} 行逐字一致)")

print(f"\n== {len(D['embed_excerpts'])} excerpts, {bad} mismatches ==")
raise SystemExit(1 if bad else 0)
