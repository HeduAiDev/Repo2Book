#!/usr/bin/env python3
"""
ch33 capability-boundary census — STATIC scan (no NPU needed).

This is the chapter's "trace": we cannot run the unittest suite (it needs
CANN/NPU — conftest.assign_npu binds a real device on module load). So the
one-hand evidence is the *static* set of @pytest.mark.skip/xfail/skipif
markers and their reason strings. This script counts them reproducibly.

Run from:  third_party/ascend/unittest/  of the pinned triton-ascend source.
    python3 census_skip_markers.py
Emits census_skip_markers.out.json alongside.
"""
import re, glob, os, json, sys
from collections import Counter

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."  # unittest dir

def total_py():
    per = {}
    for d in ("pytest_ut", "autotune_ut", "custom_op", "affine_map", "Conversion"):
        per[d] = len(glob.glob(os.path.join(ROOT, d, "**", "*.py"), recursive=True))
    toplevel = [p for p in glob.glob(os.path.join(ROOT, "*.py"))]
    per["<toplevel conftest.py>"] = len(toplevel)
    per["TOTAL"] = len(glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True))
    return per

def scan(subdir):
    """Return (active_files, active_occurrences, commented_occurrences)."""
    files = set()
    occ = []          # (file,line,kind,reason,commented,is_param)
    for f in sorted(glob.glob(os.path.join(ROOT, subdir, "*.py"))):
        src = open(f).read().splitlines()
        for i, l in enumerate(src):
            m = re.search(r'(#\s*)?(@)?pytest\.mark\.(skip|xfail)\b', l)
            if not m:
                continue
            if "skipif" in l:            # conditional (hardware) — counted separately
                continue
            commented = l.lstrip().startswith("#")
            is_param = "marks=" in l
            kind = m.group(3)
            blob = " ".join(src[i:i+3])   # reason may sit on a following line
            rm = re.search(r'reason\s*=\s*["\']([^"\']*)["\']', blob)
            reason = rm.group(1) if rm else None
            occ.append((os.path.basename(f), i+1, kind, reason, commented, is_param))
            if not commented:
                files.add(os.path.basename(f))
    active = [o for o in occ if not o[4]]
    commented = [o for o in occ if o[4]]
    return sorted(files), active, commented

def skipif_files(subdir):
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, subdir, "*.py"))):
        for i, l in enumerate(open(f).read().splitlines()):
            if "pytest.mark.skipif" in l:
                rm = re.search(r'reason\s*=\s*["\']([^"\']*)["\']', l)
                out.append((os.path.basename(f), i+1, rm.group(1) if rm else None))
    return out

def category(r):
    if r is None: return "NO-REASON"
    if "waiting for TA to support" in r: return "TA (upstream triton-ascend)"
    if "bishengir-compile" in r or "compiler to support" in r: return "bishengir / compiler"
    if "NPUIR is updated in April" in r: return "NPUIR April regression"
    if "ub overflow" in r.lower(): return "UB overflow (hardware)"
    if r == "attn_cp": return "attn_cp (whole batch)"
    if "randomly failed" in r.lower(): return "flaky: randomly failed"
    if "expm1 failed sometimes" in r: return "flaky: expm1 sometimes"
    if "full tensor has problem" in r: return "sporadic: atomic_cas full-tensor"
    if "allow_tf32" in r: return "xfail: allow_tf32"
    if "multi-process error" in r: return "COMMENTED: 3Dgrid multi-process"
    return "OTHER: " + r

result = {"total_py_by_dir": total_py(), "subdirs": {}}
grand_active_files = 0
for sd in ("pytest_ut", "autotune_ut", "custom_op"):
    files, active, commented = scan(sd)
    grand_active_files += len(files)
    result["subdirs"][sd] = {
        "active_unconditional_files": len(files),
        "active_files_list": files,
        "active_occurrences": len(active),
        "commented_occurrences": [(o[0], o[1], o[3]) for o in commented],
        "skipif_hardware": skipif_files(sd),
    }

# reason categories over pytest_ut active occurrences (the census the chapter uses)
_, pt_active, _ = scan("pytest_ut")
cats = Counter(category(o[3]) for o in pt_active)
result["pytest_ut_reason_categories"] = dict(cats.most_common())
result["pytest_ut_reason_categories"]["_SUM"] = sum(cats.values())
result["grand_total_active_unconditional_files_3subdirs"] = grand_active_files

# literal-wording audit: which reasons actually contain the string "bishengir"
bishengir_lines = []
for f in sorted(glob.glob(os.path.join(ROOT, "pytest_ut", "*.py"))):
    for i, l in enumerate(open(f).read().splitlines()):
        if "bishengir" in l:
            rm = re.search(r'reason\s*=\s*["\']([^"\']*)["\']', l)
            bishengir_lines.append((os.path.basename(f), i+1, rm.group(1) if rm else l.strip()))
result["reasons_literally_naming_bishengir"] = bishengir_lines

print(json.dumps(result, indent=2, ensure_ascii=False))
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "census_skip_markers.out.json")
with open(out_path, "w") as fh:
    json.dump(result, fh, indent=2, ensure_ascii=False)
