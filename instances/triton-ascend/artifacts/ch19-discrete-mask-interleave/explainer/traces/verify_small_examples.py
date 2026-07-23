#!/usr/bin/env python3
"""ch19 手工推演的自洽性校验器。

本章 skip_impl（纯 C++ MLIR pass，无 Python 精简版），无法运行真实 pass。
这里用 Python 复刻各 pattern 的 *语义*（不是源码），把 explainer 里手写的小例
逐个算一遍，确认 explainer.json 表格里的数字自洽。真实前后对照以 pin 源码 +
unittest/Conversion/General/DiscreteMaskAccess/{loadstore,atomic}.mlir 的 CHECK 为准。
输出存 verify_small_examples.out.json。
"""
import json

BLOCK = 8
idx = list(range(BLOCK))

out = {}

# ---- m3 混合掩码拆分：contMask=idx<6（连续尾块界），discMask 运行期谓词 ----
validLen = 6
contMask = [i < validLen for i in idx]
discMask = [i in (1, 3, 4, 6) for i in idx]      # 运行期值条件，parse 失败
combined = [c and d for c, d in zip(contMask, discMask)]
# 若无 contMask 全载 [0,8)，会触碰 idx>=validLen 的 OOB 位置
oob_touched = [i for i in idx if i >= validLen]   # 6,7
combined_selected = [i for i in idx if combined[i]]
out["m3"] = {
    "contMask_true": [i for i in idx if contMask[i]],
    "discMask_true": [i for i in idx if discMask[i]],
    "combined_selected": combined_selected,       # {1,3,4}，idx6 被 contMask 挡掉
    "oob_touched_if_no_contMask": oob_touched,     # {6,7}
}

# ---- m4 离散 Load fallback：mask=(idx<2)|(idx>5)，全载 + select(mask, loaded, 0) ----
mask = [(i < 2) or (i > 5) for i in idx]
mem = [10 + i for i in idx]                        # mem[idx] = 10+idx
load_result = [mem[i] if mask[i] else 0 for i in idx]
out["m4"] = {
    "mask_true": [i for i in idx if mask[i]],      # {0,1,6,7}
    "loaded": mem,
    "result": load_result,                         # [10,11,0,0,0,0,16,17]
}

# ---- m5 离散 Store 读-改-写：written = src if mask else origin ----
src = [10 + i for i in idx]                        # 待写新值
origin = [90 + i for i in idx]                     # dst 原值
written = [src[i] if mask[i] else origin[i] for i in idx]
changed = [i for i in idx if mask[i]]
out["m5"] = {
    "src": src,
    "origin": origin,
    "written": written,                            # [10,11,92,93,94,95,16,17]
    "changed_count": len(changed),                 # 4
    "dma_passes": 2,                               # load origin + store 回写
    "total_elems": BLOCK,                          # 每趟仍搬 8
}

# ---- m6 离散 Atomic min：幺元 = i32 max = 2147483647 ----
I32_MAX = 2147483647
src_min = [5, 3, 7, 2, 9, 1, 4, 8]
masked = [src_min[i] if mask[i] else I32_MAX for i in idx]
# 全段 atomic min：未选中位置填 MAX，对 min 无影响
out["m6"] = {
    "i32_max_unit": I32_MAX,
    "src": src_min,
    "maskedValue": masked,                         # [5,3,MAX,MAX,MAX,MAX,4,8]
    "unselected_are_MAX": all(masked[i] == I32_MAX for i in idx if not mask[i]),
}

# ---- m11 Deinterleave：N=4，交错缓冲 2N=8 ----
N = 4
a = [0, 1, 2, 3]        # even half
b = [100, 101, 102, 103]  # odd half
interleaved = []
for k in range(N):
    interleaved.append(a[k])
    interleaved.append(b[k])          # [0,100,1,101,2,102,3,103]
even = interleaved[0::2]              # extract offset0 stride2 size4
odd = interleaved[1::2]               # extract offset1 stride2 size4
out["m11"] = {
    "N": N, "twoN": 2 * N,
    "interleaved": interleaved,
    "even_extract_off0_stride2": even,   # [0,1,2,3]
    "odd_extract_off1_stride2": odd,     # [100,101,102,103]
    "orig_stride": 2, "orig_shape": N,
    "expanded_stride": 1, "expanded_shape": 2 * N,
}

print(json.dumps(out, indent=2))
with open(__file__.replace(".py", ".out.json"), "w") as f:
    json.dump(out, f, indent=2)
