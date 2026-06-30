#!/usr/bin/env python3
"""ch22 复用 vs 特化分区图：KV 管理 / 调度的核心循环，昇腾 90% 原样继承、只在几处开子类。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1340, 760
TXT = "#1e293b"
SUB = "#64748b"
VLLM = "#0f766e"   # 原样复用 青
ASC = "#7c3aed"    # 昇腾特化 紫

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs><marker id="ar" viewBox="0 0 12 10" refX="11" refY="5" markerWidth="11" markerHeight="9" orient="auto"><path d="M0,0 L12,5 L0,10 Z" fill="#7c3aed"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="25" font-weight="bold" fill="{TXT}">KV 管理与调度：90% 原样复用 vLLM，只在 5 处开子类</text>')
L.append(f'<text x="{W/2}" y="66" text-anchor="middle" font-size="14" fill="{SUB}">箭头 = 继承 / override；右列每个特化都「贴着」左列对应基类，只改最小落点</text>')

# 两列分区底
L.append(f'<rect x="40" y="92" width="560" height="610" rx="14" fill="#f0fdfa" stroke="{VLLM}" stroke-width="1.6"/>')
L.append(f'<rect x="740" y="92" width="560" height="610" rx="14" fill="#faf5ff" stroke="{ASC}" stroke-width="1.6"/>')
L.append(f'<text x="320" y="122" text-anchor="middle" font-size="18" font-weight="bold" fill="{VLLM}">原样复用 · vLLM 基座（只 import 不改）</text>')
L.append(f'<text x="1020" y="122" text-anchor="middle" font-size="18" font-weight="bold" fill="{ASC}">昇腾特化 · 子类 / 重映射</text>')


def box(cx, cy, w, h, fill, stroke, lines, sw=1.7):
    x = cx - w / 2
    L.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    n = len(lines)
    for i, (t, s) in enumerate(lines):
        ty = cy + h / 2 + (i - (n - 1) / 2) * (s + 5) + s * 0.34
        col = stroke if i == 0 else TXT
        fw = "bold" if i == 0 else "normal"
        L.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="{s}" font-weight="{fw}" fill="{col}">{esc(t)}</text>')


def arrow(y, label):
    L.append(f'<line x1="605" y1="{y}" x2="735" y2="{y}" stroke="{ASC}" stroke-width="2" marker-end="url(#ar)"/>')
    L.append(f'<text x="670" y="{y-9}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="{ASC}">{esc(label)}</text>')


rows = [
    (175,
     ("FullAttentionManager", "per-spec KV 管理基类"),
     ("CompressAttentionManager", "压缩 MLA：//compress_ratio 后调 super()"),
     "继承 + 新增"),
    (270,
     ("spec_manager_map", "spec → manager 查表"),
     ("get_manager_for_kv_cache_spec", "重映射：压缩 MLA 改选 + 设 admission cap"),
     "复用查表"),
    (365,
     ("Scheduler.schedule", "RUNNING/WAITING 双循环骨架"),
     ("SchedulerDynamicBatch", "查表动态 token 预算 + decode-first 重排"),
     "override"),
    (460,
     ("Scheduler / update_from_output", "抢占 + 输出组装骨架"),
     ("RecomputeScheduler (+Config/Output)", "kv_consumer 丢弃重算 → 回吐 PD proxy"),
     "override"),
    (555,
     ("Scheduler.schedule", "num_new_tokens 计算骨架"),
     ("ProfilingChunkScheduler", "二次模型预测 chunk size 收窄"),
     "override"),
]
for y, lft, rgt, lab in rows:
    box(320, y, 500, 70, "#ecfeff", VLLM, [(lft[0], 15), (lft[1], 12.5)])
    box(1020, y, 500, 70, "#f5f3ff", ASC, [(rgt[0], 15), (rgt[1], 12.5)])
    arrow(y + 35, lab)

# 底部：原样复用块池
box(320, 645, 500, 44, "#ecfeff", VLLM, [("BlockPool · 物理 block 分配 + 前缀缓存池（零改动）", 14)])
L.append(f'<text x="1020" y="660" text-anchor="middle" font-size="13.5" fill="{SUB}">（无对应特化——物理 block 计数与硬件无关，整块原样继承）</text>')

L.append(f'<text x="{W/2}" y="730" text-anchor="middle" font-size="14" font-weight="bold" fill="{TXT}">立意：成熟插件对宿主「核心循环」该尽量少碰——昇腾这里只动 KV 换算比例与三种调度策略的循环落点。</text>')

L.append('</svg>')
open("reuse_map.svg", "w").write('\n'.join(L))
print("wrote reuse_map.svg")
