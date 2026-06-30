#!/usr/bin/env python3
"""ch22 RecomputeScheduler 分叉：block 不够时，kv_consumer 丢弃重算而非本地抢占。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1280, 760
TXT = "#1e293b"
SUB = "#64748b"
ASC = "#7c3aed"
VLLM = "#0f766e"
RED = "#dc2626"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs><marker id="ar" viewBox="0 0 12 10" refX="11" refY="5" markerWidth="11" markerHeight="9" orient="auto"><path d="M0,0 L12,5 L0,10 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="24" font-weight="bold" fill="{TXT}">block 不够时的分叉：抢占（复用）vs 丢弃重算（昇腾特化）</text>')


def box(cx, cy, w, h, fill, stroke, lines, sw=1.7):
    x = cx - w / 2
    L.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    n = len(lines)
    for i, (t, s) in enumerate(lines):
        ty = cy + h / 2 + (i - (n - 1) / 2) * (s + 5) + s * 0.34
        col = stroke if i == 0 else TXT
        fw = "bold" if i == 0 else "normal"
        L.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="{s}" font-weight="{fw}" fill="{col}">{esc(t)}</text>')


def vline(x, y1, y2, color="#475569"):
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#ar)"/>')


CX = W / 2
box(CX, 70, 380, 50, "#f1f5f9", "#475569", [("RUNNING 循环：allocate_slots(request)", 15)])
vline(CX, 120, 152)
# 菱形判定
L.append(f'<polygon points="{CX},155 {CX+110},205 {CX},255 {CX-110},205" fill="#fffbeb" stroke="#d97706" stroke-width="1.7"/>')
L.append(f'<text x="{CX}" y="200" text-anchor="middle" font-size="14" font-weight="bold" fill="#d97706">new_blocks</text>')
L.append(f'<text x="{CX}" y="218" text-anchor="middle" font-size="14" fill="#d97706">is None?</text>')

# 右：成功 break
L.append(f'<line x1="{CX+110}" y1="205" x2="{CX+300}" y2="205" stroke="#16a34a" stroke-width="2" marker-end="url(#ar)"/>')
L.append(f'<text x="{CX+205}" y="195" text-anchor="middle" font-size="13" font-weight="bold" fill="#16a34a">否（够）</text>')
box(CX+390, 182, 230, 48, "#f0fdf4", "#16a34a", [("调度该请求，break", 14)])

# 下：None → 判 kv_producer
vline(CX, 255, 290)
L.append(f'<text x="{CX+14}" y="278" font-size="13" font-weight="bold" fill="{RED}">是（不够）</text>')
L.append(f'<polygon points="{CX},293 {CX+130},345 {CX},397 {CX-130},345" fill="#faf5ff" stroke="{ASC}" stroke-width="1.7"/>')
L.append(f'<text x="{CX}" y="340" text-anchor="middle" font-size="14" font-weight="bold" fill="{ASC}">is_kv_producer?</text>')
L.append(f'<text x="{CX}" y="360" text-anchor="middle" font-size="12.5" fill="{SUB}">（PD 角色）</text>')

# 左分支：producer / 非 PD → 常规抢占（复用）
L.append(f'<line x1="{CX-130}" y1="345" x2="{CX-300}" y2="345" stroke="{VLLM}" stroke-width="2" marker-end="url(#ar)"/>')
L.append(f'<text x="{CX-215}" y="335" text-anchor="middle" font-size="13" font-weight="bold" fill="{VLLM}">是 / 非 PD</text>')
box(CX-450, 318, 280, 60, "#ecfeff", VLLM, [("常规 preempt（逐字复用 vLLM）", 13.5), ("PRIORITY/FCFS 选最低优先级踢回 waiting", 11.5)])

# 右分支：consumer → 丢弃重算
L.append(f'<line x1="{CX+130}" y1="345" x2="{CX+300}" y2="345" stroke="{ASC}" stroke-width="2" marker-end="url(#ar)"/>')
L.append(f'<text x="{CX+215}" y="335" text-anchor="middle" font-size="13" font-weight="bold" fill="{ASC}">否 (consumer)</text>')
box(CX+455, 312, 300, 72, "#f5f3ff", ASC, [("丢弃重算（昇腾特化）", 13.5), ("running.pop() + kv_cache_manager.free()", 11.5), ("→ append recomputed_reqs", 11.5)])

# consumer 往下到 update_from_output
vline(CX+455, 384, 470)
box(CX+455, 472, 320, 66, "#f5f3ff", ASC, [("update_from_output 回吐", 13.5), ("EngineCoreOutput(stop_reason='recomputed')", 11), ("finish_reason=STOP, new_token_ids=[]", 11)])
vline(CX+455, 538, 600)
box(CX+455, 602, 320, 66, "#fef2f2", RED, [("PD proxy 收到 'recomputed'", 13.5), ("把请求改投他节点重新 prefill+decode", 11.5), ("回指 ch10/ch11 PD 分离与 remote KV", 11)])

# 底注
L.append(f'<text x="{W/2}" y="720" text-anchor="middle" font-size="13.5" fill="{SUB}">PD 分离下 decode 节点的 KV 来自 prefill 节点；本地抢占重算无意义，丢回 proxy 改投他处更划算。</text>')

L.append('</svg>')
open("recompute_flow.svg", "w").write('\n'.join(L))
print("wrote recompute_flow.svg")
