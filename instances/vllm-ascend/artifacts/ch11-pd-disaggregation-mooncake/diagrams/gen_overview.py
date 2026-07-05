#!/usr/bin/env python3
"""PD 分离三层全景：proxy 负载均衡（顶）→ 连接器分发（中）→ mooncake P2P 传输（底），侧挂 KV 亲和。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1280, 760
S = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
S.append('<defs>')
S.append('<marker id="arr" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#475569"/></marker>')
S.append('<marker id="arrP" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#7c3aed"/></marker>')
S.append('</defs>')
S.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10, sw=2):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x, y, t, size=14, anchor="middle", weight="normal", fill="#1e293b", italic=False):
    st = ' font-style="italic"' if italic else ''
    S.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}"{st}>{esc(t)}</text>')


def lines(cx, y0, items, size=13, dy=20, fill="#334155", anchor="middle"):
    for i, t in enumerate(items):
        txt(cx, y0 + i*dy, t, size=size, fill=fill, anchor=anchor)


txt(W/2, 42, "PD 分离三层：proxy 分发 → 连接器分发 → mooncake P2P 传输", 23, "middle", "bold", "#0f172a")
txt(W/2, 70, "一个请求自顶向下穿过三层；KV 亲和在连接器层旁挂一问，决定省多少传输", 14, "middle", "normal", "#64748b")

# ── 第三层（顶）：proxy 负载均衡 ──
box(70, 100, 860, 120, "#eef2ff", "#6366f1", 14, 2.5)
txt(90, 128, "③ proxy / router 负载均衡层", 16, "start", "bold", "#4338ca")
txt(90, 150, "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py", 12, "start", "normal", "#6366f1", italic=True)
box(110, 162, 360, 44, "#e0e7ff", "#818cf8", 8, 1.5)
txt(290, 182, "SharedProxyScheduler · 每角色一个最少负载堆", 12.5, "middle", "bold", "#3730a3")
txt(290, 199, "prefill=tokens+0.3·kv ｜ decode=tokens", 11.5, "middle", "normal", "#4338ca")
box(500, 162, 410, 44, "#e0e7ff", "#818cf8", 8, 1.5)
txt(705, 182, "assign_instances · 挑 prefiller→握手→挑 decoder", 12.5, "middle", "bold", "#3730a3")
txt(705, 199, "build_prefill_request 盖 P 角色章 (max_tokens=1)", 11.5, "middle", "normal", "#4338ca")

# ── 第一层（中）：连接器分发 ──
box(70, 270, 860, 150, "#ecfdf5", "#10b981", 14, 2.5)
txt(90, 298, "① 连接器分发层（vLLM v1 KVConnector 契约）", 16, "start", "bold", "#047857")
box(110, 314, 280, 90, "#d1fae5", "#34d399", 8, 1.5)
txt(250, 338, "AscendMultiConnector", 13.5, "middle", "bold", "#065f46")
lines(250, 360, ["子类化 MultiConnector + SupportsHMA", "首个命中子连接器赢 load·save 给全体", "layerwise 永远拿真 blocks"], 11, 17, "#047857")
# fan-out to three
for i, (nm1, nm2, note, hi) in enumerate([
    ("Mooncake", "Connector", "同契约", False),
    ("Mooncake", "HybridConnector", "同契约", False),
    ("Mooncake", "LayerwiseConnector", "本章讲透 ▼", True),
]):
    bx = 430 + i*168
    fill = "#fde68a" if hi else "#ecfccb"
    stroke = "#f59e0b" if hi else "#a3e635"
    box(bx, 324, 156, 70, fill, stroke, 8, 2 if hi else 1.5)
    txt(bx+78, 346, nm1, 11.5, "middle", "bold", "#713f12" if hi else "#3f6212")
    txt(bx+78, 362, nm2, 11.5, "middle", "bold", "#713f12" if hi else "#3f6212")
    txt(bx+78, 382, note, 11, "middle", "normal", "#92400e" if hi else "#65a30d")
    # arrow from AscendMultiConnector
    S.append(f'<path d="M390,359 L{bx},359" stroke="#34d399" stroke-width="1.6" fill="none" marker-end="url(#arr)"/>')

# ── 第二层（底）：mooncake P2P ──
box(70, 470, 860, 120, "#fef2f2", "#f87171", 14, 2.5)
txt(90, 498, "② mooncake P2P 传输层（prefill ↔ decode 跨节点直传 KV）", 16, "start", "bold", "#b91c1c")
box(110, 514, 360, 60, "#fee2e2", "#fca5a5", 8, 1.5)
txt(290, 537, "GlobalTE · 进程级单例 TransferEngine", 12.5, "middle", "bold", "#991b1b")
txt(290, 556, "backend 'ascend' · P2PHANDSHAKE", 11.5, "middle", "normal", "#b91c1c")
box(500, 514, 410, 60, "#fee2e2", "#fca5a5", 8, 1.5)
txt(705, 537, "get_transfer_meta · base_addr + block_id·block_len", 12, "middle", "bold", "#991b1b")
txt(705, 556, "group_concurrent_contiguous 连续块合批", 11.5, "middle", "normal", "#b91c1c")

# down arrows between layers
S.append(f'<path d="M500,220 L500,266" stroke="#475569" stroke-width="2.2" fill="none" marker-end="url(#arr)"/>')
txt(560, 248, "下发 prefill / decode 请求", 11.5, "start", "normal", "#475569")
S.append(f'<path d="M500,420 L500,466" stroke="#475569" stroke-width="2.2" fill="none" marker-end="url(#arr)"/>')
txt(560, 448, "save_kv_layer / start_load_kv 回调驱动传输", 11.5, "start", "normal", "#475569")

# ── 侧挂：KV 亲和（climax）──
box(975, 270, 250, 320, "#faf5ff", "#a855f7", 14, 2.5)
txt(1100, 298, "★ KV 亲和路由", 15, "middle", "bold", "#7e22ce")
txt(1100, 318, "（本章高潮）", 12, "middle", "normal", "#9333ea")
lines(1100, 350, [
    "KVPoolScheduler",
    ".get_num_new_matched_tokens",
    "",
    "client.lookup 先问：",
    "「这条 prompt 有多少",
    "token 已在外部 KV 命中？」",
    "",
    "need_to_allocate",
    "  = 命中 − 已算",
    "只拉缺口，KV 在哪",
    "就从哪加载",
], 11.5, 21, "#6b21a8")
txt(1100, 575, "跨节点字节 ∝ 缺口，非整段", 11, "middle", "italic", "#7e22ce")
# dashed link from connector layer to affinity
S.append(f'<path d="M930,345 L971,345" stroke="#a855f7" stroke-width="2" stroke-dasharray="6,4" fill="none" marker-end="url(#arrP)"/>')

# bottom note
txt(W/2, 640, "三层各自独立扩缩：prefill 实例算得快、decode 实例吐得稳，代价是把 prompt 的 KV 搬一次", 13, "middle", "normal", "#475569")
txt(W/2, 662, "亲和路由让「搬一次」尽量小——这是 PD 分离真正省钱的地方", 13, "middle", "bold", "#7e22ce")

S.append('</svg>')
open("overview.svg", "w", encoding="utf-8").write("\n".join(S))
print("wrote overview.svg")
