#!/usr/bin/env python3
"""同一个 MooncakeLayerwiseConnector 类，按 KVConnectorRole 只活半边：Scheduler vs Worker。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1180, 680
S = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
S.append('<defs>')
S.append('<marker id="arr" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#475569"/></marker>')
S.append('<marker id="arrD" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#0ea5e9"/></marker>')
S.append('</defs>')
S.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10, sw=2):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x, y, t, size=14, anchor="middle", weight="normal", fill="#1e293b", italic=False, mono=False):
    st = ' font-style="italic"' if italic else ''
    fam = ' font-family="monospace"' if mono else ''
    S.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}"{st}{fam}>{esc(t)}</text>')


txt(W/2, 40, "一个类，两半身：MooncakeLayerwiseConnector 按 role 只活半边", 21, "middle", "bold", "#0f172a")
txt(W/2, 68, "同一份代码在调度进程与 worker 进程各实例化一次，role 决定哪半是活的", 14, "middle", "normal", "#64748b")

# facade
box(420, 92, 340, 56, "#f1f5f9", "#64748b", 10, 2)
txt(590, 116, "MooncakeLayerwiseConnector（facade）", 14, "middle", "bold", "#334155")
txt(590, 136, "每个 base-class 钩子按 role 转发到下面半边", 11.5, "middle", "italic", "#64748b")

# ── 左：SCHEDULER ──
box(70, 190, 480, 360, "#eff6ff", "#3b82f6", 12, 2.5)
txt(310, 220, "role == SCHEDULER（调度进程）", 16, "middle", "bold", "#1d4ed8")
txt(310, 242, "决定收/发什么、建元数据、发握手", 12, "middle", "normal", "#2563eb")
hooks_s = [
    ("get_num_new_matched_tokens", "do_remote_prefill → 拉整段 prompt（异步）"),
    ("update_state_after_alloc", "按 do_remote_prefill / do_remote_decode 分入 recv / send 队列 + POST 握手"),
    ("build_connector_meta", "把待 recv/send 化成 ReqMeta，逐步下发"),
    ("request_finished_all_groups", "HMA 逐组聚合，回收"),
]
y = 264
for name, desc in hooks_s:
    box(92, y, 436, 56, "#dbeafe", "#60a5fa", 8, 1.4)
    txt(112, y + 22, name, 13, "start", "bold", "#1e40af", mono=True)
    txt(112, y + 42, desc, 11, "start", "normal", "#1d4ed8")
    y += 66

# ── 右：WORKER ──
box(630, 190, 480, 360, "#fef2f2", "#ef4444", 12, 2.5)
txt(870, 220, "role == WORKER（worker 进程）", 16, "middle", "bold", "#b91c1c")
txt(870, 242, "碰真实 KV 张量、跑收发后台线程", 12, "middle", "normal", "#dc2626")
hooks_w = [
    ("start_load_kv", "forward 前调：consumer 登记收 / producer 解析块映射"),
    ("save_kv_layer", "每算完一层立刻入发送队列 = 逐层 push（本连接器的招牌）"),
    ("get_finished", "把后台线程已完成的传输回收上报，好释放块"),
    ("register_kv_caches", "把 KV 张量内存登记给 mooncake 引擎"),
]
y = 264
for name, desc in hooks_w:
    box(652, y, 436, 56, "#fee2e2", "#f87171", 8, 1.4)
    txt(672, y + 22, name, 13, "start", "bold", "#991b1b", mono=True)
    txt(672, y + 42, desc, 11, "start", "normal", "#b91c1c")
    y += 66

# facade -> two halves
S.append(f'<path d="M500,148 L350,186" stroke="#64748b" stroke-width="1.8" fill="none" marker-end="url(#arr)"/>')
S.append(f'<path d="M680,148 L830,186" stroke="#64748b" stroke-width="1.8" fill="none" marker-end="url(#arr)"/>')

# metadata dashed arrow scheduler -> worker, placed below both columns
txt(W/2, 588, "MooncakeLayerwiseConnectorMetadata：每步 scheduler 装好 → worker 取用", 12.5, "middle", "bold", "#0369a1")
S.append(f'<path d="M310,604 L870,604" stroke="#0ea5e9" stroke-width="2.4" stroke-dasharray="7,5" fill="none" marker-end="url(#arrD)"/>')
txt(310, 624, "SCHEDULER", 11, "middle", "bold", "#1d4ed8")
txt(870, 624, "WORKER", 11, "middle", "bold", "#b91c1c")

txt(W/2, 658, "这就是 vLLM v1 连接器的标准切分：scheduler 出主意、worker 干体力活，元数据当信使", 13, "middle", "bold", "#475569")

S.append('</svg>')
open("split.svg", "w", encoding="utf-8").write("\n".join(S))
print("wrote split.svg")
