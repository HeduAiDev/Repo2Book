#!/usr/bin/env python3
"""EPLB 在线热迁移流水线总览：主进程 ①节拍 + ③P2P ↔ 两条队列 ↔ ②子进程(④策略) + shared_dict。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1200, 800
S = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
S.append('<defs>')
S.append('<marker id="arr" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#475569"/></marker>')
S.append('<marker id="arrG" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#15803d"/></marker>')
S.append('<marker id="arrB" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#1d4ed8"/></marker>')
S.append('</defs>')
S.append(f'<rect width="{W}" height="{H}" fill="white"/>')

S.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="24" font-weight="bold" fill="#0f172a">EPLB 在线热迁移流水线：重规划进子进程，靠两条队列与异步 P2P 与推理解耦</text>')


def box(x, y, w, h, fill, stroke, rx=10, sw=2):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x, y, t, size=14, anchor="middle", weight="normal", fill="#1e293b"):
    S.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(t)}</text>')


def lines(cx, y0, items, size=13, dy=20, fill="#334155"):
    for i, t in enumerate(items):
        txt(cx, y0 + i*dy, t, size=size, fill=fill)


# ── 推理主进程容器 ──
box(40, 70, 680, 320, "#eff6ff", "#3b82f6", 14, 2.5)
txt(60, 98, "推理主进程（vllm_ascend/worker/model_runner_v1.py）", 15, "start", "bold", "#1d4ed8")

# ① EplbUpdator
box(70, 118, 300, 250, "#dbeafe", "#2563eb")
txt(220, 144, "① EplbUpdator · 节拍状态机", 14, "middle", "bold", "#1e3a8a")
lines(220, 172, [
    "cur_iterations + 3 个 flag",
    "forward_before()：取规划 / 发 P2P",
    "forward_end()：gather+点火 / 收权重",
    "update_iteration()：推进 / 归零",
], size=12.5, dy=22)
txt(220, 290, "一整轮 = interval + algo", 12, "middle", "normal", "#64748b")
txt(220, 308, "+ num_moe_layers 拍", 12, "middle", "normal", "#64748b")
txt(220, 344, "eplb_updator.py:L77-L148", 11, "middle", "italic", "#94a3b8")

# ③ D2DExpertWeightLoader
box(400, 118, 300, 250, "#dbeafe", "#2563eb")
txt(550, 144, "③ D2DExpertWeightLoader", 14, "middle", "bold", "#1e3a8a")
lines(550, 172, [
    "异步 P2P · 三态机",
    "WAITING → READY → TRANSFERRING",
    "generate → asyn → update",
    "dist.P2POp(isend/irecv)",
], size=12.5, dy=22)
txt(550, 290, "搬的是 expert 权重张量", 12, "middle", "normal", "#64748b")
txt(550, 308, "每 step 只搬一层", 12, "middle", "normal", "#64748b")
txt(550, 344, "eplb_device_transfer_loader.py", 11, "middle", "italic", "#94a3b8")

# ── shared_dict ──
box(770, 110, 390, 168, "#fef9c3", "#ca8a04", 12, 2.5)
txt(965, 140, "shared_dict = Manager().dict()", 15, "middle", "bold", "#854d0e")
txt(965, 162, "跨进程共享态（主进程 ↔ 子进程）", 12, "middle", "normal", "#a16207")
for i, (k, v) in enumerate([
    ("expert_maps", "各层各卡当前 expert 放置"),
    ("moe_load", "all_gather 后的全局负载"),
]):
    yy = 188 + i*38
    box(795, yy, 340, 30, "#fffbeb", "#eab308", 6, 1.5)
    txt(810, yy+20, k, 13, "start", "bold", "#854d0e")
    txt(960, yy+20, v, 12, "start", "normal", "#a16207")

# ── EPLB 子进程容器 ──
box(40, 470, 720, 290, "#f0fdf4", "#16a34a", 14, 2.5)
txt(60, 498, "② EPLB 子进程（daemon=True Process）", 15, "start", "bold", "#15803d")

# worker_process loop
box(70, 518, 280, 220, "#dcfce7", "#22c55e")
txt(210, 544, "worker_process()", 14, "middle", "bold", "#166534")
lines(210, 572, [
    "while True:",
    "  planner_q.get()  ← 阻塞等唤醒",
    "  info = worker.do_update()",
    "  block_update_q.put(info)",
    "  （背压：队列非空则自旋）",
], size=12, dy=24, fill="#166534")
txt(210, 716, "eplb_worker.py:L342-L378", 11, "middle", "italic", "#86998a")

# EplbWorker.do_update
box(375, 518, 175, 220, "#dcfce7", "#22c55e")
txt(462, 544, "EplbWorker", 14, "middle", "bold", "#166534")
lines(462, 574, [
    "do_update()",
    "读 shared_dict",
    "→ 算新放置",
    "→ compose",
    "  send/recv",
    "→ pack",
], size=12, dy=22, fill="#166534")

# ④ policy
box(575, 518, 165, 220, "#dcfce7", "#16a34a", 10, 2.5)
txt(657, 544, "④ PolicyFactory", 14, "middle", "bold", "#166534")
lines(657, 574, [
    "int → 策略类",
    "rebalance_experts",
    "（唯一接口）",
    "DefaultEplb",
    "贪心装箱 + 冗余",
], size=12, dy=24, fill="#166534")

# ── NPU 间 D2D 搬运 ──
box(810, 470, 350, 290, "#faf5ff", "#9333ea", 14, 2.5)
txt(985, 498, "NPU 间 D2D 权重搬运", 15, "middle", "bold", "#7e22ce")
# two NPUs
box(845, 540, 110, 70, "#f3e8ff", "#a855f7", 10)
txt(900, 575, "NPU 0", 14, "middle", "bold", "#6b21a8")
txt(900, 596, "持有 expert", 11, "middle", "normal", "#7e22ce")
box(1015, 540, 110, 70, "#f3e8ff", "#a855f7", 10)
txt(1070, 575, "NPU 1", 14, "middle", "bold", "#6b21a8")
txt(1070, 596, "接收 expert", 11, "middle", "normal", "#7e22ce")
S.append(f'<line x1="955" y1="565" x2="1010" y2="565" stroke="#9333ea" stroke-width="2.5" marker-end="url(#arr)"/>')
txt(985, 558, "isend/irecv", 11, "middle", "normal", "#7e22ce")
lines(985, 645, [
    "batch_isend_irecv 批量搬运",
    "借 get_dynamic_eplb_group()",
    "= _DYNAMIC_EPLB（第 8 章建）",
    "拓扑复用 MC2 group_ranks",
], size=12, dy=23, fill="#6b21a8")

# ── 队列箭头：主进程 ↔ 子进程 ──
# planner_q：主 → 子（向下）
S.append(f'<line x1="170" y1="390" x2="170" y2="514" stroke="#15803d" stroke-width="3" marker-end="url(#arrG)"/>')
box(85, 415, 230, 50, "#ffffff", "#15803d", 8, 1.5)
txt(200, 435, "planner_q.put(1)", 12.5, "middle", "bold", "#15803d")
txt(200, 453, "唤醒信号（无界队列）", 11.5, "middle", "normal", "#15803d")

# block_update_q：子 → 主（向上）
S.append(f'<line x1="430" y1="514" x2="430" y2="372" stroke="#1d4ed8" stroke-width="3" marker-end="url(#arrB)"/>')
box(330, 415, 250, 50, "#ffffff", "#1d4ed8", 8, 1.5)
txt(455, 435, "block_update_q", 12.5, "middle", "bold", "#1d4ed8")
txt(455, 453, "规划结果（maxsize=1 背压）", 11.5, "middle", "normal", "#1d4ed8")

# ── shared_dict 连线 ──
# 主进程写 moe_load（虚线）
S.append(f'<path d="M 700 200 L 768 200" stroke="#ca8a04" stroke-width="2" stroke-dasharray="5,4" fill="none" marker-end="url(#arr)"/>')
txt(720, 192, "写 moe_load", 10.5, "middle", "normal", "#a16207")
# 子进程读（虚线）
S.append(f'<path d="M 600 518 Q 980 340 965 280" stroke="#ca8a04" stroke-width="2" stroke-dasharray="5,4" fill="none" marker-end="url(#arr)"/>')
txt(820, 400, "do_update 读 expert_maps/moe_load", 11, "middle", "normal", "#a16207")

# ── ③ → NPU 搬运连线 ──
S.append(f'<path d="M 700 230 Q 870 250 900 536" stroke="#9333ea" stroke-width="2" stroke-dasharray="5,4" fill="none" marker-end="url(#arr)"/>')

S.append('</svg>')
open("pipeline_overview.svg", "w").write("\n".join(S))
print("wrote pipeline_overview.svg")
