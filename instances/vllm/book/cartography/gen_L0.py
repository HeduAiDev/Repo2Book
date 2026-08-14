#!/usr/bin/env python3
"""L0 全书唯一权威架构图 —— v3 图系的根。

一张图 = 一个请求的一生：API 进程 → ZMQ 边界 → EngineCore 进程。
所有 L1(Part)/L2(章) 图必须是本图某一块的**放大**（同坐标/配色/术语）。
改本图须同步 ARCHITECTURE.md §1。
"""
import sys
from pathlib import Path
import xml.sax.saxutils as xs

FONT = 'Microsoft YaHei, SimHei, Noto Sans CJK SC, PingFang SC, sans-serif'
C_TXT, C_MUTE = '#0f172a', '#475569'
C_API_S, C_API_F = '#2563eb', '#eff6ff'      # API 进程 = 蓝
C_ENG_S, C_ENG_F = '#ea580c', '#fff7ed'      # EngineCore 进程 = 橙
C_ZMQ_S, C_ZMQ_F = '#7c3aed', '#f5f3ff'      # ZMQ 边界 = 紫
C_GPU_S, C_GPU_F = '#16a34a', '#f0fdf4'      # GPU 执行臂 = 绿
C_KV_S, C_KV_F = '#0891b2', '#ecfeff'        # 显存 = 青
W = 1180
L = []
def esc(s): return xs.escape(str(s))
def tw(s, fs, bold=False):
    n = sum(1 for c in str(s) if ord(c) > 0x2E80)
    return (n * fs + (len(str(s)) - n) * fs * 0.58) * (1.07 if bold else 1.0)
def text(x, y, s, fs=12, fill=C_TXT, anchor='middle', bold=False):
    b = ' font-weight="bold"' if bold else ''
    L.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}"{b}>{esc(s)}</text>')
def fit(cx, y, s, box_w, base, fill, bold=False, minfs=7.5, anchor='middle'):
    fs = base
    while fs > minfs and tw(s, fs, bold) > box_w:
        fs -= 0.3
    if tw(s, fs, bold) > box_w:
        keep = max(3, int(len(s) * box_w / max(1e-6, tw(s, fs, bold))) - 1)
        s = s[:keep] + '…'
    text(cx, y, s, round(fs, 1), fill, anchor=anchor, bold=bold)
def box(x, y, w, h, fill, stroke, r=8, sw=1.6, dash=False):
    d = ' stroke-dasharray="5,4"' if dash else ''
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def arrow(x1, y1, x2, y2, color='#64748b', sw=1.8, marker='a'):
    L.append(f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" stroke="{color}" stroke-width="{sw}" marker-end="url(#{marker})"/>')
def subbox(x, y, w, h, title, sub, fs=10.5, subfs=8.6, fill='#ffffff', stroke='#94a3b8', badge=''):
    box(x, y, w, h, fill, stroke, r=6, sw=1.2)
    fit(x + 8, y + h / 2 + (3.5 if sub else 0), title, w - 16 - (tw(badge, 8, True) + 10 if badge else 0), fs, C_TXT, bold=True, anchor='start')
    if sub:
        fit(x + 8, y + h - 7, sub, w - 16, subfs, C_MUTE, anchor='start')
    if badge:
        text(x + w - 7, y + 15, badge, 8, stroke, anchor='end', bold=True)

# ---------- 画布 ----------
L.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 1010">')
L.append(f'<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
         f'<marker id="dn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>'
         f'<marker id="up" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#ea580c"/></marker></defs>')
L.append(f'<rect width="{W}" height="1010" fill="white"/>')

# 标题
text(24, 34, 'vLLM v1 全书架构图（L0·唯一权威图）：一个请求的一生', 17, C_TXT, anchor='start', bold=True)
text(24, 56, '蓝框 = API 进程（零 GPU）｜紫框 = ZMQ 边界｜橙框 = EngineCore 进程｜绿框 = GPU 执行臂｜青框 = 显存账本', 10.5, C_MUTE, anchor='start')

MX, MW = 24, W - 48          # 主框
y = 78

# ---------- API 进程 ----------
box(MX, y, MW, 208, C_API_F, C_API_S, sw=2.2)
text(MX + 14, y + 24, 'API 进程（frontend · 零 GPU）', 13, C_API_S, anchor='start', bold=True)
text(MX + MW - 14, y + 24, 'HTTP/JSON/SSE 的 CPU 活全在这里', 9.5, C_MUTE, anchor='end')
# 左列：下行
subbox(MX + 14, y + 40, 330, 34, 'OpenAI server / LLM / AsyncLLM', '两个使用面，同一套三件套', fs=10)
subbox(MX + 14, y + 82, 330, 34, 'Renderer.render', 'tokenize 在这里！文本不过 IPC', fs=10)
subbox(MX + 14, y + 124, 330, 34, 'InputProcessor', '校验 + request_id 双轨 + 组装 EngineCoreRequest', fs=10)
subbox(MX + 14, y + 166, 330, 30, 'AsyncMPClient（下行发送端）', '', fs=10)
# 右列：上行
subbox(MX + 356, y + 40, 330, 34, 'RequestOutputCollector', '单槽 + Event + 生产侧合并（刻意不用 Queue）', fs=10)
subbox(MX + 356, y + 82, 330, 34, 'OutputProcessor', 'detokenize（增量）+ logprobs + 组装 RequestOutput', fs=10)
subbox(MX + 356, y + 124, 330, 34, 'output_handler（单任务分发）', '按 chunk 切片 + sleep(0) 让事件循环喘气', fs=10)
subbox(MX + 356, y + 166, 330, 30, 'PULL socket（上行接收端）', '', fs=10)
# 用户
subbox(MX + 712, y + 82, 330, 60, 'user（HTTP 客户端）', '断连 = 反向 abort（把幽灵请求从引擎抠掉）', fs=11)
arrow(MX + 877, y + 82, MX + 877, y + 74, '#64748b', 1.4)
arrow(MX + 696, y + 99, MX + 712, y + 99, '#2563eb', 2, 'dn')
text(MX + 704, y + 94, '进', 9, C_API_S)
arrow(MX + 712, y + 124, MX + 696, y + 124, '#ea580c', 2, 'up')
text(MX + 704, y + 137, '出', 9, C_ENG_S)
# 列间箭头（下行链/上行链）
for yy in (y + 57, y + 99):
    arrow(MX + 179, yy + 8, MX + 179, yy + 16, '#2563eb', 1.5, 'dn')
for yy in (y + 99, y + 141):
    arrow(MX + 521, yy + 8, MX + 521, yy + 16, '#ea580c', 1.5, 'up')
arrow(MX + 521, y + 57, MX + 521, y + 82, '#ea580c', 1.5, 'up')
y += 208

# ---------- ZMQ 边界 ----------
gap = 66
zy = y + 6
box(MX, zy, MW, gap - 12, C_ZMQ_F, C_ZMQ_S, sw=1.8)
text(MX + 14, zy + 22, 'ZMQ 边界（进程间）', 12, C_ZMQ_S, anchor='start', bold=True)
text(MX + MW / 2, zy + 22, '下行: ROUTER(bind) → DEALER(connect, identity=rank)', 10, C_TXT)
text(MX + MW / 2, zy + 40, '上行: engine PUSH(connect) → client PULL(bind) · HWM=0 无反压', 10, C_TXT)
text(MX + MW - 14, zy + 22, 'msgpack + 零拷贝帧 + OOB 旁路', 9, C_MUTE, anchor='end')
# 两侧穿边界的消息箭头
cx_dn = MX + 179
cx_up = MX + 521
arrow(cx_dn, y + 2, cx_dn, zy + gap - 4, '#2563eb', 2.4, 'dn')
text(cx_dn + 6, y + gap / 2 + 2, 'EngineCoreRequest（只有 token ids）', 9, '#2563eb', anchor='start')
arrow(cx_up, zy + gap - 4, cx_up, y + 2, '#ea580c', 2.4, 'up')
text(cx_up - 6, y + gap / 2 + 14, 'EngineCoreOutputs（每步整批聚合）', 9, '#ea580c', anchor='end')
y += gap

# ---------- EngineCore 进程 ----------
box(MX, y, MW, 560, C_ENG_F, C_ENG_S, sw=2.2)
text(MX + 14, y + 24, 'EngineCore 进程（busy loop · 只做调度+执行）', 13, C_ENG_S, anchor='start', bold=True)
text(MX + MW - 14, y + 24, 'GPU 上下文只存在于此进程及 worker', 9.5, C_MUTE, anchor='end')

# 循环五段（横向流程）
cy = y + 38
box(MX + 14, cy, MW - 28, 74, '#ffffff', C_ENG_S, r=6, sw=1.4)
text(MX + 26, cy + 18, 'EngineCore.step() 逐拍循环', 11, C_ENG_S, anchor='start', bold=True)
steps = ['① schedule()', '② execute_model\n(non_block→Future)', '③ get_grammar_bitmask\n(CPU 藏进 GPU 窗口)', '④ sample_tokens\n(logits+bitmask)', '⑤ update_from_output\n(状态推进+free 块)']
sw_ = (MW - 28 - 40 - 4 * 26) / 5
for i, s in enumerate(steps):
    sx = MX + 14 + 20 + i * (sw_ + 26)
    box(sx, cy + 28, sw_, 36, '#fff7ed', '#fdba74', r=5, sw=1.1)
    lines = s.split('\n')
    text(sx + sw_ / 2, cy + 40, lines[0], 9.2, C_TXT, bold=True)
    if len(lines) > 1:
        text(sx + sw_ / 2, cy + 54, lines[1], 7.6, C_MUTE)
    if i < 4:
        arrow(sx + sw_, cy + 46, sx + sw_ + 26, cy + 46, C_ENG_S, 1.6)

# 调度器 + KV 账本（左半）
sy = cy + 84
subbox(MX + 14, sy, 380, 64, 'Scheduler', '只认 token 数、无 prefill/decode 相位 · RUNNING 先于 WAITING', fs=11, subfs=8.8)
subbox(MX + 14, sy + 74, 380, 64, 'KVCacheManager / BlockPool / 前缀缓存', '固定块池 + 链式哈希（非 radix 树）· 分配失败→抢占 recompute-only', fs=10.5, subfs=8.8, fill=C_KV_F, stroke=C_KV_S)
arrow(MX + 204, sy + 64, MX + 204, sy + 74, '#64748b', 1.6)
text(MX + 212, sy + 72, '每拍对账', 8.5, C_MUTE, anchor='start')
arrow(MX + 204, sy + 74, MX + 204, sy + 64, '#64748b', 1.6)
# SchedulerOutput 差量协议箭头（下行到执行臂）
arrow(MX + 404, sy + 32, MX + 428, sy + 32, C_ENG_S, 2.2)
text(MX + 416, sy + 24, 'SchedulerOutput', 8.2, C_ENG_S, anchor='middle', bold=True)
text(MX + 416, sy + 44, '差量协议', 8, C_MUTE)

# GPU 执行臂（右半）
gx, gy, gw, gh = MX + 428, sy, MW - 28 - 428 + 14, 206
box(gx, gy, gw, gh, C_GPU_F, C_GPU_S, r=8, sw=2)
text(gx + 12, gy + 18, 'GPU 执行臂', 11.5, C_GPU_S, anchor='start', bold=True)
subbox(gx + 10, gy + 26, (gw - 30) / 2, 44, 'Executor → Worker', '进程拓扑 / 设备生命周期', fs=9.5, subfs=8)
subbox(gx + 20 + (gw - 30) / 2, gy + 26, (gw - 30) / 2, 44, 'GPUModelRunner', 'InputBatch 持久批次 + CpuGpuBuffer 固定地址', fs=9.5, subfs=8)
subbox(gx + 10, gy + 76, (gw - 30) / 2, 44, 'piecewise compile + CUDA Graph', '注意力处切图 · 按形状查表回放', fs=9.5, subfs=8)
subbox(gx + 20 + (gw - 30) / 2, gy + 76, (gw - 30) / 2, 44, '模型层', 'DecoderLayer 拼装 · Attention=插座 · MLA/GQA 变体', fs=9.5, subfs=8)
subbox(gx + 10, gy + 126, gw - 20, 34, 'compute_logits → Sampler 9 步管线 → 结构化输出位掩码 / spec decode', 'logits 只在需要的位置物化 · GPU 全程不落 CPU', fs=9.5, subfs=8)
arrow(gx + gw / 2, gy + 126, gx + gw / 2, gy + 126, C_GPU_S)
# KV↔执行臂（block_table）
arrow(MX + 404, sy + 106, gx - 2, gy + 60, C_KV_S, 1.6)
text(MX + 420, sy + 100, 'block_table / slot_mapping', 8.2, C_KV_S, anchor='start')

# 回程箭头（执行臂 → 循环）
arrow(gx + gw / 2, gy - 2, gx + gw / 2, cy + 100 + 2, C_ENG_S, 1.6, 'a')
# 循环→scheduler
arrow(MX + 204, cy + 102, MX + 204, sy - 2, C_ENG_S, 1.6)

y += 560

# ---------- 图例 ----------
ly = y + 26
text(24, ly, '读图：请求自左上进入，沿蓝箭头下行穿 ZMQ 进引擎；每拍循环调度→执行→采样→回收；新 token 沿橙箭头上行回 API 进程拼字流出。', 10, C_MUTE, anchor='start')
text(24, ly + 18, '第一设计原则：GPU 是最贵的员工，一切 CPU 活都不能让它等。第二原则：显存是共享账本，一切调度先对账。', 10, C_MUTE, anchor='start')

H = ly + 34
L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">'
L[1] = f'<rect width="{W}" height="{H}" fill="white"/>' if L[1].startswith('<rect width="1180" height="1010"') else L[1]
L.append('</svg>')

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'L0-architecture.svg'
out.write_text('\n'.join(L), encoding='utf-8')
print(f'OK {out} (viewBox 0 0 {W} {H})')
