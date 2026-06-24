#!/usr/bin/env python3
"""请求端到端旅程图（ch01 导读）。

中间一条三段式主链 InputProcessor -> EngineCore -> OutputProcessor；
上方分叉两个使用面入口（离线 LLM / 服务 AsyncLLM），各自的驱动方式；
EngineCore 内画 step 三步 schedule/execute/update + Scheduler + 分页 KV cache。
仅概念级，节点名用真实类名。所有坐标由 Python 计算。
"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1040, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
         'markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
         '<marker id="ap" viewBox="0 0 10 6" refX="9" refY="3" '
         'markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
         '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W//2}" y="40" text-anchor="middle" font-size="22" '
         f'font-weight="bold" fill="#0f172a">一个请求的端到端旅程</text>')


def box(x, y, w, h, title, sub, fill, stroke, tcol="#0f172a", scol="#475569",
        rx=12, tw="bold", fs=16, ss=11.5):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>')
    if sub:
        L.append(f'<text x="{x+w//2}" y="{y+h//2-6}" text-anchor="middle" '
                 f'font-size="{fs}" font-weight="{tw}" fill="{tcol}">{esc(title)}</text>')
        L.append(f'<text x="{x+w//2}" y="{y+h//2+15}" text-anchor="middle" '
                 f'font-size="{ss}" fill="{scol}">{esc(sub)}</text>')
    else:
        L.append(f'<text x="{x+w//2}" y="{y+h//2+5}" text-anchor="middle" '
                 f'font-size="{fs}" font-weight="{tw}" fill="{tcol}">{esc(title)}</text>')


def arr(x1, y1, x2, y2, color="#475569", marker="a", dash=None, sw=2.2):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
             f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def label(x, y, t, color="#64748b", fs=11, anchor="middle", weight="normal"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" '
             f'font-weight="{weight}" fill="{color}">{esc(t)}</text>')


# --- 两个使用面入口（顶部分叉） ---
ent_y, ent_h, ent_w = 66, 60, 360
ofs_x = 70                    # 离线 LLM
srv_x = W - 70 - ent_w       # 服务 AsyncLLM
box(ofs_x, ent_y, ent_w, ent_h, "离线 LLM", "调用方线程 while has_unfinished: step()",
    "#ecfdf5", "#10b981", scol="#047857")
box(srv_x, ent_y, ent_w, ent_h, "服务 AsyncLLM / OpenAI server",
    "独立子进程 run_busy_loop + 背景 output_handler",
    "#eff6ff", "#3b82f6", scol="#1d4ed8")

# --- 三段式主链（中列纵向） ---
cx = W // 2
chain_x = cx - 150
chain_w = 300
ip_y, core_y, op_y = 168, 300, 580
seg_h = 56

box(chain_x, ip_y, chain_w, seg_h, "InputProcessor", "tokenize → EngineCoreRequest",
    "#f1f5f9", "#94a3b8")
# EngineCore 外框（内核）
core_h = 232
L.append(f'<rect x="{chain_x-40}" y="{core_y}" width="{chain_w+80}" height="{core_h}" '
         f'rx="16" fill="#fff7ed" stroke="#f59e0b" stroke-width="3"/>')
label(cx, core_y + 24, "EngineCore（内核 · GPU 活）", "#b45309", 15, weight="bold")
box(chain_x, op_y, chain_w, seg_h, "OutputProcessor", "detokenize → RequestOutput",
    "#f1f5f9", "#94a3b8")

# 入口 -> InputProcessor（两条汇入）
arr(ofs_x + ent_w // 2, ent_y + ent_h, chain_x + 70, ip_y, "#10b981")
arr(srv_x + ent_w // 2, ent_y + ent_h, chain_x + chain_w - 70, ip_y, "#3b82f6")

# 主链纵向箭头 + 数据类型标签
arr(cx, ip_y + seg_h, cx, core_y, "#475569")
label(cx + 92, (ip_y + seg_h + core_y) // 2 + 4, "EngineCoreRequest", "#475569", 11)

# --- EngineCore 内：step 三步 + Scheduler + 分页 KV cache ---
step_y = core_y + 44
step_h = 50
sw_ = 80
gap = 18
total = sw_ * 3 + gap * 2
sx0 = cx - total // 2
steps = [("schedule", "排哪些 token"), ("execute_model", "前向 + 采样"),
         ("update", "出 token / 回收")]
prev_r = None
for i, (t, s) in enumerate(steps):
    x = sx0 + i * (sw_ + gap)
    box(x, step_y, sw_, step_h, t, s, "#fef3c7", "#d97706", scol="#92400e",
        fs=12.5, ss=9.5, rx=9)
    if prev_r is not None:
        arr(prev_r, step_y + step_h // 2, x, step_y + step_h // 2, "#d97706")
    prev_r = x + sw_
# update -> schedule 回环（跨拍持久）
loop_y = step_y + step_h + 16
arr(sx0 + 2 * (sw_ + gap) + sw_ // 2, step_y + step_h,
    sx0 + sw_ // 2, step_y + step_h, "#d97706", dash="5 4")
L.append(f'<path d="M{sx0+2*(sw_+gap)+sw_//2},{step_y+step_h} '
         f'V{loop_y} H{sx0+sw_//2} V{step_y+step_h}" '
         f'fill="none" stroke="#d97706" stroke-width="2" stroke-dasharray="5 4" '
         f'marker-end="url(#a)"/>')
label(cx, loop_y + 14, "逐拍循环（连续批处理 · 批次跨拍持久）", "#92400e", 10.5)

# Scheduler + 分页 KV cache 两个被 EngineCore 管的子系统
sub_y = step_y + step_h + 36
sub_w = 132
sub_h = 40
sch_x = cx - sub_w - 14
kv_x = cx + 14
box(sch_x, sub_y, sub_w, sub_h, "Scheduler", "token 预算 / 抢占",
    "#ffffff", "#f59e0b", scol="#92400e", fs=12.5, ss=9.5, rx=9)
box(kv_x, sub_y, sub_w, sub_h, "分页 KV cache", "块池 / 前缀复用",
    "#ffffff", "#f59e0b", scol="#92400e", fs=12.5, ss=9.5, rx=9)

# EngineCore -> OutputProcessor
arr(cx, core_y + core_h, cx, op_y, "#475569")
label(cx + 90, (core_y + core_h + op_y) // 2 + 4, "EngineCoreOutputs", "#475569", 11)

# OutputProcessor -> 回到调用方
ret_y = op_y + seg_h + 28
arr(cx, op_y + seg_h, cx, ret_y - 4, "#475569")
label(cx, ret_y + 12, "RequestOutput → 回到你手里", "#0f172a", 13, weight="bold")

# CPU/GPU 分层旁注
label(chain_x - 70, ip_y + seg_h // 2 + 4, "CPU", "#94a3b8", 12, weight="bold")
label(chain_x - 70, op_y + seg_h // 2 + 4, "CPU", "#94a3b8", 12, weight="bold")

L.append('</svg>')
svg = '\n'.join(L)
out = __file__.rsplit('/', 1)[0] + '/request-journey.svg'
with open(out, 'w') as f:
    f.write(svg)
print("wrote", out)
