#!/usr/bin/env python3
"""fig-ch06-three-lines: 为什么通信器是 OOT 换底座最干净样本——三条线并排。
左道=基座 vLLM；中道=vllm_ascend 子类化（高亮唯一差异点 all_to_all）；
右道=ctypes 范式样本（虚线标注未接入）；底部横贯=310P 猴补层。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

W, H = 1180, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="ar" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#475569"/></marker>')
L.append('<marker id="arA" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#7c3aed"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

def box(x, y, w, h, lines, fill, stroke, tcol="#1e293b", fs=15, rx=8, dash=False, bold0=True):
    d = ' stroke-dasharray="7,5"' if dash else ''
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.8" rx="{rx}"{d}/>')
    n = len(lines)
    cy = y + h/2 - (n-1)*(fs+5)/2 + fs/2
    for i, t in enumerate(lines):
        fw = "bold" if (i == 0 and bold0) else "normal"
        L.append(f'<text x="{x+w/2}" y="{cy + i*(fs+5)}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="{tcol}">{esc(t)}</text>')

# 三道标题
lane_y = 64
L.append(f'<text x="200" y="40" text-anchor="middle" font-size="18" font-weight="bold" fill="#0f172a">① 基座 vLLM</text>')
L.append(f'<text x="590" y="40" text-anchor="middle" font-size="18" font-weight="bold" fill="#0f172a">② vllm_ascend 子类化</text>')
L.append(f'<text x="980" y="40" text-anchor="middle" font-size="18" font-weight="bold" fill="#0f172a">③ ctypes 范式样本</text>')

# ---- 左道：基座 ----
# Platform 默认回调
box(70, lane_y, 260, 64,
    ["Platform.get_device_communicator_cls()", "→ \"...DeviceCommunicatorBase\"（字符串）"],
    "#f8fafc", "#94a3b8", fs=13)
# GroupCoordinator resolve
box(70, lane_y+110, 260, 70,
    ["GroupCoordinator.__init__", "resolve_obj_by_qualname(字符串)", "→ 类 → 实例化"],
    "#f1f5f9", "#94a3b8", fs=13)
# 基类集合通信一列灰框
base_x, base_y = 70, lane_y+248
box(base_x, base_y, 260, 32, ["DeviceCommunicatorBase（基类）"], "#e2e8f0", "#64748b", fs=14)
methods = ["all_reduce", "all_gather", "reduce_scatter", "gather / send / recv", "broadcast"]
my = base_y + 40
for m in methods:
    box(base_x+20, my, 220, 30, [f"{m} → dist.{m if m!='gather / send / recv' else 'xxx'}(group=…)"], "#f1f5f9", "#cbd5e1", tcol="#64748b", fs=12, rx=5, bold0=False)
    my += 36
L.append(f'<text x="{base_x+130}" y="{my+18}" text-anchor="middle" font-size="12.5" fill="#64748b">底层 backend 由进程组决定（NPU→HCCL）</text>')

# ---- 中道：NPUCommunicator ----
mid_x = 460
box(mid_x, lane_y, 260, 64,
    ["NPUPlatform.get_device_communicator_cls()", "→ \"...NPUCommunicator\"（覆写）"],
    "#ede9fe", "#7c3aed", tcol="#5b21b6", fs=13)
# 字符串箭头 覆写指向 NPUCommunicator
npu_x, npu_y = 460, lane_y+248
box(npu_x, npu_y, 260, 32, ["NPUCommunicator（子类）"], "#ddd6fe", "#7c3aed", tcol="#5b21b6", fs=14)
# 继承的方法（灰显，指回基类）
box(npu_x+20, npu_y+40, 220, 56,
    ["继承基类全部集合通信", "（all_reduce … broadcast）零修改"],
    "#f1f5f9", "#cbd5e1", tcol="#64748b", fs=12, rx=5, bold0=False)
# __init__ 微调
box(npu_x+20, npu_y+104, 220, 50,
    ["__init__：device=npu", "ca_comm=None（占位）"],
    "#faf5ff", "#a78bfa", tcol="#6d28d9", fs=12, rx=5, bold0=False)
# 唯一差异点 all_to_all 高亮
box(npu_x+20, npu_y+162, 220, 50,
    ["★ 新增 all_to_all", "split→dist.all_to_all→cat"],
    "#fef3c7", "#f59e0b", tcol="#92400e", fs=13, rx=5)

# 箭头：覆写回调 → NPUCommunicator
L.append(f'<line x1="{mid_x+130}" y1="{lane_y+64}" x2="{mid_x+130}" y2="{npu_y-6}" stroke="#7c3aed" stroke-width="2" marker-end="url(#arA)"/>')
L.append(f'<text x="{mid_x+138}" y="{lane_y+150}" font-size="12" fill="#7c3aed">一个字符串</text>')
L.append(f'<text x="{mid_x+138}" y="{lane_y+168}" font-size="12" fill="#7c3aed">换底座</text>')
# 箭头：继承 → 基类（横向虚线）
L.append(f'<line x1="{npu_x}" y1="{npu_y+68}" x2="{base_x+260}" y2="{base_y+90}" stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="6,4" marker-end="url(#ar)"/>')
L.append(f'<text x="{(npu_x+base_x+260)/2}" y="{(npu_y+68+base_y+90)/2-8}" text-anchor="middle" font-size="11.5" fill="#64748b">继承 = 复用</text>')
# 箭头：resolve → 实例化 NPUCommunicator
L.append(f'<line x1="{base_x+130}" y1="{lane_y+180}" x2="{npu_x+130}" y2="{npu_y-6}" stroke="#475569" stroke-width="1.6" marker-end="url(#ar)"/>')

# ---- 右道：ctypes 范式 ----
rx0 = 850
box(rx0, lane_y, 260, 56, ["PyHcclCommunicator", "unique_id 建组 / warmup / disabled"],
    "#f8fafc", "#94a3b8", tcol="#475569", fs=13, dash=True)
box(rx0, lane_y+96, 260, 56, ["HCCLLibrary（ctypes 绑定）", "exported_functions C 签名表"],
    "#f8fafc", "#94a3b8", tcol="#475569", fs=13, dash=True)
box(rx0, lane_y+192, 260, 44, ["libhccl.so（CDLL 加载）"],
    "#f1f5f9", "#94a3b8", tcol="#475569", fs=13, dash=True)
L.append(f'<line x1="{rx0+130}" y1="{lane_y+56}" x2="{rx0+130}" y2="{lane_y+94}" stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="6,4" marker-end="url(#ar)"/>')
L.append(f'<line x1="{rx0+130}" y1="{lane_y+152}" x2="{rx0+130}" y2="{lane_y+190}" stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="6,4" marker-end="url(#ar)"/>')
# 未接入标签
box(rx0+30, lane_y+260, 200, 56,
    ["TODO 未接入 NPUCommunicator", "对位 pynccl 的移植样本"],
    "#fff7ed", "#fb923c", tcol="#9a3412", fs=12.5, dash=True, rx=10)

# ---- 底部横贯：310P 猴补 ----
patch_y = H - 92
L.append(f'<rect x="60" y="{patch_y}" width="{W-120}" height="64" fill="#fef2f2" stroke="#ef4444" stroke-width="1.8" stroke-dasharray="8,5" rx="10"/>')
L.append(f'<text x="{W/2}" y="{patch_y+26}" text-anchor="middle" font-size="15" font-weight="bold" fill="#991b1b">④ 仅 310P：patch_distributed 拦截 torch.distributed.broadcast / all_reduce(int64)</text>')
L.append(f'<text x="{W/2}" y="{patch_y+48}" text-anchor="middle" font-size="13" fill="#b91c1c">用 all_gather 模拟补硬件能力缺口（复用 ch03 两段式猴补；A2/A3/A5 不触发）</text>')

L.append('</svg>')
open("three_lines.svg", "w").write('\n'.join(L))
print("wrote three_lines.svg", W, H)
