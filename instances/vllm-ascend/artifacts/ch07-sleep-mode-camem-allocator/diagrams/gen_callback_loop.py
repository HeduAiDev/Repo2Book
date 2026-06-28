#!/usr/bin/env python3
"""pluggable allocator 回调闭环：分配走绿线记账，GC 走红线弹账，init_module 登记回调。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

W, H = 1080, 600
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="#15803d"/></marker>')
L.append('<marker id="r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="#b91c1c"/></marker>')
L.append('<marker id="b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="#7c3aed"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">pluggable allocator 的回调闭环</text>')
L.append(f'<text x="{W/2}" y="60" text-anchor="middle" font-size="13" fill="#64748b">绿线＝分配记账　红线＝GC 弹账　紫线＝init_module 一次性登记回调</text>')

def box(x, y, w, h, lines, fill, stroke, fs=13, bold0=True):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    n = len(lines)
    cy = y + h/2 - (n-1)*9 + 4
    for i, ln in enumerate(lines):
        fw = "bold" if (i == 0 and bold0) else "normal"
        col = "#0f172a" if (i == 0 and bold0) else "#334155"
        L.append(f'<text x="{x+w/2}" y="{cy+i*18}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="{col}">{esc(ln)}</text>')

# Boxes
# torch region (top-left)
box(40, 90, 250, 70, ["torch（use_mem_pool 区间）","权重加载 / KV 分配"], "#e0f2fe", "#0284c7")
# NPUPluggableAllocator (top-mid)
box(415, 90, 250, 70, ["NPUPluggableAllocator","接管 malloc / free"], "#ede9fe", "#7c3aed")
# C extension (top-right)
box(790, 90, 250, 70, ["vllm_ascend_C（C 扩展）","reserve + map / unmap 物理页"], "#fef3c7", "#d97706", fs=12)
# physical page (mid-right)
box(820, 230, 190, 60, ["物理页 + handle","(dev,size,VA,phys)"], "#fff7ed", "#ea580c", fs=12)
# malloc callback (mid)
box(400, 250, 280, 64, ["python_malloc_callback","记 handle + 贴 current_tag"], "#dcfce7", "#16a34a", fs=12.5)
# ledger (bottom-mid)
box(370, 410, 340, 90, ["pointer_to_data 账本","{ 设备VA : AllocationData(handle, tag,","cpu_backup_tensor) }"], "#f1f5f9", "#475569", fs=12.5)
# free callback (bottom-right)
box(790, 410, 250, 64, ["python_free_callback","弹账本 · 交还 handle"], "#fee2e2", "#dc2626", fs=12.5)
# GC trigger (bottom-left)
box(40, 410, 250, 64, ["张量被 GC","torch 调 C 扩展 free"], "#fef2f2", "#ef4444", fs=12.5)

# ---- ARROWS ----
# green path: torch -> allocator -> C ext -> physical page -> malloc cb -> ledger
L.append(f'<line x1="290" y1="125" x2="415" y2="125" stroke="#15803d" stroke-width="2" marker-end="url(#g)"/>')
L.append(f'<text x="352" y="118" text-anchor="middle" font-size="11" fill="#15803d">分配</text>')
L.append(f'<line x1="665" y1="125" x2="790" y2="125" stroke="#15803d" stroke-width="2" marker-end="url(#g)"/>')
# C ext -> physical page
L.append(f'<line x1="915" y1="160" x2="915" y2="230" stroke="#15803d" stroke-width="2" marker-end="url(#g)"/>')
# physical page -> malloc callback (回调)
L.append(f'<line x1="820" y1="262" x2="680" y2="276" stroke="#15803d" stroke-width="2" marker-end="url(#g)"/>')
L.append(f'<text x="752" y="258" text-anchor="middle" font-size="11" fill="#15803d">回调</text>')
# malloc callback -> ledger
L.append(f'<line x1="540" y1="314" x2="540" y2="410" stroke="#15803d" stroke-width="2" marker-end="url(#g)"/>')
L.append(f'<text x="556" y="365" text-anchor="start" font-size="11" fill="#15803d">写入</text>')

# red path: GC -> C ext free? Actually GC -> free callback -> ledger pop
# GC trigger -> free callback (along bottom) ; show GC -> C ext -> free callback
L.append(f'<line x1="290" y1="442" x2="370" y2="455" stroke="#b91c1c" stroke-width="2" marker-end="url(#r)"/>')
# free callback -> ledger (pop)
L.append(f'<line x1="790" y1="448" x2="710" y2="455" stroke="#b91c1c" stroke-width="2" marker-end="url(#r)"/>')
L.append(f'<text x="750" y="440" text-anchor="middle" font-size="11" fill="#b91c1c">弹出</text>')
# free callback -> C ext (return handle to unmap) upward
L.append(f'<line x1="915" y1="410" x2="915" y2="290" stroke="#b91c1c" stroke-width="2" marker-end="url(#r)"/>')
L.append(f'<text x="980" y="352" text-anchor="middle" font-size="11" fill="#b91c1c">交还 handle</text>')

# purple registration: allocator (get_pluggable_allocator/init_module)登记 malloc & free callbacks
L.append(f'<line x1="500" y1="160" x2="510" y2="250" stroke="#7c3aed" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#b)"/>')
L.append(f'<text x="430" y="205" text-anchor="middle" font-size="11" fill="#7c3aed">init_module</text>')
L.append(f'<text x="430" y="220" text-anchor="middle" font-size="11" fill="#7c3aed">登记回调</text>')

# bottom note
L.append(f'<text x="{W/2}" y="560" text-anchor="middle" font-size="13" fill="#475569">账本是 sleep / wake_up 能逐块 offload、逐块重映射的唯一依据</text>')
L.append(f'<text x="{W/2}" y="582" text-anchor="middle" font-size="12" fill="#7c3aed">回调存进 C 扩展的全局变量 → CaMemAllocator 必须是单例</text>')

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm-ascend/artifacts/ch07-sleep-mode-camem-allocator/diagrams/callback_loop.svg","w").write('\n'.join(L))
print("ok")
