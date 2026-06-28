#!/usr/bin/env python3
"""三态时间线：清醒 → sleep(level1) → wake_up。强调虚拟地址 VA 三态不变、物理页可换。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

W, H = 1180, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="#475569"/></marker>')
L.append('<marker id="arR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="#b91c1c"/></marker>')
L.append('<marker id="arG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="#15803d"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# Title
L.append(f'<text x="{W/2}" y="34" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">sleep / wake_up：虚拟地址不动，物理页可换</text>')

panels = [
    {"x": 30,  "title": "① 清醒", "sub": "VA ← map → 物理页"},
    {"x": 410, "title": "② sleep(level 1)", "sub": "解绑物理页、VA 保留"},
    {"x": 790, "title": "③ wake_up", "sub": "重映射回原 VA"},
]
PW = 350
VA_LABEL = "VA  0x7f..e000"

for i, p in enumerate(panels):
    x = p["x"]
    # panel frame
    L.append(f'<rect x="{x}" y="60" width="{PW}" height="470" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>')
    L.append(f'<text x="{x+PW/2}" y="90" text-anchor="middle" font-size="17" font-weight="bold" fill="#1e293b">{esc(p["title"])}</text>')
    L.append(f'<text x="{x+PW/2}" y="111" text-anchor="middle" font-size="12.5" fill="#64748b">{esc(p["sub"])}</text>')

    cx = x + PW/2
    # VA bar — IDENTICAL across all three
    va_y = 132
    L.append(f'<rect x="{x+30}" y="{va_y}" width="{PW-60}" height="40" rx="6" fill="#ede9fe" stroke="#7c3aed" stroke-width="2"/>')
    L.append(f'<text x="{cx}" y="{va_y+18}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#5b21b6">虚拟地址区间（不变）</text>')
    L.append(f'<text x="{cx}" y="{va_y+34}" text-anchor="middle" font-size="12" font-family="monospace" fill="#6d28d9">{esc(VA_LABEL)}</text>')

    # physical pages region
    phys_y = 230
    wx = x + 40
    kx = x + PW/2 + 10
    bw = (PW-100)/2
    if i == 0:  # awake: both solid mapped
        for bx, lbl, sc in [(wx,"weights\n物理页",("#bfdbfe","#2563eb")), (kx,"kv_cache\n物理页",("#bbf7d0","#16a34a"))]:
            fill, st = sc
            L.append(f'<rect x="{bx}" y="{phys_y}" width="{bw}" height="56" rx="6" fill="{fill}" stroke="{st}" stroke-width="2"/>')
            parts = lbl.split("\n")
            L.append(f'<text x="{bx+bw/2}" y="{phys_y+24}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#0f172a">{esc(parts[0])}</text>')
            L.append(f'<text x="{bx+bw/2}" y="{phys_y+42}" text-anchor="middle" font-size="11.5" fill="#334155">{esc(parts[1])}</text>')
            # map arrow VA -> page
            L.append(f'<line x1="{bx+bw/2}" y1="{va_y+40}" x2="{bx+bw/2}" y2="{phys_y}" stroke="#475569" stroke-width="1.6" marker-end="url(#ar)"/>')
        L.append(f'<text x="{cx}" y="{va_y+62}" text-anchor="middle" font-size="11" fill="#64748b">map</text>')
    elif i == 1:  # sleep: VA dashed-retained; weights->CPU; kv discarded
        # weights physical -> dashed gone box + arrow to CPU
        L.append(f'<rect x="{wx}" y="{phys_y}" width="{bw}" height="56" rx="6" fill="none" stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="5 4"/>')
        L.append(f'<text x="{wx+bw/2}" y="{phys_y+27}" text-anchor="middle" font-size="11.5" fill="#94a3b8">weights 页</text>')
        L.append(f'<text x="{wx+bw/2}" y="{phys_y+44}" text-anchor="middle" font-size="11.5" fill="#94a3b8">已解绑</text>')
        L.append(f'<rect x="{kx}" y="{phys_y}" width="{bw}" height="56" rx="6" fill="none" stroke="#ef4444" stroke-width="1.6" stroke-dasharray="5 4"/>')
        L.append(f'<text x="{kx+bw/2}" y="{phys_y+27}" text-anchor="middle" font-size="11.5" fill="#dc2626">kv_cache 页</text>')
        L.append(f'<text x="{kx+bw/2}" y="{phys_y+44}" text-anchor="middle" font-size="11.5" fill="#dc2626">丢弃 ✗</text>')
        # CPU backup box (weights only)
        cpu_y = 360
        L.append(f'<rect x="{wx}" y="{cpu_y}" width="{bw}" height="50" rx="6" fill="#fef9c3" stroke="#ca8a04" stroke-width="2"/>')
        L.append(f'<text x="{wx+bw/2}" y="{cpu_y+21}" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#854d0e">CPU pin 备份</text>')
        L.append(f'<text x="{wx+bw/2}" y="{cpu_y+38}" text-anchor="middle" font-size="11" fill="#a16207">weights 拷回</text>')
        # D2H arrow weights page -> CPU
        L.append(f'<line x1="{wx+bw/2}" y1="{phys_y+56}" x2="{wx+bw/2}" y2="{cpu_y}" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#arR)"/>')
        L.append(f'<text x="{wx+bw/2+6}" y="{(phys_y+56+cpu_y)/2+4}" text-anchor="start" font-size="11" font-weight="bold" fill="#b91c1c">D2H</text>')
        # device freed note
        L.append(f'<rect x="{kx}" y="{cpu_y}" width="{bw}" height="50" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="3 3"/>')
        L.append(f'<text x="{kx+bw/2}" y="{cpu_y+21}" text-anchor="middle" font-size="11" fill="#64748b">无备份</text>')
        L.append(f'<text x="{kx+bw/2}" y="{cpu_y+38}" text-anchor="middle" font-size="11" fill="#64748b">唤醒后重算</text>')
        L.append(f'<text x="{cx}" y="445" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#15803d">设备物理显存 → 全部还给系统</text>')
        L.append(f'<text x="{cx}" y="466" text-anchor="middle" font-size="11.5" fill="#64748b">unmap_and_release（unmap 在 if 之外）</text>')
    else:  # wake_up
        # weights remapped solid + restored
        L.append(f'<rect x="{wx}" y="{phys_y}" width="{bw}" height="56" rx="6" fill="#bfdbfe" stroke="#2563eb" stroke-width="2"/>')
        L.append(f'<text x="{wx+bw/2}" y="{phys_y+24}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#0f172a">weights 页</text>')
        L.append(f'<text x="{wx+bw/2}" y="{phys_y+42}" text-anchor="middle" font-size="11.5" fill="#15803d">已复原</text>')
        # kv empty page
        L.append(f'<rect x="{kx}" y="{phys_y}" width="{bw}" height="56" rx="6" fill="#f1f5f9" stroke="#16a34a" stroke-width="2"/>')
        L.append(f'<text x="{kx+bw/2}" y="{phys_y+24}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#0f172a">kv_cache 页</text>')
        L.append(f'<text x="{kx+bw/2}" y="{phys_y+42}" text-anchor="middle" font-size="11.5" fill="#64748b">空页 · 待重算</text>')
        # remap arrows VA -> pages
        for bx in (wx+bw/2, kx+bw/2):
            L.append(f'<line x1="{bx}" y1="{va_y+40}" x2="{bx}" y2="{phys_y}" stroke="#15803d" stroke-width="1.8" marker-end="url(#arG)"/>')
        L.append(f'<text x="{cx}" y="{va_y+62}" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d">create_and_map</text>')
        # H2D from CPU for weights
        cpu_y = 360
        L.append(f'<rect x="{wx}" y="{cpu_y}" width="{bw}" height="50" rx="6" fill="#fef9c3" stroke="#ca8a04" stroke-width="2"/>')
        L.append(f'<text x="{wx+bw/2}" y="{cpu_y+30}" text-anchor="middle" font-size="11.5" fill="#854d0e">CPU 备份释放</text>')
        L.append(f'<line x1="{wx+bw/2}" y1="{cpu_y}" x2="{wx+bw/2}" y2="{phys_y+56}" stroke="#15803d" stroke-width="1.8" marker-end="url(#arG)"/>')
        L.append(f'<text x="{wx+bw/2+6}" y="{(phys_y+56+cpu_y)/2+4}" text-anchor="start" font-size="11" font-weight="bold" fill="#15803d">H2D</text>')

# bottom unifying note
L.append(f'<text x="{W/2}" y="552" text-anchor="middle" font-size="13" fill="#5b21b6">三态里那一行紫色虚拟地址值从不改变 → PyTorch 现存的张量指针、计算图引用全程有效，无需重建模型</text>')

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm-ascend/artifacts/ch07-sleep-mode-camem-allocator/diagrams/virtual_memory_sleep_wake.svg","w").write('\n'.join(L))
print("ok")
