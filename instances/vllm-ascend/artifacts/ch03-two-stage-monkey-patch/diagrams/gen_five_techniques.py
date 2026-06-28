#!/usr/bin/env python3
"""5 种重绑技法对照卡片：按「被替换引用的形态」选招。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1280, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="42" font-family="sans-serif" font-size="24" font-weight="bold" fill="#0f172a" text-anchor="middle">5 种重绑技法：按「被替换引用的形态」选招</text>')
L.append(f'<text x="{W/2}" y="72" font-family="sans-serif" font-size="14.5" fill="#64748b" text-anchor="middle">monkey-patch 改的是「某命名空间里的名字指向哪个对象」，不改对象本身</text>')

cards = [
    {
        "no": "①", "name": "整类替换", "col": "#2563eb", "bg": "#eff6ff",
        "form": "类名属性",
        "code": ["MultiprocExecutor", "= AscendMultiprocExecutor"],
        "file": "platform/patch_multiproc_executor.py",
        "note": "子类整体重写，把模块属性指向子类",
    },
    {
        "no": "②", "name": "工厂(注册表)替换", "col": "#7c3aed", "bg": "#f5f3ff",
        "form": "类名 + 派发表",
        "code": ["spec_manager_map[MambaSpec]", "= AscendMambaManager"],
        "file": "platform/patch_mamba_manager.py",
        "note": "只换类名不够：工厂照旧表 new 旧类",
    },
    {
        "no": "③", "name": "方法替换", "col": "#0891b2", "bg": "#ecfeff",
        "form": "类的单个属性",
        "code": ["Scheduler.", "_mamba_block_aligned_split", "= fn"],
        "file": "platform/patch_scheduler.py",
        "note": "不建子类，只换一个绑定方法",
    },
    {
        "no": "④", "name": "库函数 wrapper", "col": "#d97706", "bg": "#fff7ed",
        "form": "模块里的函数",
        "code": ["triton.next_power_of_2", "= next_power_of_2", "broadcast = wrapper(fn)"],
        "file": "worker/patch_triton.py · platform/patch_distributed.py",
        "note": "闭包捕获原 fn，前后增量包裹 / 直接补函数",
    },
    {
        "no": "⑤", "name": "from-import 缓存陷阱", "col": "#dc2626", "bg": "#fef2f2",
        "form": "同对象的所有别名",
        "code": ["torch.distributed.broadcast", "= w", "distributed_c10d.broadcast", "= w"],
        "file": "platform/patch_distributed.py",
        "note": "调用方可能已缓存引用：所有别名都要重绑",
    },
]

n = len(cards)
margin = 28
gap = 16
cw = (W - 2 * margin - (n - 1) * gap) / n
cy = 100
ch = 540
for i, c in enumerate(cards):
    x = margin + i * (cw + gap)
    L.append(f'<rect x="{x}" y="{cy}" width="{cw}" height="{ch}" rx="12" fill="{c["bg"]}" stroke="{c["col"]}" stroke-width="2"/>')
    # header band
    L.append(f'<rect x="{x}" y="{cy}" width="{cw}" height="58" rx="12" fill="{c["col"]}"/>')
    L.append(f'<rect x="{x}" y="{cy+30}" width="{cw}" height="28" fill="{c["col"]}"/>')
    L.append(f'<text x="{x+cw/2}" y="{cy+26}" font-family="sans-serif" font-size="22" font-weight="bold" fill="white" text-anchor="middle">{esc(c["no"])}</text>')
    L.append(f'<text x="{x+cw/2}" y="{cy+50}" font-family="sans-serif" font-size="15" font-weight="bold" fill="white" text-anchor="middle">{esc(c["name"])}</text>')

    # form label
    yy = cy + 88
    L.append(f'<text x="{x+cw/2}" y="{yy}" font-family="sans-serif" font-size="12.5" fill="#64748b" text-anchor="middle">被替换引用形态</text>')
    L.append(f'<text x="{x+cw/2}" y="{yy+22}" font-family="sans-serif" font-size="14" font-weight="bold" fill="{c["col"]}" text-anchor="middle">{esc(c["form"])}</text>')

    # code block
    cby = yy + 42
    cbh = 22 * len(c["code"]) + 18
    L.append(f'<rect x="{x+10}" y="{cby}" width="{cw-20}" height="{cbh}" rx="7" fill="#0f172a"/>')
    for j, line in enumerate(c["code"]):
        L.append(f'<text x="{x+20}" y="{cby+24+j*22}" font-family="monospace" font-size="11.5" fill="#e2e8f0">{esc(line)}</text>')

    # note
    ny = cby + cbh + 26
    # wrap note manually
    note = c["note"]
    L.append(f'<text x="{x+cw/2}" y="{ny}" font-family="sans-serif" font-size="12" fill="#64748b" text-anchor="middle">代表 patch</text>')
    # file path — split on ' · '
    parts = c["file"].split(" · ")
    for j, p in enumerate(parts):
        L.append(f'<text x="{x+cw/2}" y="{ny+18+j*16}" font-family="monospace" font-size="9.5" fill="#475569" text-anchor="middle">{esc(p)}</text>')

    # bottom note box
    bny = cy + ch - 78
    L.append(f'<rect x="{x+10}" y="{bny}" width="{cw-20}" height="66" rx="7" fill="white" stroke="{c["col"]}" stroke-width="1" stroke-dasharray="3 3"/>')
    # wrap note into <=2 lines by char count
    words = note
    # naive wrap at ~12 chars
    line1, line2 = words, ""
    if len(words) > 13:
        cut = words.rfind("：", 0, 14)
        if cut < 0:
            cut = 13
        else:
            cut += 1
        line1, line2 = words[:cut], words[cut:]
    L.append(f'<text x="{x+cw/2}" y="{bny+28}" font-family="sans-serif" font-size="12" fill="#334155" text-anchor="middle">{esc(line1)}</text>')
    if line2:
        L.append(f'<text x="{x+cw/2}" y="{bny+48}" font-family="sans-serif" font-size="12" fill="#334155" text-anchor="middle">{esc(line2)}</text>')

L.append('</svg>')
open("five_techniques.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote five_techniques.svg")
