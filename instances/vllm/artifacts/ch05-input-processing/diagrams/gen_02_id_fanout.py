#!/usr/bin/env python3
"""02-request-id-and-fanout: assign_request_id 唯一化 + n>1 并行采样 fan-out。"""
import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs


def esc(s): return xs.escape(s)


C_BOX = "#ffffff"
C_BD = "#475569"
C_TXT = "#1e293b"
C_MUT = "#64748b"
C_ID = "#dbeafe"
C_ID_BD = "#2563eb"
C_CHILD = "#dcfce7"
C_CHILD_BD = "#16a34a"
C_PARENT = "#fef3c7"
C_PARENT_BD = "#d97706"


def box(L, x, y, w, h, title, sub="", src="", fill=C_BOX, bd=C_BD, tcol=C_TXT, fs=13, mono_title=False):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{bd}" stroke-width="2"/>')
    lines = [l for l in [sub, src] if l]
    cy = y + (h // 2 if not lines else h // 2 - 6 * len(lines))
    ff = "monospace" if mono_title else "sans-serif"
    L.append(f'<text x="{x+w//2}" y="{cy+5}" text-anchor="middle" font-family="{ff}" font-size="{fs}" font-weight="bold" fill="{tcol}">{esc(title)}</text>')
    off = cy + 21
    if sub:
        L.append(f'<text x="{x+w//2}" y="{off}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="{C_MUT}">{esc(sub)}</text>')
        off += 15
    if src:
        L.append(f'<text x="{x+w//2}" y="{off}" text-anchor="middle" font-family="monospace" font-size="9.5" fill="{C_MUT}">{esc(src)}</text>')


def arrow(L, x1, y1, x2, y2, mid="", color=C_BD, mid_dy=-6):
    L.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="2.2" marker-end="url(#a)"/>')
    if mid:
        L.append(f'<text x="{(x1+x2)//2}" y="{(y1+y2)//2+mid_dy}" text-anchor="middle" font-family="monospace" font-size="10" fill="{color}">{esc(mid)}</text>')


def build():
    w, h = 1080, 720
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

    # ===== 上半：assign_request_id =====
    L.append(f'<text x="30" y="36" font-family="sans-serif" font-size="18" font-weight="bold" fill="{C_TXT}">{esc("① 请求 id 唯一化：assign_request_id()")}</text>')
    L.append(f'<text x="30" y="56" font-family="monospace" font-size="10" fill="{C_MUT}">{esc("vllm/v1/engine/input_processor.py:L214-L232")}</text>')

    box(L, 60, 80, 230, 60, '"req-abc"', "外部传入 request_id（可能重复）", fill="#f1f5f9", mono_title=True, fs=15)
    box(L, 420, 80, 250, 60, "external_req_id ← request_id", "原 id 留存，供输出回传 / abort", fill=C_ID, bd=C_ID_BD, fs=12)
    box(L, 780, 80, 250, 60, '"req-abc-3f9a2b1c"', "request_id += 8 字符随机后缀", fill=C_ID, bd=C_ID_BD, mono_title=True, fs=13)
    arrow(L, 290, 110, 420, 110, "拷贝", color=C_ID_BD)
    arrow(L, 670, 110, 780, 110, "random_uuid():.8", color=C_ID_BD)

    L.append(f'<line x1="30" y1="175" x2="1050" y2="175" stroke="#e2e8f0" stroke-width="1.5"/>')

    # ===== 下半：fan-out =====
    L.append(f'<text x="30" y="210" font-family="sans-serif" font-size="18" font-weight="bold" fill="{C_TXT}">{esc("② n>1 并行采样 fan-out：ParentRequest")}</text>')
    L.append(f'<text x="30" y="230" font-family="monospace" font-size="10" fill="{C_MUT}">{esc("async_llm.py:L381-L398 · parallel_sampling.py:L52-L94")}</text>')

    box(L, 410, 255, 260, 70, "父 EngineCoreRequest", "sampling_params.n = 4", "is_pooling or n==1 → 单请求直发", fill=C_PARENT, bd=C_PARENT_BD, fs=13)

    # 四个子请求
    cw, gap = 220, 20
    total = 4 * cw + 3 * gap
    start = (w - total) // 2
    for i in range(4):
        x = start + i * (cw + gap)
        last = (i == 3)
        box(L, x, 430, cw, 92,
            f'child {i}',
            f'request_id = "{i}_req-abc-3f9a..."',
            "复用父对象" if last else "copy(request)",
            fill=C_CHILD, bd=C_CHILD_BD, fs=13)
        L.append(f'<text x="{x+cw//2}" y="{430+62}" text-anchor="middle" font-family="monospace" font-size="9.5" fill="{C_MUT}">{esc("sampling n=1; seed→seed+idx")}</text>')
        arrow(L, 540, 325, x + cw // 2, 430, color=C_CHILD_BD)

    cinfo = esc('get_child_info(idx)：child id = "{idx}_{request_id}"，子参数 n=1')
    L.append(f'<text x="{w//2}" y="395" text-anchor="middle" font-family="sans-serif" font-size="12" fill="{C_MUT}">{cinfo}</text>')

    # seed 说明
    L.append(f'<rect x="60" y="560" width="960" height="120" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
    L.append(f'<text x="78" y="586" font-family="sans-serif" font-size="13" font-weight="bold" fill="{C_TXT}">{esc("子采样参数派生（_get_child_sampling_params）")}</text>')
    L.append(f'<text x="78" y="612" font-family="sans-serif" font-size="12" fill="{C_TXT}">{esc("• seed is None：所有子请求共享同一份 n=1 克隆（缓存复用，省内存；各自采样天然独立）")}</text>')
    L.append(f'<text x="78" y="636" font-family="sans-serif" font-size="12" fill="{C_TXT}">{esc("• seed 已设：每个子请求 seed = seed + index（结果各异且可复现）")}</text>')
    L.append(f'<text x="78" y="660" font-family="sans-serif" font-size="12" fill="{C_TXT}">{esc("• 输出聚合：流式逐个转发；FINAL_ONLY 按 index 攒齐 n 个再整批返回")}</text>')

    L.append('</svg>')
    return '\n'.join(L)


if __name__ == '__main__':
    base = sys.argv[1]
    svg = build()
    Path(base + '.svg').write_text(svg)
    print(f"SVG {len(svg)}B")
    assert subprocess.run(['xmllint', '--noout', base + '.svg']).returncode == 0
    subprocess.run(['rsvg-convert', '-z', '2', base + '.svg', '-o', base + '.png'], check=True)
    print(f"PNG {os.path.getsize(base+'.png')//1024}KB")
