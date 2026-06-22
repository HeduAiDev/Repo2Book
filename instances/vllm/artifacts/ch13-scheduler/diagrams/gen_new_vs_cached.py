#!/usr/bin/env python3
"""NewRequestData(首次全量) vs CachedRequestData(后续增量) 对照。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


w, h = 880, 470
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
         'markerWidth="8" markerHeight="6" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{w//2}" y="32" text-anchor="middle" font-size="21" '
         'font-weight="bold" fill="#0f172a">SchedulerOutput 二分：全量首发 vs 增量续发</text>')

# Left: NewRequestData
lx, ly, lw = 40, 70, 360
L.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="340" rx="9" '
         'fill="#fef2f2" stroke="#f87171" stroke-width="2"/>')
L.append(f'<text x="{lx+lw/2}" y="{ly+28}" text-anchor="middle" font-size="16" '
         'font-weight="bold" fill="#b91c1c">NewRequestData（首次调度 · 全量）</text>')
L.append(f'<text x="{lx+lw/2}" y="{ly+50}" text-anchor="middle" font-size="12.5" '
         'fill="#7f1d1d">scheduled_new_reqs — WAITING 第一次进 running</text>')

new_fields = [
    ("prompt_token_ids", "全部 prompt token"),
    ("sampling_params", "采样参数（温度/max_tokens…）"),
    ("mm_features", "多模态特征"),
    ("block_ids", "首批 KV 块"),
    ("num_computed_tokens", "已算 token 数"),
    ("lora_request", "LoRA 适配器"),
]
fy = ly + 70
for name, desc in new_fields:
    L.append(f'<rect x="{lx+16}" y="{fy}" width="{lw-32}" height="38" rx="5" '
             'fill="#fee2e2" stroke="#fca5a5" stroke-width="1.2"/>')
    L.append(f'<text x="{lx+28}" y="{fy+17}" font-size="13" font-weight="bold" '
             f'font-family="monospace" fill="#991b1b">{esc(name)}</text>')
    L.append(f'<text x="{lx+28}" y="{fy+33}" font-size="11" fill="#7f1d1d">{esc(desc)}</text>')
    fy += 44

# Right: CachedRequestData
rx, rw = 480, 360
L.append(f'<rect x="{rx}" y="{ly}" width="{rw}" height="340" rx="9" '
         'fill="#f0fdf4" stroke="#4ade80" stroke-width="2"/>')
L.append(f'<text x="{rx+rw/2}" y="{ly+28}" text-anchor="middle" font-size="16" '
         'font-weight="bold" fill="#15803d">CachedRequestData（后续步 · 增量 diff）</text>')
L.append(f'<text x="{rx+rw/2}" y="{ly+50}" text-anchor="middle" font-size="12.5" '
         'fill="#14532d">scheduled_cached_reqs — running / resumed 续发</text>')

cached_fields = [
    ("req_ids", "只发 id，静态数据 worker 已缓存"),
    ("new_block_ids", "本拍新增的 KV 块（追加）"),
    ("num_computed_tokens", "更新后的已算 token 数"),
    ("num_output_tokens", "已出 token 数（含占位）"),
    ("resumed_req_ids", "抢占恢复者：替换 block 而非追加"),
]
fy = ly + 70
for name, desc in cached_fields:
    L.append(f'<rect x="{rx+16}" y="{fy}" width="{rw-32}" height="38" rx="5" '
             'fill="#dcfce7" stroke="#86efac" stroke-width="1.2"/>')
    L.append(f'<text x="{rx+28}" y="{fy+17}" font-size="13" font-weight="bold" '
             f'font-family="monospace" fill="#166534">{esc(name)}</text>')
    L.append(f'<text x="{rx+28}" y="{fy+33}" font-size="11" fill="#14532d">{esc(desc)}</text>')
    fy += 44

# middle arrow
my = 240
L.append(f'<line x1="{lx+lw+4}" y1="{my}" x2="{rx-6}" y2="{my}" stroke="#7c3aed" '
         'stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(lx+lw+rx)/2}" y="{my-12}" text-anchor="middle" font-size="12.5" '
         'font-weight="bold" fill="#7c3aed">worker 缓存全量后</text>')
L.append(f'<text x="{(lx+lw+rx)/2}" y="{my+22}" text-anchor="middle" font-size="12.5" '
         'font-weight="bold" fill="#7c3aed">每拍只收 diff</text>')

# bottom note
L.append(f'<text x="{w//2}" y="442" text-anchor="middle" font-size="13" '
         'fill="#475569" font-style="italic">'
         'prev_step_scheduled_req_ids 判定谁「上一拍没调度」，需补传 all_token_ids —— 把 IPC 通信量压到最小</text>')

L.append('</svg>')
open("13-new-vs-cached.svg", "w").write('\n'.join(L))
print("wrote 13-new-vs-cached.svg")
