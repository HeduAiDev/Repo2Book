#!/usr/bin/env python3
"""配置改写时序：从 VllmConfig.__post_init__ 到 NPUPlatform.check_and_update_config 的就地改写编排。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

W, H = 1040, 800
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
         '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
         '<marker id="arp" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
         '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# title
L.append(f'<text x="{W/2}" y="36" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold" fill="#0f172a">配置改写时序：平台 = 构图前最后一道改写器</text>')
L.append(f'<text x="{W/2}" y="60" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#64748b">VllmConfig 按引用传入，平台就地改它（passed by reference, modified in place）</text>')

# left vertical spine of __post_init__
spine_x = 130
top = 100
bot = 700
L.append(f'<line x1="{spine_x}" y1="{top}" x2="{spine_x}" y2="{bot}" stroke="#cbd5e1" stroke-width="3"/>')
L.append(f'<text x="{spine_x-12}" y="{top-12}" text-anchor="start" font-family="sans-serif" font-size="13" font-weight="bold" fill="#475569">VllmConfig.__post_init__</text>')

# two windows on the spine: defaults (early) and check (late)
def dot(y, label, sub):
    L.append(f'<circle cx="{spine_x}" cy="{y}" r="7" fill="#7c3aed"/>')
    L.append(f'<text x="{spine_x+16}" y="{y-4}" font-family="sans-serif" font-size="13" font-weight="bold" fill="#5b21b6">{esc(label)}</text>')
    L.append(f'<text x="{spine_x+16}" y="{y+13}" font-family="sans-serif" font-size="11.5" fill="#64748b">{esc(sub)}</text>')

dot(150, 'L983  apply_config_platform_defaults(self)', '改写窗口①：先注入 Ascend 默认值')
dot(225, 'L1197  check_and_update_config(self)', '改写窗口②：拿到完整 config 做最后修正  ← 本章主场')

# box: NPUPlatform.check_and_update_config orchestration column
bx = 420
bw = 540
steps = [
    ('守卫早退', 'device_type≠npu / model_config is None → return', '#f1f5f9', '#64748b'),
    ('_validate_*  一致性校验', 'layer_sharding / draft_decode_cp  不符则 raise', '#fef9c3', '#a16207'),
    ('_fix_incompatible_config', '9 段 cascade reset：GPU/ROCm 专属参数 → 安全值', '#dbeafe', '#1d4ed8'),
    ('init_ascend_config', '开放 dict additional_config → 强类型 AscendConfig（单例）', '#dcfce7', '#15803d'),
    ('cudagraph / 编译改写', 'enforce_eager→CompilationMode.NONE；非法 mode→CUDAGraphMode.NONE', '#ede9fe', '#6d28d9'),
    ("worker_cls 'auto' → NPUWorker", '把抽象哨兵落成具体 Worker 的 qualname 字符串', '#fae8ff', '#a21caf'),
    ('设进程环境变量', 'PYTORCH_NPU_ALLOC_CONF += expandable_segments:True', '#ffedd5', '#c2410c'),
]
sy = 290
sh = 52
gap = 9
# arrow from window2 dot into the box top
L.append(f'<path d="M {spine_x} 232 C 200 270, 280 280, {bx-6} {sy+sh/2}" fill="none" stroke="#7c3aed" stroke-width="2.2" marker-end="url(#arp)"/>')
prev_cy = None
for i, (t, s, fill, stroke) in enumerate(steps):
    y = sy + i*(sh+gap)
    L.append(f'<rect x="{bx}" y="{y}" width="{bw}" height="{sh}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{bx+16}" y="{y+22}" font-family="sans-serif" font-size="14" font-weight="bold" fill="{stroke}">{esc(t)}</text>')
    L.append(f'<text x="{bx+16}" y="{y+41}" font-family="sans-serif" font-size="11.5" fill="#475569">{esc(s)}</text>')
    cy = y + sh/2
    if prev_cy is not None:
        L.append(f'<line x1="{bx-14}" y1="{prev_cy}" x2="{bx-14}" y2="{cy}" stroke="#94a3b8" stroke-width="1.6"/>')
        L.append(f'<line x1="{bx-14}" y1="{cy}" x2="{bx-3}" y2="{cy}" stroke="#94a3b8" stroke-width="1.6" marker-end="url(#ar)"/>')
    prev_cy = cy

# bottom: rewritten config flows on
out_y = sy + len(steps)*(sh+gap) + 14
L.append(f'<rect x="{bx}" y="{out_y}" width="{bw}" height="44" rx="8" fill="#0f172a"/>')
L.append(f'<text x="{bx+bw/2}" y="{out_y+27}" text-anchor="middle" font-family="sans-serif" font-size="13.5" font-weight="bold" fill="#e2e8f0">被改写后的 VllmConfig → 后续构图 / 建 Worker / 选注意力后端</text>')
L.append(f'<line x1="{bx+bw/2}" y1="{out_y-9}" x2="{bx+bw/2}" y2="{out_y-2}" stroke="#94a3b8" stroke-width="1.6" marker-end="url(#ar)"/>')

L.append('</svg>')
open('/mnt/e/Laboratory/Repo2Book/instances/vllm-ascend/artifacts/ch05-check-and-update-config/diagrams/rewrite_timeline.svg','w').write('\n'.join(L))
print('ok')
