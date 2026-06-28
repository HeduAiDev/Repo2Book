#!/usr/bin/env python3
"""三级取值漏斗：additional_config -> env -> default，且 env/default 在 envs.py lambda 里先塌缩。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

W, H = 1000, 540
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
         '<marker id="arg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
         '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="38" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold" fill="#0f172a">三级取值：additional_config → env → default</text>')
L.append(f'<text x="{W/2}" y="62" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#64748b">_get_config_value 只裁决一层；env 与 default 早在 envs.py 的 lambda 里塌缩成一个值</text>')

# Step 1: _get_config_value decision (additional_config)
dx, dy, dw, dh = 60, 120, 380, 160
L.append(f'<rect x="{dx}" y="{dy}" width="{dw}" height="{dh}" rx="10" fill="#dcfce7" stroke="#15803d" stroke-width="2"/>')
L.append(f'<text x="{dx+dw/2}" y="{dy+28}" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="bold" fill="#14532d">_get_config_value 裁决这一层</text>')
L.append(f'<text x="{dx+20}" y="{dy+62}" font-family="sans-serif" font-size="13.5" fill="#166534">config_key in additional_config ?</text>')
L.append(f'<text x="{dx+38}" y="{dy+94}" font-family="sans-serif" font-size="13" fill="#15803d" font-weight="bold">是 → 用 additional_config[key]（最高优先级）</text>')
L.append(f'<text x="{dx+38}" y="{dy+126}" font-family="sans-serif" font-size="13" fill="#b91c1c">否 → 返回传入的 env_value（已含 env/default）</text>')

# Step 2: envs.py lambda collapse
ex, ey, ew, eh = 560, 120, 380, 160
L.append(f'<rect x="{ex}" y="{ey}" width="{ew}" height="{eh}" rx="10" fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<text x="{ex+ew/2}" y="{ey+28}" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="bold" fill="#1e3a8a">envs.py：env 与 default 在此塌缩</text>')
L.append(f'<text x="{ex+24}" y="{ey+64}" font-family="monospace" font-size="13" fill="#1e40af">lambda: bool(int(</text>')
L.append(f'<text x="{ex+36}" y="{ey+88}" font-family="monospace" font-size="13" fill="#1e40af">os.getenv(KEY, "0")))</text>')
L.append(f'<text x="{ex+24}" y="{ey+124}" font-family="sans-serif" font-size="12.5" fill="#475569">环境里有 KEY → 取环境值；没有 → 取 default</text>')

# arrow: env_value feeds into _get_config_value's else branch
L.append(f'<path d="M {ex} {ey+115} C 510 {ey+150}, 470 {dy+140}, {dx+dw+4} {dy+120}" fill="none" stroke="#15803d" stroke-width="2" marker-end="url(#arg)"/>')
L.append(f'<text x="{(ex+dx+dw)/2}" y="{ey+108}" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#15803d">env_value（实参先求好再传入）</text>')

# outcome priority strip — manual stepping, no messy loop
oy = 350
L.append(f'<text x="60" y="{oy}" font-family="sans-serif" font-size="15" font-weight="bold" fill="#0f172a">逻辑优先级（高 → 低）：</text>')
cy = oy + 18
chips = [('additional_config[key]', '#15803d', '#dcfce7'),
         ('环境变量 KEY（DEPRECATED）', '#a16207', '#fef9c3'),
         ('default（lambda 内默认值）', '#475569', '#f1f5f9')]
cx = 60
for i,(t,stroke,fill) in enumerate(chips):
    cw = 24 + len(t)*9.5
    L.append(f'<rect x="{cx}" y="{cy}" width="{cw:.0f}" height="42" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    L.append(f'<text x="{cx+cw/2:.0f}" y="{cy+27}" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="bold" fill="{stroke}">{esc(t)}</text>')
    cx = cx + cw
    if i < len(chips)-1:
        L.append(f'<text x="{cx+16:.0f}" y="{cy+29}" text-anchor="middle" font-family="sans-serif" font-size="22" fill="#94a3b8">&gt;</text>')
        cx = cx + 32

# failure-of-validation callout
fy = oy + 90
L.append(f'<rect x="60" y="{fy}" width="{W-120}" height="56" rx="10" fill="#fff1f2" stroke="#e11d48" stroke-width="1.8"/>')
L.append(f'<text x="80" y="{fy+24}" font-family="sans-serif" font-size="13.5" font-weight="bold" fill="#9f1239">失校验的代价（无 schema 后门）</text>')
L.append(f'<text x="80" y="{fy+45}" font-family="sans-serif" font-size="13" fill="#881337">把 additional_config 键名拼错 → 静默落回 env_value，不报错；你以为开了，其实没开。</text>')

L.append('</svg>')
out = '/mnt/e/Laboratory/Repo2Book/instances/vllm-ascend/artifacts/ch05-check-and-update-config/diagrams/three_level.svg'
open(out,'w').write('\n'.join(L))
print('ok')
