#!/usr/bin/env python3
"""ch33 机制图共享脚手架：标题带 + L0 指北 chip + 页脚 + 收尾写盘。

角色色沿 l0_common（同源强制）：采样出口=C_SAM_S 品红、GPU 执行臂=C_GPU_S 绿、
调度/显存账本=C_KV_S 青。数字全部来自 explainer figure_spec.numbers（逐字）。
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402


def header(W, title, subtitle, zoom):
    """标题带（左标题+副标题，右上 L0 指北虚线 chip）。返回 (MX, BXR)。"""
    MX = 46.0
    BXR = W - MX
    lc.text(MX, 33, title, 15.5, lc.C_TXT, 'start', True, maxw=W - 270, tag='title')
    lc.text(MX, 55, subtitle, 10, lc.C_MUTE, 'start', maxw=W - MX - 250, tag='subtitle')
    cw = lc.chip_w(zoom)
    lc.rect(BXR - cw, 11, cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(BXR - cw / 2, 25.5, zoom, 9.5, lc.C_BEAT_T, 'middle', True, maxw=cw - 4,
            tag='chip')
    return MX, BXR


def footer(MX, BXR, y, lines):
    """页脚出处行（可多行，行距 14）。返回下一可用 y。"""
    for i, s in enumerate(lines):
        lc.text(MX, y + i * 14, s, 8.5, lc.C_FAINT, 'start', maxw=BXR - MX,
                tag='foot%d' % i)
    return y + len(lines) * 14


def conclusion(MX, BXR, y, s):
    """图注结论行（正文由 writer 配图注，这里给图内结论一行）。"""
    lc.text(MX, y, s, 10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='conc')


def write(figure_id, W, H):
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}">',
           f'<rect width="{W:.0f}" height="{H:.0f}" fill="white"/>', lc.DEFS]
    svg += [s for _, s in lc.ELEMS]
    svg.append('</svg>')
    out = HERE / f'{figure_id}.svg'
    out.write_bytes('\n'.join(svg).encode('utf-8'))
    print(f'wrote {out.name}  ({W:.0f}x{H:.0f}, {len(lc.ELEMS)} elems)')
    if lc.WARN:
        print('--- OVERFLOW WARNINGS ---')
        for w_ in lc.WARN:
            print(' ', w_)
        sys.exit(1)
