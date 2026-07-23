#!/usr/bin/env python3
"""before-after 模板:昇腾 scope+事件 vs Hopper warp+mbarrier 对位。
右栏是**姊妹篇基座《Triton 源码解读》**讲 warp specialization 的那一章,不是本书
(triton-ascend)自己的后文章节——本书自己的 ch31 讲 torch_npu/mindspore 策略注册表,
另一主题。徽标写「对位·姊妹篇基座」,不带 ch 号,避免读者翻本书 ch31 扑空
(2026-07-23 writer figure-requests 改稿)。虚线框沿用「非本章正文机制、外部对照」的
视觉含义,保留不动。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
ASC = "#1e40af"
ASC_BG = "#dbeafe"
HOP = "#7c3aed"
HOP_BG = "#ede9fe"
PREVIEW = "#b45309"

TITLE = "同一个「生产者/消费者分工 + 同步」思想,两种硬件两套物化"
SUB = "昇腾:两颗物理核跨核 + block 级事件;Hopper:同一 SM 内 warp 分组 + 共享内存 mbarrier"

PANEL_W, PAD, TOP = 430, 40, 150
GAP = 90
W = PAD * 2 + PANEL_W * 2 + GAP
H = TOP + 400

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="12.5" fill="{GRAY}">{esc(SUB)}</text>']

# ---- LEFT: Ascend (this chapter, solid style) ----
px = PAD
cx = px + PANEL_W / 2
L.append(f'<text x="{cx}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14.5" font-weight="bold" fill="{ASC}">昇腾(本章,ch17)</text>')
L.append(f'<rect x="{px}" y="{TOP}" width="{PANEL_W}" height="300" rx="10" '
          f'fill="{ASC_BG}" stroke="{ASC}" stroke-width="1.8"/>')
ay = TOP + 26
L.append(f'<text x="{px+20}" y="{ay}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="{INK}">分域:2 个物理核</text>')
L.append(f'<text x="{px+20}" y="{ay+22}" font-family="sans-serif" font-size="11.5" fill="{ASC}">'
          f'CUBE scope.scope + VECTOR scope.scope</text>')
L.append(f'<text x="{px+20}" y="{ay+52}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="{INK}">同步原语</text>')
L.append(f'<text x="{px+20}" y="{ay+74}" font-family="sans-serif" font-size="11.5" fill="{ASC}">'
          f'sync_block_set / sync_block_wait</text>')
L.append(f'<text x="{px+20}" y="{ay+104}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="{INK}">事件旗池</text>')
L.append(f'<text x="{px+20}" y="{ay+126}" font-family="sans-serif" font-size="11.5" fill="{ASC}">'
          f'14 个 flag id(syncFlag%14 循环复用)</text>')
L.append(f'<text x="{px+20}" y="{ay+156}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="{INK}">数据搬运</text>')
L.append(f'<text x="{px+20}" y="{ay+178}" font-family="sans-serif" font-size="11.5" fill="{ASC}">'
          f'显式跨地址空间(fixpipe/copy),因跨物理核、无共享内存</text>')

# ---- RIGHT: Hopper (forward reference to ch31 > ch17, must use "预告" style) ----
px2 = PAD + PANEL_W + GAP
cx2 = px2 + PANEL_W / 2
L.append(f'<text x="{cx2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14.5" font-weight="bold" fill="{HOP}">Hopper(基座对位)</text>')
# 外部对照样式:虚线边框 + 顶部标签「对位·姊妹篇基座」(不带 ch 号,避免与本书自己
# 的 ch31——torch_npu/mindspore 策略注册表,另一主题——混淆)
L.append(f'<rect x="{px2}" y="{TOP}" width="{PANEL_W}" height="300" rx="10" '
          f'fill="{HOP_BG}" stroke="{HOP}" stroke-width="1.8" stroke-dasharray="8,5"/>')
BADGE_TEXT = "对位 · 姊妹篇基座"
# 宽度按字符估算:CJK/中点约 1em,英文/空格约 0.55em;字号 11、粗体、左右各留 16px 内边距
_cjk_n = sum(1 for c in BADGE_TEXT if ord(c) > 0x2E7F)
_other_n = len(BADGE_TEXT) - _cjk_n
BADGE_W = round(_cjk_n * 11 * 1.05 + _other_n * 11 * 0.55) + 32
L.append(f'<rect x="{px2+PANEL_W-BADGE_W}" y="{TOP-16}" width="{BADGE_W}" height="26" rx="13" fill="{PREVIEW}"/>')
L.append(f'<text x="{px2+PANEL_W-BADGE_W/2}" y="{TOP+2}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="white">{esc(BADGE_TEXT)}</text>')
by = TOP + 26
L.append(f'<text x="{px2+20}" y="{by}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="{INK}">分域:同一 SM 内</text>')
L.append(f'<text x="{px2+20}" y="{by+22}" font-family="sans-serif" font-size="11.5" fill="{HOP}">'
          f'producer warp 组 + consumer warp 组</text>')
L.append(f'<text x="{px2+20}" y="{by+52}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="{INK}">同步原语</text>')
L.append(f'<text x="{px2+20}" y="{by+74}" font-family="sans-serif" font-size="11.5" fill="{HOP}">'
          f'共享内存 mbarrier</text>')
L.append(f'<text x="{px2+20}" y="{by+104}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="{INK}">事件资源</text>')
L.append(f'<text x="{px2+20}" y="{by+126}" font-family="sans-serif" font-size="11.5" fill="{HOP}">'
          f'barrier 对象(SM 内共享内存分配)</text>')
L.append(f'<text x="{px2+20}" y="{by+156}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="{INK}">数据搬运</text>')
L.append(f'<text x="{px2+20}" y="{by+178}" font-family="sans-serif" font-size="11.5" fill="{HOP}">'
          f'同一 SM 共享内存直接可见,无需跨地址空间搬运</text>')

CAP1 = "同一个「生产者/消费者分工 + 同步」思想，两种硬件两套物化：昇腾是两颗异构物理核跨核 + block 级事件 +"
CAP2 = "显式跨地址空间数据搬运；跨物理核这一点决定了昇腾必须显式搬数据而非共享内存。"
cap_y = TOP + 340
L.append(f'<text x="{PAD}" y="{cap_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP1)}</text>')
L.append(f'<text x="{PAD}" y="{cap_y+22}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP2)}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m16-warp-vs-scope.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
