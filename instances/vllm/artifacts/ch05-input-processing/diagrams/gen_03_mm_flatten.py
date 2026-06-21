#!/usr/bin/env python3
"""03-multimodal-flatten: dict-of-list → argsort_mm_positions(按 offset) → list[MultiModalFeatureSpec]。"""
import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs


def esc(s): return xs.escape(s)


C_BD = "#475569"
C_TXT = "#1e293b"
C_MUT = "#64748b"
C_IMG = "#dbeafe"
C_IMG_BD = "#2563eb"
C_AUD = "#fce7f3"
C_AUD_BD = "#db2777"
C_HDR = "#e2e8f0"


def cell(L, x, y, w, hgt, txt, fill="#ffffff", bd=C_BD, fs=11, bold=False, mono=True):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{hgt}" fill="{fill}" stroke="{bd}" stroke-width="1.5"/>')
    ff = "monospace" if mono else "sans-serif"
    fw = "bold" if bold else "normal"
    L.append(f'<text x="{x+w//2}" y="{y+hgt//2+4}" text-anchor="middle" font-family="{ff}" font-size="{fs}" font-weight="{fw}" fill="{C_TXT}">{esc(txt)}</text>')


def build():
    w, h = 1120, 720
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    L.append(f'<text x="30" y="34" font-family="sans-serif" font-size="19" font-weight="bold" fill="{C_TXT}">{esc("多模态展平：dict-of-list → 按 offset 排序 → list[MultiModalFeatureSpec]")}</text>')
    L.append(f'<text x="30" y="55" font-family="monospace" font-size="10" fill="{C_MUT}">{esc("input_processor.py:L324-L360 · argsort_mm_positions → vllm/multimodal/utils.py")}</text>')

    # ===== 左：dict-of-list（按 modality 分组，offset 交错） =====
    lx = 40
    L.append(f'<text x="{lx}" y="95" font-family="sans-serif" font-size="13" font-weight="bold" fill="{C_TXT}">{esc("输入：decoder_inputs（按 modality 分组）")}</text>')
    # image 组
    cell(L, lx, 110, 280, 28, 'mm_placeholders["image"]', fill=C_HDR, bd=C_IMG_BD, bold=True, mono=False)
    cell(L, lx, 138, 140, 30, "offset=5", fill=C_IMG, bd=C_IMG_BD)
    cell(L, lx + 140, 138, 140, 30, "offset=80", fill=C_IMG, bd=C_IMG_BD)
    L.append(f'<text x="{lx+70}" y="186" text-anchor="middle" font-family="monospace" font-size="9.5" fill="{C_MUT}">{esc("image[0]")}</text>')
    L.append(f'<text x="{lx+210}" y="186" text-anchor="middle" font-family="monospace" font-size="9.5" fill="{C_MUT}">{esc("image[1]")}</text>')
    # audio 组
    cell(L, lx, 215, 280, 28, 'mm_placeholders["audio"]', fill=C_HDR, bd=C_AUD_BD, bold=True, mono=False)
    cell(L, lx, 243, 280, 30, "offset=40", fill=C_AUD, bd=C_AUD_BD)
    L.append(f'<text x="{lx+140}" y="291" text-anchor="middle" font-family="monospace" font-size="9.5" fill="{C_MUT}">{esc("audio[0]")}</text>')
    L.append(f'<text x="{lx}" y="335" font-family="sans-serif" font-size="11" fill="{C_MUT}">{esc("mm_hashes / mm_kwargs 同样按 modality 分组")}</text>')

    # ===== 中：argsort =====
    mx = 410
    L.append(f'<rect x="{mx}" y="170" width="220" height="80" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="2"/>')
    L.append(f'<text x="{mx+110}" y="200" text-anchor="middle" font-family="monospace" font-size="13" font-weight="bold" fill="{C_TXT}">{esc("argsort_mm_positions")}</text>')
    L.append(f'<text x="{mx+110}" y="222" text-anchor="middle" font-family="sans-serif" font-size="11" fill="{C_MUT}">{esc("摊平 → 按 offset 升序")}</text>')
    L.append(f'<text x="{mx+110}" y="240" text-anchor="middle" font-family="monospace" font-size="10" fill="{C_MUT}">{esc("O(M log M)")}</text>')
    L.append(f'<path d="M320,210 L{mx},210" fill="none" stroke="{C_BD}" stroke-width="2.2" marker-end="url(#a)"/>')

    # 排序结果序列
    L.append(f'<text x="{mx+110}" y="280" text-anchor="middle" font-family="sans-serif" font-size="11" fill="{C_TXT}">{esc("排序键序列：")}</text>')
    L.append(f'<text x="{mx+110}" y="300" text-anchor="middle" font-family="monospace" font-size="10.5" fill="{C_TXT}">{esc("(image,0)=5 → (audio,0)=40")}</text>')
    L.append(f'<text x="{mx+110}" y="318" text-anchor="middle" font-family="monospace" font-size="10.5" fill="{C_TXT}">{esc("→ (image,1)=80")}</text>')

    # ===== 右：flatten list =====
    rx = 680
    L.append(f'<text x="{rx}" y="95" font-family="sans-serif" font-size="13" font-weight="bold" fill="{C_TXT}">{esc("输出：list[MultiModalFeatureSpec]（按序列位置有序）")}</text>')
    L.append(f'<path d="M630,210 L{rx-10},150" fill="none" stroke="{C_BD}" stroke-width="2.2" marker-end="url(#a)"/>')

    rows = [
        ("image", "offset=5", C_IMG, C_IMG_BD),
        ("audio", "offset=40", C_AUD, C_AUD_BD),
        ("image", "offset=80", C_IMG, C_IMG_BD),
    ]
    ry = 115
    for i, (mod, off, fill, bd) in enumerate(rows):
        y = ry + i * 100
        L.append(f'<rect x="{rx}" y="{y}" width="400" height="88" rx="6" fill="{fill}" stroke="{bd}" stroke-width="2"/>')
        L.append(f'<text x="{rx+14}" y="{y+24}" font-family="monospace" font-size="12" font-weight="bold" fill="{C_TXT}">{esc(f"MultiModalFeatureSpec  #{i}")}</text>')
        L.append(f'<text x="{rx+14}" y="{y+44}" font-family="monospace" font-size="10.5" fill="{C_TXT}">{esc(f"modality={mod}  mm_position.{off}")}</text>')
        L.append(f'<text x="{rx+14}" y="{y+62}" font-family="monospace" font-size="10.5" fill="{C_TXT}">{esc("data  mm_hash  identifier")}</text>')
        L.append(f'<text x="{rx+14}" y="{y+80}" font-family="monospace" font-size="9.5" fill="{C_MUT}">{esc("identifier = _get_mm_identifier(hash, lora)")}</text>')

    L.append(f'<text x="{rx+200}" y="425" text-anchor="middle" font-family="sans-serif" font-size="11" fill="{C_MUT}">{esc("下游单遍按序处理交错的图/音/视频")}</text>')

    # 底部说明条
    L.append(f'<rect x="40" y="455" width="1040" height="100" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
    L.append(f'<text x="58" y="481" font-family="sans-serif" font-size="13" font-weight="bold" fill="{C_TXT}">{esc("为什么要展平排序？")}</text>')
    L.append(f'<text x="58" y="506" font-family="sans-serif" font-size="12" fill="{C_TXT}">{esc("下游（EngineCore / 调度 / 编码器缓存）需按 item 在 prompt 中的真实位置有序处理；")}</text>')
    L.append(f'<text x="58" y="528" font-family="sans-serif" font-size="12" fill="{C_TXT}">{esc("不同模态可能交错出现，dict-of-list 无法表达全局顺序，按 PlaceholderRange.offset 升序统一成单条扁平列表。")}</text>')

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
