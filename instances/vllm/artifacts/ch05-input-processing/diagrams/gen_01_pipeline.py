#!/usr/bin/env python3
"""01-input-processing-pipeline: process_inputs() 竖向流水线（prompt → EngineCoreRequest）。"""
import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs


def esc(s): return xs.escape(s)


C_BOX = "#ffffff"
C_BD = "#475569"
C_TXT = "#1e293b"
C_MUT = "#64748b"
C_VALID = "#fef3c7"
C_VALID_BD = "#d97706"
C_BRANCH = "#eef2ff"
C_BRANCH_BD = "#6366f1"
C_OUT = "#dcfce7"
C_OUT_BD = "#16a34a"
C_ID = "#dbeafe"
C_ID_BD = "#2563eb"


def box(L, x, y, w, h, title, sub="", src="", fill=C_BOX, bd=C_BD, tcol=C_TXT, fs=13):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{bd}" stroke-width="2"/>')
    lines = [l for l in [sub, src] if l]
    cy = y + (h // 2 if not lines else h // 2 - 6 * len(lines))
    L.append(f'<text x="{x+w//2}" y="{cy+5}" text-anchor="middle" font-family="sans-serif" font-size="{fs}" font-weight="bold" fill="{tcol}">{esc(title)}</text>')
    off = cy + 21
    if sub:
        L.append(f'<text x="{x+w//2}" y="{off}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="{C_MUT}">{esc(sub)}</text>')
        off += 15
    if src:
        L.append(f'<text x="{x+w//2}" y="{off}" text-anchor="middle" font-family="monospace" font-size="9.5" fill="{C_MUT}">{esc(src)}</text>')


def arrow(L, x1, y1, x2, y2, mid="", mid_anchor="middle", mid_dx=8, color=C_BD):
    L.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="2.2" marker-end="url(#a)"/>')
    if mid:
        my = (y1 + y2) // 2 + 4
        L.append(f'<text x="{(x1+x2)//2+mid_dx}" y="{my}" text-anchor="{mid_anchor}" font-family="monospace" font-size="10" fill="{color}">{esc(mid)}</text>')


def build():
    w, h = 1020, 1180
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    L.append(f'<text x="30" y="36" font-family="sans-serif" font-size="20" font-weight="bold" fill="{C_TXT}">{esc("Stage 1 输入处理流水线：process_inputs() 把 prompt 变成 EngineCoreRequest")}</text>')
    L.append(f'<text x="30" y="58" font-family="monospace" font-size="11" fill="{C_MUT}">{esc("vllm/v1/engine/input_processor.py")}</text>')

    cx = w // 2
    bw = 440
    bx = cx - bw // 2

    # 入口
    box(L, bx, 80, bw, 50, "用户 prompt + SamplingParams / PoolingParams", "supported_tasks / lora_request / data_parallel_rank", fill="#f1f5f9", fs=13)

    # 校验三连
    arrow(L, cx, 130, cx, 165, color=C_MUT)
    box(L, bx, 165, bw, 64, "_validate_params + _validate_lora", "生成/池化任务路由 → params.verify；LoRA 一致性", "L234-L249 / L81 / L138", fill=C_VALID, bd=C_VALID_BD)
    arrow(L, cx, 229, cx, 264, color=C_MUT)
    box(L, bx, 264, bw, 50, "校验 data_parallel_rank ∈ [0, num_ranks)", "", "L251-L259", fill=C_VALID, bd=C_VALID_BD)

    # 分流
    q = '"type"'
    L.append(f'<text x="{cx}" y="345" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="bold" fill="{C_BRANCH_BD}">{esc("prompt 是带 " + q + " 的 dict 吗？")}</text>')
    box(L, 80, 365, 380, 70, "是：已渲染 EngineInput", "Renderer 已 tokenize/多模态 → 直接透传", "L261-L272", fill=C_BRANCH, bd=C_BRANCH_BD, fs=12)
    box(L, 560, 365, 380, 70, "否：raw prompt（deprecated）", "现场 input_preprocessor.preprocess() 兜底 tokenize", "L273-L286 → preprocess.py", fill="#fde8e8", bd="#dc2626", fs=12)
    arrow(L, cx, 314, 270, 365, color=C_BRANCH_BD)
    arrow(L, cx, 314, 750, 365, color="#dc2626")
    arrow(L, 270, 435, cx, 475, color=C_MUT)
    arrow(L, 750, 435, cx, 475, color=C_MUT)

    # 平台校验 + 拆分 + 模型输入校验
    box(L, bx, 475, bw, 50, "current_platform.validate_request(...)", "平台级请求校验", "L288", fill=C_VALID, bd=C_VALID_BD)
    box(L, bx, 555, bw, 50, "split_enc_dec_input → (encoder, decoder)", "", "L290", fill=C_BOX)
    box(L, bx, 635, bw, 70, "_validate_model_inputs", "空 prompt / 超 max_model_len / 等长 / token id 越界 / 编码器缓存超限", "L291 → L426 / L379", fill=C_VALID, bd=C_VALID_BD, fs=12)

    arrow(L, cx, 525, cx, 555, color=C_MUT)
    arrow(L, cx, 605, cx, 635, color=C_MUT)

    # 取 prompt 字段
    box(L, bx, 735, bw, 56, "取 prompt_token_ids / prompt_embeds / is_token_ids", '按 decoder_inputs["type"]=="embeds" 区分', "L293-L301", fill=C_BOX)
    arrow(L, cx, 705, cx, 735, color=C_MUT)

    # 参数补全
    box(L, bx, 821, bw, 70, "SamplingParams.clone() 后补全", "max_tokens←max_model_len-seq_len；eos/stop；bad_words", "L303-L322", fill="#e0e7ff", bd=C_BRANCH_BD, fs=12)
    arrow(L, cx, 791, cx, 821, color=C_MUT)

    # 多模态展平
    box(L, bx, 921, bw, 64, '多模态展平（type=="multimodal"）', "argsort_mm_positions → list[MultiModalFeatureSpec]", "L324-L360", fill="#e0e7ff", bd=C_BRANCH_BD, fs=12)
    arrow(L, cx, 891, cx, 921, color=C_MUT)

    # 组装
    box(L, bx, 1015, bw, 56, "return EngineCoreRequest(...)", "cache_salt / priority / trace_headers 透传", "L362-L377", fill=C_OUT, bd=C_OUT_BD)
    arrow(L, cx, 985, cx, 1015, color=C_MUT)

    # assign_request_id（在 add_request 里，后置）
    box(L, bx, 1101, bw, 56, "assign_request_id：注入 8 字符随机后缀", "request_id → \"{external}-{随机}\"", "L214-L232", fill=C_ID, bd=C_ID_BD)
    arrow(L, cx, 1071, cx, 1101, "由 add_request() 调用", mid_anchor="middle", mid_dx=0, color=C_ID_BD)

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
