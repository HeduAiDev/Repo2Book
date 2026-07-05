#!/usr/bin/env python3
"""fig29-1: 工厂分发 —— get_spec_decode_method 一处 if-elif 把 method 字符串映射到 8 个 proposer，
右侧按三类着色：纯薄壳 / 中等薄壳·no-op / 走重量级 base。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1480, 700
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=9, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=14, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


text(W / 2, 40, "一处工厂分发：method 字符串 → 8 个 Ascend*Proposer", 23, "middle", "#0f172a", "bold")
text(W / 2, 66, "绿 = 纯薄壳（提议成本近零）   橙 = 中等薄壳 / no-op   红 = 走重量级 base（要跑 draft 前向）", 13.5, "middle", "#64748b")

# 列标题
text(150, 104, 'speculative_config.method', 14, "middle", "#475569", "bold", mono=True)
text(W / 2, 104, "get_spec_decode_method", 14, "middle", "#475569", "bold", mono=True)
text(1130, 104, "返回的 proposer 实例", 14, "middle", "#475569", "bold")

# 8 行：method 字符串 → 类 ，category: g/a/r
rows = [
    ('"ngram"', "AscendNgramProposer", "g", "CPU n-gram，调父类 batch_propose"),
    ('"ngram_gpu"', "AscendNgramProposerNPU", "a", "继承 GPU 类但三方法全 no-op"),
    ('"suffix"', "AscendSuffixDecodingProposer", "g", "propose 一行转发父类"),
    ('"medusa"', "AscendMedusaProposer", "a", "gather 末位 hidden + super()"),
    ('"eagle" / "eagle3" / "mtp"', "AscendEagleProposer", "r", "三 method 共用一类"),
    ('"dflash"', "AscendDflashProposer", "r", "继承 AscendEagleProposer 再叠层"),
    ('"draft_model"', "AscendDraftModelProposer", "r", "pass_hidden_states=False + 校验"),
    ('"extract_hidden_states"', "AscendExtractHiddenStatesProposer", "a", "只抽取 hidden，不真投机"),
]

palette = {
    "g": ("#ecfdf5", "#10b981", "#047857"),
    "a": ("#fffbeb", "#f59e0b", "#b45309"),
    "r": ("#fef2f2", "#ef4444", "#b91c1c"),
}

# 中间 dispatcher 竖条
disp_x, disp_y, disp_w = 600, 130, 140
top_y = 140
row_h = 56
gap = 8
n = len(rows)
disp_h = n * (row_h + gap) - gap + 16
box(disp_x, top_y - 8, disp_w, disp_h, "#f1f5f9", "#94a3b8", 12, 1.8)
text(disp_x + disp_w / 2, top_y + disp_h / 2 - 28, "if method ==", 13, "middle", "#475569", "bold", mono=True)
text(disp_x + disp_w / 2, top_y + disp_h / 2 - 6, "elif … elif …", 13, "middle", "#475569", "bold", mono=True)
text(disp_x + disp_w / 2, top_y + disp_h / 2 + 16, "else: raise", 12.5, "middle", "#94a3b8", "normal", mono=True)
text(disp_x + disp_w / 2, top_y + disp_h / 2 + 34, "ValueError", 12.5, "middle", "#94a3b8", "normal", mono=True)

for i, (m, cls, cat, note) in enumerate(rows):
    y = top_y + i * (row_h + gap)
    # 左：method 字符串
    box(40, y, 230, row_h, "#f8fafc", "#cbd5e1", 8, 1.4)
    text(155, y + row_h / 2 + 5, m, 13.5, "middle", "#334155", "bold", mono=True)
    # 左→中 箭头
    L.append(f'<line x1="270" y1="{y + row_h/2}" x2="{disp_x}" y2="{y + row_h/2}" stroke="#cbd5e1" stroke-width="1.4" marker-end="url(#ar)"/>')
    # 右：proposer 类
    fill, stroke, txt = palette[cat]
    rx = disp_x + disp_w + 60
    rw = 600
    box(rx, y, rw, row_h, fill, stroke, 8, 1.7)
    text(rx + 16, y + 23, cls, 14, "start", txt, "bold", mono=True)
    text(rx + 16, y + 43, note, 12, "start", "#64748b")
    # 中→右 箭头
    L.append(f'<line x1="{disp_x + disp_w}" y1="{y + row_h/2}" x2="{rx}" y2="{y + row_h/2}" stroke="#cbd5e1" stroke-width="1.4" marker-end="url(#ar)"/>')

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm-ascend/artifacts/ch29-speculative-decode-npu/diagrams/fig29-1-factory.svg", "w").write('\n'.join(L))
print("wrote fig29-1-factory.svg")
