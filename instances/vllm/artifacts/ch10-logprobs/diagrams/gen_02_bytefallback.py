#!/usr/bin/env python3
"""02-byte-fallback-reconstruction: _correct_decoded_token 逐步追踪表。
以 "中" = e4 b8 ad 三字节为例，上下文 [X(干净), e4, b8]，完成 token = ad。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 940, 470
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="30" font-family="sans-serif" font-size="18" font-weight="bold" '
         f'text-anchor="middle" fill="#0f172a">_correct_decoded_token：上下文增长 → 拼出完整字符 → 剥干净前缀</text>')
L.append(f'<text x="{W/2}" y="52" font-family="sans-serif" font-size="13" text-anchor="middle" fill="#475569">'
         f'例：完成 token = 0xAD，上下文 context_token_ids = [X(0x58, 干净), 0xE4, 0xB8]　目标字符「中」= E4 B8 AD</text>')

# 表格
cols = ["num_ctx", "context (取末 num_ctx 个)", "decode(context+[0xAD])", "结尾 �?", "回溯 clean_end", "clean_prefix", "返回值"]
cw = [70, 175, 175, 70, 130, 110, 110]
x0, y0 = 20, 75
rh = 64
cx = [x0]
for w in cw:
    cx.append(cx[-1] + w)

# 表头
L.append(f'<rect x="{x0}" y="{y0}" width="{sum(cw)}" height="34" fill="#1e293b"/>')
for i, c in enumerate(cols):
    L.append(f'<text x="{cx[i]+cw[i]/2}" y="{y0+22}" font-family="sans-serif" font-size="12.5" '
             f'font-weight="bold" text-anchor="middle" fill="white">{esc(c)}</text>')

rows = [
    ("1", "[0xB8]", "decode([B8,AD])", "是 �", "—", "—", "continue", "#fee2e2", "#dc2626"),
    ("2", "[0xE4, 0xB8]", "decode([E4,B8,AD])", "否 → '中'", "E4,B8 单解=� → clean_end=0", "'' (空)", "'中'", "#dcfce7", "#15803d"),
]
# 但实际上下文是 [X,E4,B8]，max_ctx=min(3,4)=3，逐步增长。补一行 num_ctx=3 终态。
rows = [
    ("1", "[0xB8]", "decode([B8, AD])", "是 �", "—（continue）", "—", "继续增大 ctx", "#fee2e2", "#b91c1c"),
    ("2", "[0xE4, 0xB8]", "decode([E4, B8, AD])", "否 → 「中」", "E4、B8 单解皆 � → clean_end=0", "'' 空串", "「中」", "#dcfce7", "#15803d"),
]
yy = y0 + 34
for r in rows:
    nctx, ctx, dec, end, clean, prefix, ret, bg, retcol = r
    L.append(f'<rect x="{x0}" y="{yy}" width="{sum(cw)}" height="{rh}" fill="{bg}" stroke="#cbd5e1" stroke-width="1"/>')
    vals = [nctx, ctx, dec, end, clean, prefix, ret]
    for i, v in enumerate(vals):
        # 多行：clean 列可能较长，简单按字符断
        col = "#0f172a"
        wcol = "normal"
        if i == 6:
            col = retcol
            wcol = "bold"
        # wrap long
        maxc = max(8, int(cw[i] / 7.5))
        words = v
        chunks = []
        cur = ""
        for ch in words:
            cur += ch
            if len(cur) >= maxc and (ch in " ,)]"):
                chunks.append(cur)
                cur = ""
        if cur:
            chunks.append(cur)
        if not chunks:
            chunks = [v]
        n = len(chunks)
        cyt = yy + rh/2 - (n-1)*15/2 + 5
        for k, ch in enumerate(chunks):
            L.append(f'<text x="{cx[i]+cw[i]/2}" y="{cyt + k*15}" font-family="sans-serif" '
                     f'font-size="12" font-weight="{wcol}" text-anchor="middle" fill="{col}">{esc(ch.strip())}</text>')
    yy += rh

# 列分隔线
for i in range(1, len(cw)):
    L.append(f'<line x1="{cx[i]}" y1="{y0}" x2="{cx[i]}" y2="{yy}" stroke="#cbd5e1" stroke-width="1"/>')

# 终态说明
ty = yy + 30
L.append(f'<text x="{x0}" y="{ty}" font-family="sans-serif" font-size="14" font-weight="bold" fill="#0f172a">归属结论</text>')
notes = [
    "· 前两个字节 token（E4、B8）单独解码 = U+FFFD，当时各自记为空串 ''。",
    "· 完成 token（AD）凭增长的上下文拼出「中」，剥掉干净前缀后，整个「中」归属它一个。",
    "· max_ctx = min(len(context), 4)：UTF-8 单字符至多 4 字节，4 个上下文 token 必能拼全。",
    "· 若到 4 个上下文仍以 � 结尾，说明此处字节序列确实不完整 → 返回 ''，留给后续完成 token 补出。",
]
for i, t in enumerate(notes):
    L.append(f'<text x="{x0+8}" y="{ty+26+i*22}" font-family="sans-serif" font-size="12.5" fill="#334155">{esc(t)}</text>')

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch10-logprobs/diagrams/02-byte-fallback-reconstruction.svg", "w").write('\n'.join(L))
print("ok")
