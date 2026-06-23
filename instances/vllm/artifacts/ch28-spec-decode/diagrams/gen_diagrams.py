"""ch28 投机解码插图生成器。普通 sans-serif，rsvg-convert 逐字 CJK 回退。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


def header(w, h, arrow_fill="#64748b"):
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
    L.append(
        f'<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
        f'markerWidth="7" markerHeight="5" orient="auto">'
        f'<path d="M0,0 L10,3 L0,6 Z" fill="{arrow_fill}"/></marker></defs>'
    )
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    return L


def box(L, x, y, w, h, fill, stroke, label, sub=None, fs=15, tcol="#1e293b", rx=8):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    cy = y + h / 2 + (fs * 0.35 if sub is None else -2)
    L.append(f'<text x="{x + w/2}" y="{cy}" text-anchor="middle" font-size="{fs}" fill="{tcol}" font-weight="bold">{esc(label)}</text>')
    if sub:
        L.append(f'<text x="{x + w/2}" y="{cy + fs + 3}" text-anchor="middle" font-size="{fs-3}" fill="#475569">{esc(sub)}</text>')


def text(L, x, y, s, fs=13, col="#334155", anchor="middle", weight="normal"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" fill="{col}" font-weight="{weight}">{esc(s)}</text>')


def harrow(L, x1, x2, y, col="#64748b"):
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{col}" stroke-width="1.8" marker-end="url(#a)"/>')


# ======================================================================
# 01 — 投机解码流水线全景
# ======================================================================
def diagram_01():
    w, h = 1240, 360
    L = header(w, h)
    text(L, w / 2, 30, "投机解码一轮：一次目标前向产出 k+1 个 token", fs=18, col="#0f172a", weight="bold")

    bw, bh = 178, 78
    gap = 22
    x0 = 26
    ytop = 90
    stages = [
        ("proposer 产草稿", "ngram / EAGLE / MTP", "#dbeafe", "#3b82f6"),
        ("摊平 + 建 index", "SpecDecodeMetadata", "#e0e7ff", "#6366f1"),
        ("目标模型一次前向", "扁平 logits", "#fef9c3", "#ca8a04"),
        ("切 target / bonus", "RejectionSampler.forward", "#dcfce7", "#16a34a"),
        ("逐位接受 / 拒绝", "rejection_sample", "#fee2e2", "#dc2626"),
        ("还原变长", "parse_output", "#f3e8ff", "#9333ea"),
    ]
    shapes = [
        "list[list[int]]",
        "draft [num_tokens]",
        "[num_tokens+batch, vocab]",
        "target / bonus logits",
        "[batch, max_spec_len+1]",
        "list[list[int]]",
    ]
    xs_ = []
    for i, (lab, sub, fill, st) in enumerate(stages):
        x = x0 + i * (bw + gap)
        xs_.append(x)
        box(L, x, ytop, bw, bh, fill, st, lab, sub, fs=14)
        text(L, x + bw / 2, ytop + bh + 26, shapes[i], fs=11, col="#64748b")
        if i > 0:
            harrow(L, xs_[i - 1] + bw, x, ytop + bh / 2)

    # 注脚说明 bonus / k+1
    text(L, w / 2, 300, "全 k 个草稿被接受时，最后一行 logits 顺手再采一个 bonus token —— 输出第二维 max_spec_len+1 的那个 +1",
         fs=13, col="#7c2d12")
    text(L, w / 2, 326, "承接 ch27 采样层（target/bonus 都过普通 sampler）与 ch25 的 MTP draft 头", fs=12, col="#64748b")
    return w, h, L


# ======================================================================
# 02 — index 间接（用源码注释里的真实数字）
# ======================================================================
def diagram_02():
    w, h = 1260, 560
    L = header(w, h)
    text(L, w / 2, 30, "index 间接：把 5 个请求的变长草稿摊平进一条扁平 logits", fs=18, col="#0f172a", weight="bold")
    text(L, w / 2, 54, "num_draft_tokens = [3, 0, 2, 0, 1]   （req1、req3 草稿数为 0）", fs=14, col="#475569")

    # 11 行扁平 logits（每请求 num_draft+1 行）；左留白给 index 标签
    n = 11
    cw = 78
    x0 = 300
    yrow = 110
    rh = 56
    # 请求归属：req0 占 0..3, req1 占 4, req2 占 5..7, req3 占 8, req4 占 9..10
    req_of = [0, 0, 0, 0, 1, 2, 2, 2, 3, 4, 4]
    is_bonus = [False, False, False, True, True, False, False, True, True, False, True]
    req_fill = ["#dbeafe", "#fef3c7", "#dcfce7", "#fae8ff", "#fee2e2"]
    req_stroke = ["#3b82f6", "#d97706", "#16a34a", "#a21caf", "#dc2626"]
    for i in range(n):
        x = x0 + i * cw
        r = req_of[i]
        fill = "#bfdbfe" if is_bonus[i] else req_fill[r]
        # bonus 用更深底 + 蓝边突出
        st = "#2563eb" if is_bonus[i] else req_stroke[r]
        L.append(f'<rect x="{x}" y="{yrow}" width="{cw-6}" height="{rh}" rx="6" fill="{fill}" stroke="{st}" stroke-width="1.6"/>')
        text(L, x + (cw - 6) / 2, yrow + 24, f"row {i}", fs=13, col="#0f172a", weight="bold")
        tag = "bonus" if is_bonus[i] else "draft"
        text(L, x + (cw - 6) / 2, yrow + 43, f"req{r} {tag}", fs=11, col="#475569")

    text(L, x0 - 14, yrow + rh / 2 + 5, "扁平 logits", fs=13, col="#334155", anchor="end")
    text(L, x0 - 14, yrow + rh / 2 + 22, "11 = 6+5", fs=11, col="#94a3b8", anchor="end")

    # target_logits_indices = [0,1,2,5,6,9] -> 指向草稿行
    tgt = [0, 1, 2, 5, 6, 9]
    ty = yrow + rh + 120
    text(L, x0 - 14, ty, "target_logits_indices", fs=13, col="#16a34a", anchor="end", weight="bold")
    text(L, x0 - 14, ty + 18, "[0,1,2,5,6,9]", fs=12, col="#64748b", anchor="end")
    for idx in tgt:
        x = x0 + idx * cw + (cw - 6) / 2
        L.append(f'<line x1="{x}" y1="{yrow + rh}" x2="{x}" y2="{ty - 14}" stroke="#16a34a" stroke-width="1.8" marker-end="url(#a)"/>')

    # bonus_logits_indices = [3,4,7,8,10]
    by = ty + 80
    bon = [3, 4, 7, 8, 10]
    text(L, x0 - 14, by, "bonus_logits_indices", fs=13, col="#2563eb", anchor="end", weight="bold")
    text(L, x0 - 14, by + 18, "[3,4,7,8,10]", fs=12, col="#64748b", anchor="end")
    for idx in bon:
        x = x0 + idx * cw + (cw - 6) / 2
        # 从行底走一条折线到 bonus 标注带
        L.append(f'<line x1="{x}" y1="{yrow + rh}" x2="{x}" y2="{by - 14}" stroke="#2563eb" stroke-width="1.6" stroke-dasharray="5,4" marker-end="url(#a)"/>')

    text(L, w / 2, by + 70, "草稿数为 0 的 req1、req3 在 target_logits_indices 里没有任何条目 —— 只在 bonus 带里各占一格",
         fs=13, col="#7c2d12")
    text(L, w / 2, by + 94, "每请求要采 num_draft+1 个位置：cu_num_sampled_tokens-1 正好落在每段最后一行 = bonus 位",
         fs=12, col="#64748b")
    return w, h, L


# ======================================================================
# 03 — 逐位置接受/拒绝状态机
# ======================================================================
def diagram_03():
    w, h = 1180, 470
    L = header(w, h)
    text(L, w / 2, 30, "单请求逐位置接受 / 拒绝：首个拒绝即截断，全接受补 bonus", fs=18, col="#0f172a", weight="bold")

    # 上排：random 路径，4 个草稿位 + bonus 槽
    def lane(yc, title, cells, note):
        text(L, 40, yc - 44, title, fs=15, col="#0f172a", anchor="start", weight="bold")
        cw, ch = 150, 64
        x0 = 90
        gap = 18
        for i, (lab, sub, fill, st) in enumerate(cells):
            x = x0 + i * (cw + gap)
            box(L, x, yc - 30, cw, ch, fill, st, lab, sub, fs=14)
            if i > 0:
                harrow(L, x0 + i * (cw + gap) - gap, x, yc + 2)
        text(L, w / 2, yc + 58, note, fs=12.5, col="#475569")

    rnd = [
        ("pos0 accept", "p_t/p_d ≥ u", "#dcfce7", "#16a34a"),
        ("pos1 accept", "p_t/p_d ≥ u", "#dcfce7", "#16a34a"),
        ("pos2 REJECT", "取 recovered，停", "#fee2e2", "#dc2626"),
        ("pos3 = -1", "PLACEHOLDER", "#f1f5f9", "#94a3b8"),
        ("无 bonus", "rejected=True", "#f1f5f9", "#94a3b8"),
    ]
    lane(140, "random 路径（rejection_random_sample_kernel）", rnd,
         "接受准则 accepted = (target_prob/draft_prob ≥ uniform)；拒绝处写 recovered，后续位保留 -1，不补 bonus")

    grd = [
        ("pos0 accept", "draft==argmax", "#dcfce7", "#16a34a"),
        ("pos1 accept", "draft==argmax", "#dcfce7", "#16a34a"),
        ("pos2 accept", "draft==argmax", "#dcfce7", "#16a34a"),
        ("全部接受", "rejected=False", "#dbeafe", "#3b82f6"),
        ("+ bonus", "采自最后行", "#bfdbfe", "#2563eb"),
    ]
    lane(330, "greedy 路径（rejection_greedy_sample_kernel）", grd,
         "draft_token_id == target_argmax_id 即接受；3 个草稿全中 → 在第 num_draft 槽追加 bonus → 这一轮吐 4 个 token")

    text(L, w / 2, 440, "两条 kernel 用 is_greedy mask 在同一 batch 内共存：各自跳过不属于自己的请求", fs=13, col="#7c2d12")
    return w, h, L


# ======================================================================
# 04 — recovered 残差分布
# ======================================================================
def diagram_04():
    w, h = 1180, 480
    L = header(w, h)
    text(L, w / 2, 30, "拒绝时从残差分布 (p_target − p_draft)+ 采 recovered token", fs=18, col="#0f172a", weight="bold")
    text(L, w / 2, 54, "接受覆盖公共质量 min(p,q)，残差补上 p 超出 q 的部分 —— 拼起来正好是目标分布 p", fs=13, col="#475569")

    vocab = ["a", "b", "c", "d", "e"]
    q = [0.10, 0.35, 0.30, 0.15, 0.10]   # draft
    p = [0.30, 0.15, 0.30, 0.05, 0.20]   # target
    n = len(vocab)
    bw = 60
    grp = 200
    x0 = 140
    baseTop = 230
    scale = 320

    def bars(yb, vals, fill, st, title):
        text(L, 70, yb - scale - 4, title, fs=14, col=st, anchor="start", weight="bold")
        # 轴
        L.append(f'<line x1="{x0-20}" y1="{yb}" x2="{x0 + (n-1)*grp + bw + 30}" y2="{yb}" stroke="#cbd5e1" stroke-width="1.2"/>')
        for i, v in enumerate(vals):
            x = x0 + i * grp
            bh = v * scale
            L.append(f'<rect x="{x}" y="{yb - bh}" width="{bw}" height="{bh}" rx="3" fill="{fill}" stroke="{st}" stroke-width="1.3"/>')
            text(L, x + bw / 2, yb + 18, vocab[i], fs=13, col="#334155")
            text(L, x + bw / 2, yb - bh - 6, f"{v:.2f}", fs=11, col=st)

    # 上：p_draft 与 p_target 并列（用偏移区分）
    text(L, 70, 90, "p_draft (q) 与 p_target (p)", fs=14, col="#0f172a", anchor="start", weight="bold")
    yb1 = baseTop
    L.append(f'<line x1="{x0-20}" y1="{yb1}" x2="{x0 + (n-1)*grp + bw + 70}" y2="{yb1}" stroke="#cbd5e1" stroke-width="1.2"/>')
    for i in range(n):
        x = x0 + i * grp
        # q 蓝
        bhq = q[i] * scale
        L.append(f'<rect x="{x}" y="{yb1 - bhq}" width="{bw}" height="{bhq}" rx="3" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.3"/>')
        text(L, x + bw / 2, yb1 - bhq - 6, f"{q[i]:.2f}", fs=10, col="#3b82f6")
        # p 橙
        xp = x + bw + 8
        bhp = p[i] * scale
        L.append(f'<rect x="{xp}" y="{yb1 - bhp}" width="{bw}" height="{bhp}" rx="3" fill="#fed7aa" stroke="#ea580c" stroke-width="1.3"/>')
        text(L, xp + bw / 2, yb1 - bhp - 6, f"{p[i]:.2f}", fs=10, col="#ea580c")
        text(L, x + bw + 4, yb1 + 18, vocab[i], fs=13, col="#334155")
    # 图例
    L.append(f'<rect x="{w-260}" y="78" width="16" height="16" fill="#dbeafe" stroke="#3b82f6"/>')
    text(L, w - 238, 91, "q = p_draft", fs=12, col="#334155", anchor="start")
    L.append(f'<rect x="{w-140}" y="78" width="16" height="16" fill="#fed7aa" stroke="#ea580c"/>')
    text(L, w - 118, 91, "p = p_target", fs=12, col="#334155", anchor="start")

    # 下：残差
    yb2 = 430
    text(L, 70, yb2 - 150 - 4, "残差 (p − q)+   （归一化后即 recovered 的采样分布）", fs=14, col="#9333ea", anchor="start", weight="bold")
    L.append(f'<line x1="{x0-20}" y1="{yb2}" x2="{x0 + (n-1)*grp + bw + 70}" y2="{yb2}" stroke="#cbd5e1" stroke-width="1.2"/>')
    resid = [max(p[i] - q[i], 0.0) for i in range(n)]
    rscale = 320
    for i in range(n):
        x = x0 + i * grp + bw / 2 + 4
        v = resid[i]
        bh = v * rscale
        fill = "#f3e8ff" if v > 0 else "#f1f5f9"
        st = "#9333ea" if v > 0 else "#cbd5e1"
        L.append(f'<rect x="{x}" y="{yb2 - bh}" width="{bw}" height="{max(bh,1)}" rx="3" fill="{fill}" stroke="{st}" stroke-width="1.3"/>')
        text(L, x + bw / 2, yb2 + 18, vocab[i], fs=13, col="#334155")
        text(L, x + bw / 2, yb2 - bh - 6, f"{v:.2f}", fs=11, col=st)
    text(L, w / 2, yb2 + 42, "Gumbel-max：score = prob × inv_q（inv_q=1/Exp(1)），argmax(score) 等价于按残差分布采样，省去显式归一化",
         fs=12.5, col="#7c2d12")
    return w, h, L


for name, fn in [
    ("01-spec-decode-pipeline", diagram_01),
    ("02-index-indirection", diagram_02),
    ("03-rejection-accept-reject", diagram_03),
    ("04-recovered-residual-dist", diagram_04),
]:
    w, h, L = fn()
    L.append("</svg>")
    out = "/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch28-spec-decode/diagrams/" + name + ".svg"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("wrote", out)
