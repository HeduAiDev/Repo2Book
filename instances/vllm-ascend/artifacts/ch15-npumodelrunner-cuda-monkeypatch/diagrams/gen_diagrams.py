"""生成 ch14 三张配图：
  1) inherit-vs-rewrite —— ch13 钉死 vs ch14 可换接缝 的对照
  2) symbol-replacement —— 两个 wrapper 的符号替换表（作用域仅 with 内）
  3) seam-sequence    —— capture_model 的装/卸时间轴
全部坐标由 Python 计算，无手填魔数。"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(str(s))

FONT = 'font-family="sans-serif"'


def box(x, y, w, h, fill, stroke, rx=10, sw=2):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, s, size=15, anchor="middle", fill="#1e293b", weight="normal", mono=False):
    fam = 'font-family="monospace"' if mono else FONT
    return (f'<text x="{x}" y="{y}" {fam} font-size="{size}" '
            f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>')


# ----------------------------------------------------------------------------
# 图 1：钉死 vs 可换接缝（ch13 / ch14 对照）
# ----------------------------------------------------------------------------
def diagram_inherit_vs_rewrite(path):
    W, H = 1080, 560
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    L.append('<defs>'
             '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
             '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
             '</defs>')
    L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    L.append(text(W / 2, 38, "父类怎么表达设备差异，决定了子类继承还是重写", 22, weight="bold"))
    L.append(text(W / 2, 64, "同一个仓库，Worker 重写、ModelRunner 继承——分水岭在父类那一侧", 14, fill="#64748b"))

    cw, ch = 470, 410
    gx, gy = 40, 96
    nx = W - gx - cw

    # 左：ch13 Worker —— 钉死
    L.append(box(gx, gy, cw, ch, "#fef2f2", "#ef4444", sw=2.5))
    L.append(text(gx + cw / 2, gy + 32, "ch13 · GPU Worker（基座父类）", 17, weight="bold", fill="#b91c1c"))
    L.append(text(gx + cw / 2, gy + 56, "设备差异 = else-raise 钉死", 14, fill="#b91c1c"))
    # 代码块
    cby = gy + 78
    L.append(box(gx + 24, cby, cw - 48, 132, "#ffffff", "#fecaca", rx=6, sw=1.5))
    code_l = [
        ('if device_type == "cuda":', "#334155"),
        ('    self.device = "cuda:N"', "#334155"),
        ('    init_distributed("nccl")', "#334155"),
        ("else:", "#334155"),
        ("    raise RuntimeError(...)", "#b91c1c"),
    ]
    for i, (c, col) in enumerate(code_l):
        L.append(text(gx + 40, cby + 26 + i * 21, c, 12.5, anchor="start", fill=col,
                      weight="bold" if "raise" in c else "normal", mono=True))
    L.append(text(gx + cw / 2, cby + 168,
                  "昇腾 device_type==\"npu\" → 第一脚掉进 else", 13, fill="#b91c1c"))
    L.append(text(gx + cw / 2, cby + 192,
                  "无接缝可换 → 只能整段重写", 14, fill="#b91c1c", weight="bold"))
    # 结论
    L.append(box(gx + 70, gy + ch - 56, cw - 140, 38, "#fee2e2", "#ef4444", rx=8, sw=2))
    L.append(text(gx + cw / 2, gy + ch - 32, "NPUWorker(WorkerBase) · 平级重写", 14, weight="bold", fill="#991b1b"))

    # 右：ch14 ModelRunner —— 可换接缝
    L.append(box(nx, gy, cw, ch, "#f0fdf4", "#10b981", sw=2.5))
    L.append(text(nx + cw / 2, gy + 32, "ch14 · GPU ModelRunner（基座父类）", 17, weight="bold", fill="#047857"))
    L.append(text(nx + cw / 2, gy + 56, "设备差异 = 三类「可换接缝（seam）」", 14, fill="#047857"))
    seams = [
        ("① override 钩子", '_init_device_properties / _sync_device',
         "父类亲口标注 used for model runner override"),
        ("② 散落的 torch.cuda.*", 'mem_get_info / Event / Stream / synchronize',
         "进程级符号，可临时改向 torch.npu.*"),
        ("③ 模块级符号", 'graph_capture / CUDAGraphWrapper',
         "可 setattr 换成 NPU / ACL 同形版"),
    ]
    sby = gy + 76
    for i, (t, code, note) in enumerate(seams):
        yy = sby + i * 78
        L.append(box(nx + 24, yy, cw - 48, 66, "#ffffff", "#a7f3d0", rx=6, sw=1.5))
        L.append(text(nx + 40, yy + 22, t, 13.5, anchor="start", weight="bold", fill="#065f46"))
        L.append(text(nx + 40, yy + 41, code, 11.5, anchor="start", fill="#334155", mono=True))
        L.append(text(nx + 40, yy + 58, note, 11, anchor="start", fill="#64748b"))
    L.append(box(nx + 50, gy + ch - 56, cw - 100, 38, "#dcfce7", "#10b981", rx=8, sw=2))
    L.append(text(nx + cw / 2, gy + ch - 32,
                  "NPUModelRunner(GPUModelRunner) · 继承 + 临时猴补", 13.5, weight="bold", fill="#065f46"))

    L.append('</svg>')
    open(path, "w").write("\n".join(L))


# ----------------------------------------------------------------------------
# 图 2：符号替换表（两个 wrapper）
# ----------------------------------------------------------------------------
def diagram_symbol_replacement(path):
    W, H = 1080, 600
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    L.append('<defs>'
             '<marker id="map" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
             '<path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
             '</defs>')
    L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    L.append(text(W / 2, 38, "两个 wrapper 临时换掉哪些符号（作用域仅 with 内有效）", 22, weight="bold"))

    def table(x, y, w, title, subtitle, rows, accent, accent_bg):
        nonlocal L
        rowh = 34
        head = 70
        h = head + rowh * len(rows) + 16
        L.append(box(x, y, w, h, "#ffffff", accent, rx=10, sw=2.5))
        L.append(box(x, y, w, 46, accent_bg, accent, rx=10, sw=2.5))
        L.append(text(x + w / 2, y + 22, title, 15, weight="bold", fill=accent, mono=True))
        L.append(text(x + w / 2, y + 40, subtitle, 11.5, fill="#475569"))
        # 列标题
        cy = y + head - 6
        lx = x + 28
        rx = x + w - 28
        L.append(text(lx, cy, "原符号（父类引用）", 11.5, anchor="start", weight="bold", fill="#64748b"))
        L.append(text(rx, cy, "with 内临时指向", 11.5, anchor="end", weight="bold", fill="#64748b"))
        L.append(f'<line x1="{x+16}" y1="{cy+8}" x2="{x+w-16}" y2="{cy+8}" stroke="#e2e8f0" stroke-width="1"/>')
        for i, (a, b) in enumerate(rows):
            ry = cy + 30 + i * rowh
            L.append(text(lx, ry, a, 12.5, anchor="start", fill="#334155", mono=True))
            L.append(text(rx, ry, b, 12.5, anchor="end", fill=accent, mono=True, weight="bold"))
            # 中间映射箭头
            mxl = x + w * 0.46
            mxr = x + w * 0.56
            L.append(f'<line x1="{mxl}" y1="{ry-4}" x2="{mxr}" y2="{ry-4}" '
                     f'stroke="#7c3aed" stroke-width="1.6" marker-end="url(#map)"/>')
        return h

    colw = 500
    lx = 40
    rx = W - 40 - colw
    ty = 72
    rowsA = [
        ("torch.cuda.Event", "torch.npu.Event"),
        ("torch.cuda.Stream", "torch.npu.Stream"),
        ("torch.cuda.synchronize", "torch.npu.synchronize"),
        ("torch.cuda.mem_get_info", "torch.npu.mem_get_info"),
    ]
    hA = table(lx, ty, colw, "_torch_cuda_wrapper()",
               "进程级换 torch 设备符号（代表 4 个，源码共 8 个：7 torch.cuda.* + torch.Event）",
               rowsA, "#2563eb", "#dbeafe")
    rowsB = [
        ("<父模块>.graph_capture", "NPU 版 graph_capture"),
        ("<父模块>.CUDAGraphWrapper", "ACLGraphWrapper"),
    ]
    hB = table(rx, ty, colw, "_replace_gpu_model_runner_function_wrapper()",
               "换父类所在模块的模块级符号（经 MRO 定位）",
               rowsB, "#7c3aed", "#ede9fe")

    # 失败兜底 + 退出说明
    noteY = ty + max(hA, hB) + 36
    L.append(box(lx, noteY, colw, 96, "#fef9c3", "#eab308", rx=10, sw=2))
    L.append(text(lx + 20, noteY + 26, "失败兜底（except 分支）", 13.5, anchor="start", weight="bold", fill="#854d0e"))
    L.append(text(lx + 20, noteY + 48, "Event/Stream → _EventPlaceholder / _StreamPlaceholder", 11.5,
                  anchor="start", fill="#713f12", mono=True))
    L.append(text(lx + 20, noteY + 68, "record/wait/synchronize 全 no-op、query 恒 True", 11.5,
                  anchor="start", fill="#713f12"))
    L.append(text(lx + 20, noteY + 86, "→ 异常路径 torch.cuda.* 仍可调用不崩", 11.5,
                  anchor="start", fill="#854d0e", weight="bold"))

    L.append(box(rx, noteY, colw, 96, "#dcfce7", "#16a34a", rx=10, sw=2))
    L.append(text(rx + 20, noteY + 26, "可逆性（finally 分支）", 13.5, anchor="start", weight="bold", fill="#166534"))
    L.append(text(rx + 20, noteY + 48, "_replace：逐一 setattr 还原 original_attrs（旧值）", 11.5,
                  anchor="start", fill="#14532d", mono=True))
    L.append(text(rx + 20, noteY + 68, "_torch_cuda：还原成一组稳态缺省（非原样 cuda）", 11.5,
                  anchor="start", fill="#14532d"))
    L.append(text(rx + 20, noteY + 86, "→ 置换生存期 ⊆ with 作用域，不泄漏到作用域外", 11.5,
                  anchor="start", fill="#166534", weight="bold"))

    L.append('</svg>')
    open(path, "w").write("\n".join(L))


# ----------------------------------------------------------------------------
# 图 3：capture_model 的装 / 卸时间轴
# ----------------------------------------------------------------------------
def diagram_seam_sequence(path):
    W, H = 1120, 560
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    L.append('<defs>'
             '<marker id="fa" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
             '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
             '</defs>')
    L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    L.append(text(W / 2, 38, "capture_model：进入「装」符号、退出「卸」符号，对称成对", 22, weight="bold"))

    # 中央时间轴
    axx = W / 2
    top = 78
    bot = H - 56
    L.append(f'<line x1="{axx}" y1="{top}" x2="{axx}" y2="{bot}" stroke="#cbd5e1" stroke-width="3"/>')

    steps = [
        ("① NPUModelRunner.capture_model(self)", "进入 override", "#2563eb", "#dbeafe", "in"),
        ("② _get_gpu_model_runner_module_name(self)", "沿 MRO 取父类模块名", "#7c3aed", "#ede9fe", "in"),
        ("③ with _torch_cuda_wrapper():", "torch.cuda.* ↦ torch.npu.*", "#0891b2", "#cffafe", "in"),
        ("④ with _replace_..._wrapper(mod):", "父模块 graph_capture/CUDAGraphWrapper ↦ NPU/ACL", "#0891b2", "#cffafe", "in"),
        ("⑤ GPUModelRunner.capture_model(self)", "父类巨方法一行不改地跑在替换态下", "#16a34a", "#dcfce7", "run"),
        ("⑥ 退出 _replace_...：finally 还原 original_attrs", "父模块符号复原", "#ea580c", "#ffedd5", "out"),
        ("⑦ 退出 _torch_cuda_wrapper：finally 落稳态缺省", "torch.cuda.* 复位", "#ea580c", "#ffedd5", "out"),
    ]
    n = len(steps)
    band = (bot - top - 20) / n
    bw, bh = 470, band - 14
    for i, (title, note, stroke, fill, side) in enumerate(steps):
        cy = top + 18 + i * band
        left = side == "out"  # 卸载放左、装载放右、run 居中
        if side == "run":
            bx = axx - bw / 2
            L.append(box(bx, cy, bw, bh, fill, stroke, rx=9, sw=3))
            L.append(text(axx, cy + bh / 2 - 4, title, 13, weight="bold", fill=stroke, mono=True))
            L.append(text(axx, cy + bh / 2 + 16, note, 11.5, fill="#166534"))
        else:
            if left:
                bx = axx - 30 - bw
                tx_anchor = "end"; tx = bx + bw - 16
            else:
                bx = axx + 30
                tx_anchor = "start"; tx = bx + 16
            L.append(box(bx, cy, bw, bh, fill, stroke, rx=9, sw=2))
            L.append(text(tx, cy + bh / 2 - 4, title, 12, weight="bold", fill=stroke, anchor=tx_anchor, mono=True))
            L.append(text(tx, cy + bh / 2 + 15, note, 11, fill="#475569", anchor=tx_anchor))
            # 连到中轴的小节点
            nodex = bx + bw if not left else bx
            L.append(f'<circle cx="{axx}" cy="{cy + bh/2}" r="5" fill="{stroke}"/>')
            L.append(f'<line x1="{nodex}" y1="{cy + bh/2}" x2="{axx}" y2="{cy + bh/2}" '
                     f'stroke="{stroke}" stroke-width="1.6" stroke-dasharray="4,3"/>')

    # 右侧「装」/左侧「卸」标注
    L.append(text(axx + 300, top + 6, "进入：装符号 →", 13, fill="#0891b2", weight="bold"))
    L.append(text(axx - 300, top + 6, "← 退出：卸符号", 13, fill="#ea580c", weight="bold"))

    L.append('</svg>')
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    import os
    d = os.path.dirname(os.path.abspath(__file__))
    diagram_inherit_vs_rewrite(os.path.join(d, "inherit-vs-rewrite.svg"))
    diagram_symbol_replacement(os.path.join(d, "symbol-replacement.svg"))
    diagram_seam_sequence(os.path.join(d, "seam-sequence.svg"))
    print("done")
