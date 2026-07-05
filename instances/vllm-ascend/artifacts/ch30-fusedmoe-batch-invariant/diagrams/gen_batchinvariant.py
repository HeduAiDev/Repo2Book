#!/usr/bin/env python3
"""fig26-4: batch-invariant 两步走 —— 关漂移源 env + torch.library 替换算子；同一行逐位复现。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1340, 700
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=15, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


text(W / 2, 42, "batch-invariant：两步消除批内非确定性", 24, "middle", "#0f172a", "bold")
text(W / 2, 70, "init_batch_invariance（VLLM_BATCH_INVARIANT=1 时启动期一次性）", 13.5, "middle", "#64748b", mono=True)

# 左块：关漂移源
lx, ly, lw, lh = 70, 110, 580, 290
box(lx, ly, lw, lh, "#fff7ed", "#f59e0b", 14, 2)
text(lx + lw / 2, ly + 34, "① override_envs_for_invariance — 关漂移源", 16, "middle", "#b45309", "bold")
env_lines = [
    ("weight_nz_mode = 0", "关 NZ 权重重排（不同排布→不同累加序）"),
    ("enable_matmul_allreduce = False", "关 matmul-allreduce 融合"),
    ('HCCL_DETERMINISTIC = "strict"', "集合通信强制确定性 reduce 顺序"),
    ('LCCL_DETERMINISTIC = "1"', "同上（卡内通信库）"),
]
for i, (code, desc) in enumerate(env_lines):
    yy = ly + 70 + i * 54
    text(lx + 28, yy, code, 13.5, "start", "#7c2d12", "bold", mono=True)
    text(lx + 28, yy + 22, desc, 12, "start", "#92400e")

# 右块：替换算子
rx, ry, rw, rh = 690, 110, 580, 290
box(rx, ry, rw, rh, "#eff6ff", "#3b82f6", 14, 2)
text(rx + rw / 2, ry + 34, "② enable_batch_invariant_mode — 替换算子", 16, "middle", "#1d4ed8", "bold")
text(rx + 28, ry + 70, "torch.library.Library(\"aten\", \"IMPL\") 在 NPU 上替换：", 13, "start", "#1e3a8a", mono=True)
ops = ["aten::mm / matmul / addmm / bmm", "aten::softmax / _softmax", "aten::sum（+ torch.sum 猴补）"]
for i, op in enumerate(ops):
    text(rx + 40, ry + 102 + i * 30, "• " + op, 13, "start", "#1e40af", mono=True)
text(rx + 28, ry + 210, "→ 固定分块 kernel（AscendC 优先、triton 回退）", 13, "start", "#1e3a8a")
text(rx + 28, ry + 236, "matmul_persistent: BLOCK_K 常量 · float32 累加 · allow_tf32=False",
     12, "start", "#1d4ed8", mono=True)

# 底部：同一行逐位复现
by = 430
box(70, by, W - 140, 230, "#f8fafc", "#cbd5e1", 14, 1.6)
text(W / 2, by + 32, "效果：同一行 token 无论和谁组 batch，reduce 顺序相同 ⇒ 逐位相同", 15, "middle", "#0f172a", "bold")

# 两个 batch 场景
cx1 = 360
cx2 = 980
for cx, label, rows in [
    (cx1, "batch = [ 行A ]", ["行A"]),
    (cx2, "batch = [ 行A, 行B, 行C ]", ["行A", "行B", "行C"]),
]:
    text(cx, by + 70, label, 13.5, "middle", "#334155", "bold")
    for i, r in enumerate(rows):
        rxx = cx - 60 + i * 0
        yy = by + 90 + i * 34
        fill = "#fde68a" if r == "行A" else "#e2e8f0"
        box(cx - 70, yy, 140, 26, fill, "#94a3b8", 6, 1.4)
        text(cx, yy + 18, r, 12.5, "middle", "#334155", "bold")
    # 行A 输出
    text(cx, by + 200, "行A 输出 = bits(行A)", 12.5, "middle", "#b45309", "bold", mono=True)

# 中间等号
text(W / 2, by + 150, "=", 40, "middle", "#10b981", "bold")
text(W / 2, by + 196, "逐位相同", 13, "middle", "#059669", "bold")

L.append('</svg>')
open("fig26-4-batchinvariant.svg", "w").write('\n'.join(L))
print("wrote fig26-4-batchinvariant.svg")
