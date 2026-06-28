#!/usr/bin/env python3
"""cumem.py vs camem.py 逐行对位：移植=同构换符号 + 几处结构性差异。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

W, H = 1140, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">移植 = 同构换符号</text>')
L.append(f'<text x="{W/2}" y="60" text-anchor="middle" font-size="13" fill="#64748b">vllm/device_allocator/cumem.py　vs　vllm_ascend/device_allocator/camem.py</text>')

# column headers
colL_x, colR_x = 60, 600
colW = 480
hdr_y = 84
L.append(f'<rect x="{colL_x}" y="{hdr_y}" width="{colW}" height="38" rx="7" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>')
L.append(f'<text x="{colL_x+colW/2}" y="{hdr_y+25}" text-anchor="middle" font-size="15" font-weight="bold" fill="#1e3a8a">vLLM cumem.py（GPU 原版）</text>')
L.append(f'<rect x="{colR_x}" y="{hdr_y}" width="{colW}" height="38" rx="7" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>')
L.append(f'<text x="{colR_x+colW/2}" y="{hdr_y+25}" text-anchor="middle" font-size="15" font-weight="bold" fill="#166534">vllm-ascend camem.py（NPU 移植）</text>')

# swap rows: (concept, left, right)
rows = [
    ("拷贝原语", "libcudart.cudaMemcpy(dst, src, n)", "acl.rt.memcpy(dst, destMax, src, n, kind)"),
    ("C 扩展", "vllm.cumem_allocator", "vllm_ascend.vllm_ascend_C"),
    ("pluggable allocator", "torch.cuda.memory.CUDAPluggableAllocator", "torch.npu.memory.NPUPluggableAllocator"),
    ("内存池 / 上下文", "torch.cuda.memory.MemPool / use_mem_pool", "torch.npu.memory.MemPool / use_mem_pool"),
    ("清缓存", "torch.cuda.empty_cache()", "torch.npu.empty_cache()"),
    ("环境变量", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_NPU_ALLOC_CONF"),
    (".so 反查名", 'find_loaded_library("cumem_allocator")', 'find_loaded_library("vllm_ascend_C")'),
]
ry = hdr_y + 50
rh = 40
for i, (concept, lft, rgt) in enumerate(rows):
    y = ry + i*rh
    bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
    L.append(f'<rect x="{colL_x}" y="{y}" width="{colW}" height="{rh-6}" rx="5" fill="{bg}" stroke="#cbd5e1" stroke-width="1"/>')
    L.append(f'<rect x="{colR_x}" y="{y}" width="{colW}" height="{rh-6}" rx="5" fill="{bg}" stroke="#cbd5e1" stroke-width="1"/>')
    # concept tag at far left of left box
    L.append(f'<text x="{colL_x+10}" y="{y+13}" text-anchor="start" font-size="10.5" fill="#94a3b8">{esc(concept)}</text>')
    L.append(f'<text x="{colL_x+10}" y="{y+30}" text-anchor="start" font-size="12" font-family="monospace" fill="#1e3a8a">{esc(lft)}</text>')
    L.append(f'<text x="{colR_x+10}" y="{y+30}" text-anchor="start" font-size="12" font-family="monospace" fill="#166534">{esc(rgt)}</text>')
    # swap arrow between
    midx = colL_x + colW + (colR_x - (colL_x+colW))/2
    L.append(f'<text x="{midx}" y="{y+26}" text-anchor="middle" font-size="16" fill="#7c3aed">→</text>')

# structural differences band
sd_y = ry + len(rows)*rh + 18
L.append(f'<rect x="{colL_x}" y="{sd_y}" width="{colR_x+colW-colL_x}" height="34" rx="7" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{(colL_x+colR_x+colW)/2}" y="{sd_y+22}" text-anchor="middle" font-size="14" font-weight="bold" fill="#92400e">结构性差异（非换符号——ascend 基于较早 vLLM 移植，之后上游又演进）</text>')

diffs = [
    ("expandable_segments", "__init__ 期硬 assert 拒绝", "use_memory_pool 内临时关闭再恢复"),
    ("回调强引用（PR 22724）", "无", "有（防 GC 回收 bound method）"),
    ("use_memory_pool 退出", "只 restore tag", "snapshot() 手动释放零分配"),
]
# note: left=ascend, right=vLLM per evidence; label accordingly
dy = sd_y + 56
dh = 36
L.append(f'<text x="{colL_x+10}" y="{dy-4}" text-anchor="start" font-size="11" fill="#64748b">差异点</text>')
L.append(f'<text x="{colL_x+330}" y="{dy-4}" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#166534">ascend camem</text>')
L.append(f'<text x="{colR_x+220}" y="{dy-4}" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#1e3a8a">vLLM cumem（新版）</text>')
for i, (concept, asc, vl) in enumerate(diffs):
    y = dy + i*dh
    bg = "#fffbeb" if i % 2 == 0 else "#ffffff"
    L.append(f'<rect x="{colL_x}" y="{y}" width="{colR_x+colW-colL_x}" height="{dh-6}" rx="5" fill="{bg}" stroke="#fcd34d" stroke-width="1"/>')
    L.append(f'<text x="{colL_x+10}" y="{y+20}" text-anchor="start" font-size="11.5" fill="#92400e">{esc(concept)}</text>')
    L.append(f'<text x="{colL_x+330}" y="{y+20}" text-anchor="middle" font-size="11.5" fill="#166534">{esc(asc)}</text>')
    L.append(f'<text x="{colR_x+220}" y="{y+20}" text-anchor="middle" font-size="11.5" fill="#1e3a8a">{esc(vl)}</text>')

# bottom takeaway
L.append(f'<text x="{W/2}" y="{H-16}" text-anchor="middle" font-size="13" fill="#475569">类 / 方法 / 控制流逐行对位——sleep mode 的本质（保留 VA、解绑重绑物理页）与硬件无关，照搬最易跟上游同步</text>')

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm-ascend/artifacts/ch07-sleep-mode-camem-allocator/diagrams/symbol_swap.svg","w").write('\n'.join(L))
print("ok")
