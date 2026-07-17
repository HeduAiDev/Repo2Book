import triton, triton.language as tl, torch, re

@triton.jit
def heavy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    a = tl.load(x_ptr + off, mask=m)
    # long chain of many independent live accumulators -> register pressure
    acc = a
    for i in tl.static_range(1, 64):
        acc = acc * a + tl.sin(a * float(i)) + tl.cos(acc)
    tl.store(out_ptr + off, acc, mask=m)

x = torch.randn(8192, device='cuda')
o = torch.empty_like(x)
grid = (triton.cdiv(8192, 256),)
c = heavy_kernel[grid](x, o, 8192, BLOCK=256)
ptx = c.asm['ptx']
ptx = re.sub(r'\.version \d+\.\d+', '.version 8.7', ptx)
open('heavy_v87.ptx','w').write(ptx)
print("triton n_regs:", c.n_regs, "n_spills:", c.n_spills, "name:", c.metadata.name)
