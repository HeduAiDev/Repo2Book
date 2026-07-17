import triton, triton.language as tl, torch, os

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < n_elements
    x = tl.load(x_ptr + off, mask=mask)
    y = tl.load(y_ptr + off, mask=mask)
    tl.store(out_ptr + off, x + y, mask=mask)

x = torch.randn(4096, device='cuda')
y = torch.randn(4096, device='cuda')
o = torch.empty_like(x)
grid = (triton.cdiv(4096, 1024),)
compiled = add_kernel[grid](x, y, o, 4096, BLOCK_SIZE=1024)
print("=== triton version:", triton.__version__)
print("=== asm keys:", list(compiled.asm.keys()))
ptx = compiled.asm.get('ptx')
if ptx:
    with open('/tmp/claude-0/-mnt-e-Laboratory-Repo2Book/2aa0927d-d054-4a4d-bf5e-bed9e7a4bae6/scratchpad/add_kernel.ptx','w') as f:
        f.write(ptx)
    print("=== PTX head (first 25 lines) ===")
    print("\n".join(ptx.splitlines()[:25]))
print("=== n_regs:", getattr(compiled, 'n_regs', None), " n_spills:", getattr(compiled, 'n_spills', None))
print("=== metadata name:", compiled.metadata.name if hasattr(compiled,'metadata') else None)
print("=== shared:", compiled.metadata.shared if hasattr(compiled,'metadata') else None)
