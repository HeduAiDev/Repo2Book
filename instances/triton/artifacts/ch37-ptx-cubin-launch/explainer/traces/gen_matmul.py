import triton, triton.language as tl, torch, re

@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m*BM + tl.arange(0, BM)
    offs_n = pid_n*BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:,None]*K + offs_k[None,:]
    b_ptrs = b_ptr + offs_k[:,None]*N + offs_n[None,:]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs); b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BK; b_ptrs += BK*N
    c_ptrs = c_ptr + offs_m[:,None]*N + offs_n[None,:]
    tl.store(c_ptrs, acc)

M=N=K=512
a=torch.randn(M,K,device='cuda'); b=torch.randn(K,N,device='cuda'); c=torch.empty(M,N,device='cuda')
grid=(triton.cdiv(M,128), triton.cdiv(N,128))
comp = mm_kernel[grid](a,b,c,M,N,K, BM=128,BN=128,BK=32, num_warps=4)
ptx = re.sub(r'\.version \d+\.\d+', '.version 8.7', comp.asm['ptx'])
open('mm_v87.ptx','w').write(ptx)
print("mm n_regs:", comp.n_regs, "n_spills:", comp.n_spills, "shared:", comp.metadata.shared, "num_warps:", comp.metadata.num_warps, "name:", comp.metadata.name)
