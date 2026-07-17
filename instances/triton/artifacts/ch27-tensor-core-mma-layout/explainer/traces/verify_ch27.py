#!/usr/bin/env python3
"""ch27 素材自核脚本(host 纯算术/控制流,无 CUDA/无目标仓运行时)。

三件事全部只依赖 dossier 里逐字引的源码常量/算式,不查 PTX、不编硬件坐标:
  1) C accumulator 逐 lane 坐标公式 (g,2h)/(g,2h+1)/(g+8,2h)/(g+8,2h+1),g=lane>>2,h=lane&3
     —— 复现出 16x8 lane 矩阵,逐格比对源码注释里那张矩阵(TritonGPUAttrDefs.td:L1105-L1126)。
  2) warpsPerTileV2 贪心(AccelerateMatmul.cpp:L82-L104)—— 逐迭代打印 ret,得 warpsPerCTA。
  3) kWidth=32/bitwidth(TritonGPUAttrDefs.td:L1340)与 getSizePerThreadForOperand
     (Dialect.cpp:L2144-L2159)的每线程元素数验算(A=8/B=4 for f16)。

本脚本是 explainer 数字的 host-side 交叉验证(trace_source=manual:无 subtract-only 精简版,
数字均溯源到源码常量 file:Lxxx;此脚本仅复算这些常量的直接后果以自核,非目标仓运行轨迹)。
"""


def c_accumulator_matrix():
    """从公式复现 m16n8 的 C accumulator 单块 16x8 lane 矩阵,并比对源码注释矩阵前 8 行。"""
    rows, cols = 16, 8
    grid = [[None] * cols for _ in range(rows)]
    for lane in range(32):
        g = lane >> 2   # 组号
        h = lane & 3    # 组内序
        for (r, c) in [(g, 2 * h), (g, 2 * h + 1), (g + 8, 2 * h), (g + 8, 2 * h + 1)]:
            grid[r][c] = lane
    # 源码注释里逐字印的前 8 行(warp 0 左上 8x8 块,每格是 lane id)
    src_top8 = [
        [0, 0, 1, 1, 2, 2, 3, 3],
        [4, 4, 5, 5, 6, 6, 7, 7],
        [8, 8, 9, 9, 10, 10, 11, 11],
        [12, 12, 13, 13, 14, 14, 15, 15],
        [16, 16, 17, 17, 18, 18, 19, 19],
        [20, 20, 21, 21, 22, 22, 23, 23],
        [24, 24, 25, 25, 26, 26, 27, 27],
        [28, 28, 29, 29, 30, 30, 31, 31],
    ]
    print("=== (1) C accumulator lane 矩阵(公式复现,16 行 x 8 列)===")
    for r in range(rows):
        print("row %2d: %s" % (r, " ".join("%2d" % grid[r][c] for c in range(cols))))
    # 比对前 8 行
    ok = all(grid[r][c] == src_top8[r][c] for r in range(8) for c in range(cols))
    print("前 8 行 vs 源码注释矩阵(.td:L1105-L1126)逐格一致:", ok)
    # 第 8-15 行应是 0-31 重复(+8 行偏移),即与前 8 行相同的 lane 图
    rep = all(grid[r][c] == grid[r - 8][c] for r in range(8, 16) for c in range(cols))
    print("第 8-15 行 = 前 8 行 lane 重复(+8 偏移):", rep)
    # 抽查几个 lane 的 4 个坐标
    print("--- 抽查 lane -> 4 个 (row,col) 坐标 ---")
    for lane in [0, 1, 4, 8, 31]:
        g, h = lane >> 2, lane & 3
        coords = [(g, 2 * h), (g, 2 * h + 1), (g + 8, 2 * h), (g + 8, 2 * h + 1)]
        print("lane %2d: g=%d h=%d -> %s (每线程 4 个 fp32)" % (lane, g, h, coords))
    print()


def warps_per_tile_v2(shape, num_warps):
    """AccelerateMatmul.cpp:L82-L104 的贪心,逐迭代打印。rank=2。"""
    print("=== (2) warpsPerTileV2  shape=%s numWarps=%d ===" % (list(shape), num_warps))
    ret = [1, 1]
    spw = [16, 8]  # shapePerWarp[rank-2]=16, [rank-1]=8  (== instrShape)
    it = 0
    print("init: ret=%s shapePerWarp=%s" % (ret, spw))
    while True:
        if ret[0] * ret[1] >= num_warps:
            print("iter %d: ret[0]*ret[1]=%d >= numWarps=%d  -> break" %
                  (it, ret[0] * ret[1], num_warps))
            break
        lhs = shape[0] // spw[0] // ret[0]
        rhs = shape[1] // (spw[1] * 2) // ret[1]
        if lhs >= rhs:
            if ret[0] < shape[0] // spw[0]:
                ret[0] *= 2
                action = "ret[0]*=2"
            else:
                ret[1] *= 2
                action = "ret[1]*=2 (M 轴已满)"
        else:
            ret[1] *= 2
            action = "ret[1]*=2 (N 轴优先)"
        it += 1
        print("iter %d: cmp %d>=%d ? %s -> %s => ret=%s" %
              (it, lhs, rhs, lhs >= rhs, action, ret))
    print("=> warpsPerCTA = %s  (总 warp = %d)\n" % (ret, ret[0] * ret[1]))
    return ret


def kwidth_and_elems():
    print("=== (3) kWidth = 32/bitwidth 与每线程元素数验算 ===")
    # 默认 builder: kWidth = 32 / bitwidth (TritonGPUAttrDefs.td:L1340)
    for name, bitwidth in [("f16", 16), ("fp8(default builder)", 8)]:
        kw = 32 // bitwidth
        print("%-22s bitwidth=%2d -> kWidth = 32/%d = %d" % (name, bitwidth, bitwidth, kw))
    # 低精度显式挑更大 kWidth (AccelerateMatmul.cpp:L473-L481)
    print("E2M1 (mxfp, 4-bit)     显式 kWidth = 4  (AccelerateMatmul.cpp:L474)")
    print("E5M2/E4M3 (fp8)        显式 kWidth = 8  (AccelerateMatmul.cpp:L481)")
    # getSizePerThreadForOperand (Dialect.cpp:L2144-L2159), f16 kWidth=2
    kw = 2
    A_M, A_K = 2, 2 * kw          # opIdx=0: [rank-2]=2, [rank-1]=2*kWidth
    B_K, B_N = 2 * kw, 1          # opIdx=1: [rank-2]=2*kWidth, [rank-1]=1
    A_elems = A_M * A_K
    B_elems = B_K * B_N
    print("--- f16 (kWidth=2) 每线程元素数 ---")
    print("A(opIdx=0): sizePerThread=[M=%d, K=%d] -> %d 个 f16   (核对 16x16=256 /32 = %d)"
          % (A_M, A_K, A_elems, 16 * 16 // 32))
    print("B(opIdx=1): sizePerThread=[K=%d, N=%d] -> %d 个 f16   (核对 16x8 =128 /32 = %d)"
          % (B_K, B_N, B_elems, 16 * 8 // 32))
    # shapePerWarp K = 4*64/bitwidth (Dialect.cpp:L2021)
    print("getMMAv2RepForOperand shapePerWarp K = 4*64/16 = %d  (== m16n8k16 的 K=16)"
          % (4 * 64 // 16))
    print("A 每行沿 K=16 由 4 threads per row 分担 -> 每线程 16/4 = %d 个连续 K 元素"
          % (16 // 4))
    print()


if __name__ == "__main__":
    c_accumulator_matrix()
    warps_per_tile_v2([128, 128], 8)
    kwidth_and_elems()
