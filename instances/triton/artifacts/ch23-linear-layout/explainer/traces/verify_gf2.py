#!/usr/bin/env python3
"""ch23 explainer 手工推演的 GF(2) 自核脚本(NOT the chapter's source code).

本章无精简版(primer);此脚本是一个几十行的 GF(2) 玩具算子,只为**验证 explainer.json
里手工填的 4 个 worked_example 的算术无误**——它复刻 LinearLayout.h/.cpp 注释里描述的
数学(xor 求值 / 复合 / 拼接矩阵 RREF / 求秩),不是 Triton 的 LinearLayout 实现本身。
因此 explainer 的 trace_source 仍标 "manual"(这是推演自核,不是跑 pin 源码)。

跑法: python3 verify_gf2.py   全部 assert 通过则打印每个例子的结果表。
"""

# ---- GF(2) 线性映射:一个布局 = 一组 bases(每个 base 是输出比特串的整数编码) ----

def apply_layout(bases, x):
    """L(x) = 把 x 的每个 1-bit 选中的 base 全部 xor 起来(xor 线性律)。"""
    out = 0
    for i, b in enumerate(bases):
        if (x >> i) & 1:
            out ^= b
    return out


# ================= m3: 4x4 swizzle 亲手填表 =================
# bases: L(0,1)=(0,1), L(0,2)=(0,2), L(1,0)=(1,1), L(2,0)=(2,2)
# 2D->2D。把 (t,w) 编码成输入整数 (w 低 2 bit, t 高 2 bit),输出 (x,y) 编码同理。
# 输入 bit 顺序: bit0=w1, bit1=w2, bit2=t1, bit3=t2。
def m3_L(t, w):
    # 4 bases indexed by input bit: [L(0,1), L(0,2), L(1,0), L(2,0)]
    bases_xy = [(0, 1), (0, 2), (1, 1), (2, 2)]
    x = y = 0
    inp = (w & 1) | ((w >> 1 & 1) << 1) | ((t & 1) << 2) | ((t >> 1 & 1) << 3)
    for i, (bx, by) in enumerate(bases_xy):
        if (inp >> i) & 1:
            x ^= bx
            y ^= by
    return (x, y)

def m3_check():
    # source 头注释 LinearLayout.h:56-59 亲手填的 4 格
    assert m3_L(0, 0) == (0, 0)
    assert m3_L(0, 3) == (0, 3)
    assert m3_L(3, 0) == (3, 3)
    assert m3_L(3, 3) == (3, 0)
    # 整表应等于 (t,w) -> (t, w^t)  (LinearLayout.h:66-74)
    table = {}
    for t in range(4):
        for w in range(4):
            got = m3_L(t, w)
            assert got == (t, w ^ t), (t, w, got)
            table[(t, w)] = got
    return table


# ================= m6: compose  O∘L = 把每个 base 喂给 outer 求值 =================
# 1D->1D，2 输入 bit。L bases=[2,1]，O bases=[1,3]。
def m6_check():
    L = [2, 1]
    O = [1, 3]
    # compose: new base i = O(L(2^i)) = apply(O, L[i])
    composed = [apply_layout(O, b) for b in L]
    # 直接定义验证: (O∘L)(x) == apply(composed, x) for all x
    for x in range(4):
        direct = apply_layout(O, apply_layout(L, x))
        via_bases = apply_layout(composed, x)
        assert direct == via_bases, (x, direct, via_bases)
    return {"L": L, "O": O, "composed": composed,
            "table": [(x, apply_layout(L, x), apply_layout(O, apply_layout(L, x)))
                      for x in range(4)]}


# ---- 比特矩阵工具:bases <-> N×M 矩阵(每列一个 base,行=输出 bit) ----

def bases_to_rows(bases, nrows):
    """返回 nrows 个整数,每个整数第 c bit = base c 的第 r bit(little-endian 列打包,
    对应 f2reduce.h 语义: 第 r 行第 c 列 = (rows[r] >> c) & 1)。"""
    rows = [0] * nrows
    for c, b in enumerate(bases):
        for r in range(nrows):
            if (b >> r) & 1:
                rows[r] |= (1 << c)
    return rows

def rref(rows, ncols):
    """GF(2) 原地 RREF(教学版,复刻 f2reduce::inplace_rref_strided 的效果)。"""
    rows = list(rows)
    nrows = len(rows)
    pivot_row = 0
    for col in range(ncols):
        sel = None
        for r in range(pivot_row, nrows):
            if (rows[r] >> col) & 1:
                sel = r
                break
        if sel is None:
            continue
        rows[pivot_row], rows[sel] = rows[sel], rows[pivot_row]
        for r in range(nrows):
            if r != pivot_row and ((rows[r] >> col) & 1):
                rows[r] ^= rows[pivot_row]
        pivot_row += 1
        if pivot_row == nrows:
            break
    return rows


# ================= m7: invertAndCompose  C = A.invertAndCompose(B) =================
# 语义: A(x) = B(C(x))；B 可逆时 C = B^{-1}∘A。1D->1D,2 bit。
# A(this) bases=[3,2]，B(outer) bases=[2,3]。拼接 [matB | matA] → RREF → 左半单位阵、右半=C。
def m7_check():
    A = [3, 2]
    B = [2, 3]
    nrows = 2
    matA = bases_to_rows(A, nrows)          # matThis
    matB = bases_to_rows(B, nrows)          # matOuter
    # 横向拼接: matOuter 在左(col 0..1)，matThis 在右(col 2..3)
    ncolsB, ncolsA = 2, 2
    combined = [matB[r] | (matA[r] << ncolsB) for r in range(nrows)]
    reduced = rref(combined, ncolsB + ncolsA)
    # 左半应为单位阵
    left = [reduced[r] & ((1 << ncolsB) - 1) for r in range(nrows)]
    assert left == [1 << r for r in range(nrows)], left
    # 右半 = C 的比特矩阵,读回 bases
    right_rows = [reduced[r] >> ncolsB for r in range(nrows)]
    C = [0] * ncolsA
    for c in range(ncolsA):
        for r in range(nrows):
            if (right_rows[r] >> c) & 1:
                C[c] |= (1 << r)
    # 验证语义 A(x) = B(C(x)) 对所有 x
    for x in range(4):
        assert apply_layout(A, x) == apply_layout(B, apply_layout(C, x)), x
    return {"A": A, "B": B, "C": C,
            "table": [(x, apply_layout(C, x), apply_layout(A, x),
                       apply_layout(B, apply_layout(C, x))) for x in range(4)]}


# ================= m8: getMatrixRank = RREF 后非零行数 =================
def matrix_rank(bases, nrows, ncols):
    rows = bases_to_rows(bases, nrows)
    reduced = rref(rows, ncols)
    return sum(1 for r in reduced if r != 0), reduced

def m8_check():
    # (a) 我们构造的例子: 1D->1D 8 元素,bases=[1,2,3],输出 3 bit。
    #     base4=3=1^2 → 线性相关 → rank 2 < 3 bases → 非单射(L(3)=L(4)=3)。
    rank_a, red_a = matrix_rank([1, 2, 3], nrows=3, ncols=3)
    assert rank_a == 2, rank_a
    assert apply_layout([1, 2, 3], 3) == apply_layout([1, 2, 3], 4) == 3   # 碰撞
    # (b) source getMatrix 例(LinearLayout.cpp:79-93): 4 bases,3×4 矩阵,
    #     两个 base 相同 L(0,2)=L(1,0)=(0b10,0b0) → 非单射;满行秩(surjective)。
    #     bases 编码: 输出 (x[2bit], y[1bit]) → 整数 y<<2 | x。
    #     L(0,1)=(0b01,0b1)=0b101=5, L(0,2)=(0b10,0b0)=2, L(1,0)=(0b10,0b0)=2, L(2,0)=(0b11,0b0)=3
    rank_b, _ = matrix_rank([5, 2, 2, 3], nrows=3, ncols=4)
    return {"rank_a": rank_a, "red_a": [bin(r) for r in red_a], "rank_b": rank_b}


if __name__ == "__main__":
    t3 = m3_check()
    print("== m3 4x4 swizzle 填满(全部 == (t, w^t)) ==")
    for t in range(4):
        print("  t=%d: " % t + "  ".join(str(t3[(t, w)]) for w in range(4)))
    r6 = m6_check()
    print("\n== m6 compose L=%s ∘ 先, O=%s → composed bases=%s ==" % (r6["L"], r6["O"], r6["composed"]))
    print("  x | L(x) | O(L(x))")
    for x, lx, olx in r6["table"]:
        print("  %d |  %d   |   %d" % (x, lx, olx))
    r7 = m7_check()
    print("\n== m7 invertAndCompose A=%s, B=%s → C=%s ==" % (r7["A"], r7["B"], r7["C"]))
    print("  x | C(x) | A(x) | B(C(x))  (末两列须相等)")
    for x, cx, ax, bcx in r7["table"]:
        print("  %d |  %d   |  %d   |   %d" % (x, cx, ax, bcx))
    r8 = m8_check()
    print("\n== m8 rank via RREF ==")
    print("  (a) bases=[1,2,3] 输出3bit → rank=%d (RREF行: %s), L(3)=L(4)=3 碰撞" % (r8["rank_a"], r8["red_a"]))
    print("  (b) source getMatrix 例 bases=[5,2,2,3] 3×4 → rank=%d (满行秩,但两 base 相同→非单射)" % r8["rank_b"])
    print("\nALL ASSERTS PASSED")
