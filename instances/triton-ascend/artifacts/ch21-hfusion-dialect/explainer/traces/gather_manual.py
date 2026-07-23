#!/usr/bin/env python3
"""ch21 m4 — hfusion.gather 语义手工推演(非编译器 dump)。

依据:HFusionStructuredOps.td:L202-L222 的三重循环等价语义
  for i: for j: for k:  output[i][j] = (index[i][j]==k) ? src[i][k] : output[i][j]
即等价于 output[i][j] = src[i][ index[i][j] ]  (沿 axis=1 按 index 选列)。
本脚本纯 host python 复算,坐实展示用的小例数值——非 bishengir-opt 编译产物。
"""

# 小而具体:src 3x4, index 3x2, axis=1。src[i][k] = 10*(i+1)+k(方便肉眼核对来源)
src = [[10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33]]
index = [[3, 0],
         [1, 2],
         [0, 3]]
axis = 1
M, K = len(src), len(src[0])      # 3, 4
J = len(index[0])                 # 2

# 忠实照 .td 三重循环执行(不走捷径),统计比较次数
output = [[None] * J for _ in range(M)]
compares = 0
for i in range(M):
    for j in range(J):
        for k in range(K):        # gather 轴:不可 tile,否则某片可能只见到部分 k
            compares += 1
            if index[i][j] == k:
                output[i][j] = src[i][k]

print("src   =", src)
print("index =", index, " axis =", axis)
print("output=", output)
print("output_shape =", (M, J), " (= index 形状:除 gather 轴外与 src 同,gather 轴换成 index 长度)")
print("iterations = M*J*K =", M * J * K, " compares_executed =", compares)
