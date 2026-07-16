#!/usr/bin/env python3
"""ch10 explainer 驱动 —— source-hash-cachekey 机制真值取证。

cache_key 属性（jit.py:L717-L725）：无被调 JITFunction、无可疑全局量时，
DependenciesFinder.ret == sha256(src).hexdigest()（hasher 初值即 sha256(src)，
见 jit.py:L38），最终 cache_key = ret + str(starting_line_number)。
本驱动对『同一 kernel 改动一个常量前后』的两份 src 各算 sha256，证明源码一变
哈希即变 → 缓存键失效 → 重编。只用 stdlib。
"""
import hashlib
import json

# add_kernel 两版源码：仅 BLOCK 相关常量注释不同（模拟改了一行 kernel 体）
src_v1 = (
    "def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):\n"
    "    pid = tl.program_id(axis=0)\n"
    "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
    "    mask = offsets < n_elements\n"
    "    x = tl.load(x_ptr + offsets, mask=mask)\n"
    "    y = tl.load(y_ptr + offsets, mask=mask)\n"
    "    tl.store(output_ptr + offsets, x + y, mask=mask)\n"
)
# v2：把最后一行的加法改成乘法（源码实变）
src_v2 = src_v1.replace("x + y", "x * y")

# SOURCE: jit.py:L38  self.hasher = hashlib.sha256(src.encode("utf-8"))
# SOURCE: jit.py:L46-48 ret == hasher.hexdigest()（无依赖/无可疑全局量时）
ret_v1 = hashlib.sha256(src_v1.encode("utf-8")).hexdigest()
ret_v2 = hashlib.sha256(src_v2.encode("utf-8")).hexdigest()

# SOURCE: jit.py:L723  self.hash = dependencies_finder.ret + str(self.starting_line_number)
starting_line_number = 12  # 假设 kernel def 在文件第 12 行（inspect.getsourcelines 得）
cache_key_v1 = ret_v1 + str(starting_line_number)
cache_key_v2 = ret_v2 + str(starting_line_number)

out = {
    "note": "sha256 over verbatim src; no called JITFunction/no suspicious globals => ret == hexdigest",
    "src_v1_first_line": src_v1.splitlines()[0],
    "changed_line": "x + y  ->  x * y",
    "sha256_v1_first12": ret_v1[:12],
    "sha256_v2_first12": ret_v2[:12],
    "hashes_differ": ret_v1 != ret_v2,
    "starting_line_number": starting_line_number,
    "cache_key_v1_first12": cache_key_v1[:12],
    "cache_key_v2_first12": cache_key_v2[:12],
    "cache_keys_differ": cache_key_v1 != cache_key_v2,
    # 同一份 src 再算一次 → 哈希稳定（幂等，命中同缓存）
    "sha256_v1_recomputed_first12": hashlib.sha256(src_v1.encode("utf-8")).hexdigest()[:12],
    "v1_recompute_stable": hashlib.sha256(src_v1.encode("utf-8")).hexdigest() == ret_v1,
}
print(json.dumps(out, indent=2))
