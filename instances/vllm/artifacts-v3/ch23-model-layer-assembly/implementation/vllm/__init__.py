# ch23 精简版 vllm 包骨架（HOST SEAM 载体）
# 真实 vllm/__init__.py 是数百行延迟导入门面（vllm/__init__.py:L1-L~200）；
# 本章精简版不重建门面——各模块按真实路径就地承载（同名同构子集）。
