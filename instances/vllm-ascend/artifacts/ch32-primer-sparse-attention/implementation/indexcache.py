"""ch32 落地 IndexCache / skip_topk —— 层间复用 top-k 索引(supporting 机制)。

论文正文未展开这项工程优化,它属于落地代码的调度层:sfa_v1.py forward(L1328-1347)
`if self.skip_topk: topk_indices = self._get_indexcache_topk_indices(...) else:
topk_indices = self.indexer_select_post_process(...)`,dsa_v1.py(L1476-1485)有对应的
`use_index_cache` 开关。相邻层的注意力选择模式相近,后续层可以直接复用前一个(真正跑过
indexer 打分 + top-k 的)层缓存下来的 topk_indices_buffer,省去每层重复打分的开销。
这不改变 Eq.(1)-(2) 的算法本身,只是"每层都算一次"还是"隔层复用"的调度选择,故 dossier
标为 supporting、无需图示/无需数值推演。
"""
import numpy as np


# 落地对应 sfa_v1.py AscendSFAImpl._get_indexcache_topk_indices ——
# PAPER: skip_topk=True 时必须已有缓存的 topk_indices_buffer,否则直接报错(不能凭空复用不存在的索引)
def get_cached_topk_indices(topk_indices_buffer):
    if topk_indices_buffer is None:
        raise RuntimeError("IndexCache requires topk_indices_buffer when skip_topk is enabled.")
    return topk_indices_buffer


# 落地对应 sfa_v1.py forward 里 `if self.skip_topk: ... else: indexer_select_post_process(...)`
# PAPER: 每层要么复用缓存,要么真正跑一遍 indexer 打分 + top-k(由调用方传入 compute_topk_fn,
# 即 lightning_indexer.py + dsa_topk_selection.py 那条真链路)
def layer_topk_indices(skip_topk: bool, topk_indices_buffer, compute_topk_fn):
    if skip_topk:
        return get_cached_topk_indices(topk_indices_buffer)
    return compute_topk_fn()
