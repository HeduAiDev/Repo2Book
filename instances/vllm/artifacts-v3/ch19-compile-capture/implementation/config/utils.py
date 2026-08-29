# Subtract-only companion for v3 ch19 — vllm/config/utils.py
# (pin v0.27.1 / 6e448d0ea). Only the Range class is on this chapter's
# surface (compile_ranges 的区间载体)；其余（hash_factors/config 装饰器/
# get_hash_factors）是 ch03/ch17 域，SUBTRACTED。
from __future__ import annotations

from dataclasses import dataclass


# SUBTRACTED: vllm/config/utils.py 的 config/hash 装饰器族与
#   get_hash_factors/hash_factors（L1-L385——ch03 配置基建域，编译缓存
#   hash 的消费侧已随 delete[3] 缓存块删除）。


# SOURCE: vllm/config/utils.py:L386-L409 Range —— 编译区间（含端点）
@dataclass
class Range:
    """
    A range of numbers.
    Inclusive of start, inclusive of end.
    """

    start: int
    end: int

    def is_single_size(self) -> bool:  # SOURCE: vllm/config/utils.py:L396-L397
        return self.start == self.end

    def __contains__(self, size: int) -> bool:  # SOURCE: vllm/config/utils.py:L399-L401
        # Inclusive of start, inclusive of end
        return self.start <= size <= self.end

    def __eq__(self, other: object) -> bool:  # SOURCE: vllm/config/utils.py:L403-L406
        if not isinstance(other, Range):
            return False
        return self.start == other.start and self.end == other.end

    def __hash__(self) -> int:  # SOURCE: vllm/config/utils.py:L408-L409
        return hash((self.start, self.end))
