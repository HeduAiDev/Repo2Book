# SOURCE: vllm/v1/metrics/stats.py（本章切面：命中率观测口径）
# PrefixCacheStats（m19）：准入时 record（num_tokens/num_hits/preempted 旗标；
# 跳过查询的请求不计数），make_prefix_cache_stats 取走清零——官方命中率分母
# 是「查询 token」、分子是「命中 token」，被抢占者单独记账（preempted_* 三位）。
# SUBTRACTED: BaseCacheStats 之外的统计族（CachingMetrics/MultiModal 等）与
#   scheduler 侧 record 调用点（dossier.delete 第 10 条——纯统计旁路）。
from dataclasses import dataclass


# SOURCE: vllm/v1/metrics/stats.py:L18 BaseCacheStats
@dataclass
class BaseCacheStats:
    """Stores cache hit statistics."""

    # SOURCE: vllm/v1/metrics/stats.py:L22-L32
    reset: bool = False
    """Whether the cache was reset."""

    requests: int = 0
    """The number of requests in this update."""

    queries: int = 0
    """The number of queries in these requests."""

    hits: int = 0
    """The number of hits in these requests."""


# SOURCE: vllm/v1/metrics/stats.py:L115 PrefixCacheStats
@dataclass
class PrefixCacheStats(BaseCacheStats):
    """
    Stores prefix cache hit statistics.
    - `reset`: Whether `reset_prefix_cache` was invoked.
    - `queries`: Refers to the number of tokens that were queried.
    """

    # SOURCE: vllm/v1/metrics/stats.py:L122-L129
    preempted_requests: int = 0
    """The number of previously preempted requests in this update."""

    preempted_queries: int = 0
    """The `queries` number for preempted requests."""

    preempted_hits: int = 0
    """The `hits` number for preempted requests."""

    # SOURCE: vllm/v1/metrics/stats.py:L131 record
    def record(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        """Aggregate request information into the stats."""
        # SOURCE: vllm/v1/metrics/stats.py:L133-L142
        if preempted:
            # Previously preempted request
            self.preempted_requests += 1
            self.preempted_queries += num_tokens
            self.preempted_hits += num_hits
        else:
            # New request
            self.requests += 1
            self.queries += num_tokens
            self.hits += num_hits
