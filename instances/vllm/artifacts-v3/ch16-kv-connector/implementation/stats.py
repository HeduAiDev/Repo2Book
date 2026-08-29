# SOURCE: vllm/v1/metrics/stats.py
# PrefixCacheStats——外部命中率的官方口径（m17：connector_prefix_cache_
# stats 的载体；准入时 record、未调度的查询不计数；preempted 分账）。
# SUBTRACTED: BaseCacheStats 其余子类（MultiModal/Spec/Perf 等观测族）
#   与 CachingMetrics——第 3 条观测旁路。
from dataclasses import dataclass


# SOURCE: vllm/v1/metrics/stats.py:L19 BaseCacheStats
@dataclass
class BaseCacheStats:
    """Stores cache hit statistics."""

    # SOURCE: vllm/v1/metrics/stats.py:L21-L31
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

    # SOURCE: vllm/v1/metrics/stats.py:L121-L128 preempted 分账字段
    preempted_requests: int = 0
    """The number of previously preempted requests in this update."""

    preempted_queries: int = 0
    """The `queries` number for preempted requests."""

    preempted_hits: int = 0
    """The `hits` number for preempted requests."""

    # SOURCE: vllm/v1/metrics/stats.py:L130 record（preempted 分账——官方口径）
    def record(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        """Aggregate request information into the stats."""
        # SOURCE: vllm/v1/metrics/stats.py:L132-L142
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
