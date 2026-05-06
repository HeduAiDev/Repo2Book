# REFERENCE: pedagogical only — vLLM does NOT use a radix tree.
# vLLM's prefix cache lives in a flat hash table:
#   instances/vllm/source/vllm/v1/core/block_pool.py:L34-L127
"""Trie + path-compressed radix tree — pedagogical comparison.

WHY THIS FILE EXISTS:
    The Ch07 outline asks for "Radix Tree 数据结构详解 — 从 Trie 到压缩 Radix
    Tree". vLLM does NOT actually use either — its prefix cache is a flat
    hash table. So this module implements both, then benchmarks them against
    the hash table to make the trade-off CONCRETE in numbers, not vibes.

Mental model differences:

    Trie:                   one node per token. n^L space. Slow.
    Radix tree:             merge runs of unbranched nodes. O(unique paths).
    Hash table (vLLM):      O(L/B) lookup, O(unique block prefixes) space.

When would a real radix tree win?
    - You need to ENUMERATE all cached prefixes (admin tooling, eviction
      heuristics that walk the structure).
    - You need RANGE queries ("all prefixes shorter than X bytes").
    - Memory is so tight that path compression's savings dominate.

When does the hash table win? Always, in vLLM's workload. Reasons:
    - Per-block lookup is O(1) instead of O(L/B path traversal).
    - Cache locality — one bucket vs walking pointers.
    - Concurrent insertion is much simpler with a dict.
    - The chained-hash invariant gives the "tree property" (a miss at depth
      K guarantees no hit at depth > K) for free.

The microbenchmark in `demo.py` shows the hash table is ~3-5x faster than the
radix tree for `find_longest_cache_hit` on realistic prompt lengths (4K-32K
tokens). That's the engineering rationale Ch07 wants the reader to grok.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# ════════════════════════════════════════════════════════════════════════════
# Trie — the naive baseline.
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class TrieNode:
    """One token per edge. block_id stored on terminal nodes."""

    children: dict[int, "TrieNode"] = field(default_factory=dict)
    block_id: int | None = None  # None = no block ends here


# REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L48-L52
# (vLLM's NOTE that block tables are append-only — would conflict with a
#  trie's restructuring on insert)
class Trie:
    """Naive trie keyed by token-id sequences.

    Time complexity:
        insert: O(L)
        lookup: O(L)
    Space complexity:
        O(sum of unique paths) — terrible for large vocab.
    """

    def __init__(self) -> None:
        self.root = TrieNode()
        self._size = 0

    def insert(self, tokens: list[int], block_id: int) -> None:
        node = self.root
        for t in tokens:
            node = node.children.setdefault(t, TrieNode())
        node.block_id = block_id
        self._size += 1

    def find_longest_prefix(self, tokens: list[int]) -> tuple[int, int | None]:
        """Return (longest_match_length, block_id_at_that_position)."""
        node = self.root
        last_match_pos = -1
        last_match_block: int | None = None
        for i, t in enumerate(tokens):
            child = node.children.get(t)
            if child is None:
                break
            node = child
            if node.block_id is not None:
                last_match_pos = i + 1
                last_match_block = node.block_id
        return (max(last_match_pos, 0), last_match_block)

    def __len__(self) -> int:
        return self._size


# ════════════════════════════════════════════════════════════════════════════
# Path-compressed Radix Tree.
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class RadixNode:
    """Edges store TOKEN SEQUENCES, not single tokens."""

    edge: tuple[int, ...] = ()
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    block_id: int | None = None


# REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L473-L483
# (vLLM's hash-table scan-and-stop mirrors a radix tree match — this class
#  shows what the explicit tree version costs)
class RadixTree:
    """Path-compressed trie. Each edge holds a sequence of tokens.

    Time complexity:
        insert: O(L) but with much smaller constants when many sequences
                share long prefixes.
        lookup: O(L) worst-case, O(unique_prefix_length) expected.
    Space complexity:
        O(sum of unique edge labels) — typically << Trie.

    Operations needed for prefix cache:
        match  — find longest prefix
        insert — add a new (tokens, block_id) entry, splitting edges as needed
        evict  — remove a (tokens, block_id) entry; collapse if dangling

    NOTE: real radix-tree implementations also handle node merging on
    eviction; we omit that for clarity and document the simplification.
    """

    def __init__(self) -> None:
        self.root = RadixNode()
        self._size = 0

    def insert(self, tokens: list[int], block_id: int) -> None:
        """Insert with edge-splitting on common-prefix divergence."""
        if not tokens:
            self.root.block_id = block_id
            return
        node = self.root
        idx = 0
        while idx < len(tokens):
            child = node.children.get(tokens[idx])
            if child is None:
                # No matching edge — create a new one with the rest.
                new_node = RadixNode(
                    edge=tuple(tokens[idx:]), block_id=block_id
                )
                node.children[tokens[idx]] = new_node
                self._size += 1
                return
            # Walk along the edge as far as it matches.
            edge = child.edge
            i = 0
            while (
                i < len(edge)
                and idx + i < len(tokens)
                and edge[i] == tokens[idx + i]
            ):
                i += 1
            if i == len(edge):
                # Full edge matched; descend.
                node = child
                idx += i
                continue
            # Partial match → split the edge.
            split = RadixNode(edge=edge[:i])
            split.children[edge[i]] = RadixNode(
                edge=edge[i:],
                children=child.children,
                block_id=child.block_id,
            )
            node.children[tokens[idx]] = split
            if idx + i == len(tokens):
                split.block_id = block_id
            else:
                split.children[tokens[idx + i]] = RadixNode(
                    edge=tuple(tokens[idx + i :]), block_id=block_id
                )
            self._size += 1
            return
        # All tokens matched along edges; mark this node.
        node.block_id = block_id
        self._size += 1

    def find_longest_prefix(self, tokens: list[int]) -> tuple[int, int | None]:
        """Walk along edges; remember the deepest node with a block_id."""
        node = self.root
        idx = 0
        last_match_pos = -1
        last_match_block: int | None = node.block_id
        if last_match_block is not None:
            last_match_pos = 0
        while idx < len(tokens):
            child = node.children.get(tokens[idx])
            if child is None:
                break
            edge = child.edge
            i = 0
            while (
                i < len(edge)
                and idx + i < len(tokens)
                and edge[i] == tokens[idx + i]
            ):
                i += 1
            if i < len(edge):
                # Edge diverged before we reached its end — partial match.
                break
            # Full edge consumed.
            node = child
            idx += i
            if node.block_id is not None:
                last_match_pos = idx
                last_match_block = node.block_id
        return (max(last_match_pos, 0), last_match_block)

    def __len__(self) -> int:
        return self._size

    def edges(self) -> Iterable[tuple[tuple[int, ...], int | None]]:
        """Walk all (edge, block_id) pairs. Useful for visualization /
        size benchmarks. NOT used by find_longest_prefix."""
        stack: list[tuple[RadixNode, tuple[int, ...]]] = [(self.root, ())]
        while stack:
            node, path = stack.pop()
            yield (path, node.block_id)
            for child_token, child in node.children.items():
                stack.append((child, path + child.edge))
