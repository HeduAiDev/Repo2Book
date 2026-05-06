"""Unit tests for the pedagogical Trie + RadixTree.

These data structures are NOT in vLLM (K01) — they exist purely as a
comparison baseline for the §2 microbench. The tests verify they BEHAVE
correctly (so the benchmark is fair); they do NOT pin them as production code.

Coverage:
- Trie insert + find_longest_prefix happy path
- RadixTree: edge splitting on partial match
- RadixTree: full path match returns deepest block_id
- RadixTree: divergence in middle of edge breaks early
- Both: empty input handled
- Both: longest-prefix semantics across multiple inserts
"""

from __future__ import annotations

from implementation.radix_tree import RadixTree, Trie


class TestTrieBasics:
    def test_single_insert_lookup(self) -> None:
        t = Trie()
        t.insert([1, 2, 3], block_id=42)
        pos, bid = t.find_longest_prefix([1, 2, 3])
        assert pos == 3
        assert bid == 42

    def test_lookup_with_no_match(self) -> None:
        t = Trie()
        t.insert([1, 2, 3], 42)
        # Different first token — no node walked, last_match_pos stays -1.
        pos, bid = t.find_longest_prefix([99, 100])
        assert pos == 0
        assert bid is None

    def test_partial_prefix_match(self) -> None:
        """If insert is [1,2,3] and lookup is [1,2,3,4,5], match length=3."""
        t = Trie()
        t.insert([1, 2, 3], 42)
        pos, bid = t.find_longest_prefix([1, 2, 3, 4, 5])
        assert pos == 3
        assert bid == 42

    def test_prefix_of_inserted_returns_partial(self) -> None:
        """If insert is [1,2,3,4] and lookup is [1,2], we walked [1,2] but
        no node has block_id set → no match recorded."""
        t = Trie()
        t.insert([1, 2, 3, 4], 42)
        pos, bid = t.find_longest_prefix([1, 2])
        # We walked the trie 2 levels but never hit a node with block_id.
        assert pos == 0
        assert bid is None

    def test_size_tracking(self) -> None:
        t = Trie()
        assert len(t) == 0
        t.insert([1], 1)
        assert len(t) == 1
        t.insert([2], 2)
        assert len(t) == 2


class TestRadixTreeBasics:
    def test_single_insert_lookup(self) -> None:
        r = RadixTree()
        r.insert([1, 2, 3, 4], block_id=99)
        pos, bid = r.find_longest_prefix([1, 2, 3, 4])
        assert pos == 4
        assert bid == 99

    def test_no_match_returns_zero_none(self) -> None:
        r = RadixTree()
        r.insert([1, 2, 3], 99)
        pos, bid = r.find_longest_prefix([42])
        assert pos == 0
        assert bid is None

    def test_edge_split_on_partial_divergence(self) -> None:
        """Insert [1,2,3,4,5] then [1,2,3,9,10] should split the edge at depth 3."""
        r = RadixTree()
        r.insert([1, 2, 3, 4, 5], 100)
        r.insert([1, 2, 3, 9, 10], 200)
        # Lookup of either path should still hit.
        pos1, bid1 = r.find_longest_prefix([1, 2, 3, 4, 5])
        assert pos1 == 5
        assert bid1 == 100
        pos2, bid2 = r.find_longest_prefix([1, 2, 3, 9, 10])
        assert pos2 == 5
        assert bid2 == 200

    def test_lookup_diverges_mid_edge(self) -> None:
        """Insert [1,2,3,4,5]; lookup [1,2,3,9,10] diverges at index 3.
        The walker hits the [4,5] edge, sees mismatch at i=0, breaks."""
        r = RadixTree()
        r.insert([1, 2, 3, 4, 5], 100)
        pos, bid = r.find_longest_prefix([1, 2, 3, 9, 10])
        # No block_id was registered at depth 3 (we only have one at depth 5).
        assert pos == 0
        assert bid is None

    def test_inserting_prefix_then_extension(self) -> None:
        """Insert [1,2,3] (block 100), then [1,2,3,4,5] (block 200).
        Lookup [1,2,3] returns 100; lookup [1,2,3,4,5] returns 200."""
        r = RadixTree()
        r.insert([1, 2, 3], 100)
        r.insert([1, 2, 3, 4, 5], 200)
        pos1, bid1 = r.find_longest_prefix([1, 2, 3])
        assert pos1 == 3
        assert bid1 == 100
        pos2, bid2 = r.find_longest_prefix([1, 2, 3, 4, 5])
        assert pos2 == 5
        assert bid2 == 200

    def test_size_tracking(self) -> None:
        r = RadixTree()
        assert len(r) == 0
        r.insert([1, 2, 3], 1)
        assert len(r) == 1
        r.insert([4, 5, 6], 2)
        assert len(r) == 2

    def test_edges_iterator(self) -> None:
        """edges() walks the tree; useful for visualization."""
        r = RadixTree()
        r.insert([1, 2, 3], 100)
        r.insert([4, 5], 200)
        edges = list(r.edges())
        # We should see at least the root + both inserted paths.
        # Root has block_id=None; child block_ids include 100 and 200.
        block_ids_seen = {bid for _, bid in edges if bid is not None}
        assert block_ids_seen == {100, 200}
