"""Unit tests for KVCacheBlock and FreeKVCacheBlockQueue.

Critical invariants from vLLM kv_cache_utils.py:L113-L370:
- Doubly-linked list — popleft, append, remove are all O(1).
- Fake head/tail eliminate null-pointer branching.
- block_hash setter asserts existing hash is None (catches re-cache bugs).
- popleft on empty queue raises ValueError.
- remove() on a block not in the queue raises RuntimeError (caller bug).
"""

from __future__ import annotations

import pytest

from implementation.kv_cache_block import FreeKVCacheBlockQueue, KVCacheBlock


class TestKVCacheBlock:
    def test_default_state(self) -> None:
        """Fresh block: ref_cnt=0, no hash, no free-list pointers, not null."""
        b = KVCacheBlock(block_id=7)
        assert b.block_id == 7
        assert b.ref_cnt == 0
        assert b.block_hash is None
        assert b.prev_free_block is None
        assert b.next_free_block is None
        assert b.is_null is False

    def test_block_hash_set_then_reset(self) -> None:
        """Set hash → read it back → reset_hash clears it."""
        b = KVCacheBlock(block_id=1)
        b.block_hash = b"abc"
        assert b.block_hash == b"abc"
        b.reset_hash()
        assert b.block_hash is None

    def test_block_hash_double_set_asserts(self) -> None:
        """Re-setting a hash without reset must hit the assert (vLLM L137-L142)."""
        b = KVCacheBlock(block_id=1)
        b.block_hash = b"abc"
        with pytest.raises(AssertionError):
            b.block_hash = b"def"


class TestFreeQueueBasic:
    def test_initial_state_links_blocks_in_order(self) -> None:
        """Construction puts blocks 0..N-1 in head→tail order."""
        blocks = [KVCacheBlock(block_id=i) for i in range(4)]
        q = FreeKVCacheBlockQueue(blocks)
        assert len(q) == 4
        assert q.num_free_blocks == 4
        # Pop them all in order.
        ids = [q.popleft().block_id for _ in range(4)]
        assert ids == [0, 1, 2, 3]
        assert len(q) == 0

    def test_popleft_empty_raises(self) -> None:
        """Empty queue popleft raises ValueError (vLLM kv_cache_utils.py:L221)."""
        q = FreeKVCacheBlockQueue([])
        with pytest.raises(ValueError):
            q.popleft()

    def test_popleft_n_returns_n_in_order(self) -> None:
        """popleft_n(3) returns 3 blocks in head order."""
        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        q = FreeKVCacheBlockQueue(blocks)
        got = q.popleft_n(3)
        assert [b.block_id for b in got] == [0, 1, 2]
        assert len(q) == 2

    def test_popleft_n_zero_no_op(self) -> None:
        """popleft_n(0) returns empty list and doesn't touch the queue."""
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        q = FreeKVCacheBlockQueue(blocks)
        assert q.popleft_n(0) == []
        assert len(q) == 3


class TestFreeQueueAppend:
    def test_append_goes_to_tail(self) -> None:
        """append puts the block at the tail; popleft still returns LRU first."""
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        q = FreeKVCacheBlockQueue(blocks)
        # Pop block 0 (LRU), then append it back.
        popped = q.popleft()
        assert popped.block_id == 0
        q.append(popped)
        # Order is now [1, 2, 0].
        assert [q.popleft().block_id for _ in range(3)] == [1, 2, 0]

    def test_append_n_preserves_input_order(self) -> None:
        """append_n([a,b,c]) ends with tail order ...,a,b,c."""
        blocks = [KVCacheBlock(block_id=i) for i in range(2)]
        q = FreeKVCacheBlockQueue(blocks)
        # Empty the queue.
        q.popleft(); q.popleft()
        new = [KVCacheBlock(block_id=10), KVCacheBlock(block_id=20), KVCacheBlock(block_id=30)]
        q.append_n(new)
        assert [q.popleft().block_id for _ in range(3)] == [10, 20, 30]


class TestFreeQueueRemoveMiddle:
    """W: O(1) middle-removal is what justifies the linked-list design.
    These tests pin the fast path that BlockPool.touch relies on."""

    def test_remove_middle_block(self) -> None:
        """remove(b) where b is not at head or tail."""
        blocks = [KVCacheBlock(block_id=i) for i in range(4)]
        q = FreeKVCacheBlockQueue(blocks)
        # blocks[2] is in the middle.
        q.remove(blocks[2])
        assert len(q) == 3
        # Order is now [0, 1, 3].
        assert [q.popleft().block_id for _ in range(3)] == [0, 1, 3]

    def test_remove_head_block(self) -> None:
        """remove(b) at head (just behind fake head)."""
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        q = FreeKVCacheBlockQueue(blocks)
        q.remove(blocks[0])
        assert [q.popleft().block_id for _ in range(2)] == [1, 2]

    def test_remove_tail_block(self) -> None:
        """remove(b) at tail (just before fake tail)."""
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        q = FreeKVCacheBlockQueue(blocks)
        q.remove(blocks[2])
        assert [q.popleft().block_id for _ in range(2)] == [0, 1]

    def test_remove_unlinked_block_raises(self) -> None:
        """remove() on a block whose pointers are None raises RuntimeError."""
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        q = FreeKVCacheBlockQueue(blocks)
        # Pop block 0; its pointers are now None → re-removing must raise.
        popped = q.popleft()
        with pytest.raises(RuntimeError):
            q.remove(popped)

    def test_remove_then_reinsert_re_links(self) -> None:
        """Pop → append → remove must work; pointers are correctly re-set."""
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        q = FreeKVCacheBlockQueue(blocks)
        b1 = blocks[1]
        # Pop, append back at tail, then remove from middle (it IS the tail now after the
        # original tail's still in place... actually, it's not — appending puts b1 at tail).
        # To make it middle: pop b1, append, append a new block.
        q.remove(b1)
        q.append(b1)
        new = KVCacheBlock(block_id=99)
        q.append(new)
        # Order: [0, 2, 1, 99]. Remove the middle (1).
        q.remove(b1)
        assert [q.popleft().block_id for _ in range(3)] == [0, 2, 99]


class TestEmptyQueue:
    def test_empty_init(self) -> None:
        """Empty queue: len 0, popleft raises."""
        q = FreeKVCacheBlockQueue([])
        assert len(q) == 0
        with pytest.raises(ValueError):
            q.popleft()

    def test_empty_then_append_works(self) -> None:
        """Append to an empty queue and popleft retrieves it."""
        q = FreeKVCacheBlockQueue([])
        b = KVCacheBlock(block_id=42)
        q.append(b)
        assert len(q) == 1
        assert q.popleft() is b
