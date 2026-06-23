# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Subtract-only reduced companion for ch18.
# SOURCE: vllm/v1/sample/logits_processor/state.py (pin f3fef123)
# BatchUpdateBuilder is the real slot-reuse / condense bookkeeper. Kept verbatim;
# only the get_and_reset BatchUpdate construction (which feeds logits processors,
# off the ch18 spine) is reduced to a plain reset.

import enum


# SOURCE: vllm/v1/sample/logits_processor/interface.py MoveDirectionality
class MoveDirectionality(enum.Enum):
    UNIDIRECTIONAL = enum.auto()
    SWAP = enum.auto()


# SOURCE: vllm/v1/sample/logits_processor/state.py:L18  BatchUpdateBuilder
class BatchUpdateBuilder:
    """Helps track persistent batch state changes.

    Guarantees (under the documented call ordering):
      * self.removed is always sorted in descending order
      * self.pop_removed() and self.peek_removed() both return the lowest
        removed request index in the current step
    """

    def __init__(self, removed=None, added=None, moved=None) -> None:
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L44
        self._removed = removed or []
        self.added = added or []
        self.moved = moved or []
        self._is_removed_sorted = False
        self.batch_changed = False

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L59  _ensure_removed_sorted
    def _ensure_removed_sorted(self) -> None:
        if not self._is_removed_sorted:
            self._removed.sort(reverse=True)
            self._is_removed_sorted = True

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L69  removed
    @property
    def removed(self) -> list[int]:
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L69
        """Removed request indices sorted in descending order"""
        self._ensure_removed_sorted()
        return self._removed

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L76  removed_append
    def removed_append(self, index: int) -> None:
        if self._is_removed_sorted:
            raise RuntimeError(
                "Cannot register new removed request after self.removed has been read."
            )
        self._removed.append(index)
        self.batch_changed = True

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L92  has_removed
    def has_removed(self) -> bool:
        return bool(self._removed)

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L95  peek_removed
    def peek_removed(self) -> int | None:
        """Return lowest removed request index"""
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed[-1]
        return None

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L102  pop_removed
    def pop_removed(self) -> int | None:
        """Pop lowest removed request index"""
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed.pop()
        return None

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L109  reset
    def reset(self) -> bool:
        """Returns True if there were any changes to the batch."""
        self._is_removed_sorted = False
        self._removed.clear()
        self.added.clear()
        self.moved.clear()
        batch_changed = self.batch_changed
        self.batch_changed = False
        return batch_changed

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L119  get_and_reset
    def get_and_reset(self, batch_size: int):
        """Generate a logitsprocs batch update and reset internal state.

        Returns a truthy summary tuple when the batch changed, else None.
        """
        self._is_removed_sorted = False
        self.batch_changed = False
        if not any((self._removed, self.moved, self.added)):
            return None
        # SUBTRACTED: BatchUpdate dataclass construction for logits processors.
        #   Approved: logits-processor wiring is off the ch18 persistent-batch
        #   spine; the reduced companion only needs the "batch changed" signal.
        #   Orig: vllm/v1/sample/logits_processor/state.py:L136-L141
        batch_update = (batch_size, self._removed, self.moved, self.added)
        self._removed = []
        self.moved = []
        self.added = []
        return batch_update
