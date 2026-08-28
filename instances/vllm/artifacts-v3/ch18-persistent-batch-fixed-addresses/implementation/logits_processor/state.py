# SOURCE: vllm/v1/sample/logits_processor/state.py
# 本章主角文件之一：BatchUpdateBuilder（m03）——本拍批次变更记录器，
# removed 恒降序、pop_removed()/peek_removed() 返回当前最小空 slot；
# slot 复用与压实的索引真相源。全文逐字（无删除项）。
from collections.abc import Iterable, Iterator
from itertools import chain
from typing import TYPE_CHECKING

from .interface import (
    AddedRequest,
    BatchUpdate,
    MovedRequest,
    RemovedRequest,
)

if TYPE_CHECKING:
    from .interface import LogitsProcessor


# SOURCE: vllm/v1/sample/logits_processor/state.py:L18 BatchUpdateBuilder
class BatchUpdateBuilder:
    """Helps track persistent batch state changes and build
    a batch update data structure for logitsprocs
    Assumptions:
    * All information about requests removed from persistent batch
      during a step is aggregated in self._removed through calls to
      self.removed_append() at the beginning of a step. This must happen
      before the first time that self.removed, self.pop_removed()
      or self.peek_removed() are invoked in a given step
    * After the first time that self.removed, self.pop_removed()
      or self.peek_removed() are read in a step, no new removals
      are registered using self.removed_append()
    * Elements of self._removed are never directly modified, added or
      removed (i.e. modification is only via self.removed_append() and
      self.pop_removed())
    Guarantees under above assumptions:
    * self.removed is always sorted in descending order
    * self.pop_removed() and self.peek_removed() both return
      the lowest removed request index in the current step
    """

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L39-L42 字段声明
    _removed: list[RemovedRequest]
    _is_removed_sorted: bool
    added: list[AddedRequest]
    moved: list[MovedRequest]

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L44-L57 __init__
    def __init__(
        self,
        removed: list[RemovedRequest] | None = None,
        added: list[AddedRequest] | None = None,
        moved: list[MovedRequest] | None = None,
    ) -> None:
        self._removed = removed or []
        self.added = added or []
        self.moved = moved or []
        self._is_removed_sorted = False

        # Used to track changes in the pooling case
        # where we don't populate the added list.
        self.batch_changed = False

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L59-L67 _ensure_removed_
    #   sorted
    def _ensure_removed_sorted(self) -> None:
        """Sort removed request indices in
        descending order.
        Idempotent after first call in a
        given step, until reset.
        """
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L65-L67
        if not self._is_removed_sorted:
            self._removed.sort(reverse=True)
            self._is_removed_sorted = True

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L69-L74 removed property
    @property
    def removed(self) -> list[RemovedRequest]:
        """Removed request indices sorted in
        descending order"""
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L73-L74
        self._ensure_removed_sorted()
        return self._removed

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L76-L89 removed_append
    def removed_append(self, index: int) -> None:
        """Register the removal of a request from the persistent batch.

        Must not be called after the first time self.removed,
        self.pop_removed() or self.peek_removed() are invoked.

        Args:
          index: request index
        """
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L84-L90
        if self._is_removed_sorted:
            raise RuntimeError(
                "Cannot register new removed request after self.removed has been read."
            )
        self._removed.append(index)
        self.batch_changed = True

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L92-L93 has_removed
    def has_removed(self) -> bool:
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L93
        return bool(self._removed)

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L95-L100 peek_removed
    def peek_removed(self) -> int | None:
        """Return lowest removed request index"""
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L97-L100
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed[-1]
        return None

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L102-L107 pop_removed
    def pop_removed(self) -> int | None:
        """Pop lowest removed request index"""
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L104-L107
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed.pop()
        return None

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L109-L117 reset
    def reset(self) -> bool:
        """Returns True if there were any changes to the batch."""
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L111-L117
        self._is_removed_sorted = False
        self._removed.clear()
        self.added.clear()
        self.moved.clear()
        batch_changed = self.batch_changed
        self.batch_changed = False
        return batch_changed

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L119-L145 get_and_reset
    def get_and_reset(self, batch_size: int) -> BatchUpdate | None:
        """Generate a logitsprocs batch update data structure and reset
        internal batch update builder state.

        Args:
          batch_size: current persistent batch size

        Returns:
          Frozen logitsprocs batch update instance; `None` if no updates
        """
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L129-L131
        # Reset removal-sorting logic
        self._is_removed_sorted = False
        self.batch_changed = False
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L132-L134
        if not any((self._removed, self.moved, self.added)):
            # No update; short-circuit
            return None
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L135-L141 Build
        #   batch state update
        batch_update = BatchUpdate(
            batch_size=batch_size,
            removed=self._removed,
            moved=self.moved,
            added=self.added,
        )
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L142-L145
        self._removed = []
        self.moved = []
        self.added = []
        return batch_update


# SOURCE: vllm/v1/sample/logits_processor/state.py:L148 LogitsProcessors
class LogitsProcessors:
    """Encapsulates initialized logitsproc objects."""

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L151-L160 __init__
    def __init__(self, logitsprocs: Iterable["LogitsProcessor"] | None = None) -> None:
        self.argmax_invariant: list[LogitsProcessor] = []
        self.non_argmax_invariant: list[LogitsProcessor] = []
        if logitsprocs:
            for logitproc in logitsprocs:
                (
                    self.argmax_invariant
                    if logitproc.is_argmax_invariant()
                    else self.non_argmax_invariant
                ).append(logitproc)

    # SOURCE: vllm/v1/sample/logits_processor/state.py:L162-L165 all property
    @property
    def all(self) -> Iterator["LogitsProcessor"]:
        """Iterator over all logits processors."""
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L165
        return chain(self.argmax_invariant, self.non_argmax_invariant)
