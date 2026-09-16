# SOURCE: vllm/v1/core/sched/utils.py
# 只做减法的忠实精简版：本章消费面 = remove_all（update_from_output 移除
# 已停请求）。函数逐字。
# SUBTRACTED: SPDX 版权头；文件其余（check_stop/_has_repeating_pattern 等）。
import contextlib


# SOURCE: vllm/v1/core/sched/utils.py:L62-L86 remove_all —— 逐字
def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]
