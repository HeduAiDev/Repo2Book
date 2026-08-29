# Subtract-only companion for v3 ch19 — vllm/compilation/partition_rules.py
# (pin v0.27.1 / 6e448d0ea). Kept surface: should_split (the split-point
# predicate, m08). inductor_partition_rule_context is the use_inductor_
# graph_partition route's rule registration (kept verbatim for
# CompilerManager.compile_context; unreachable on the default Dynamo route).
from __future__ import annotations

import contextlib
from collections.abc import Generator

import torch

from .._host_seams import init_logger

logger = init_logger(__name__)


# SOURCE: vllm/compilation/partition_rules.py:L14-L38 should_split —— 切点
#   判定：node.target 是 OpOverload(Packet) 且限定名在 splitting_ops 中
def should_split(node: torch.fx.Node, splitting_ops: list[str]) -> bool:  # SOURCE: vllm/compilation/partition_rules.py:L14-L38
    """
    Check if a node should be split for dynamo graph partition.
    It operates on dynamo graph, so the node.target can be anything.
    We need to check and split only on OpOverload and OpOverloadPacket.
    """

    if node.op != "call_function":
        return False

    target = node.target

    if isinstance(target, torch._ops.OpOverloadPacket):
        # Example: "aten::add"
        return target._qualified_op_name in splitting_ops

    if isinstance(target, torch._ops.OpOverload):
        # Example: "aten::add"
        packet_name = target.name()

        # Example: "aten::add.default"
        op_overload_name = f"{packet_name}.{target._overloadname}"
        return op_overload_name in splitting_ops or packet_name in splitting_ops

    return False


# SOURCE: vllm/compilation/partition_rules.py:L41-L75 inductor_partition_rule_
#   context —— use_inductor_graph_partition 路线的分区规则临时注册
@contextlib.contextmanager
def inductor_partition_rule_context(  # SOURCE: vllm/compilation/partition_rules.py:L41-L75
    splitting_ops: list[str] | None,
) -> Generator[None, None, None]:
    """Context manager to temporarily register Inductor partition rules.

    Registers custom partition rules for specified operators, forcing the
    Inductor scheduler to partition the graph at these operators. The rules
    are automatically restored to their previous state on exit.

    Args:
        splitting_ops: List of operator names to partition on.
    """
    if not splitting_ops:
        logger.debug("No partition ops provided; skipping rule registration.")
        yield
        return

    # Save current state before registering

    saved_splitting_ops: list[str] = list(
        torch._inductor.config.custom_should_partition_ops
    )
    torch._inductor.config.custom_should_partition_ops = splitting_ops

    logger.debug(
        "Registered inductor partition rules for %d operators", len(splitting_ops)
    )

    try:
        yield
    finally:
        # Clear and restore previous state
        torch._inductor.config.custom_should_partition_ops = saved_splitting_ops
        logger.debug("Restored previous partition rules state.")
