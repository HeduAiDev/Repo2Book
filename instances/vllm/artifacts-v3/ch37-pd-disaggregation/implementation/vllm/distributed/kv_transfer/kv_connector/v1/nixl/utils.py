# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL connector 的共享常量、ZMQ 上下文与请求 id 归一。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py:L1-L68
# SUBTRACTED: 平台 OOT 设备表扩展（`_NIXL_SUPPORTED_DEVICE.update(
#   current_platform.get_nixl_supported_devices())`，L24）——本章只跑宿主平台。
"""

import contextlib
from collections.abc import Iterator
from typing import Any

import regex as re
import zmq

from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py:L16-L27
# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": (
        "cpu",
        "xpu",
    ),
    "cpu": ("cpu",),
}


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py:L30-L43
@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py:L30-L43
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py:L45-L52
def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        # All inner specs are the same type; pick any.
        inner = next(iter(spec.kv_cache_specs.values()))
        return type(inner)
    return type(spec)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py:L54-L68
# Trailing 8-hex randomization suffix appended by
# ``input_processor.assign_request_id`` as ``-{random_uuid():.8}``.
_RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py:L64-L68
def get_base_request_id(request_id: str) -> str:
    """Strip the per-request ``-<8 hex>`` randomization suffix, if present."""
    return _RANDOM_SUFFIX_RE.sub("", request_id)


__all__ = [
    "zmq_ctx",
    "get_representative_spec_type",
    "get_base_request_id",
    "_NIXL_SUPPORTED_DEVICE",
]
