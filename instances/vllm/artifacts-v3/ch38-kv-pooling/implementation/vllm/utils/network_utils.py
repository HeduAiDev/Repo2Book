# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""本章用到的网络小工具：ZMQ 套接字构造（Mooncake LookupKey 管理通道）。

# SOURCE: vllm/utils/network_utils.py:L284-L340（make_zmq_socket）
# SUBTRACTED: psutil 内存探测与 SNDBUF/RCVBUF 大缓冲调优、XPUB 分支、
#   get_open_port 族——本章只跑 localhost 小消息（ch37 同款删法）。
"""

from typing import Any

import zmq


# SOURCE: vllm/utils/network_utils.py:L284-L340
def make_zmq_socket(
    ctx: zmq.Context,
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    identity: bytes | None = None,
    linger: int | None = None,
    router_handover: bool = False,
) -> zmq.Socket:
    # SOURCE: vllm/utils/network_utils.py:L284-L340
    """Make a ZMQ socket with the proper bind/connect semantics."""

    socket = ctx.socket(socket_type)

    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket


__all__ = ["make_zmq_socket"]
