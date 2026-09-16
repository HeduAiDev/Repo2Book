# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""本章用到的网络小工具：ZMQ 端点构造与套接字。

D 与 P 的带外握手（side channel）就走这里造出来的 ROUTER/REQ 套接字——
**真的走网络栈**，本章唯一没有替换的通信面。

# SOURCE: vllm/utils/network_utils.py:L265-L340（make_zmq_path / make_zmq_socket）
# SUBTRACTED: psutil 内存探测与 SNDBUF/RCVBUF 大缓冲调优（L295-L305）、
#   XPUB 分支、get_open_port 族——本章只跑 localhost 小消息，系统默认缓冲足够。
"""

import ipaddress
import socket
from typing import Any

import zmq


# SOURCE: vllm/utils/network_utils.py:L265-L282
def make_zmq_path(scheme: str, host: str, port: int | None = None) -> str:
    """Make a ZMQ path from its parts."""
    if port is None:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


# SOURCE: vllm/utils/network_utils.py:L284-L340
# SUBTRACTED: psutil 缓冲尺寸计算与 SNDHWM/RCVBUF 设置——消息只有握手元数据与
#   notif 几十字节，默认 HWM 足够；行为（bind/connect 语义）逐字保留。
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

    if socket_type == zmq.ROUTER and router_handover:
        # Let a new connection take over an identity left behind by a dead one.
        socket.setsockopt(zmq.ROUTER_HANDOVER, 1)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    if socket_type == zmq.XPUB:
        socket.setsockopt(zmq.XPUB_VERBOSE, True)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket


# SOURCE: vllm/utils/network_utils.py:L150-L160
def get_open_port() -> int:
    """取一个空闲 TCP 端口（测试装配用）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# SOURCE: vllm/utils/network_utils.py:L103-L120
def is_valid_ipv6_address(address: str) -> bool:
    return isinstance(ipaddress.ip_address(address), ipaddress.IPv6Address)


__all__ = ["make_zmq_path", "make_zmq_socket", "get_open_port", "is_valid_ipv6_address"]
