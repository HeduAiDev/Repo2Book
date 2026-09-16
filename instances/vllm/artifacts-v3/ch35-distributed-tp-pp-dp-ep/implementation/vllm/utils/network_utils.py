# SOURCE: vllm/utils/network_utils.py —— 本章触碰面的真实子集：make_zmq_socket
# （ch05 域的物理层工具，coordinator/core_client 消费）+ tcp/ipv6 地址族。
# 真实文件的其余面（端口池/PING/分层探测）不携带。

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse

import psutil
import zmq

import vllm.envs as envs


# SOURCE: vllm/utils/network_utils.py:L103-L108 is_valid_ipv6_address — 逐字
def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


# SOURCE: vllm/utils/network_utils.py:L130-L131 get_distributed_init_method — 逐字
def get_distributed_init_method(ip: str, port: int) -> str:
    return get_tcp_uri(ip, port)


# SOURCE: vllm/utils/network_utils.py:L134-L138 get_tcp_uri — 逐字
def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"


# SOURCE: vllm/utils/network_utils.py:L141-L143 get_open_zmq_ipc_path —— HOST SEAM
#   （win32）：libzmq 在 Windows 无 ipc:// 传输，退回 loopback tcp（ch05/ch09
#   win32 先例；Linux 上为真实 ipc 路径）。
def get_open_zmq_ipc_path() -> str:
    # SOURCE: vllm/utils/network_utils.py:L141-L143（锚点双置）
    import sys

    if sys.platform == "win32":  # HOST SEAM
        return f"tcp://127.0.0.1:{get_open_port()}"
    base_rpc_path = envs.VLLM_RPC_BASE_PATH
    return f"ipc://{base_rpc_path}/{__import__('uuid').uuid4()}"


# SOURCE: vllm/utils/network_utils.py envs.VLLM_RPC_BASE_PATH —— HOST SEAM
envs.VLLM_RPC_BASE_PATH = "/tmp/vllm"  # HOST SEAM（win32 分支不触达）


# SOURCE: vllm/utils/network_utils.py:L146-L147 get_open_zmq_inproc_path — 逐字
def get_open_zmq_inproc_path() -> str:
    return f"inproc://{__import__('uuid').uuid4()}"


# SOURCE: vllm/utils/network_utils.py:L150+ get_open_port — 逐字
def get_open_port() -> int:
    """
    Get an open port for the vLLM process to listen on.
    An edge case to handle, is when we run data parallel,
    we need to avoid ports that are potentially used by the
    data parallel master process.
    Right now we reserve 10 ports for the data parallel master
    process. Currently it uses 2 ports.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# SOURCE: vllm/utils/network_utils.py:L242-L256 split_zmq_path —— 语义子集
#   （真实版经 vllm.utils.url.parse_url 解析；HOST SEAM 直接 urlparse，同值）。
def split_zmq_path(path: str) -> tuple[str, str, str]:  # HOST SEAM (parse_url → urlparse)
    """Split a zmq path into its parts."""
    # SOURCE: vllm/utils/network_utils.py:L242-L262（锚点双置）
    parsed = urlparse(path)
    if not parsed.scheme:
        raise ValueError(f"Invalid zmq path: {path}")

    scheme = parsed.scheme
    host = parsed.hostname or ""
    port = "" if parsed.port is None else str(parsed.port)
    if host.startswith("[") and host.endswith("]"):
        host = host[1:]  # Remove brackets for IPv6 address
    return scheme, host, port


# SOURCE: vllm/utils/network_utils.py:L284-L341 make_zmq_socket — 逐字
def make_zmq_socket(
    ctx: Any,
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    identity: bytes | None = None,
    linger: int | None = None,
    router_handover: bool = False,
) -> Any:
    """Make a ZMQ socket with the proper bind/connect semantics."""

    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1

    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if socket_type == zmq.ROUTER and router_handover:
        # Let a new connection take over an identity left behind by a dead one.
        socket.setsockopt(zmq.ROUTER_HANDOVER, 1)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    if socket_type == zmq.XPUB:
        socket.setsockopt(zmq.XPUB_VERBOSE, True)

    # Determine if the path is a TCP socket with an IPv6 address.
    # Enable IPv6 on the zmq socket if so.
    scheme, host, _ = split_zmq_path(path)
    if scheme == "tcp" and is_valid_ipv6_address(host):
        socket.setsockopt(zmq.IPV6, 1)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket
