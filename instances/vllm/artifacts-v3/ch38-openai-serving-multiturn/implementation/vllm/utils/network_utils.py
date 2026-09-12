# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/utils/network_utils.py —— HOST SEAM（最小承载）：本章消费面
# 只有 create_server_socket 的 IPv6 地址判定（api_server.py:L584）。
# SUBTRACTED: find_process_using_port（launcher.py 端口占用排查块，
# dossier elide 已省）与 socket 工具族其余。
import ipaddress


# SOURCE: vllm/utils/network_utils.py:L103-L107 —— is_valid_ipv6_address 逐字
def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False
