# SOURCE: vllm/v1/utils.py
# ch34 切面：get_engine_client_zmq_addr（coordinator 三地址的生成器）与 shutdown
# （进程收尸）。真实文件的其余面（.usage 统计/APIServerProcessManager）属 ch04/ch05
# 域，不携带。

from __future__ import annotations

import time
from multiprocessing.process import BaseProcess

from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path
from vllm.utils.network_utils import get_tcp_uri  # noqa: F401  (同一地址族)

logger = init_logger(__name__)


# SOURCE: vllm/v1/utils.py:L152-L163 get_engine_client_zmq_addr — 逐字
def get_engine_client_zmq_addr(
    local_only: bool,
    host: str,
    port: int = 0,
) -> str:
    """Return an IPC path (``local_only=True``) or ``tcp://host:port``.

    ``port=0`` lets the kernel assign the port at ``bind()`` time; the
    caller must recover it via ``getsockopt(zmq.LAST_ENDPOINT)``."""
    if local_only:
        return get_open_zmq_ipc_path()
    return get_tcp_uri(host, port)


# SOURCE: vllm/v1/utils.py:L590-L626+ shutdown — 逐字（前段 + 收尸循环主干）
def shutdown(procs: list[BaseProcess], timeout: float | None = None) -> None:
    """Shutdown processes with timeout.

    Args:
        procs: List of processes to shutdown
        timeout: Maximum time in seconds for graceful shutdown
    """
    if timeout is None:
        # Keep a small grace period for best-effort cleanup paths that do
        # not have a user-configured shutdown timeout.
        timeout = 5.0

    logger.debug(
        "[shutdown] Process manager: start process_count=%d timeout=%ss names=%s",
        len(procs),
        timeout,
        (",").join([proc.name for proc in procs]),
    )

    # Shutdown the process.
    for proc in procs:
        if proc.is_alive():
            logger.info(
                "[shutdown] Process manager: send sigterm to process %s", proc.name
            )
            proc.terminate()

    # Allow time for remaining procs to terminate.
    deadline = time.monotonic() + timeout
    for proc in procs:
        remaining = deadline - time.monotonic()
        proc.join(timeout=max(0, remaining))
        if proc.is_alive():
            # Not terminated after timeout: kill it.
            logger.info(
                "[shutdown] Process manager: failed to terminate process %s, "
                "killing it",
                proc.name,
            )
            proc.kill()
            proc.join()
            # FIXME: unifying failed and successful termination status
            # for simplicity.
        if proc.exitcode is not None:
            logger.debug(
                "[shutdown] Process manager: process %s exited with code %d",
                proc.name,
                proc.exitcode,
            )
