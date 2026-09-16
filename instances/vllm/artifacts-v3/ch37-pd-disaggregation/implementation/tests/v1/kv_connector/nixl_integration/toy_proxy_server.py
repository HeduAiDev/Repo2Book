# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""disaggregator 的最小参考实现（vLLM 集成测试里的 P/D 路由器）。

控制面在这里的全部工作 = 搬运一只信封：
  1. 发单给 P（改写 kv_transfer_params：do_remote_decode=True、max_tokens=1、
     stream=False——P 只算 prefill 就收工）；
  2. 从 P 的响应里掏出 kv_transfer_params 原样附加到同一请求体，转交 D。

# SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L1-L290
# SUBTRACTED: XPU/shutdown/健康检查端点与启动样板（lifespan 客户端池的关闭段、
#   parse_args 的 host/port 校验、uvicorn 启动块、healthcheck 端点）——语义由
#   两个核心函数与 round-robin 选择完整表达，运维样板与教学无关（删除项 10）。
"""

import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L19-L82
@asynccontextmanager
async def lifespan(app: FastAPI):
    # SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L19-L82
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=prefiller_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=decoder_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )

    yield


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


# SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L134-L152
def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


# SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L155-L197
async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    # These args are not supported for P
    min_tokens = req_data.pop("min_tokens", None)
    min_completion_tokens = req_data.pop("min_completion_tokens", None)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()

    # read/consume the response body to release the connection
    # otherwise, it would http.ReadError
    await response.aread()

    # Add back the min_tokens and min_completion_tokens so D can use them
    req_data["min_tokens"] = min_tokens
    req_data["min_completion_tokens"] = min_completion_tokens

    return response


# SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L200-L216
async def stream_service_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


# SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L219-L261
async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        # Get the next prefill client in round-robin fashion
        prefill_client_info = get_next_client(request.app, "prefill")

        # Send request to prefill service
        response = await send_request_to_service(
            prefill_client_info, api, req_data, request_id
        )

        # Extract the needed fields
        response_json = response.json()
        await response.aclose()  # CRITICAL: Release connection back to pool
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, "decode")

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        # Stream response from decode service
        # SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L245-L249
        async def generate_stream():
            # SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L245-L249
            async for chunk in stream_service_response(
                decode_client_info, api, req_data, request_id=request_id
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


# SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L264-L266
@app.post("/v1/completions")
async def handle_completions(request: Request):
    # SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L264-L266
    return await _handle_completions("/completions", request)


# SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L269-L271
@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    # SOURCE: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:L269-L271
    return await _handle_completions("/chat/completions", request)


# SUBTRACTED: /healthcheck 端点与 __main__ 启动块（删除项 10）。
