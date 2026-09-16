# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/serve/__init__.py —— HOST SEAM（最小承载）：
# build_app 消费 register_vllm_serve_api_routers（健康检查等服务路由族）
# 与 register_vllm_dev_api_routers（--dev 模式路由族）。真实实现从
# serve 子包各 api_router 装配；精简环境注册最小 /health 面。
from fastapi import FastAPI
from fastapi.responses import JSONResponse


# SOURCE: vllm/entrypoints/serve/__init__.py —— HOST SEAM：
# register_vllm_serve_api_routers 退化（真实挂 /health、/load_lora_adapter
# 等服务路由；精简环境挂 /health 占位——服务编排主线在 chat 面）
def register_vllm_serve_api_routers(app: FastAPI) -> None:
    # SUBTRACTED: vllm/entrypoints/serve/api_router.py 的完整服务路由族
    # （health/起停/tokenize）——按 m18 同构点到为止。
    @app.get("/health")
    # SOURCE: vllm/entrypoints/serve/api_router.py —— health 位
    async def health() -> JSONResponse:
        return JSONResponse(content={}, status_code=200)


# SOURCE: vllm/entrypoints/serve/__init__.py —— HOST SEAM：
# register_vllm_dev_api_routers 退化（--dev 模式路由族，无消费点）
def register_vllm_dev_api_routers(app: FastAPI) -> None:
    return
