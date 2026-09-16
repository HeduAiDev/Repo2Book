# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/openai/models/api_router.py —— 逐字承载（29 行
# 全量，无删改）：/v1/models 路由（build_app 的 models 装配位）。


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


# SOURCE: vllm/entrypoints/openai/models/api_router.py:L16-L17
def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


# SOURCE: vllm/entrypoints/openai/models/api_router.py:L20-L24
@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler = models(raw_request)

    models_ = await handler.show_available_models()
    return JSONResponse(content=models_.model_dump())


# SOURCE: vllm/entrypoints/openai/models/api_router.py:L26-L27
def attach_router(app: FastAPI):
    app.include_router(router)
