# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/openai/models/serving.py —— HOST SEAM（消费面）：
# OpenAIServingModels 是 init_app_state 挂 state 的三件之一
# （openai_serving_models），OpenAIServingChat 构造期消费其
# model_name/is_base_model/lora_requests 面；/v1/models 的
# show_available_models 逐字保留；LoRA 装载路由（load/unload_lora_
# adapter/resolve_lora）删——m18 平行面按同构点到为止。OpenAIModelRegistry
# 逐字。
from http import HTTPStatus
from itertools import count

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.lora.request import LoRARequest


# SOURCE: vllm/entrypoints/openai/models/serving.py —— HOST SEAM：
# AtomicCounter（真实自 utils；itertools.count 载体）
class AtomicCounter:
    # SOURCE: vllm/entrypoints/openai/models/serving.py —— HOST SEAM 位
    def __init__(self, start: int = 0):
        self._counter = count(start)

    def __next__(self) -> int:
        return next(self._counter)

    # SOURCE: vllm/entrypoints/openai/models/serving.py —— HOST SEAM 位
    def get(self) -> int:
        return next(self._counter)


# SOURCE: vllm/entrypoints/openai/models/serving.py:L31-L80 —— OpenAIModelRegistry 逐字
class OpenAIModelRegistry:
    """Read-only view of the loaded base models with no engine dependency.

    Suitable for CPU-only / render-only contexts that have no engine client
    and no LoRA support.
    """

    # SOURCE: vllm/entrypoints/openai/models/serving.py:L39-L44
    def __init__(
        self,
        model_config,
        base_model_paths: list[BaseModelPath],
    ) -> None:
        self.model_config = model_config
        self.base_model_paths = base_model_paths
        self.lora_requests: dict[str, LoRARequest] = {}

    # SOURCE: vllm/entrypoints/openai/models/serving.py:L46-L48
    def model_name(self, lora_request: LoRARequest | None = None) -> str:
        return self.base_model_paths[0].name

    # SOURCE: vllm/entrypoints/openai/models/serving.py:L50-L51
    def is_base_model(self, model_name: str) -> bool:
        return any(model.name == model_name for model in self.base_model_paths)

    # SOURCE: vllm/entrypoints/openai/models/serving.py:L53-L66
    async def check_model(self, model_name: str | None) -> ErrorResponse | None:
        """Return an ErrorResponse if model_name is not served, else None."""
        if not model_name or self.is_base_model(model_name):
            return None
        return create_error_response(
            message=f"The model `{model_name}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    # SOURCE: vllm/entrypoints/openai/models/serving.py:L68-L79
    async def show_available_models(self) -> ModelList:
        """Show available models (base models only)."""
        max_model_len = self.model_config.max_model_len
        return ModelList(
            data=[
                ModelCard(
                    id=base_model.name,
                    max_model_len=max_model_len,
                    root=base_model.model_path,
                    permission=[ModelPermission()],
                )
                for base_model in self.base_model_paths
            ]
        )

    # SOURCE: vllm/entrypoints/openai/models/serving.py:L79-L80
    async def resolve_lora(self, lora_name: str):
        raise RuntimeError("The OpenAIModelRegistry has no LoRA support.")


# SOURCE: vllm/entrypoints/openai/models/serving.py:L83-L160 —— OpenAIServingModels
# 消费面（HOST SEAM：LoRA resolver 注册表族与 load/unload 路由删——
# m18 平行面；lora_requests/model_name/is_base_model/init_static_loras 保留）
class OpenAIServingModels:
    """Shared instance to hold data about the loaded base model(s) and adapters.

    Handles the routes:
    - /v1/models
    - /v1/load_lora_adapter
    - /v1/unload_lora_adapter
    """

    # SOURCE: vllm/entrypoints/openai/models/serving.py:L91-L123（裁剪：
    # lora_resolvers 注册表段删，HOST SEAM）
    def __init__(
        self,
        engine_client: EngineClient,
        base_model_paths: list[BaseModelPath],
        *,
        lora_modules=None,
    ):
        super().__init__()

        self.registry = OpenAIModelRegistry(
            model_config=engine_client.model_config,
            base_model_paths=base_model_paths,
        )

        self.engine_client = engine_client
        self.base_model_paths = base_model_paths

        self.static_lora_modules = lora_modules
        self.lora_requests: dict[str, LoRARequest] = {}
        self.lora_id_counter = AtomicCounter(0)

        # SUBTRACTED: vllm/entrypoints/openai/models/serving.py:L114-L119
        # LoRAResolver 注册表装配——LoRA 动态解析面，本章主线不触达。

        self.model_config = self.engine_client.model_config
        self.renderer = self.engine_client.renderer
        self.input_processor = self.engine_client.input_processor

    # SOURCE: vllm/entrypoints/openai/models/serving.py:L124-L140 —— HOST SEAM：
    # init_static_loras 退化（真实装载静态 LoRA 模块；精简环境 lora_modules
    # 恒 None → 早退语义一致）
    async def init_static_loras(self):
        """Loads all static LoRA modules.
        Raises if any fail to load"""
        if self.static_lora_modules is None:
            return
        raise NotImplementedError(
            "static LoRA loading is out of scope of the ch38 reduced build"
        )

    # SOURCE: vllm/entrypoints/openai/models/serving.py:L141-L142
    def is_base_model(self, model_name: str) -> bool:
        return self.registry.is_base_model(model_name)

    # SOURCE: vllm/entrypoints/openai/models/serving.py:L144-L147
    def model_name(self, lora_request: LoRARequest | None = None) -> str:
        if lora_request is not None:
            return lora_request.lora_name
        return self.base_model_paths[0].name

    # SOURCE: vllm/entrypoints/openai/models/serving.py:L148-L162 —— show_
    # available_models 逐字（/v1/models 路由消费）
    async def show_available_models(self) -> ModelList:
        """Show available models. This includes the base model and all
        adapters."""
        model_list = await self.registry.show_available_models()
        lora_cards = [
            ModelCard(
                id=lora.lora_name,
                root=lora.path,
                parent=lora.base_model_name
                if lora.base_model_name
                else self.base_model_paths[0].name,
                permission=[ModelPermission()],
            )
            for lora in self.lora_requests.values()
        ]
        model_list.data.extend(lora_cards)
        return model_list

    # SUBTRACTED: vllm/entrypoints/openai/models/serving.py 其余
    # （load_lora_adapter / unload_lora_adapter / resolve_lora）——LoRA
    # 装载路由（m18 平行面；OpenAIModelRegistry.show_available_models 的
    # max_model_len 装配位在上方 HOST SEAM 段内按消费面承载）。
