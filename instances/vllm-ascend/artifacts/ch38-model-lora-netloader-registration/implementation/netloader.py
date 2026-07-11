# ch30 变体(3) 网络加载 loader 注册 —— subtract-only 精简版
#
# 真实源码 vllm_ascend/model_loader/netloader/netloader.py（369 行）：
#   @register_model_loader("netloader") 把 ModelNetLoaderElastic 注册成 load_format='netloader'，
#   继承 BaseModelLoader（与 vLLM DefaultModelLoader 平级）。load_model 从网络弹性拉权重加速
#   冷启动，失败优雅回退 revert_to_default → DefaultModelLoader。
#
# 按 subtraction_plan.delete 批准项删去：
#   - __init__ 里 CONFIG_FILE / extra_config 的 SOURCE/MODEL/LISTEN_PORT/INT8_CACHE/... 逐项
#     校验+类型转换循环（L60-L126）——配置解析样板，与「注册到 loader 扩展点 + 弹性加载/回退」无关；
#   - load_model 里 elastic server 启动块（L241-L313）——把本实例变成下一个实例的 source 的副作用；
#   - draft 模型把 sources 端口 +DRAFT_PORT_OFFSET 重写块（L192-L206）+ quant/model_config 深拷贝备份。
# 保留：装饰器、类签名、load_model 主干（initialize_model→elastic_load→失败回退）、revert_to_default。
# 真正的网络传输要 NPU+网络，host 不真跑，只读这层控制流与回退策略。

import torch
from torch import nn
from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
from vllm.utils.torch_utils import set_default_torch_dtype

from copy import deepcopy

from .load import elastic_load


# SOURCE: vllm_ascend/model_loader/netloader/netloader.py:L39-L126
@register_model_loader("netloader")
class ModelNetLoaderElastic(BaseModelLoader):
    """
    A model loader that uses elastic loading for loading weights.
    """

    source: list[dict] | None
    model_path: str | None
    listen_port: int | None
    int8_cache: str
    int8_cache_name: list[str] | None
    output_prefix: str | None

    def __init__(self, load_config: LoadConfig):
        # SOURCE: vllm_ascend/model_loader/netloader/netloader.py:L52-L126
        super().__init__(load_config)
        # SUBTRACTED: netloader.py:L61-L126 —— 从 CONFIG_FILE / model_loader_extra_config 读取并
        #   逐项校验+类型转换 SOURCE/MODEL/LISTEN_PORT/INT8_CACHE/INT8_CACHE_NAME/OUTPUT_PREFIX，
        #   setattr 到 self.{source,model_path,listen_port,...}。纯配置解析样板，与「注册到 loader
        #   扩展点 + 弹性加载/回退」主题无关。精简版直接置默认值占位（真值由上面那段解析填）。
        self.source = None
        self.model_path = None
        self.listen_port = None
        self.int8_cache = "no"
        self.int8_cache_name = None
        self.output_prefix = None

    # SOURCE: vllm_ascend/model_loader/netloader/netloader.py:L128-L131
    @staticmethod
    def _is_draft_model(model_config: ModelConfig) -> bool:
        # SOURCE: vllm_ascend/model_loader/netloader/netloader.py:L128-L131
        """Check whether the model_config corresponds to a draft model for speculative decoding."""
        return getattr(model_config, "runner_type", None) == "draft"

    # SOURCE: vllm_ascend/model_loader/netloader/netloader.py:L133-L322
    def load_model(self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = "") -> nn.Module:
        device_config = vllm_config.device_config
        parallel_config = vllm_config.parallel_config

        need_process_weights_after_loading = False

        if self.model_path is None:
            self.model_path = model_config.model
            logger.info("model_path is set to %s", self.model_path)

        device_id = torch.distributed.get_rank()
        is_draft = self._is_draft_model(model_config)

        if is_draft:
            logger.info("Loading draft model via netloader, model_path: %s", model_config.model)
        else:
            logger.info("Loading target model via netloader, model_path: %s", model_config.model)

        if (
            self.source is None
            or not isinstance(self.source, list)
            or device_id
            not in [
                one_device["device_id"]
                for one_device in self.source
                if isinstance(one_device, dict) and "device_id" in one_device
            ]
        ):
            logger.warning("Did not get valid source info, use DefaultModelLoader")
            model, need_process_weights_after_loading = self.revert_to_default(
                model_config, vllm_config, device_config, prefix
            )

        else:
            target_device = torch.device(device_config.device)

            # SUBTRACTED: netloader.py:L181-L183 —— quant_config / model_config 深拷贝备份
            #   （供 elastic_load 失败回退时还原），边角分支。

            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)

                sources = self.source
                # SUBTRACTED: netloader.py:L192-L206 —— is_draft 时把每个 source 的端口
                #   +DRAFT_PORT_OFFSET 重写出 sources（投机解码 draft 边角分支）。

                model = elastic_load(
                    model=model,
                    device_id=device_id,
                    model_path=model_config.model,
                    sources=sources,
                    tp=parallel_config.tensor_parallel_size,
                    pp=parallel_config.pipeline_parallel_size,
                    group_name="netloader_draft" if is_draft else "netloader",
                )
                need_process_weights_after_loading = True

                if model is None:
                    logger.warning("Netloader elastic loading fails, use load format DefaultModelLoader")
                    # SUBTRACTED: netloader.py:L224-L235 —— 还原 quant_config/model_config 备份 +
                    #   del model / gc.collect() / 清 NPU(或 CUDA) cache 的释放细节。
                    model, need_process_weights_after_loading = self.revert_to_default(
                        model_config, vllm_config, device_config, prefix
                    )

        # SUBTRACTED: netloader.py:L241-L313 —— elastic server 启动块（get_ip / 端口分配 /
        #   写地址文件 / ElasticServer.start 的 try/except）。把本实例变成下一个实例的 source 的
        #   副作用，偏离「本实例如何弹性拉权重」主干。

        if need_process_weights_after_loading:
            process_weights_after_loading(model, model_config, torch.device(device_config.device))

        if model is None:
            logger.error("NetLoader elastic loads model fails")
            return None

        return model.eval()

    # SOURCE: vllm_ascend/model_loader/netloader/netloader.py:L324-L363
    def revert_to_default(self, model_config, vllm_config, device_config, prefix: str = "") -> tuple[nn.Module, bool]:
        # SUBTRACTED: docstring（回退语义说明，纯注释）。
        load_config = deepcopy(self.load_config)
        load_config.model_loader_extra_config = {}
        load_config.load_format = "auto"
        default_model_loader = DefaultModelLoader(load_config)

        if model_config.quantization is None:
            model = default_model_loader.load_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
            need_process_weights_after_loading = False
        else:
            logger.warning("Quantization is set, netloader use DefaultModelLoader with process_weights_after_loading ")
            need_process_weights_after_loading = True
            target_device = torch.device(device_config.device)
            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
                default_model_loader.load_weights(model, model_config)
            model = model.eval()

        return model, need_process_weights_after_loading

    # SOURCE: vllm_ascend/model_loader/netloader/netloader.py:L365-L366
    def download_model(self, model_config: ModelConfig) -> None:
        pass

    # SOURCE: vllm_ascend/model_loader/netloader/netloader.py:L368-L369
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass
