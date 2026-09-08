# SOURCE: vllm/model_executor/models/registry.py
# ch23 主角文件之七（m8/m9）：arch 字符串→模型类的唯一查表处。
# SUBTRACTED：delete[8]——条目表收缩为五条（LlamaForCausalLM/DeepseekV32/
#   V4/DSparkDraftModel/DeepSeekV4MTPModel）+ _TEXT_GENERATION_MODELS 单表
#   （其余九类分类字典与数百条目删除；『LlamaModel』键属 _EMBEDDING_MODELS
#   （registry.py:L234，embedding 面），不进保留单表）；_RegisteredModel
#   （L864-L882，已导入条目壳——探测族生态位）与 _try_resolve_transformers/
#   terratorch 后端、_normalize_arch 的 runner_type/convert_type 匹配、接口
#   探测全家族（inspect_model_cls/modelinfo 文件 hash 缓存/_run_in_subprocess）
#   删除——load_model_cls 与 _resolve_module_name 原样保留（本章插座面）。
#   _PREVIOUSLY_SUPPORTED/OOT 名单面删除。
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Set, Union

from torch import nn

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend

logger = init_logger(__name__)


# SOURCE: vllm/model_executor/models/registry.py:L72 _TEXT_GENERATION_MODELS
#   条目表 —— 减法子集（delete[8]：只留 delete 批准的五条目；格式逐字——
#   arch → (mod_relname, cls_name)。真实表的 L92-L95 四行新旧布局对照
#   （DeepseekV2/V3/V32 扁平 vs V4 全限定）由正文 embed 摘录直引真源码）
_TEXT_GENERATION_MODELS = {
    # 新旧布局对照（registry.py:L92-L95 选段：V32 扁平 vs V4 全限定）
    "DeepseekV32ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    "DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),
    # llama 条目（registry.py:L147）
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    # 同款全限定条目：DSparkDraftModel（L617）/DeepSeekV4MTPModel（L643）
    "DSparkDraftModel": ("vllm.models.deepseek_v4", "DSparkDeepseekV4ForCausalLM"),
    "DeepSeekV4MTPModel": ("vllm.models.deepseek_v4", "DeepSeekV4MTP"),
}

# SUBTRACTED: _VLLM_MODELS 十表合并（registry.py:L723-L733）——delete[8] 收缩
#   为单表直用
_VLLM_MODELS = {
    **_TEXT_GENERATION_MODELS,
}


# SUBTRACTED: _SUBPROCESS_COMMAND/_PREVIOUSLY_SUPPORTED_MODELS/
#   _OOT_SUPPORTED_MODELS（registry.py:L736-L~850）——探测与弃用名单面


# SUBTRACTED: _RegisteredModel（registry.py:L864-L882）——「已导入」条目壳，
#   探测族生态位（delete[8]）

# SOURCE: vllm/model_executor/models/registry.py:L851 _BaseRegisteredModel
#   减法子集（load_model_cls 抽象位逐字；inspect_model_cls 抽象删除——
#   接口探测族归 delete[8]）
class _BaseRegisteredModel(ABC):
    @abstractmethod
    # SOURCE: vllm/model_executor/models/registry.py:L851 _BaseRegisteredModel
    def load_model_cls(self) -> type[nn.Module]:
        # SOURCE: vllm/model_executor/models/registry.py:L851 _BaseRegisteredModel
        raise NotImplementedError


# SOURCE: vllm/model_executor/models/registry.py:L884 _LazyRegisteredModel
#   减法子集（dataclass 壳与 load_model_cls 保留；modelinfo 缓存族
#   （_get_cache_dir/_get_modelinfo_module_hash/inspect_model_cls 等 L893-L1014）
#   删除——delete[8]：探测缓存是启动优化，叙事保留实现删除）
@dataclass(frozen=True)
class _LazyRegisteredModel(_BaseRegisteredModel):
    """
    Represents a model that has not been imported in the main process.
    """

    module_name: str
    class_name: str

    # SOURCE: vllm/model_executor/models/registry.py:L1017-L1019 load_model_cls
    #   （逐字——importlib.import_module → getattr：import 推迟到真要类的一刻）
    def load_model_cls(self) -> type[nn.Module]:
        # SOURCE: vllm/model_executor/models/registry.py:L1017-L1019 load_model_cls
        mod = importlib.import_module(self.module_name)
        return getattr(mod, self.class_name)


# SUBTRACTED: _try_load_model_cls 模块级函数（registry.py:L1023-L1033）——
#   current_platform.verify_model_arch 包裹位随探测族删除


# SOURCE: vllm/model_executor/models/registry.py:L1050 _ModelRegistry
#   子集（models 字段与 resolve 主路径保留；探测方法族 is_*_model/
#   inspect_model_cls L1244-L1435 删除——delete[8]）
@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    # SOURCE: vllm/model_executor/models/registry.py:L1051-L1052（逐字）
    models: dict[str, _BaseRegisteredModel] = field(default_factory=dict)

    # SOURCE: vllm/model_executor/models/registry.py:L1054-L1055
    #   get_supported_archs（逐字——返回键视图）
    def get_supported_archs(self) -> Set[str]:
        # SOURCE: vllm/model_executor/models/registry.py:L1054-L1055
        return self.models.keys()

    # SOURCE: vllm/model_executor/models/registry.py:L1136-L1140 _try_load_
    #   model_cls 方法（减法子集：查表命中即惰性拿类，未命中 None；异常
    #   吞噬面删除——精简版让异常直抛）
    def _try_load_model_cls(self, model_arch: str) -> type[nn.Module] | None:
        # SOURCE: vllm/model_executor/models/registry.py:L1136-L1140 _try_load_
        if model_arch not in self.models:
            return None

        return self.models[model_arch].load_model_cls()

    # SUBTRACTED: _normalize_arch（registry.py:L1218-L1242）——delete[8]：
    #   runner_type/convert_type 的后缀匹配族

    # SOURCE: vllm/model_executor/models/registry.py:L1296-L1356 resolve_model_cls
    #   —— 减法子集（transformers/terratorch 后端与 fallback 分支删除
    #   ——delete[8]；主循环逐字：arch 命中即 load_model_cls）
    def resolve_model_cls(
        self,
        architectures: Union[str, list[str]],
        model_config,
    ) -> tuple[type[nn.Module], str]:
        # SOURCE: vllm/model_executor/models/registry.py:L1296-L1356 resolve_model_cls
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            raise ValueError("No model architectures are specified")

        # SUBTRACTED: transformers/terratorch 后端与 fallback 分支
        #   （registry.py:L1306-L1339）——delete[8]，另一条产品线

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)

    # SOURCE: vllm/model_executor/models/registry.py:L1103-L1133
    #   _raise_for_unsupported —— 减法子集（未支持主消息逐字；
    #   _PREVIOUSLY_SUPPORTED/_OOT 名单支删除）
    def _raise_for_unsupported(self, architectures: list[str]):
        # SOURCE: vllm/model_executor/models/registry.py:L1103-L1133
        all_supported_archs = self.get_supported_archs()

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )


# SOURCE: vllm/model_executor/models/registry.py:L1439-L1445 _resolve_module_name
#   （逐字——新旧布局归一：vllm. 开头原样返回（新布局全限定路径），否则拼
#   vllm.model_executor.models. 前缀（旧扁平））
def _resolve_module_name(mod_relname: str) -> str:
    # Allow registry entries to point at fully-qualified module paths (e.g.
    # ``vllm.models.deepseek_v4``) for models that live outside the legacy
    # ``vllm.model_executor.models`` flat layout.
    # SOURCE: vllm/model_executor/models/registry.py:L1439-L1445 _resolve_module_name
    if mod_relname.startswith("vllm."):
        return mod_relname
    return f"vllm.model_executor.models.{mod_relname}"


# SOURCE: vllm/model_executor/models/registry.py:L1448-L1458 ModelRegistry 构造
#   （逐字——逐条 _resolve_module_name 包成 _LazyRegisteredModel：此刻一行
#   源码都没 import）
ModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=_resolve_module_name(mod_relname),
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
    }
)
