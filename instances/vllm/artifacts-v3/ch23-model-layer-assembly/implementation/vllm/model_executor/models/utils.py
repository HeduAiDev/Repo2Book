# SOURCE: vllm/model_executor/models/utils.py
# ch23 主角文件之六（m6/m14）：WeightsMapper（hf_to_vllm 权重名映射——六类
# 映射的 stacked 形是本章主角）+ AutoWeightsLoader（递归分发器：只遍历一遍
# 权重流、按子模块名分组下发、子模块自带 load_weights 即委派）+ PPMissingLayer
# 哨兵 + make_layers（get_pp_indices 均分 + Identity 占空段——参数名跨 PP
# 拓扑稳定）+ make_empty_intermediate_tensors_factory + maybe_prefix +
# extract_layer_index。
# SUBTRACTED：多模态嵌入合面/专家权重路由/EAGLE spec 层工具（maybe_fuse_
#   shared_experts 等 utils.py:L428-L780）——MoE/多模态/spec 域；StageMissingLayer
#   （enc-dec 分段）——encoder-decoder 域。
from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeAlias

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

ShardId: TypeAlias = str | int | tuple[int, ...]


# SOURCE: vllm/model_executor/models/utils.py:L46 WeightsMapper
@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns.

    If a key maps to a value of `None`, the corresponding weight is ignored."""

    # SOURCE: vllm/model_executor/models/utils.py:L53-L59 六类映射字段（逐字）
    orig_to_new_renaming: list = field(default_factory=list)
    orig_to_new_regex: Mapping = field(default_factory=dict)
    orig_to_new_substr: Mapping[str, str | None] = field(default_factory=dict)
    orig_to_new_stacked: Mapping[str, tuple[str, ShardId]] = field(default_factory=dict)
    orig_to_new_prefix: Mapping[str, str | None] = field(default_factory=dict)
    orig_to_new_suffix: Mapping[str, str | None] = field(default_factory=dict)

    # SOURCE: vllm/model_executor/models/utils.py:L59-L80 __or__ 合并（逐字）
    def __or__(self, other: "WeightsMapper") -> "WeightsMapper":
        """Combine two `WeightsMapper`s by merging their mappings."""
        return WeightsMapper(
            orig_to_new_renaming=[
                *self.orig_to_new_renaming,
                *other.orig_to_new_renaming,
            ],
            orig_to_new_regex={**self.orig_to_new_regex, **other.orig_to_new_regex},
            orig_to_new_substr={**self.orig_to_new_substr, **other.orig_to_new_substr},
            orig_to_new_stacked={
                **self.orig_to_new_stacked,
                **other.orig_to_new_stacked,
            },
            orig_to_new_prefix={**self.orig_to_new_prefix, **other.orig_to_new_prefix},
            orig_to_new_suffix={**self.orig_to_new_suffix, **other.orig_to_new_suffix},
        )

    # SOURCE: vllm/model_executor/models/utils.py:L76-L80 _map_name
    #   兼容包装：丢 shard_id
    def _map_name(self, key: str) -> str | None:
        """Map a weight name (backward-compatible wrapper that discards shard_id)."""
        # SOURCE: vllm/model_executor/models/utils.py:L76-L80 _map_name
        result = self._map_name_with_shard(key)
        return result[0] if result is not None else None

    # SOURCE: vllm/model_executor/models/utils.py:L81-L134 _map_name_with_shard
    #   —— 减法子集（kv_scale 弃用告警与 renaming/regex 面删除；substr/
    #   stacked/prefix/suffix 四类逐字——stacked 是本章主角：q/k/v→qkv_proj）
    def _map_name_with_shard(self, key: str) -> tuple[str, ShardId | None] | None:
        """Map a weight name and extract any shard_id metadata.

        Returns:
            (mapped_name, shard_id) if the name should be kept.
            None if the name should be dropped.
        """
        # SUBTRACTED: kv_scale 弃用告警与 orig_to_new_renaming/regex 两类映射
        #   （models/utils.py:L99-L115）——本章无消费面

        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        shard_id: ShardId | None = None
        # SOURCE: vllm/model_executor/models/utils.py:L116-L120 stacked 映射
        #   （逐字——shard_id 元数据由此挂出）
        for substr, (new_key, new_shard_id) in self.orig_to_new_stacked.items():
            if substr in key:
                key = key.replace(substr, new_key, 1)
                shard_id = new_shard_id

        # SOURCE: vllm/model_executor/models/utils.py:L122-L137 prefix/suffix
        #   （逐字）
        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key, shard_id

    # SOURCE: vllm/model_executor/models/utils.py:L136-L152 apply
    #   shard_id 盖章到张量属性：weight_loader 的定位键
    def apply(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        # SOURCE: vllm/model_executor/models/utils.py:L136-L152 apply
        for name, data in weights:
            result = self._map_name_with_shard(name)
            if result is None:
                continue
            out_name, shard_id = result
            if shard_id is not None:
                data.shard_id = shard_id
            yield out_name, data

    # SUBTRACTED: apply_list/apply_dict/get_unstacked_mapper
    #   （models/utils.py:L154-L175）——LoRA/量化侧消费面

    # SUBTRACTED: get_unstacked_mapper（models/utils.py:L177-L~190）——LoRA
    #   名字解析与量化层表的 constituent 名保留面（ch27/LoRA 域）


# SOURCE: vllm/model_executor/models/utils.py:L173 AutoWeightsLoader
class AutoWeightsLoader:
    """
    Helper class to load weights into a [`torch.nn.Module`][]. It is able
    to automatically detect child modules and parameters while iterating over
    the weights only once.

    The weight loading logic for individual modules can be overridden
    by defining a `load_weights` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a `weight_loader` method.

    Detailed weight loading information can be viewed by setting
    the environment variable `VLLM_LOGGING_LEVEL=DEBUG`.
    """

    # SOURCE: vllm/model_executor/models/utils.py:L189-L196 早期训练器遗留
    #   跳过表（逐字）
    # Models trained using early version ColossalAI or quantized by
    # GPTQModel may include these tensors in checkpoint. Skip them.
    ROTARY_EMBEDS_UNUSED_WEIGHTS = [
        "rotary_pos_emb.inv_freq",
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    ]

    # SOURCE: vllm/model_executor/models/utils.py:L198-L215 __init__（逐字）
    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: list[str] | None = None,
        skip_substrs: list[str] | None = None,
        ignore_unexpected_prefixes: list[str] | None = None,
        ignore_unexpected_suffixes: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.module = module
        self.skip_prefixes = skip_prefixes or []
        self.skip_substrs = skip_substrs or []
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []
        self.ignore_unexpected_suffixes = ignore_unexpected_suffixes or []
        # update default skip_substrs
        self.skip_substrs += self.ROTARY_EMBEDS_UNUSED_WEIGHTS

    # SOURCE: vllm/model_executor/models/utils.py:L217-L235 _groupby_prefix
    #   （逐字——按首段前缀分组：递归分发的骨架）
    def _groupby_prefix(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
        # SOURCE: vllm/model_executor/models/utils.py:L217-L235 _groupby_prefix
        weights_by_parts = (
            (weight_name.split(".", 1), weight_data)
            for weight_name, weight_data in weights
        )

        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):
            yield (
                prefix,
                # Because maxsplit=1 in weight_name.split(...),
                # the length of `parts` must either be 1 or 2
                (
                    ("" if len(parts) == 1 else parts[1], weights_data)
                    for parts, weights_data in group
                ),
            )

    # SOURCE: vllm/model_executor/models/utils.py:L237-L243 _get_qualname（逐字）
    def _get_qualname(self, prefix: str, rest: str) -> str:
        if prefix == "":
            return rest
        if rest == "":
            return prefix

        return ".".join((prefix, rest))

    # SOURCE: vllm/model_executor/models/utils.py:L245-L248 _can_skip（逐字）
    def _can_skip(self, qualname: str) -> bool:
        return any(qualname.startswith(p) for p in self.skip_prefixes) or any(
            substr in qualname for substr in self.skip_substrs
        )

    # SOURCE: vllm/model_executor/models/utils.py:L250-L253 _can_ignore_unexpected
    #   （逐字）
    def _can_ignore_unexpected(self, qualname: str) -> bool:
        # SOURCE: vllm/model_executor/models/utils.py:L250-L253 _can_ignore_unexpected
        iup = (qualname.startswith(p) for p in self.ignore_unexpected_prefixes)
        ius = (qualname.endswith(s) for s in self.ignore_unexpected_suffixes)
        return any(iup) or any(ius)

    # SOURCE: vllm/model_executor/models/utils.py:L255-L280 _load_param
    #   子集（主干逐字：weight_loader 属性优先、default 兜底）
    def _load_param(
        self,
        base_prefix: str,
        param: nn.Parameter,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        # SOURCE: vllm/model_executor/models/utils.py:L255-L280 _load_param
        for weight_name, weight_data in weights:
            weight_qualname = self._get_qualname(base_prefix, weight_name)

            if self._can_skip(weight_qualname):
                logger.debug("Skipping weight %s", weight_qualname)

                continue

            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    logger.debug("Ignoring weight %s", weight_qualname)

                    continue

                raise ValueError(
                    f"Attempted to load nested weight {weight_qualname!r} "
                    f"into a single parameter {base_prefix!r}"
                )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight_data)

            logger.debug("Loaded weight %s with shape %s", weight_qualname, param.shape)

            yield weight_qualname

    # SUBTRACTED: _add_loadable_non_param_tensors（models/utils.py:L282-L317）
    #   ——batchnorm 统计等非参数张量面（本章模型无此面）

    # SOURCE: vllm/model_executor/models/utils.py:L317-L398 _load_module
    #   子集（主干逐字：PPMissingLayer 早退、子模块 load_weights 委派、
    #   前缀分组递归、无主错误；StageMissingLayer/batchnorm 面删除）
    def _load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        # SOURCE: vllm/model_executor/models/utils.py:L322-L323 PPMissingLayer
        #   早退（逐字——空段权重直接跳过）
        if isinstance(module, PPMissingLayer):
            return

        # Avoid infinite recursion since this function is typically
        # called inside load_weights of the module itself
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    logger.warning(
                        "Unable to collect loaded parameters for module %s", module
                    )
                else:
                    yield from map(
                        lambda x: self._get_qualname(base_prefix, x),
                        loaded_params,
                    )

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))

        # SUBTRACTED: _add_loadable_non_param_tensors（models/utils.py:L338）
        #   ——batchnorm 面删除

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:
                if self._can_skip(prefix + "."):
                    logger.debug("Skipping module %s", prefix)

                    continue

                yield from self._load_module(
                    prefix, child_modules[child_prefix], child_weights
                )
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(
                    prefix, child_params[child_prefix], child_weights
                )
            else:
                can_skip_module = self._can_skip(prefix + ".")
                can_skip_param = self._can_skip(prefix)
                if can_skip_module or can_skip_param:
                    logger.debug("Skipping missing %s", prefix)

                    continue

                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    logger.debug("Ignoring missing %s", prefix)

                    continue

                named_parameters = module.named_parameters(recurse=True)
                desc_param_keys = {
                    maybe_prefix(base_prefix, k) for k, _ in named_parameters
                }
                msg = (
                    f"There is no module or parameter named {prefix!r} "
                    f"in {self.module._get_name()}. "
                    f"The available parameters belonging to {base_prefix} "
                    f"({module._get_name()}) are: {desc_param_keys}"
                )
                raise ValueError(msg)

    # SOURCE: vllm/model_executor/models/utils.py:L398-L424 load_weights
    #   子集（@support_quantized_model_reload_from_hp_weights 装饰删除——torchao
    #   量化 reload 域；主体逐字：ignore bias 尾、quant_config 的 cache scale
    #   mapper 判位、mapper.apply、skip 过滤、_load_module 收集）
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: WeightsMapper | None = None,
    ) -> set[str]:
        # SUBTRACTED: @support_quantized_model_reload_from_hp_weights（L397）
        #   ——torchao 高精度 reload 面（ch27）
        # Ignore unexpected biases (typically from GPTQ models)
        # SOURCE: vllm/model_executor/models/utils.py:L398-L424 load_weights
        self.ignore_unexpected_suffixes.append(".bias")

        # SUBTRACTED: quant_config.get_cache_scale_mapper 合并段
        #   （models/utils.py:L407-L416）——KV cache scale 映射归 ch27
        if mapper is not None:
            weights = mapper.apply(weights)
        # filter out weights with first-prefix/substr to skip in name
        weights = (
            (name, weight) for name, weight in weights if not self._can_skip(name)
        )

        autoloaded_weights = set(self._load_module("", self.module, weights))
        return autoloaded_weights


# SUBTRACTED: maybe_fuse_shared_experts/get_spec_layer_idx_from_weight_name/
#   skip_spec_layers/init_vllm_registered_model/flatten_bn/多模态嵌入合面/
#   isin_list（models/utils.py:L428-L686）——MoE/spec/多模态域
# SUBTRACTED: StageMissingLayer/collect_children/no_init_weights
#   （models/utils.py:L687-L780）——enc-dec 分段与初始化域


# SOURCE: vllm/model_executor/models/utils.py:L781 LayerFn 协议（逐字）
class LayerFn(Protocol):
                                                        # SOURCE: vllm/model_executor/models/utils.py:L781 LayerFn 协议（逐字）
    def __call__(self, prefix: str) -> torch.nn.Module: ...


# SOURCE: vllm/model_executor/models/utils.py:L785 PPMissingLayer
#   torch.nn.Identity 子类哨兵：占空段保参数名稳定
class PPMissingLayer(torch.nn.Identity):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    # SOURCE: vllm/model_executor/models/utils.py:L785 PPMissingLayer
    def __init__(self, *args, **kwargs):
        # SOURCE: vllm/model_executor/models/utils.py:L785 PPMissingLayer
        super().__init__()

    def forward(self, *args, **kwargs):
        """Return the first arg from args or the first value from kwargs."""
        # SOURCE: vllm/model_executor/models/utils.py:L793-L796 PPMissingLayer.forward
        return args[0] if args else next(iter(kwargs.values()))


# SOURCE: vllm/model_executor/models/utils.py:L798 make_layers
#   get_pp_indices 均分 [start,end) + ModuleList 三段拼装：PPMissingLayer×头
#   + 真层×(end-start) + PPMissingLayer×尾；offloader 包装删除——offload 域）
def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str,
) -> tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.

    Args:
        num_hidden_layers: Total number of hidden layers in the model.
        layer_fn: Function to create a layer given its index.
        prefix: Prefix for layer names.

    Returns:
        Tuple of (start_layer, end_layer, modules).
    """
    # SOURCE: vllm/model_executor/models/utils.py:L798 make_layers
    from vllm.distributed import get_pp_group, get_pp_indices

    # SUBTRACTED: from vllm.model_executor.offloader import get_offloader
    #   （models/utils.py:L818）——权重 offload 域；wrap_modules 恒等包装

    start_layer, end_layer = get_pp_indices(
        num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
    )

    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)]
        + [layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)]
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
    )

    return start_layer, end_layer, modules


# SUBTRACTED: get_pp_missing_layer_names（models/utils.py:L837-L853）——
#   loader 侧跳过名单缓存（本章 _load_module 的 PPMissingLayer isinstance
#   早退已覆盖）；is_pp_missing_parameter（L855-L863）——同域


# SOURCE: vllm/model_executor/models/utils.py:L866-L877
#   make_empty_intermediate_tensors_factory（逐字——PP 空中间张量钩子工厂：
#   ['hidden_states','residual'] 两字段零张量）
def make_empty_intermediate_tensors_factory(keys: list[str], hidden_size: int):
    # SOURCE: vllm/model_executor/models/utils.py:L866-L877
    def make_empty_intermediate_tensors(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # SOURCE: vllm/model_executor/models/utils.py:L866-L877
        return IntermediateTensors(
            {
                key: torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
                for key in keys
            }
        )

    return make_empty_intermediate_tensors


# SOURCE: vllm/model_executor/models/utils.py:L882-L892 maybe_prefix（逐字）
def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"


# SUBTRACTED: get_draft_quant_config/cast_overflow_tensors/fast_topk/
#   sequence_parallel_chunk/process_eagle_weight/get_layer_index/
#   scatter_output_slices/diarized 解析（models/utils.py:L895-L~1110）——
#   spec/序列并行/多模态域


# SOURCE: vllm/model_executor/models/utils.py:L917 extract_layer_index
#   子集（主干逐字：层名里的整数抽取；num_attn_module>1 的复合层号支删除）
def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """
    Extract the layer index from the module name.
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError if num_attn_module == 1
    """
    # SOURCE: vllm/model_executor/models/utils.py:L917 extract_layer_index
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    if num_attn_module == 1 or "attn" not in layer_name:
        assert len(int_vals) == 1, (
            f"layer name {layer_name} should only contain one integer"
        )

        return int_vals[0]
    # SUBTRACTED: num_attn_module>1 的复合层号支（models/utils.py:L938-L950，
    #   else 分支——多注意力模块层的双整数解析）——特例模型域，本章无消费位
