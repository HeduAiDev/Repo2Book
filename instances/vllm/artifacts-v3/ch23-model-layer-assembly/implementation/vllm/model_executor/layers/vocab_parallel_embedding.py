# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py
# ch23 主角文件之四（m12）：词表分片积木——VocabParallelEmbedding（词表按 TP
# 均分、pad 到 64 倍数）+ ParallelLMHead（其子类；compute_logits 的投影头）。
# SUBTRACTED：量化 embedding 方法族、LoRA 追加词表面的大半 docstring 例、
#   packed_dim 装载分支（delete[7]）、get_sharded_to_full_mapping（采样重排
#   归 ch29）。
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

# SUBTRACTED: import vllm.model_executor.layers.batch_invariant 的
#   linear_batch_invariant（vocab_parallel_embedding.py:L17-L19）——ch20 域


# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L32
#   DEFAULT_VOCAB_PADDING_SIZE —— 词表 padding 基数（m5 裁剪的对面）
DEFAULT_VOCAB_PADDING_SIZE = 64


# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L35
#   UnquantizedEmbeddingMethod
class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L37-L61
    #   create_weights（逐字——[sum(output_partition_sizes), input] 权重 +
    #   input_dim=1/output_dim=0 戳记）
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L37-L61
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L61-L65
    #   process_weights_after_loading —— 减法子集（CPU dispatch 分支删除）
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # SUBTRACTED: CPU dispatch_cpu_unquantized_gemm（L62-L65）——平台域
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L61-L65
        return

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L67-L75 apply
    #   —— 减法子集（batch invariant 分支删除——ch20 域；dispatch 主路径逐字）
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SUBTRACTED: envs.VLLM_BATCH_INVARIANT 分支（L68-L69）——ch20 域
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L67-L75 apply
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L77-L78
    #   embedding（逐字——F.embedding）
    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L77-L78
        return F.embedding(input_, layer.weight)

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L80-L84
    #   tie_weights（逐字——lm_head tie 的委托目标）
    def tie_weights(
        self, layer: torch.nn.Module, embed_tokens: "VocabParallelEmbedding"
    ):
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L80-L84
        layer.weight = embed_tokens.weight
        return layer


# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L87 pad_vocab_size
def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L90
#   vocab_range_from_per_partition_vocab_size（逐字）
def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank: int, offset: int = 0
) -> Sequence[int]:
    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L90
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f + offset, index_l + offset


# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L98
#   vocab_range_from_global_vocab_size（逐字）
def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int, offset: int = 0
) -> Sequence[int]:
    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L98
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank, offset=offset
    )


# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L110
#   VocabParallelEmbeddingShardIndices（逐字——rank 的词表区间账）
@dataclass
class VocabParallelEmbeddingShardIndices:
    """Indices for a shard of a vocab parallel embedding."""

    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int

    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L128-L131
    #   num_org_elements（逐字）
    @property
    def num_org_elements(self) -> int:
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L128-L131
        return self.org_vocab_end_index - self.org_vocab_start_index

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L133-L136
    #   num_added_elements（逐字）
    @property
    def num_added_elements(self) -> int:
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L133-L136
        return self.added_vocab_end_index - self.added_vocab_start_index

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L138-L141
    #   num_org_elements_padded（逐字）
    @property
    def num_org_elements_padded(self) -> int:
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L138-L141
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L143-L146
    #   num_added_elements_padded（逐字）
    @property
    def num_added_elements_padded(self) -> int:
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L143-L146
        return (
            self.padded_added_vocab_end_index - self.padded_added_vocab_start_index
        )

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L148-L151
    #   num_org_vocab_padding（逐字——get_top_tokens/weight_loader 的 padding 边界）
    @property
    def num_org_vocab_padding(self) -> int:
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L148-L151
        return self.num_org_elements_padded - self.num_org_elements

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L153-L156
    #   num_added_vocab_padding（逐字）
    @property
    def num_added_vocab_padding(self) -> int:
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L153-L156
        return self.num_added_elements_padded - self.num_added_elements

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L158-L161
    #   num_elements_padded（逐字）
    @property
    def num_elements_padded(self) -> int:
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L158-L161
        return self.num_org_elements_padded + self.num_added_elements_padded

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L163-L169
    #   __post_init__ sanity 检查（逐字）
    def __post_init__(self):
        # sanity checks
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L163-L169
        assert self.padded_org_vocab_start_index <= self.padded_org_vocab_end_index
        assert self.padded_added_vocab_start_index <= self.padded_added_vocab_end_index

        assert self.org_vocab_start_index <= self.org_vocab_end_index
        assert self.added_vocab_start_index <= self.added_vocab_end_index

        assert self.org_vocab_start_index <= self.padded_org_vocab_start_index
        assert self.added_vocab_start_index <= self.padded_added_vocab_start_index
        assert self.org_vocab_end_index <= self.padded_org_vocab_end_index
        assert self.added_vocab_end_index <= self.padded_added_vocab_end_index

        assert self.num_org_elements <= self.num_org_elements_padded
        assert self.num_added_elements <= self.num_added_elements_padded


# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L169-L193
#   get_masked_input_and_mask —— 减法子集（@torch.compile 装饰删除——ch19 域；
#   mask 数学逐字）
def get_masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # torch.compile will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L169-L193
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index
    )
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


# --8<-- [start:vocab_parallel_embedding]
# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L198
#   @PluggableLayer.register("vocab_parallel_embedding")
@PluggableLayer.register("vocab_parallel_embedding")
# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L198
#   VocabParallelEmbedding
class VocabParallelEmbedding(PluggableLayer):
    """Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    In order to support various loading methods, we ensure that LoRA-added
    embeddings are always at the end of TP-sharded tensors. In other words,
    we shard base embeddings and LoRA embeddings separately (both padded),
    and place them in the same tensor.
    """

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L240-L334
    #   __init__（逐字——pad_vocab_size 两次、shard_indices 区间账、
    #   量化方法选择位）
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        disable_tp: bool = False,
    ):
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L240-L334
        super().__init__()

        # Keep the input dimensions.
        self.disable_tp = disable_tp
        if disable_tp:
            tp_rank, self.tp_size = 0, 1
        else:
            tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = tp_rank
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size, self.padding_size
        )
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim

        quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()

        # If we are making an embedding layer, then our quantization linear
        # method must implement the embedding operation. If we are another
        # layer type like ParallelLMHead, this is not important.
        is_embedding_layer = type(self) is VocabParallelEmbedding
        quant_method_implements_embedding = method_has_implemented_embedding(
            type(quant_method)
        )
        if is_embedding_layer and not quant_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )

        self.quant_method: QuantizeMethodBase = quant_method

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        # Divide the weight matrix along the vocabulary dimension.
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded, self.tp_size
        )
        assert (
            self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        )
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index
            - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index
            - self.shard_indices.added_vocab_start_index
        )

        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )
        self.update_param_tp_status()

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L336-L340
    #   update_param_tp_status（逐字）
    def update_param_tp_status(self):
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L336-L340
        for param in self.parameters():
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L342-L399
    #   _get_indices（逐字——rank 区间账的推导本体）
    @classmethod
    def _get_indices(
        cls,
        vocab_size_padded: int,
        org_vocab_size_padded: int,
        vocab_size: int,
        org_vocab_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> VocabParallelEmbeddingShardIndices:
        """Get start and end indices for vocab parallel embedding, following the
        layout outlined in the class docstring, based on the given tp_rank and
        tp_size."""
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L342-L399
        num_added_embeddings_padded = vocab_size_padded - org_vocab_size_padded
        padded_org_vocab_start_index, padded_org_vocab_end_index = (
            vocab_range_from_global_vocab_size(org_vocab_size_padded, tp_rank, tp_size)
        )
        padded_added_vocab_start_index, padded_added_vocab_end_index = (
            vocab_range_from_global_vocab_size(
                num_added_embeddings_padded, tp_rank, tp_size, offset=org_vocab_size
            )
        )
        # remove padding
        org_vocab_start_index = min(padded_org_vocab_start_index, org_vocab_size)
        org_vocab_end_index = min(padded_org_vocab_end_index, org_vocab_size)
        added_vocab_start_index = min(padded_added_vocab_start_index, vocab_size)
        added_vocab_end_index = min(padded_added_vocab_end_index, vocab_size)
        return VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index,
            padded_org_vocab_end_index,
            padded_added_vocab_start_index,
            padded_added_vocab_end_index,
            org_vocab_start_index,
            org_vocab_end_index,
            added_vocab_start_index,
            added_vocab_end_index,
        )

    # SUBTRACTED: get_sharded_to_full_mapping（vocab_parallel_embedding.py:
    #   L380-L443）——采样 gather 后的 index→token_id 重排，归 ch29 采样域

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L445-L502
    #   weight_loader —— 减法子集（delete[7]：packed_dim 分支删除；本秩区间
    #   narrow + 尾部 zero-fill 主干逐字）
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L445-L502
        output_dim = getattr(param, "output_dim", None)
        # SUBTRACTED: packed_dim 判整与 packed_factor 换算（L467-L492）——
        #   delete[7]，量化装载归 ch27

        # If parameter does not have output dim, then it should
        # be copied onto all gpus (e.g. g_idx for act_order gptq).
        if output_dim is None:
            if (
                loaded_weight.ndim == 0
                and param.data.ndim == 1
                and param.data.numel() == 1
            ):
                loaded_weight = loaded_weight.reshape(1)
            assert param.data.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)
            return

        # Shard indexes for loading the weight
        start_idx = self.shard_indices.org_vocab_start_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx

        assert loaded_weight.shape[output_dim] == self.org_vocab_size

        # Copy the data. Select chunk corresponding to current shard.
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
        param[loaded_weight.shape[0] :].data.fill_(0)

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L487-L502
    #   forward（逐字——TP>1 时的 mask+all_reduce / tp=1 直通）
    def forward(self, input_):
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L487-L502
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self, masked_input.long())
        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            # Reduce across all the model parallel GPUs.
            return tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L509-L515
    #   extra_repr（逐字）
    def extra_repr(self) -> str:
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L509-L515
        s = f"num_embeddings={self.num_embeddings_per_partition}"
        s += f", embedding_dim={self.embedding_dim}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", num_embeddings_padded={self.num_embeddings_padded}"
        s += f", tp_size={self.tp_size}"
        return s


# --8<-- [start:parallel_lm_head]
# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L519
#   @PluggableLayer.register("parallel_lm_head")
@PluggableLayer.register("parallel_lm_head")
# SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L520 ParallelLMHead
class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
        disable_tp: If true, tensor parallelism will be disabled for this layer.
    """

    # --8<-- [end:parallel_lm_head]

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L539-L568
    #   __init__（逐字）
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        disable_tp: bool = False,
    ):
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L539-L568
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
            disable_tp=disable_tp,
        )
        self.quant_config = quant_config
        if bias:
            self._register_bias()
        else:
            self.register_parameter("bias", None)

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L570-L575
    #   _register_bias（逐字）
    def _register_bias(self):
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L570-L575
        data = torch.empty(self.num_embeddings_per_partition, dtype=self.params_dtype)
        self.bias = Parameter(data, requires_grad=False)
        weight_attrs = dict(output_dim=0, weight_loader=self.weight_loader)
        set_weight_attrs(weight=self.bias, weight_attrs=weight_attrs)

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L577-L579
    #   tie_weights（逐字——tie_word_embeddings 的复用通道）
    def tie_weights(self, embed_tokens: VocabParallelEmbedding):
        """Tie the weights with word embeddings."""
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L577-L579
        return self.quant_method.tie_weights(self, embed_tokens)

    # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L581-L582
    #   forward（逐字——权重只在 sampler/compute_logits 用）
    def forward(self, input_):
        # SOURCE: vllm/model_executor/layers/vocab_parallel_embedding.py:L581-L582
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
