# SOURCE: vllm/utils/mistral.py
# HOST SEAM：本章消费面只有 is_mistral_tokenizer（sampling_params 的
# _is_non_tekken_mistral 判据与 guidance 显式分支的拒单检查）。函数本体逐字；
# 其依赖的 vllm.tokenizers.mistral（MistralTokenizer 类）属 Mistral 生态域、
# 不进精简版——非 Mistral 分词器下第一合取项 False 短路，mt 永不加载。
# SUBTRACTED: SPDX 版权头；is_mistral_tool_parser 等文件其余。
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    import vllm.tokenizers.mistral as mt
else:
    mt = LazyLoader("mt", globals(), "vllm.tokenizers.mistral")

# HOST SEAM：真实类型是 vllm/tokenizers/protocol.py 的 TokenizerLike 协议
TokenizerLike = Any


# SOURCE: vllm/utils/mistral.py:L19-L28 is_mistral_tokenizer —— 逐字
def is_mistral_tokenizer(obj: TokenizerLike | None) -> TypeGuard[Any]:
    """Return true if the tokenizer is a MistralTokenizer instance."""
    cls = type(obj)
    # Check for special class attribute, this avoids importing the class to
    # do an isinstance() check.  If the attribute is True, do an isinstance
    # check to be sure we have the correct type.
    return bool(
        getattr(cls, "IS_MISTRAL_TOKENIZER", False)
        and isinstance(obj, mt.MistralTokenizer)
    )
