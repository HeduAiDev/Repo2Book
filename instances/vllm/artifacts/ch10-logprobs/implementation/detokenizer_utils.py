# 只做减法的忠实精简版 —— 镜像 vllm/tokenizers/detokenizer_utils.py（pin f3fef123）
# 本章只用到非增量逐 token 去 token：convert_ids_list_to_tokens。
#
# SUBTRACTED: SPDX 版权头 + TokenizerLike 协议定义（detokenizer_utils.py:L1-L80）——
#             本章只需 tokenizer.decode([id]) 的鸭子类型接口，测试用桩 tokenizer 即可。
# SUBTRACTED: detokenize_incrementally 及其余增量去 token 逻辑（detokenizer_utils.py:L107+）——
#             属于 detokenizer 章节；logprobs 走的是非增量路径，删去不影响正确性。
from typing import Protocol


class TokenizerLike(Protocol):
    # SOURCE: vllm/tokenizers/detokenizer_utils.py:L83(TokenizerLike 接口)
    # 精简到本章实际使用的唯一方法：把 token id 列表解成字符串。
    def decode(self, token_ids: list[int]) -> str:  # SOURCE: vllm/tokenizers/detokenizer_utils.py:L83(TokenizerLike.decode)
        ...


def convert_ids_list_to_tokens(
    tokenizer: TokenizerLike,
    token_ids: list[int],
) -> list[str]:
    # SOURCE: vllm/tokenizers/detokenizer_utils.py:L83-L104
    """Detokenize the input ids individually.

    Args:
      tokenizer: tokenizer used by model under test
      token_ids: convert these tokens (Python list form)

    Returns:
      Python list of token string representations

    """
    token_str_lst = []
    for token_id in token_ids:
        # use default skip_special_tokens.
        token_str = tokenizer.decode([token_id])
        if token_str is None:
            token_str = ""
        token_str_lst.append(token_str)
    return token_str_lst
