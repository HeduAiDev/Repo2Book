"""Shared fixtures / path setup for ch10 logprobs tests.

精简版用 bare import（from logprobs import ...），需把 implementation/ 加进
sys.path。这里同时提供桩 tokenizer / 桩张量 / 桩 detokenizer，复现真实
vLLM 的可观察行为（不 import vllm）。
"""
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))


class ByteFallbackTokenizer:
    """复现 SentencePiece/BPE 的 byte-fallback：把多字节 UTF-8 字符拆成
    多个 byte 级 token。decode([单个 byte token]) 得到不完整字节序列 →
    UTF-8 errors='replace' 解出 U+FFFD（�），正是 logprobs decoded_token
    踩坑的来源。

    token id 约定：
      - 0..255 为 byte token，decode 出该单字节（可能不完整 → �）。
      - >=1000 为普通 ASCII 文本 token（id - 1000 是 ord）。
    decode(list) 把各 token 的字节拼起来一次性 UTF-8 解码。
    """

    def _bytes_for(self, token_id: int) -> bytes:
        if 0 <= token_id <= 255:
            return bytes([token_id])
        return chr(token_id - 1000).encode("utf-8")

    def decode(self, token_ids: list[int]) -> str:
        raw = b"".join(self._bytes_for(t) for t in token_ids)
        return raw.decode("utf-8", errors="replace")


class IdentityTokenizer:
    """ASCII 文本 tokenizer：每个 token id 直接映射为单字符（id-1000=ord）。"""

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t - 1000) for t in token_ids)


class FakeArray:
    """最小桩数组：复现本章用到的 numpy/torch 接口子集
    （.shape / .flatten().tolist() / .tolist()）。

    sample 路径用 .tolist()（1D 行）；prompt 路径用 2D .shape/.flatten()/.tolist()。
    """

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data

    @property
    def shape(self):
        rows = len(self._data)
        cols = len(self._data[0]) if rows and isinstance(self._data[0], list) else 0
        return (rows, cols)

    def flatten(self):
        flat = [x for row in self._data for x in row]
        return FakeArray(flat)


class FakeDetokenizer:
    """_new_completion_output 用到的最小 detokenizer 桩。"""

    def __init__(self, text="", output_token_ids=None):
        self._text = text
        self.output_token_ids = output_token_ids or []

    def get_next_output_text(self, finished, delta):
        return self._text


def utf8_byte_token_ids(char: str) -> list[int]:
    """把一个字符的 UTF-8 字节序列映射为 byte token id 列表（每字节一个 token）。"""
    return list(char.encode("utf-8"))
