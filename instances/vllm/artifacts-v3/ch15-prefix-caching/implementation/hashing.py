# SOURCE: vllm/utils/hashing.py
# 前缀哈希的序列化+摘要底座：默认算法 sha256（pickle 序列化任意 picklable
# 对象 → SHA-256 摘要），get_hash_fn_by_name 按配置名取哈希函数
# （config/cache.py:L95 prefix_caching_hash_algo 默认 "sha256"）。
# SUBTRACTED: sha256_cbor/xxhash/xxhash_cbor 三个变体（L43-L79——可选依赖
#   cbor2/xxhash 的序列化变体；默认 sha256，变体不属本章机制面，
#   get_hash_fn_by_name 相应分支一并删）。
import hashlib
import pickle
from collections.abc import Callable
from typing import Any


# SOURCE: vllm/utils/hashing.py:L26 sha256
def sha256(input: Any) -> bytes:
    """Hash any picklable Python object using SHA-256.

    The input is serialized using pickle before hashing, which allows
    arbitrary Python objects to be used. Note that this function does not
    use a hash seed—if you need one, prepend it explicitly to the input.

    Args:
        input: Any picklable Python object.

    Returns:
        Bytes representing the SHA-256 hash of the serialized input.
    """
    # SOURCE: vllm/utils/hashing.py:L39-L40
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(input_bytes).digest()


# SOURCE: vllm/utils/hashing.py:L82 get_hash_fn_by_name
def get_hash_fn_by_name(hash_fn_name: str) -> Callable[[Any], bytes]:
    """Get a hash function by name, or raise an error if the function is not found.

    Args:
        hash_fn_name: Name of the hash function.

    Returns:
        A hash function.
    """
    # SOURCE: vllm/utils/hashing.py:L91-L92（sha256 支；其余分支随变体删）
    if hash_fn_name == "sha256":
        return sha256

    # SOURCE: vllm/utils/hashing.py:L100
    raise ValueError(f"Unsupported hash function: {hash_fn_name}")
