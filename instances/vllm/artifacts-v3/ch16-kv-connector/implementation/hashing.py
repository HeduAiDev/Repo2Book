# SOURCE: vllm/utils/hashing.py
# 本章消费面：sha256（pickle→SHA-256——链式块哈希的 H；example_connector 的
# 文件夹名也用它）。本地前缀缓存的命中链路（ch15 主角）本章只消费。
# SUBTRACTED: sha256_cbor/xxhash 变体族与 get_hash_fn_by_name 分派——
#   可选依赖删（默认 sha256）；safe_hash 走独立折入（见下）。
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


# SOURCE: vllm/utils/hashing.py:L103 safe_hash（example_connector 的文件夹名
#   用它——配置哈希口径：md5、FIPS 环境回退 sha256）
def safe_hash(data: bytes, usedforsecurity: bool = True):
    """Hash for configs, defaulting to md5 but falling back to sha256
    in FIPS constrained environments.

    Args:
        data: bytes
        usedforsecurity: Whether the hash is used for security purposes

    Returns:
        Hash object
    """
    # SOURCE: vllm/utils/hashing.py:L114-L118
    try:
        return hashlib.md5(data, usedforsecurity=usedforsecurity)
    except ValueError:
        return hashlib.sha256(data)
