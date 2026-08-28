# ch15 m3 extra keys 驱动：mm/lora/cache_salt(仅首块) 拌进哈希——同 token 不同语义必不同哈希
# （kv_cache_utils.py:L430-L447 谓词 + L558-L593 四源并三源组装；盐差沿 parent 链传播）。
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
from implementation.request import Request  # noqa: E402

kcu.init_none_hash(sha256)
HASHER16 = kcu.get_request_block_hasher(16, sha256)

out = {"params": {
    "hash_block_size": 16,
    "prompt_tokens": 64,
    "满块数": 4,
    "salt_a": "tenant-a",
    "salt_b": "tenant-b",
    "lora_name": "adapter-1",
    "mm_identifier": "img-1",
}}

# --- 1) 谓词 need_extra_keys：mm / lora / cache_salt 任一存在 ---
plain = Request("p", [1] * 32, block_hasher=HASHER16)
salted = Request("s", list(range(64)), block_hasher=HASHER16, cache_salt="tenant-a")
lora = Request("l", list(range(32)), block_hasher=HASHER16,
               lora_request=SimpleNamespace(lora_name="adapter-1"))
mm = Request("m", [1] * 32, block_hasher=HASHER16,
             mm_features=[SimpleNamespace(
                 identifier="img-1",
                 mm_position=SimpleNamespace(offset=0, length=4))])
out["predicate"] = {
    "plain": kcu.need_extra_keys(plain),
    "salted": kcu.need_extra_keys(salted),
    "lora": kcu.need_extra_keys(lora),
    "mm": kcu.need_extra_keys(mm),
}

# --- 2) cache_salt 只拌首块（start_token_idx==0），但盐差沿链传播到所有块 ---
k0, mm0 = kcu.generate_block_hash_extra_keys(salted, 0, 16, 0)
k1, mm1 = kcu.generate_block_hash_extra_keys(salted, 16, 32, mm0)
k3, mm3 = kcu.generate_block_hash_extra_keys(salted, 48, 64, mm1)
other = Request("o", list(range(64)), block_hasher=HASHER16, cache_salt="tenant-b")
out["cache_salt"] = {
    "block0_start_token_idx": 0,
    "block0_extra_keys": list(k0),
    "block1_start_token_idx": 16,
    "block1_extra_keys_is_none": k1 is None,
    "block3_start_token_idx": 48,
    "block3_extra_keys_is_none": k3 is None,
    "hash0_a_hex_head": salted.block_hashes[0].hex()[:12],
    "hash0_b_hex_head": other.block_hashes[0].hex()[:12],
    "block0_tokens_identical": True,
    "hash0_differs": salted.block_hashes[0] != other.block_hashes[0],
    "hash1_differs_chain_propagation":
        salted.block_hashes[1] != other.block_hashes[1],
    "hash2_differs_chain_propagation":
        salted.block_hashes[2] != other.block_hashes[2],
    "hash3_differs_chain_propagation":
        salted.block_hashes[3] != other.block_hashes[3],
    "num_blocks_compared": 4,
}

# --- 3) lora 名每块都拌（不只首块） ---
lk0, _ = kcu.generate_block_hash_extra_keys(lora, 0, 16, 0)
lk1, _ = kcu.generate_block_hash_extra_keys(lora, 16, 32, 0)
out["lora"] = {
    "block0_extra_keys": list(lk0),
    "block1_extra_keys": list(lk1),
    "lora_in_every_block": lk0 == lk1 == ("adapter-1",),
}

# --- 4) mm：同图不同位置 → extra_keys 带块内偏移（offset - start_token_idx） ---
mm_req = Request("m2", [1] * 48, block_hasher=HASHER16,
                 mm_features=[SimpleNamespace(
                     identifier="img-1",
                     mm_position=SimpleNamespace(offset=8, length=4))])
mk0, mmi0 = kcu.generate_block_hash_extra_keys(mm_req, 0, 16, 0)
mm_req2 = Request("m3", [1] * 48, block_hasher=HASHER16,
                  mm_features=[SimpleNamespace(
                      identifier="img-1",
                      mm_position=SimpleNamespace(offset=20, length=4))])
mk0b, _ = kcu.generate_block_hash_extra_keys(mm_req2, 16, 32, 0)
out["mm_offset"] = {
    "img_at_token_offset": 8,
    "block0_extra_keys": [list(e) if isinstance(e, tuple) else e for e in mk0],
    "img_at_token_offset_2": 20,
    "block1_extra_keys": [list(e) if isinstance(e, tuple) else e for e in mk0b],
    "same_img_different_position_different_keys": mk0 != mk0b,
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m3.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
