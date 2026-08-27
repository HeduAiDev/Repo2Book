"""Driver for m10 (GPU 物理页布局：real_page_size_bytes 公式 + 每层
[num_blocks, 2, block_size, kv_heads, head_dim] 视图) — host run against the
ch13 companion（spec 对象为源码逐字公式；HOST SEAM 视图在 CPU 张量上验证）。

算术五连：
  1) 页公式小例：2×16×8×128×2 = 65536 B
  2) Llama-2-7B 每 token 每层：2×32×128×2 = 16384 B
  3) 每 token 全模型（×32 层）：524288 B = 0.5 MB
  4) 4096-token 序列的 KV 总量：2147483648 B = 2 GiB
  5) worker 换算：655360 B // 65536 B = 10 块 → 视图 [10, 2, 16, 8, 128]
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from implementation.kv_cache_interface import FullAttentionSpec  # noqa: E402
from implementation.torch_utils import get_dtype_size  # noqa: E402

BLOCK_SIZE = 16


def main():
    # 1) 页公式小例（fp16、8 kv_head、head 128）
    spec_small = FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
    )
    f1 = {
        "formula": "2(K,V) × block_size × num_kv_heads × head_size × dtype_bytes",
        "terms": [2, 16, 8, 128, 2],
        "real_page_size_bytes": spec_small.real_page_size_bytes,
        "page_size_bytes_no_quant_padding": spec_small.page_size_bytes,
    }

    # 2) Llama-2-7B 口径（32 kv_head = 32 层 GQA 每层 32 组？——Llama-2-7B 是 MHA，
    #    32 个注意力头全部当 kv_head 用，head_dim 128）
    spec_llama = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=32, head_size=128, dtype=torch.float16
    )
    per_token_per_layer = 2 * 32 * 128 * get_dtype_size(torch.float16)
    f2 = {
        "formula": "2 × 32(kv_heads) × 128(head_dim) × 2(fp16 B)",
        "terms": [2, 32, 128, 2],
        "per_token_per_layer_bytes": per_token_per_layer,
        "llama_page_bytes": spec_llama.real_page_size_bytes,
        "llama_page_check": f"{BLOCK_SIZE} × {per_token_per_layer} = {spec_llama.real_page_size_bytes}",
    }

    # 3) 每 token 全模型 + 4) 4096-token 序列
    NUM_LAYERS = 32
    per_token_all = per_token_per_layer * NUM_LAYERS
    seq4096 = per_token_all * 4096
    f3 = {
        "num_layers": NUM_LAYERS,
        "per_token_all_layers_bytes": per_token_all,
        "per_token_all_layers_mb": per_token_all / 1024 / 1024,
        "per_token_all_layers_mb_str": "0.5",
        "seq_len": 4096,
        "seq_kv_bytes": seq4096,
        "seq_kv_gib": seq4096 / 1024**3,
        "seq_kv_gib_str": "2",
        "note": "deepread/memory-kv.json why_chains[1] 口径的计算例（非源码断言）——权重之外的全部显存都是 KV 的粮仓",
    }

    # 5) worker 换算 + 视图（gpu_model_runner.py:L7400-L7413）
    NUM_BLOCKS = 10
    raw = torch.zeros(NUM_BLOCKS * spec_small.page_size_bytes, dtype=torch.int8)
    num_blocks = raw.numel() // spec_small.page_size_bytes
    view = raw.view(torch.float16).view(NUM_BLOCKS, 2, 16, 8, 128)
    f5 = {
        "raw_bytes": raw.numel(),
        "page_size_bytes": spec_small.page_size_bytes,
        "divisible": raw.numel() % spec_small.page_size_bytes == 0,
        "num_blocks": num_blocks,
        "view_shape": list(view.shape),
        "view_shape_str": "[10, 2, 16, 8, 128]",
        "k_half": "view[b][0] = 第 b 块的 K 半页；view[b][1] = V 半页——page_size_bytes 里的 2× 就是 K/V 两半",
        "scheduler_side_same_source": "调度器侧 num_blocks 由同一份 KVCacheConfig 定（单一事实源；池多大/profile 三步定账 → ch14）",
    }

    assert f1["real_page_size_bytes"] == 65536
    assert per_token_per_layer == 16384
    assert per_token_all == 524288 and round(f3["per_token_all_layers_mb"], 1) == 0.5
    assert seq4096 == 2147483648 and round(f3["seq_kv_gib"]) == 2
    assert num_blocks == 10 and list(view.shape) == [10, 2, 16, 8, 128]

    out = {
        "driver": "run_m10_page_shape.py",
        "mechanism": "m10 GPU 物理页布局（kv_cache_interface.py:L184-L226 / gpu_model_runner.py:L7312-L7353 + L7400-L7413）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch13 implementation/ 只做减法精简版（公式逐字；视图在 CPU 张量上验证——HOST SEAM，数值与 GPU 同）",
        "environment_note": "host CPU 张量承载同一 view 换算；真实 GPU 上 backend 可仲裁不同布局（get_kv_cache_shape → ch21），页字节数不变",
        "config": {"block_size": BLOCK_SIZE, "default_block_size_source": "vllm/config/cache.py:L47 DEFAULT_BLOCK_SIZE = 16"},
        "f1_page_formula_small": f1,
        "f2_per_token_per_layer_llama2_7b": f2,
        "f3_per_token_and_4096_seq": f3,
        "f5_worker_num_blocks_and_view": f5,
    }

    dst = Path(__file__).resolve().parent / "m10_page_shape.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(f1, ensure_ascii=False))
    print(json.dumps(f2, ensure_ascii=False))
    print(json.dumps(f3, ensure_ascii=False))
    print(json.dumps(f5, ensure_ascii=False))


if __name__ == "__main__":
    main()
