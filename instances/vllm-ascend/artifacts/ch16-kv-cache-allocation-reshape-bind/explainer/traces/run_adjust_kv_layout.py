"""Drive the精简版 _adjust_kv_layout to prove the block-dim = one-page stride geometry.

Faithfully replicates vllm_ascend/worker/model_runner_v1.py:L4111 `_adjust_kv_layout`
(single-segment form, so we can read out where each *block* lands physically).
Pure CPU torch control flow — runs on host (trace_source="run").

Two cases matching chapter.md 16.6:
  情形一 (degenerate, page==block): page_size_bytes=32 → num_element_per_page=16 == natural stride[0]
  情形二 (page padded > block):     page_size_bytes=48 → num_element_per_page=24 != natural 16, gap=8

Output = raw numbers the figure-spec / worked-example table cite.
"""
import json
import torch


def get_dtype_size(dtype):
    # torch.finfo/iinfo -> bytes; float16 -> 2
    return torch.empty((), dtype=dtype).element_size()


def block_layout(shape, dtype, page_size_bytes):
    """Return the per-block physical layout produced by target_stride[0]=num_element_per_page.

    Mirrors the loop body of _adjust_kv_layout for one segment: block i of the
    as_strided tensor starts at physical element i * target_stride[0]; its real
    data occupies natural stride[0] elements; the leftover is the page-align gap.
    """
    dtype_size = get_dtype_size(dtype)
    num_element_per_page = page_size_bytes // dtype_size

    natural_stride = tuple(torch.empty(shape).stride())
    target_stride = (num_element_per_page, *natural_stride[1:])

    # Materialise the actual as_strided tensor to VERIFY (not assert-by-hand)
    # that block i really lands at i * num_element_per_page elements.
    n_blocks = shape[0]
    raw = torch.arange(num_element_per_page * n_blocks, dtype=dtype)
    t = torch.as_strided(raw, size=shape, stride=target_stride, storage_offset=0)

    real_span = natural_stride[0]           # elements a block's real data occupies
    gap = num_element_per_page - real_span  # page-align空隙, block-index-independent

    blocks = []
    for i in range(n_blocks):
        off = i * target_stride[0]                     # storage_offset (elements) = 24*i
        # cross-check against the live tensor's own storage offset for block i
        observed = t[i].storage_offset()
        assert observed == off, f"block {i}: predicted {off} != observed {observed}"
        blocks.append({
            "block": i,
            "storage_offset_elements": off,
            "data_interval": [off, off + real_span],
            "gap_after_elements": gap,
        })

    return {
        "shape": list(shape),
        "dtype": str(dtype).replace("torch.", ""),
        "page_size_bytes": page_size_bytes,
        "dtype_size": dtype_size,
        "num_element_per_page": num_element_per_page,
        "natural_stride": list(natural_stride),
        "target_stride": list(target_stride),
        "block_stride_0": t.stride()[0],
        "real_span_elements": real_span,
        "gap": gap,
        "blocks": blocks,
    }


def main():
    SHAPE = (4, 2, 1, 8)
    DT = torch.float16
    result = {
        "source": "vllm_ascend/worker/model_runner_v1.py:L4111 _adjust_kv_layout (精简版)",
        "case1_page_eq_block": block_layout(SHAPE, DT, page_size_bytes=16 * 2),   # 32
        "case2_page_padded": block_layout(SHAPE, DT, page_size_bytes=48),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
