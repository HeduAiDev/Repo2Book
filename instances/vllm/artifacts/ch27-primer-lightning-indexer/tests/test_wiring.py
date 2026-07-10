import numpy as np

from wiring import TopkIndicesBuffer, main_attention_from_buffer, v32_indexer_step


def test_buffer_starts_filled_with_minus_one():
    buf = TopkIndicesBuffer(num_tokens=4, topk=3)
    assert (buf.data == -1).all()
    assert buf.data.shape == (4, 3)


def test_v32_indexer_step_writes_topk_into_shared_buffer():
    # PAPER: §2.1 "Instantiate DSA Under MLA" —— indexer 打分/选 top-k 是纯副作用：
    # 写共享 buffer，返回值只供示教。
    q = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])  # [T=2, H^I=1, d^I=2]
    k = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])  # [S=3, d^I=2]
    w = np.array([[1.0], [1.0]])  # [T=2, H^I=1]
    buffer = TopkIndicesBuffer(num_tokens=2, topk=2)

    scores = v32_indexer_step(q, k, w, buffer, topk=2, token_start=0)

    assert scores.shape == (2, 3)
    # buffer 应被写入（不再全是 -1），且写入的正是 top-2 索引
    assert (buffer.data != -1).all()
    from lightning_indexer import topk_select

    expected = topk_select(scores, 2)
    np.testing.assert_array_equal(buffer.data, expected)


def test_v32_indexer_step_respects_token_start_offset():
    q = np.array([[[1.0, 0.0]]])  # 只有 T=1 个 query token
    k = np.array([[2.0, 0.0], [0.0, 3.0]])
    w = np.array([[1.0]])
    buffer = TopkIndicesBuffer(num_tokens=4, topk=2)

    v32_indexer_step(q, k, w, buffer, topk=2, token_start=2)

    assert (buffer.data[0:2] == -1).all()  # 未写入的位置仍是 -1
    assert (buffer.data[2] != -1).all()  # token_start=2 处已被写入
    assert (buffer.data[3] == -1).all()  # 之后的位置仍未被写入


def test_main_attention_from_buffer_only_reads_selected_kv():
    # PAPER: mla.py:L168-169 —— 主注意力只从共享 buffer 里读 top-k 索引对应的
    # latent KV 条目，不接触 indexer 的 q/k/权重。
    buffer = TopkIndicesBuffer(num_tokens=1, topk=1)
    buffer.write(0, np.array([[1]]))  # 只选中索引 1

    q_main = np.array([[1.0, 0.0]])
    c_main = np.array([[9.0, 9.0], [2.0, 3.0], [7.0, 7.0]])

    out = main_attention_from_buffer(q_main, c_main, buffer, token_start=0, scale=1.0)

    # k=1 时 softmax 权重恒为 1，输出应恰好等于被选中的那一条 latent KV。
    np.testing.assert_allclose(out[0], c_main[1])
