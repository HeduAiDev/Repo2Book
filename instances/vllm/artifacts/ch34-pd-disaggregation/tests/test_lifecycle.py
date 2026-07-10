"""测 _get_kv_connector_output 复现 vLLM worker 侧生命周期的可观察行为。

档案记录的真实行为（vllm/v1/worker/kv_connector_model_runner_mixin.py:L81-L119）：
  - bind_connector_metadata → start_load_kv(在 forward 前) → yield(forward)
    → wait_for_save → get_finished → build_connector_worker_meta → clear。
  - maybe_get_kv_connector_output：无 kv_transfer_group 时是 nullcontext（零开销）。
  - kv_connector_no_forward：无 token 也走收发，wait_for_save=False。
  - defer_finalize（spec-decode）：主 forward 退出时跳过 wait_for_save，由
    finalize_kv_connector 补做。
  - maybe_transfer_kv_layer：进层前 wait_for_layer_load、出层后 save_kv_layer。
"""
import pytest

from implementation import runtime
from implementation.base import KVConnectorBase_V1, KVConnectorMetadata
from implementation.mixin import KVConnectorModelRunnerMixin, maybe_transfer_kv_layer


class RecordingConnector(KVConnectorBase_V1):
    """记录每个生命周期方法的调用顺序，便于断言时序。"""

    def __init__(self):
        super().__init__(vllm_config=None, role="worker")
        self.calls = []
        self._finished = (set(), set())

    def bind_connector_metadata(self, md):
        self.calls.append("bind")
        super().bind_connector_metadata(md)

    def start_load_kv(self, forward_context, **kw):
        self.calls.append("start_load_kv")

    def wait_for_layer_load(self, layer_name):
        self.calls.append(f"wait_layer:{layer_name}")

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kw):
        self.calls.append(f"save_layer:{layer_name}")

    def wait_for_save(self):
        self.calls.append("wait_for_save")

    def get_finished(self, finished_req_ids):
        self.calls.append("get_finished")
        return self._finished

    def clear_connector_metadata(self):
        self.calls.append("clear")
        super().clear_connector_metadata()


class SchedOutput:
    def __init__(self):
        self.kv_connector_metadata = KVConnectorMetadata()
        self.finished_req_ids = set()


@pytest.fixture(autouse=True)
def _reset_group():
    runtime.set_kv_transfer_group(None)
    yield
    runtime.set_kv_transfer_group(None)


def test_lifecycle_order_brackets_forward():
    conn = RecordingConnector()
    runtime.set_kv_transfer_group(conn)
    so = SchedOutput()

    with runtime.set_forward_context(attn_metadata=object()):
        with KVConnectorModelRunnerMixin._get_kv_connector_output(so) as out:
            # forward 在这里发生：start_load_kv 必须已经调用（异步发起在前）。
            assert conn.calls == ["bind", "start_load_kv"]
            conn.calls.append("FORWARD")

    # 退出 context manager 后，收尾按序发生。
    assert conn.calls == [
        "bind", "start_load_kv", "FORWARD",
        "wait_for_save", "get_finished", "clear",
    ]
    assert out.finished_sending == set()
    assert out.finished_recving == set()


def test_maybe_get_output_is_nullcontext_without_group():
    so = SchedOutput()
    # 无 kv_transfer_group → nullcontext，零开销、yield None。
    with KVConnectorModelRunnerMixin.maybe_get_kv_connector_output(so) as out:
        assert out is None


def test_defer_finalize_skips_wait_for_save_until_finalize():
    conn = RecordingConnector()
    runtime.set_kv_transfer_group(conn)
    so = SchedOutput()

    with runtime.set_forward_context(attn_metadata=object()):
        with KVConnectorModelRunnerMixin._get_kv_connector_output(
            so, defer_finalize=True
        ):
            pass

    # 主 forward 退出：跳过 wait_for_save，也不 clear。
    assert "wait_for_save" not in conn.calls
    assert "clear" not in conn.calls
    assert conn.calls == ["bind", "start_load_kv", "get_finished"]

    # draft forward 后补做。
    KVConnectorModelRunnerMixin.finalize_kv_connector()
    assert conn.calls[-2:] == ["wait_for_save", "clear"]


def test_no_forward_path_skips_wait_for_save():
    conn = RecordingConnector()
    runtime.set_kv_transfer_group(conn)
    so = SchedOutput()

    out = KVConnectorModelRunnerMixin.kv_connector_no_forward(so, vllm_config=None)
    # 无 token 也走收发，但 wait_for_save=False。
    assert "start_load_kv" in conn.calls
    assert "wait_for_save" not in conn.calls
    assert "get_finished" in conn.calls
    # 本例 get_finished 返回空 → 输出为空 → 返回 EMPTY 单例。
    assert out is runtime.EMPTY_MODEL_RUNNER_OUTPUT


def test_no_forward_path_propagates_nonempty_output():
    conn = RecordingConnector()
    conn._finished = ({"req-send"}, set())
    runtime.set_kv_transfer_group(conn)
    so = SchedOutput()

    out = KVConnectorModelRunnerMixin.kv_connector_no_forward(so, vllm_config=None)
    assert out is not runtime.EMPTY_MODEL_RUNNER_OUTPUT
    assert out.kv_connector_output.finished_sending == {"req-send"}


def test_maybe_transfer_kv_layer_brackets_attention():
    conn = RecordingConnector()
    runtime.set_kv_transfer_group(conn)
    conn.bind_connector_metadata(KVConnectorMetadata())

    @maybe_transfer_kv_layer
    def attn_layer(layer_name, x):
        conn.calls.append(f"compute:{layer_name}")
        return x * 2

    layers = {"layer.0": object()}
    with runtime.set_forward_context(attn_metadata=object(), no_compile_layers=layers):
        result = attn_layer("layer.0", 21)

    assert result == 42
    # 进层前 wait_for_layer_load、计算、出层后 save_kv_layer。
    assert conn.calls[-3:] == [
        "wait_layer:layer.0", "compute:layer.0", "save_layer:layer.0",
    ]


def test_maybe_transfer_kv_layer_noop_without_group():
    @maybe_transfer_kv_layer
    def attn_layer(layer_name, x):
        return x + 1

    # 无 connector → wrapper 是 no-op，直接执行原函数。
    assert attn_layer("layer.0", 10) == 11
