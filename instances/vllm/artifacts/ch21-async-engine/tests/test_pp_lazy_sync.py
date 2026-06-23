"""Tests for lazy PP synchronization.

Asserts the *observable behavior* the real vLLM code produces:
  * the irecv handle is NOT waited on until `.tensors` is first read;
  * the previous round's isend handles are waited at the start of the next
    execute_model;
  * a PP middle rank returns None and stashes send handles in _pp_send_work;
  * the first PP rank skips the irecv entirely.
"""

from implementation._bridge import Handle
from implementation.pp_lazy_sync import (
    AsyncIntermediateTensors,
    PPGroupCoordinator,
    Worker,
    _ModelRunnerOutputMarker,
)


# --- fakes wiring the bridge to an in-process PP link --------------------
class _Meta:
    _is_metadata = True

    def __init__(self, key):
        self.key = key

    def allocate(self):
        return {"buf_for": self.key}


class _TensorVal:
    _is_tensor = True

    def __init__(self, key):
        self.key = key

    def metadata(self):
        return _Meta(self.key)


class _Link:
    """In-process metadata side-channel + a handle factory recording waits."""

    def __init__(self, recv_keys):
        self._recv_keys = recv_keys
        self.sent_metadata = []

    def recv_metadata(self, src):
        return [(k, _Meta(k)) for k in self._recv_keys]

    def send_metadata(self, obj, dst):
        self.sent_metadata.append((dst, obj))


def _make_pp(rank, world, recv_keys=("hidden",)):
    pp = PPGroupCoordinator(rank_in_group=rank, world_size=world)
    link = _Link(recv_keys)
    pp._link = link
    handles = []
    import implementation.pp_lazy_sync as mod

    def factory():
        h = Handle()
        handles.append(h)
        return h

    mod._HANDLE_FACTORY = factory
    return pp, link, handles


class _SchedOut:
    def __init__(self, n=4):
        self.total_num_scheduled_tokens = n


def test_irecv_returns_unwaited_handles():
    pp, link, handles = _make_pp(rank=1, world=2, recv_keys=("hidden", "residual"))
    tensor_dict, comm_handles, postprocess = pp.irecv_tensor_dict()
    assert tensor_dict is not None
    assert set(tensor_dict.keys()) == {"hidden", "residual"}
    assert len(comm_handles) == 2
    # No handle has been waited on yet — that's the whole point of irecv.
    assert all(not h.waited for h in comm_handles)
    assert postprocess == []  # SP all-gather subtracted → no postprocess


def test_lazy_wait_triggers_only_on_tensors_access():
    pp, link, handles = _make_pp(rank=1, world=2)
    tensor_dict, comm_handles, postprocess = pp.irecv_tensor_dict()
    ait = AsyncIntermediateTensors(tensor_dict, comm_handles, postprocess)

    # Touching unrelated attributes must NOT wait.
    _ = ait._comm_handles
    assert all(not h.waited for h in comm_handles)
    assert ait._comm_waited is False

    # First read of .tensors triggers wait_for_comm → handle.wait().
    _ = ait.tensors
    assert all(h.waited for h in comm_handles)
    assert ait._comm_waited is True


def test_wait_for_comm_is_idempotent():
    pp, _, _ = _make_pp(rank=1, world=2)
    td, h, pp_ = pp.irecv_tensor_dict()
    ait = AsyncIntermediateTensors(td, h, pp_)
    ait.wait_for_comm()
    # second access does not re-wait (already waited)
    waited_before = [x.waited for x in h]
    ait.wait_for_comm()
    assert [x.waited for x in h] == waited_before


def test_world_size_one_recv_short_circuits():
    pp, _, _ = _make_pp(rank=0, world=1)
    assert pp.irecv_tensor_dict() == (None, [], [])
    assert pp.isend_tensor_dict({}) == []


def _runner_returning_intermediate():
    from implementation.pp_lazy_sync import IntermediateTensors

    class R:
        def execute_model(self, sched, inter):
            self.received = inter
            return IntermediateTensors({"hidden": _TensorVal("hidden")})

    return R()


def test_middle_rank_execute_model_returns_none_and_stashes_send():
    pp, link, handles = _make_pp(rank=1, world=3)
    worker = Worker(pp, _runner_returning_intermediate())
    out = worker.execute_model(_SchedOut(4))
    # Middle PP rank returns None (it forwarded to the next stage).
    assert out is None
    # The send handle is stashed, deferred to the next round.
    assert len(worker._pp_send_work) == 1
    assert worker._pp_send_work[0].waited is False


def test_previous_send_waited_at_start_of_next_execute():
    pp, link, handles = _make_pp(rank=1, world=3)
    worker = Worker(pp, _runner_returning_intermediate())
    worker.execute_model(_SchedOut(4))
    stale_send = worker._pp_send_work[0]
    assert stale_send.waited is False
    # Next call must wait the previous round's isend first.
    worker.execute_model(_SchedOut(4))
    assert stale_send.waited is True


def test_first_rank_skips_irecv():
    pp, link, handles = _make_pp(rank=0, world=3)
    worker = Worker(pp, _runner_returning_intermediate())
    worker.execute_model(_SchedOut(4))
    # First rank produced no AsyncIntermediateTensors (received None).
    assert worker.model_runner.received is None


def test_last_rank_returns_model_output():
    pp, link, handles = _make_pp(rank=2, world=3)

    class R:
        def execute_model(self, sched, inter):
            return _ModelRunnerOutputMarker()

    worker = Worker(pp, R())
    out = worker.execute_model(_SchedOut(4))
    assert isinstance(out, _ModelRunnerOutputMarker)
    assert worker._pp_send_work == []  # last rank does not send onward
