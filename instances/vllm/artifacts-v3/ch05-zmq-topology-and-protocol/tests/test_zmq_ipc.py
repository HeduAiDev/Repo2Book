"""TDD tests for the v3 ch05 subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).

These assert the *observable vLLM behavior* this chapter teaches, on a plain
CPU host with REAL ZMQ sockets and REAL engine subprocesses (pyzmq is the same
library vLLM uses; the only stand-in is the msgspec host seam, which produces
genuine msgpack wire bytes):

- m1  socket topology: client ROUTER(bind) / engine DEALER(connect, identity=
      engine_index.to_bytes(2,'little')), engine PUSH(connect) / client
      PULL(bind); engine connects *one DEALER per front-end address*;
- m2  DEALER speaks first: the first message on each data socket is the
      EngineCoreReadyResponse ("required before the front-end ROUTER ... can
      send input messages back to us") and carries post-init config back
      (num_gpu_blocks sync);
- m3  byte-tag wire format: (Identity, Type, *Payload frames); ADD=b'\\x00',
      ABORT=b'\\x01', UTILITY=b'\\x03' tag frames; msgpack multi-frame payload;
- m4  msgpack Struct protocol: array_like + omit_defaults + gc=False,
      FinishReason IntEnum, unknown types rejected with TypeError;
- m5  multi-frame zero-copy: small CPU tensors inlined (<256B RAW_VIEW), large
      tensors as aux_buffer frames that are memoryviews of the tensor storage;
      client sends copy=False; engine output side reuses bytearrays with a
      first-frame MessageTracker;
- m6  OOB torch_shm bypass: TensorIpcSender share_memory_ + mp.Queue, handle
      dict in the main frame, TensorIpcReceiver drain-and-buffer reassembly;
- m7  HWM=0 (RCVHWM/SNDHWM) on all make_zmq_socket sockets + bind defaults
      (PUSH/SUB/XSUB connect by default, everything else binds);
- m8  per-step aggregation: one EngineCoreOutputs per step per client;
- m9  client_index stamped by add_request_async and honored engine-side
      (outputs bucketed by request.client_index, sockets[client_index]);
- m10 engine-side two IO threads + two queue.Queue decoupling from the busy
      loop (e2e observable: requests flow while the busy loop idles);
- m11 ABORT dual queue: aborts reach the engine and produce finished_requests
      outputs with finish_reason ABORT;
- m12 UTILITY thin RPC: call_id pairing with futures, engine-side reflective
      getattr dispatch, failure_message path;
- m13 death contract: executor failure -> single-frame ENGINE_CORE_DEAD ->
      EngineDeadError on the client + engine_dead flag;
- m14 two-layer startup handshake: HELLO -> EngineHandshakeMetadata(addresses)
      -> READY on the dedicated handshake ROUTER, tcp://host:0 placeholders
      resolved via LAST_ENDPOINT;
- m15 make_client 2x2 factory table + VLLM_ENABLE_V1_MULTIPROCESSING default.

Engine subprocesses are spawned through the real CoreEngineProcManager /
launch_core_engines path (mp spawn). Tests play the scheduler by scripting
step outputs through the UTILITY RPC (the engine interior is the ch09
boundary); no forward pass is faked.

Run:  cd instances/vllm/artifacts-v3/ch05-zmq-topology-and-protocol
      python -m pytest tests/ -q
"""

import asyncio
import importlib
import queue as queue_mod
import sys
import threading
import time
import uuid
from concurrent.futures import Future
from pathlib import Path

import pytest
import torch
import zmq

_IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(_IMPL_DIR))

zmq_ipc = importlib.import_module("zmq_ipc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mk_config(**over):
    """Build the seam VllmConfig (assembly line itself is the ch03 product)."""
    model_over = dict(over.pop("model_config", {}) or {})
    cache_over = dict(over.pop("cache_config", {}) or {})
    par_over = dict(over.pop("parallel_config", {}) or {})
    sched_over = dict(over.pop("scheduler_config", {}) or {})
    cfg = zmq_ipc.VllmConfig(
        model_config=zmq_ipc.ModelConfig(**model_over),
        cache_config=zmq_ipc.CacheConfig(**cache_over),
        parallel_config=zmq_ipc.ParallelConfig(**par_over),
        scheduler_config=zmq_ipc.SchedulerConfig(**sched_over),
        instance_id=over.pop("instance_id", f"test-{uuid.uuid4().hex[:8]}"),
        **over,
    )
    return cfg


def core_request(request_id="req-1", token_ids=(1, 2, 3), embeds=None, **kw):
    return zmq_ipc.EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(token_ids),
        mm_features=None,
        sampling_params=zmq_ipc.SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        prompt_embeds=embeds,
        **kw,
    )


def script_step(*entries):
    """One scripted engine step: entries are (request_id, new_token_ids, finish)."""
    return [
        {
            "request_id": rid,
            "new_token_ids": list(tokens),
            "finish_reason": None if finish is None else int(finish),
        }
        for rid, tokens, finish in entries
    ]


class AsyncEngineSession:
    """One persistent event loop (its own thread) + one AsyncMPClient/engine.

    asyncio queues and tasks are bound to the loop that created them, so all
    async interactions with one client must share a single loop for the whole
    module-scoped session.
    """

    def __init__(self, **cfg_over):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        # Construct the client ON the session loop: the zmq.asyncio sockets and
        # the output-queue task must belong to the same loop all tests use.
        fut = asyncio.run_coroutine_threadsafe(
            self._make_client(zmq_ipc.UniprocExecutor, mk_config(**cfg_over)),
            self.loop,
        )
        self.client = fut.result(timeout=300)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _make_client(self, executor_cls, config):
        return zmq_ipc.AsyncMPClient(config, executor_cls, False)

    def run(self, coro_fn, *args, **kw):
        fut = asyncio.run_coroutine_threadsafe(coro_fn(*args, **kw), self.loop)
        return fut.result(timeout=120)

    def shutdown(self):
        # Clean stop, mirroring the real AsyncLLM teardown order: shut the
        # client down (engine SIGTERM + resources) WHILE the loop is still
        # running -- BackgroundResources schedules the async-socket close
        # back onto the loop via call_soon_threadsafe -- then stop the loop.
        client = getattr(self, "client", None)
        if client is not None:
            client.shutdown(timeout=30)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=30)


@pytest.fixture(scope="module")
def sync_client():
    """One SyncMPClient + one engine subprocess for the whole module."""
    client = zmq_ipc.SyncMPClient(mk_config(), zmq_ipc.UniprocExecutor, False)
    yield client
    client.shutdown(timeout=30)


@pytest.fixture(scope="module")
def async_session():
    session = AsyncEngineSession()
    yield session
    session.shutdown()


# ---------------------------------------------------------------------------
# m4/m5/m3: serialization + wire format (pure unit, no sockets)
# ---------------------------------------------------------------------------


class TestWireFormat:
    def test_request_type_byte_tags(self):
        # SOURCE-anchored enum: values are the wire tag frames themselves.
        T = zmq_ipc.EngineCoreRequestType
        assert T.ADD.value == b"\x00"
        assert T.ABORT.value == b"\x01"
        assert T.UTILITY.value == b"\x03"
        assert T.EXECUTOR_FAILED.value == b"\x04"
        assert T.WAKEUP.value == b"\x05"
        # The tag frame can be inverted straight back into the enum member --
        # that is how the engine input thread classifies messages.
        assert T(b"\x00") is T.ADD
        assert T(b"\x03") is T.UTILITY

    def test_encode_small_request_single_frame(self):
        enc = zmq_ipc.MsgpackEncoder()
        dec = zmq_ipc.MsgpackDecoder(zmq_ipc.EngineCoreRequest)
        req = core_request()
        bufs = enc.encode(req)
        # No tensor > threshold -> main frame only.
        assert len(bufs) == 1
        out = dec.decode(bufs)
        assert out.request_id == "req-1"
        assert out.prompt_token_ids == [1, 2, 3]
        # array_like encodes ALL fields: client_index rides the array as 0
        # (omit_defaults is a no-op for positional arrays -- see the
        # test_array_like_and_omit_defaults_on_the_wire wire-truth check).
        assert out.client_index == 0
        assert isinstance(out.sampling_params, zmq_ipc.SamplingParams)
        assert out.sampling_params.max_tokens == 8  # dataclass rides as a map

    def test_encode_large_tensor_multiframe_zero_copy(self):
        enc = zmq_ipc.MsgpackEncoder()
        dec = zmq_ipc.MsgpackDecoder(zmq_ipc.EngineCoreRequest)
        big = torch.arange(2048, dtype=torch.float32)  # 8192B > 256B threshold
        bufs = enc.encode(core_request(embeds=big))
        assert len(bufs) == 2  # main frame + aux backing frame
        aux = bufs[1]
        # The aux frame is a zero-copy view of the tensor storage, not a copy.
        assert isinstance(aux, memoryview)
        assert len(aux) == 2048 * 4
        assert bytes(aux) == bytes(big.flatten().view(torch.uint8).numpy().tobytes())
        out = dec.decode(bufs)
        assert out.prompt_embeds is not None
        assert out.prompt_embeds.dtype == torch.float32
        assert out.prompt_embeds.shape == (2048,)
        assert torch.equal(out.prompt_embeds.cpu(), big)

    def test_inline_threshold_256b(self):
        # Below VLLM_MSGPACK_ZERO_COPY_THRESHOLD: inlined into the main frame.
        assert zmq_ipc.envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD == 256
        enc = zmq_ipc.MsgpackEncoder()
        small = torch.arange(8, dtype=torch.float32)  # 32B
        bufs = enc.encode(core_request(embeds=small))
        assert len(bufs) == 1
        dec = zmq_ipc.MsgpackDecoder(zmq_ipc.EngineCoreRequest)
        out = dec.decode(bufs)
        assert torch.equal(out.prompt_embeds, small)

    def test_inline_threshold_boundary(self):
        enc = zmq_ipc.MsgpackEncoder()
        # 256B exactly -> NOT below threshold -> aux frame; 252B -> inline.
        just_under = torch.zeros(63, dtype=torch.float32)  # 252B
        assert len(enc.encode(core_request(embeds=just_under))) == 1
        at_threshold = torch.zeros(64, dtype=torch.float32)  # 256B
        assert len(enc.encode(core_request(embeds=at_threshold))) == 2

    def test_ndarray_aux_and_inline(self):
        import numpy as np

        enc = zmq_ipc.MsgpackEncoder()
        # Typed decode (dec_hook fires for np.ndarray); the *untyped* decoder
        # returns raw builtins -- same as real msgspec.
        dec = zmq_ipc.MsgpackDecoder(np.ndarray)
        big = np.arange(1000, dtype=np.float32)
        bufs = enc.encode(big)
        assert len(bufs) == 2
        out = dec.decode(bufs)
        assert isinstance(out, np.ndarray)
        assert (out == big).all()
        small = np.arange(10, dtype=np.float32)
        bufs = enc.encode(small)
        assert len(bufs) == 1
        out = dec.decode(bufs)
        assert (out == small).all()

    def test_array_like_and_omit_defaults_on_the_wire(self):
        import msgpack

        enc = zmq_ipc.MsgpackEncoder()
        # Wire truth (verified against real msgspec 0.19.0/0.20.0/0.21.1 in
        # the pin container): array_like structs encode ALL fields
        # positionally. omit_defaults only trims map-like keys and is a
        # no-op on positional arrays -- on the real vLLM wire (all three
        # wire structs are array_like) it is purely decorative.
        bufs = enc.encode(zmq_ipc.EngineCoreOutputs(outputs=[]))
        raw = msgpack.unpackb(bytes(bufs[0]), use_list=True)
        assert isinstance(raw, list)
        assert len(raw) == 6  # every field rides the positional array
        assert raw[0] == 0 and raw[1] == [] and raw[2] is None and raw[3] > 0.0
        assert raw[4] is None and raw[5] is None  # utility_output / finished_requests
        # A non-default field keeps its position in the array.
        bufs = enc.encode(zmq_ipc.EngineCoreOutputs(engine_index=3))
        raw = msgpack.unpackb(bytes(bufs[0]), use_list=True)
        assert raw[0] == 3
        # Trailing fields equal to defaults are NOT trimmed either: even a
        # minimal EngineCoreOutput rides the wire with all 6 fields
        # (real msgspec emits ["r",[1],None,None,None,0] for exactly this).
        bufs = enc.encode(zmq_ipc.EngineCoreOutput(request_id="r", new_token_ids=[1]))
        raw = msgpack.unpackb(bytes(bufs[0]), use_list=True)
        assert raw == ["r", [1], None, None, None, 0]
        # Decoding stays lenient: a SHORT positional array (a peer encoding
        # fewer fields) fills missing trailing elements from field defaults
        # -- real msgspec accepts those too, so interop is safe both ways.
        dec = zmq_ipc.MsgpackDecoder(zmq_ipc.EngineCoreOutput)
        out = dec.decode([msgpack.packb(["r", [1]], use_bin_type=True)])
        assert out.request_id == "r"
        assert out.new_token_ids == [1]
        assert out.pooling_output is None
        assert out.num_nans_in_logits == 0

    def test_roundtrip_outputs_with_finish_reason(self):
        enc = zmq_ipc.MsgpackEncoder()
        dec = zmq_ipc.MsgpackDecoder(zmq_ipc.EngineCoreOutputs)
        eco = zmq_ipc.EngineCoreOutputs(
            outputs=[
                zmq_ipc.EngineCoreOutput(
                    request_id="r1", new_token_ids=[7, 8], finish_reason=0
                ),
                zmq_ipc.EngineCoreOutput(
                    request_id="r2", new_token_ids=[9], finish_reason=2
                ),
            ],
            finished_requests={"r2"},
        )
        out = dec.decode(enc.encode(eco))
        assert out.outputs[0].request_id == "r1"
        # IntEnum survives the wire: finish_reason is a real FinishReason.
        assert out.outputs[0].finish_reason is zmq_ipc.FinishReason.STOP
        assert out.outputs[1].finish_reason is zmq_ipc.FinishReason.ABORT
        assert str(out.outputs[1].finish_reason) == "abort"
        assert out.finished_requests == {"r2"}
        assert out.timestamp > 0.0  # __post_init__ monotonic stamp survives

    def test_unknown_type_rejected_by_default(self):
        enc = zmq_ipc.MsgpackEncoder()

        class NotSerializable:
            pass

        with pytest.raises(TypeError, match="VLLM_ALLOW_INSECURE_SERIALIZATION"):
            enc.encode(NotSerializable())

    def test_utility_result_roundtrip(self):
        enc = zmq_ipc.MsgpackEncoder()
        dec = zmq_ipc.MsgpackDecoder(zmq_ipc.UtilityOutput)
        msg = zmq_ipc.UtilityOutput(call_id=42, result=zmq_ipc.UtilityResult(("a", "b")))
        out = dec.decode(enc.encode(msg))
        assert out.call_id == 42
        assert out.result.result == ["a", "b"]


# ---------------------------------------------------------------------------
# m7 + m1 primitives: make_zmq_socket and the ROUTER/DEALER envelope
# ---------------------------------------------------------------------------


class TestZmqSocketFactory:
    def test_hwm_zero_and_bind_defaults(self):
        ctx = zmq.Context()
        try:
            # PULL binds by default (not in the PUSH/SUB/XSUB connect set).
            pull = zmq_ipc.make_zmq_socket(ctx, "tcp://127.0.0.1:0", zmq.PULL)
            ep = pull.getsockopt(zmq.LAST_ENDPOINT).decode()
            assert ":0" not in ep  # bound to a concrete port
            # PUSH connects by default.
            push = zmq_ipc.make_zmq_socket(ctx, ep, zmq.PUSH)
            # m7: HWM=0 (unlimited, no backpressure) on both directions.
            assert pull.getsockopt(zmq.RCVHWM) == 0
            assert push.getsockopt(zmq.SNDHWM) == 0
            push.send(b"x")
            assert pull.recv() == b"x"
            # ROUTER gets both HWMs zeroed; DEALER too.
            router = zmq_ipc.make_zmq_socket(ctx, "tcp://127.0.0.1:0", zmq.ROUTER, bind=True)
            assert router.getsockopt(zmq.RCVHWM) == 0
            assert router.getsockopt(zmq.SNDHWM) == 0
            dealer_ep = router.getsockopt(zmq.LAST_ENDPOINT).decode()
            dealer = zmq_ipc.make_zmq_socket(
                ctx, dealer_ep, zmq.DEALER, identity=(7).to_bytes(2, "little"), bind=False
            )
            assert dealer.getsockopt(zmq.RCVHWM) == 0
            assert dealer.getsockopt(zmq.SNDHWM) == 0
            for s in (pull, push, router, dealer):
                s.close(linger=0)
        finally:
            ctx.destroy(linger=0)

    def test_router_envelope_and_dealer_first_speak(self):
        # m1/m2 at the raw socket level: the DEALER must speak first, the
        # ROUTER learns its identity, and replies must re-wrap the envelope.
        ctx = zmq.Context()
        try:
            router = ctx.socket(zmq.ROUTER)
            router.bind("tcp://127.0.0.1:0")
            ep = router.getsockopt(zmq.LAST_ENDPOINT).decode()
            dealer = ctx.socket(zmq.DEALER)
            identity = (0).to_bytes(2, "little")
            dealer.setsockopt(zmq.IDENTITY, identity)
            dealer.connect(ep)
            dealer.send(b"READY-first-speak")
            # ROUTER receive: [identity envelope, payload]
            got = router.recv_multipart()
            assert got == [identity, b"READY-first-speak"]
            # Reply must carry the envelope as frame 0.
            router.send_multipart((identity, b"\x00", b"payload-frames"))
            assert dealer.recv_multipart() == [b"\x00", b"payload-frames"]
            dealer.close(linger=0)
            router.close(linger=0)
        finally:
            ctx.destroy(linger=0)

    def test_linger_passthrough(self):
        ctx = zmq.Context()
        try:
            s = zmq_ipc.make_zmq_socket(ctx, "tcp://127.0.0.1:0", zmq.PUSH, linger=4000)
            assert s.getsockopt(zmq.LINGER) == 4000
            s.close(linger=0)
        finally:
            ctx.destroy(linger=0)


# ---------------------------------------------------------------------------
# m5 engine-side: manual first-frame tracking
# ---------------------------------------------------------------------------


class TestTrackingPayload:
    def test_send_msg_tracking_payload_frames_and_tracker(self):
        # The engine output thread's keep-alive trick: send the first frame
        # manually with track=True because send_multipart tracks only the last.
        ctx = zmq.Context()
        try:
            pull = ctx.socket(zmq.PULL)
            pull.bind("tcp://127.0.0.1:0")
            ep = pull.getsockopt(zmq.LAST_ENDPOINT).decode()
            push = zmq_ipc.make_zmq_socket(ctx, ep, zmq.PUSH, linger=4000)
            buffers = [bytearray(b"main-frame"), memoryview(b"aux-frame")]
            tracker = zmq_ipc.EngineCoreProc._send_msg_tracking_payload(push, buffers)
            assert isinstance(tracker, zmq.MessageTracker)
            frames = pull.recv_multipart()
            assert frames == [b"main-frame", b"aux-frame"]
            deadline = time.monotonic() + 5
            while not tracker.done and time.monotonic() < deadline:
                time.sleep(0.01)
            assert tracker.done
            push.close(linger=0)
            pull.close(linger=0)
        finally:
            ctx.destroy(linger=0)


# ---------------------------------------------------------------------------
# m13: death sentinel consumption on the client side
# ---------------------------------------------------------------------------


class TestDeathValidation:
    def test_validate_alive_single_frame_dead(self):
        res = zmq_ipc.BackgroundResources(ctx=zmq.Context())
        frame = zmq.Frame(zmq_ipc.EngineCoreProc.ENGINE_CORE_DEAD)
        with pytest.raises(zmq_ipc.EngineDeadError):
            res.validate_alive([frame])
        assert res.engine_dead is True

    def test_validate_alive_ordinary_message_passes(self):
        res = zmq_ipc.BackgroundResources(ctx=zmq.Context())
        frames = [zmq.Frame(b"main"), zmq.Frame(b"aux")]
        res.validate_alive(frames)  # no raise
        assert res.engine_dead is False

    def test_dead_sentinel_constant(self):
        assert zmq_ipc.EngineCoreProc.ENGINE_CORE_DEAD == b"ENGINE_CORE_DEAD"


# ---------------------------------------------------------------------------
# m6: OOB tensor IPC sender/receiver (in-process queue first)
# ---------------------------------------------------------------------------


class TestTensorIpc:
    def test_sender_share_memory_and_handle(self):
        import multiprocessing as mp

        q = mp.get_context("spawn").Queue()
        sender = zmq_ipc.TensorIpcSender(q)
        sender.new_message()
        t = torch.arange(64, dtype=torch.float32)
        handle = sender(t)
        assert handle == {
            "sender_id": sender._sender_id,
            "message_id": 1,
            "tensor_id": 0,
        }
        data = q.get(timeout=5)
        assert isinstance(data, zmq_ipc.TensorIpcData)
        assert data.tensor.is_shared()
        assert torch.equal(data.tensor, t)

    def test_receiver_drain_and_buffer_out_of_order(self):
        import multiprocessing as mp

        q = mp.get_context("spawn").Queue()
        sender = zmq_ipc.TensorIpcSender(q)
        receiver = zmq_ipc.TensorIpcReceiver(q)
        sender.new_message()  # message 1: tensor 0
        t0 = torch.arange(4)
        sender(t0)  # (1, 0)
        sender.new_message()  # message 2: tensors 0, 1
        t1 = torch.arange(5)
        t2 = torch.arange(6)
        sender(t1)  # (2, 0)
        sender(t2)  # (2, 1)
        # Oldest message first (the receiver advances current_message_id and
        # discards older buffered tensors, so messages are consumed in order).
        # The drain-and-buffer pattern buffers the message-2 tensors it sees
        # while looking for the requested (1, 0) handle.
        got = receiver("torch.int64", (4,), handle2b(sender, 1, 0))
        assert torch.equal(got, t0)
        # Within message 2, tensors can be requested out of arrival order.
        got = receiver("torch.int64", (6,), handle2b(sender, 2, 1))
        assert torch.equal(got, t2)
        got = receiver("torch.int64", (5,), handle2b(sender, 2, 0))
        assert torch.equal(got, t1)

    def test_receiver_discards_stale_tensors(self):
        import multiprocessing as mp

        q = mp.get_context("spawn").Queue()
        sender = zmq_ipc.TensorIpcSender(q)
        receiver = zmq_ipc.TensorIpcReceiver(q)
        sender.new_message()
        sender(torch.arange(3))  # message 1 tensor 0 -- will become stale
        sender.new_message()
        fresh = torch.arange(7)
        sender(fresh)  # message 2 tensor 0
        got = receiver("torch.int64", (7,), handle2b(sender, 2, 0))
        assert torch.equal(got, fresh)
        # current_message_id has advanced to 2; a late message-1 tensor is now
        # stale and must be ignored (drained without being returned/buffered).
        sender2 = zmq_ipc.TensorIpcSender(q)  # different sender_id
        assert sender2._sender_id != sender._sender_id
        # Re-queue an old-message tensor from the first sender manually.
        q.put(
            zmq_ipc.TensorIpcData(
                sender_id=sender._sender_id,
                message_id=1,
                tensor_id=0,
                tensor=torch.arange(3),
            )
        )
        sender2.new_message()
        wanted = torch.arange(9)
        sender2(wanted)  # sender2 message 1 tensor 0
        got = receiver("torch.int64", (9,), handle2b(sender2, 1, 0))
        assert torch.equal(got, wanted)


def handle2b(sender, message_id, tensor_id):
    return {
        "sender_id": sender._sender_id,
        "message_id": message_id,
        "tensor_id": tensor_id,
    }


# ---------------------------------------------------------------------------
# m15: client factory 2x2 table
# ---------------------------------------------------------------------------


class TestClientFactory:
    def test_asyncio_without_multiprocessing_rejected(self):
        # The NotImplementedError escape hatch of the 2x2 table.
        with pytest.raises(NotImplementedError, match="without multiprocessing"):
            zmq_ipc.EngineCoreClient.make_client(
                multiprocess_mode=False,
                asyncio_mode=True,
                vllm_config=mk_config(),
                executor_class=zmq_ipc.UniprocExecutor,
                log_stats=False,
            )

    def test_inproc_cell_of_the_table(self):
        client = zmq_ipc.EngineCoreClient.make_client(
            multiprocess_mode=False,
            asyncio_mode=False,
            vllm_config=mk_config(),
            executor_class=zmq_ipc.UniprocExecutor,
            log_stats=False,
        )
        assert isinstance(client, zmq_ipc.InprocClient)
        client.shutdown()

    def test_mp_cells_dispatch(self, monkeypatch):
        # Probe the two mp cells without spawning engines: patch the
        # constructors with sentinels and observe where the table routes.
        calls = []

        class FakeSync:
            kind = "sync"

            def __init__(self, *a):
                calls.append(self.kind)

        class FakeAsync:
            kind = "async"

            def __init__(self, *a):
                calls.append(self.kind)

        monkeypatch.setattr(zmq_ipc, "SyncMPClient", FakeSync)
        monkeypatch.setattr(zmq_ipc, "AsyncMPClient", FakeAsync)
        # make_async_mp_client routes DP=1 -> AsyncMPClient.
        result = zmq_ipc.EngineCoreClient.make_async_mp_client(
            mk_config(), zmq_ipc.UniprocExecutor, False
        )
        assert calls == ["async"]
        # make_client(mp, not asyncio) -> SyncMPClient.
        calls.clear()
        zmq_ipc.EngineCoreClient.make_client(
            multiprocess_mode=True,
            asyncio_mode=False,
            vllm_config=mk_config(),
            executor_class=zmq_ipc.UniprocExecutor,
            log_stats=False,
        )
        assert calls == ["sync"]
        # make_client(mp, asyncio) -> make_async_mp_client -> AsyncMPClient.
        calls.clear()
        zmq_ipc.EngineCoreClient.make_client(
            multiprocess_mode=True,
            asyncio_mode=True,
            vllm_config=mk_config(),
            executor_class=zmq_ipc.UniprocExecutor,
            log_stats=False,
        )
        assert calls == ["async"]

    def test_multiprocessing_default_on(self):
        assert zmq_ipc.envs.VLLM_ENABLE_V1_MULTIPROCESSING is True

    def test_engine_ready_timeout_default(self):
        assert zmq_ipc.envs.VLLM_ENGINE_READY_TIMEOUT_S == 600


# ---------------------------------------------------------------------------
# m2/m9/m12: InprocClient contrast (no ZMQ at all)
# ---------------------------------------------------------------------------


class TestInprocClient:
    def test_no_ipc_direct_step(self):
        client = zmq_ipc.InprocClient(
            mk_config(cache_config={"num_gpu_blocks": None}),
            zmq_ipc.UniprocExecutor,
            False,
        )
        try:
            assert client.get_output().outputs == []  # idle step, no requests
            req = core_request()
            client.add_request(req)
            # Engine interior is the ch9 seam: script the step directly on the
            # in-process engine object (no UTILITY RPC needed -- nothing crosses).
            client.engine_core.scheduler.enqueue_step_outputs(
                script_step(("req-1", [11, 12], zmq_ipc.FinishReason.STOP))
            )
            out = client.get_output()
            assert out.outputs[0].new_token_ids == [11, 12]
            assert out.outputs[0].finish_reason is zmq_ipc.FinishReason.STOP
            assert client.dp_engines_running() is False
        finally:
            client.shutdown()


# ---------------------------------------------------------------------------
# End-to-end: stations 1-10 over real processes. m1 m2 m3 m5 m8 m9 m10 m11 m12
# ---------------------------------------------------------------------------


class TestSyncMPClientE2E:
    def test_construction_topology_and_ready(self, sync_client):
        # Station 1: client half assembled ROUTER(bind)+PULL(bind) with
        # HWM=0; identity table holds the 2-byte little-endian engine id.
        assert sync_client.core_engine == (0).to_bytes(2, "little")
        assert sync_client.core_engines == [(0).to_bytes(2, "little")]
        assert sync_client.input_socket.getsockopt(zmq.TYPE) == zmq.ROUTER
        assert sync_client.resources.output_socket is None  # thread owns it
        assert sync_client.input_socket.getsockopt(zmq.RCVHWM) == 0
        # m2: EngineCoreReadyResponse carried post-init config back
        # (the seam executor "profiles" num_gpu_blocks=128 when unset).
        assert sync_client.vllm_config.cache_config.num_gpu_blocks == 128

    def test_roundtrip_add_script_outputs(self, sync_client):
        rid = f"sync-add-{uuid.uuid4().hex[:6]}"
        sync_client.add_request(core_request(request_id=rid))
        sync_client.call_utility("enqueue_step_outputs", script_step((rid, [5, 6], None)))
        sync_client.call_utility(
            "enqueue_step_outputs", script_step((rid, [7], zmq_ipc.FinishReason.STOP))
        )
        out1 = sync_client.get_output()
        assert out1.outputs[0].request_id == rid
        assert out1.outputs[0].new_token_ids == [5, 6]
        out2 = sync_client.get_output()
        assert out2.outputs[0].new_token_ids == [7]
        assert out2.outputs[0].finish_reason is zmq_ipc.FinishReason.STOP
        # m8 aggregation shape: finished ids ride on the same message.
        assert out2.finished_requests == {rid}
        # m9: the engine stamped its own index on the way out.
        assert out2.engine_index == 0

    def test_abort_requests(self, sync_client):
        rid = f"sync-abort-{uuid.uuid4().hex[:6]}"
        sync_client.add_request(core_request(request_id=rid))
        sync_client.abort_requests([rid])
        out = sync_client.get_output()
        assert out.finished_requests == {rid}
        assert out.outputs[0].finish_reason is zmq_ipc.FinishReason.ABORT

    def test_utility_rpc_and_failure_path(self, sync_client):
        # m12: thin RPC over the same socket; call_id pairs the future.
        # Wire truth: the executor holds a tuple, but the UTILITY result
        # rides msgpack as a plain array and decodes as a *list* (real vLLM's
        # `-> tuple` annotation only holds for the InprocClient path).
        tasks = sync_client.call_utility("get_supported_tasks")
        assert tasks == ["generate", "pooling"]
        # A method that does not exist -> failure_message -> future exception.
        with pytest.raises(Exception, match="no_such_method"):
            sync_client.call_utility("no_such_method")
        # Failing method body (raises) also lands in failure_message.
        with pytest.raises(Exception, match="boom_method"):
            sync_client.call_utility("boom_method")

    def test_client_index_stamped_on_add(self, sync_client):
        rid = f"sync-idx-{uuid.uuid4().hex[:6]}"
        sync_client.add_request(core_request(request_id=rid))
        info = sync_client.call_utility("get_request_info", rid)
        # Station 5 / m9: the client stamped its own index into the request
        # before it crossed; self-managed single front-end -> index 0.
        assert info["client_index"] == 0
        assert info["prompt_len"] == 3


class TestAsyncMPClientE2E:
    def test_construction_and_add_roundtrip(self, async_session):
        client = async_session.client
        assert client.core_engine == (0).to_bytes(2, "little")
        assert client.outputs_queue.maxsize == 0
        rid = f"async-add-{uuid.uuid4().hex[:6]}"

        async def scenario():
            await client.add_request_async(core_request(request_id=rid))
            await client.call_utility_async(
                "enqueue_step_outputs", script_step((rid, [21, 22], None))
            )
            await client.call_utility_async(
                "enqueue_step_outputs",
                script_step((rid, [23], zmq_ipc.FinishReason.LENGTH)),
            )
            out1 = await client.get_output_async()
            out2 = await client.get_output_async()
            return out1, out2

        out1, out2 = async_session.run(scenario)
        assert out1.outputs[0].request_id == rid
        assert out1.outputs[0].new_token_ids == [21, 22]
        assert out2.outputs[0].new_token_ids == [23]
        assert out2.outputs[0].finish_reason is zmq_ipc.FinishReason.LENGTH
        assert out2.finished_requests == {rid}
        assert str(out2.outputs[0].finish_reason) == "length"

    def test_utility_future_pairing(self, async_session):
        client = async_session.client

        async def scenario():
            tasks = await client.get_supported_tasks_async()
            info_client_index = client.client_index
            return tasks, info_client_index

        tasks, _ = async_session.run(scenario)
        assert "generate" in tasks

    def test_abort_async(self, async_session):
        client = async_session.client
        rid = f"async-abort-{uuid.uuid4().hex[:6]}"

        async def scenario():
            await client.add_request_async(core_request(request_id=rid))
            await client.abort_requests_async([rid])
            return await client.get_output_async()

        out = async_session.run(scenario)
        assert out.finished_requests == {rid}
        assert out.outputs[0].finish_reason is zmq_ipc.FinishReason.ABORT

    def test_many_steps_buffer_reuse_intact(self, async_session):
        # Engine output thread reuses bytearrays guarded by the first-frame
        # tracker: many steps in a row must arrive uncorrupted.
        client = async_session.client
        rid = f"async-many-{uuid.uuid4().hex[:6]}"
        n_steps = 50

        async def scenario():
            await client.add_request_async(core_request(request_id=rid))
            scripts = []
            for i in range(n_steps - 1):
                scripts.append(script_step((rid, [100 + i], None)))
            scripts.append(script_step((rid, [199], zmq_ipc.FinishReason.LENGTH)))
            for s in scripts:
                await client.call_utility_async("enqueue_step_outputs", s)
            tokens = []
            finish = None
            while finish is None:
                out = await client.get_output_async()
                o = out.outputs[0]
                assert o.request_id == rid
                tokens.extend(o.new_token_ids)
                finish = o.finish_reason
            return tokens

        tokens = async_session.run(scenario)
        assert tokens == [100 + i for i in range(n_steps - 1)] + [199]


class TestDeathContract:
    def test_executor_failure_engine_dead_error(self):
        session = AsyncEngineSession()
        client = session.client
        try:
            rid = f"dead-{uuid.uuid4().hex[:6]}"

            async def scenario():
                # Trigger the real executor_fail_callback wiring: it injects
                # EXECUTOR_FAILED into the input queue; the busy loop raises,
                # run_engine_core sends the single-frame dead sentinel.
                await client.add_request_async(core_request(request_id=rid))
                await client.call_utility_async("fail_executor")
                try:
                    await asyncio.wait_for(client.get_output_async(), timeout=30)
                    return "no-error"
                except zmq_ipc.EngineDeadError:
                    return "dead"

            assert session.run(scenario) == "dead"
            # engine_dead flag set: fast-fail on further sends.
            with pytest.raises(zmq_ipc.EngineDeadError):
                session.run(client.add_request_async, core_request(request_id="x"))
        finally:
            session.shutdown()

    def test_sync_death_via_raw_utility_frame(self):
        from concurrent.futures import ThreadPoolExecutor

        client = zmq_ipc.SyncMPClient(mk_config(), zmq_ipc.UniprocExecutor, False)
        try:
            # Fire a raw UTILITY frame without waiting on its future, then
            # watch the output path raise EngineDeadError from validate_alive.
            # The call_id must still be *registered* (as call_utility would):
            # the engine replies to the utility call before the EXECUTOR_FAILED
            # sentinel kills it, and an unregistered id would KeyError the
            # output thread on that reply -- real vLLM behavior, but not this
            # test's lesson.
            call_id = uuid.uuid1().int >> 64
            client.utility_results[call_id] = Future()
            client._send_input(
                zmq_ipc.EngineCoreRequestType.UTILITY,
                (0, call_id, "fail_executor", ()),
            )
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(client.get_output)
                try:
                    fut.result(timeout=30)
                    pytest.fail("expected EngineDeadError from get_output")
                except zmq_ipc.EngineDeadError:
                    pass
        finally:
            client.shutdown(timeout=30)


# ---------------------------------------------------------------------------
# m6 e2e: torch_shm OOB bypass end-to-end through the real client
# ---------------------------------------------------------------------------


class TestOobTensorE2E:
    def test_prompt_embeds_via_shared_memory_handle(self):
        cfg = mk_config(
            model_config={"multimodal_config": zmq_ipc.MultimodalConfig(mm_tensor_ipc="torch_shm")},
            cache_config={"num_gpu_blocks": None},
        )
        client = zmq_ipc.SyncMPClient(cfg, zmq_ipc.UniprocExecutor, False)
        try:
            assert isinstance(client.encoder.oob_tensor_consumer, zmq_ipc.TensorIpcSender)
            rid = f"oob-{uuid.uuid4().hex[:6]}"
            big = torch.arange(4096, dtype=torch.float32)  # 16KB > threshold
            client.add_request(core_request(request_id=rid, embeds=big))
            info = client.call_utility("get_request_info", rid)
            # The tensor arrived out-of-band (shared memory), referenced by a
            # handle dict in the main msgpack frame -- not as an aux frame.
            assert info["has_embeds"] is True
            assert info["embeds_numel"] == 4096
            # Roundtrip the actual values back through a plain utility result.
            values = client.call_utility("get_request_embeds_head", rid, 8)
            assert values == big[:8].tolist()
        finally:
            client.shutdown(timeout=30)

    def test_no_oob_without_torch_shm(self):
        client = zmq_ipc.SyncMPClient(mk_config(), zmq_ipc.UniprocExecutor, False)
        try:
            assert client.encoder.oob_tensor_consumer is None
        finally:
            client.shutdown(timeout=30)


# ---------------------------------------------------------------------------
# m14: address allocation + handshake payload structures
# ---------------------------------------------------------------------------


class TestAddresses:
    def test_get_engine_zmq_addresses_placeholder(self):
        addrs = zmq_ipc.get_engine_zmq_addresses(mk_config())
        assert isinstance(addrs, zmq_ipc.EngineZmqAddresses)
        assert len(addrs.inputs) == 1 and len(addrs.outputs) == 1
        # One input + one output address per front-end; coordinator opt-in.
        assert addrs.coordinator_input is None
        assert addrs.coordinator_output is None

    def test_handshake_metadata_roundtrip(self):
        import msgpack

        addrs = zmq_ipc.EngineZmqAddresses(inputs=["ipc://a"], outputs=["ipc://b"])
        meta = zmq_ipc.EngineHandshakeMetadata(addresses=addrs, parallel_config={})
        raw = msgpack.packb({"a": 1}, use_bin_type=True)  # seam encode sanity
        assert raw == b"\x81\xa1a\x01"
        assert meta.addresses.inputs == ["ipc://a"]

    def test_engine_identity_two_byte_little(self):
        # identity = engine_index.to_bytes(2, 'little')
        assert (0).to_bytes(2, "little") == b"\x00\x00"
        assert (1).to_bytes(2, "little") == b"\x01\x00"
        ce = zmq_ipc.CoreEngine(index=258)
        assert ce.identity == (258).to_bytes(2, "little") == b"\x02\x01"
        assert ce.state is zmq_ipc.CoreEngineState.NEW


# ---------------------------------------------------------------------------
# m12 internals: msgspec arg conversion + utility output future pairing
# ---------------------------------------------------------------------------


class TestUtilityInternals:
    def test_process_utility_output_pairs_future(self):
        results = {}
        fut = Future()
        call_id = 777
        results[call_id] = fut
        out = zmq_ipc.UtilityOutput(
            call_id=call_id, result=zmq_ipc.UtilityResult(("ok",))
        )
        zmq_ipc._process_utility_output(out, results)
        assert fut.result() == ("ok",)
        assert call_id not in results

    def test_process_utility_output_failure(self):
        results = {}
        fut = Future()
        results[9] = fut
        out = zmq_ipc.UtilityOutput(call_id=9, failure_message="Call to x failed: bad")
        zmq_ipc._process_utility_output(out, results)
        with pytest.raises(Exception, match="bad"):
            fut.result()

    def test_convert_msgspec_args_struct(self):
        def method(req: zmq_ipc.EngineCoreRequest, n: int):
            return req, n

        # Decoded builtin (list-encoded struct) converts to the annotated type.
        converted = zmq_ipc.EngineCoreProc._convert_msgspec_args(
            method, (["r", [1], None, None, None, 0.0, None, None, None], 3)
        )
        req, n = converted
        assert isinstance(req, zmq_ipc.EngineCoreRequest)
        assert req.request_id == "r"
        assert n == 3

    def test_scheduler_bucketing_by_client_index(self):
        # m9 seam mirror: outputs bucket per request.client_index exactly like
        # scheduler.py L1924 outputs[request.client_index].append + L2015 dict.
        sched = zmq_ipc.SchedulerSeam()
        r0 = zmq_ipc.Request(request_id="a", client_index=0, prompt_token_ids=[1])
        r1 = zmq_ipc.Request(request_id="b", client_index=1, prompt_token_ids=[2])
        sched.add_request(r0)
        sched.add_request(r1)
        sched.enqueue_step_outputs(
            script_step(("a", [1], zmq_ipc.FinishReason.STOP))
            + script_step(("b", [2], zmq_ipc.FinishReason.LENGTH))
        )
        model_executed, scheduled = sched.take_scheduled_batch()
        assert model_executed is True
        outs = sched.update_from_output(scheduled)
        assert set(outs.keys()) == {0, 1}
        assert outs[0].outputs[0].request_id == "a"
        assert outs[1].outputs[0].request_id == "b"
        assert outs[0].finished_requests == {"a"}
        assert outs[1].finished_requests == {"b"}
        assert sched.has_requests() is False

    def test_abort_marks_drive_finish_outputs(self):
        # The abort path: finish_requests marks the request, update_from_output
        # emits the EngineCoreOutput(finish_reason=ABORT) + finished_requests.
        sched = zmq_ipc.SchedulerSeam()
        r = zmq_ipc.Request(request_id="z", client_index=0, prompt_token_ids=[1])
        sched.add_request(r)
        aborted = sched.finish_requests(["z"], zmq_ipc.RequestStatus.FINISHED_ABORTED)
        assert [a.request_id for a in aborted] == ["z"]
        _executed, scheduled = sched.take_scheduled_batch()
        assert scheduled == []  # no script was provided
        outs = sched.update_from_output(scheduled)
        assert outs[0].finished_requests == {"z"}
        assert outs[0].outputs[0].finish_reason is zmq_ipc.FinishReason.ABORT
        assert outs[0].outputs[0].new_token_ids == []
