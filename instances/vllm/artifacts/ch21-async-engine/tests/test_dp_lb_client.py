"""Tests for the DP load-balancing clients.

Asserts real vLLM behavior:
  * make_async_mp_client picks AsyncMPClient / DPAsyncMPClient / DPLBAsyncMPClient.
  * add_request_async stamps current_wave and client_index, and sends FIRST_REQ
    only when engines are paused.
  * get_core_engine_for_request picks min score=waiting*4+running and locally
    pre-increments the chosen engine's waiting count.
  * explicit data_parallel_rank bypasses load balancing.
  * stats frame updates current_wave / engines_running / sliced lb_engines.
"""

import asyncio

from implementation._bridge import FakeSocket
from implementation.dp_lb_client import (
    AsyncMPClient,
    DPAsyncMPClient,
    DPLBAsyncMPClient,
    EngineCoreRequest,
    EngineCoreRequestType,
    _ParallelConfig,
    make_async_mp_client,
)


def test_factory_single_engine():
    pc = _ParallelConfig(data_parallel_size=1)
    c = make_async_mp_client(pc)
    assert type(c) is AsyncMPClient


def test_factory_external_lb():
    pc = _ParallelConfig(data_parallel_size=2, data_parallel_external_lb=True)
    c = make_async_mp_client(pc, num_engines=1)
    assert type(c) is DPAsyncMPClient


def test_factory_internal_lb():
    pc = _ParallelConfig(data_parallel_size=2, data_parallel_external_lb=False)
    c = make_async_mp_client(pc, num_engines=4)
    assert type(c) is DPLBAsyncMPClient


def test_score_picks_least_loaded_engine():
    c = DPLBAsyncMPClient(client_count=1, client_index=0, num_engines=3)
    # engine 2 is emptiest: scores = [4*2+1, 4*1+5, 4*0+0] = [9, 9, 0]
    c.lb_engines = [[2, 1], [1, 5], [0, 0]]
    eng = c.get_core_engine_for_request(EngineCoreRequest("r1"))
    assert eng == 2
    # local pre-increment of the chosen engine's waiting count (+client_count)
    assert c.lb_engines[2][0] == 1
    assert c.reqs_in_flight["r1"] == 2


def test_waiting_weighted_four_to_one():
    c = DPLBAsyncMPClient(client_count=1, client_index=0, num_engines=2)
    # eng0: 1 waiting (score 4), eng1: 3 running (score 3) → pick eng1
    c.lb_engines = [[1, 0], [0, 3]]
    assert c.get_core_engine_for_request(EngineCoreRequest("r")) == 1


def test_pre_increment_spreads_burst():
    c = DPLBAsyncMPClient(client_count=1, client_index=0, num_engines=2)
    c.lb_engines = [[0, 0], [0, 0]]
    # eng_start_index=0 → first request lands on engine 0
    e1 = c.get_core_engine_for_request(EngineCoreRequest("a"))
    # after pre-increment eng0 score=4 > eng1 score=0 → second goes to eng1
    e2 = c.get_core_engine_for_request(EngineCoreRequest("b"))
    assert {e1, e2} == {0, 1}


def test_explicit_dp_rank_bypasses_lb():
    c = DPLBAsyncMPClient(client_count=1, client_index=0, num_engines=4)
    c.lb_engines = [[0, 0], [0, 0], [0, 0], [0, 0]]
    req = EngineCoreRequest("r", data_parallel_rank=3)
    assert c.get_core_engine_for_request(req) == 3
    # LB counts untouched
    assert c.lb_engines == [[0, 0], [0, 0], [0, 0], [0, 0]]


def test_add_request_stamps_wave_and_sends_first_req_when_paused():
    sock = FakeSocket("first_req")
    c = DPAsyncMPClient(num_engines=1, first_req_send_socket=sock)
    c.current_wave = 7
    c.client_index = 2
    c.engines_running = False
    req = EngineCoreRequest("r")
    asyncio.run(c.add_request_async(req))
    assert req.current_wave == 7
    assert req.client_index == 2
    # ADD was sent
    typ, sent_req, eng = c._sent_inputs[0]
    assert typ == EngineCoreRequestType.ADD
    # FIRST_REQ wakeup sent because engines were paused
    assert sock.recv() == ("FIRST_REQ", c.core_engine)


def test_add_request_no_first_req_when_running():
    sock = FakeSocket("first_req")
    c = DPAsyncMPClient(num_engines=1, first_req_send_socket=sock)
    c.engines_running = True
    asyncio.run(c.add_request_async(EngineCoreRequest("r")))
    assert not sock.pending()  # no wakeup needed


def test_consume_stats_frame_updates_state_and_slices_counts():
    c = DPLBAsyncMPClient(client_count=1, client_index=0, num_engines=2)
    c.engine_ranks_managed = [0, 1]
    # global counts for all engines from coordinator; running=True, wave=9
    frame = ([[1, 2], [3, 4]], 9, True)
    c.consume_stats_frame(frame)
    assert c.current_wave == 9
    assert c.engines_running is True
    assert c.lb_engines == [[1, 2], [3, 4]]


def test_consume_stats_frame_none_counts_keeps_lb():
    c = DPAsyncMPClient(num_engines=1)
    before = c.lb_engines
    c.consume_stats_frame((None, 4, False))
    assert c.current_wave == 4
    assert c.engines_running is False
    assert c.lb_engines is before  # unchanged when counts is None
