"""ch31 精简版测试 —— 复现真实 vLLM 离线同步路径的【可观察行为】。

这些测试不测精简版自洽，而是钉住 dossier 记录的真实 vLLM 行为：
  1. make_client 三分支：离线默认 (mp=True, async=False) → SyncMPClient（非 InprocClient）。
  2. VLLM_ENABLE_V1_MULTIPROCESSING 默认 True → from_engine_args 强翻 multiprocess_mode=True。
  3. =0 回退 → InprocClient（真进程内、无 ZMQ）。
  4. SyncMPClient：get_output 阻塞 outputs_queue.get()，队列由后台 daemon 线程喂。
  5. _add_request 把 SamplingParams.output_kind 强设 FINAL_ONLY。
  6. _run_engine 末尾按 int(request_id) 排序还原提交序（乱序完成也能还原）。
  7. completion 路径不打物化 warning；chat 路径（prompts 物化成 list）打 warning_once。
  8. embed = encode(pooling_task='embed') 的薄封装，逐个 EmbeddingRequestOutput.from_base。
  9. n>1 在 LLMEngine.add_request 处 ParentRequest 扇出子请求。
"""

import importlib
import queue
import sys
from pathlib import Path

import pytest

IMPL = Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))


def _reload_with_env(monkeypatch, value):
    """按给定 VLLM_ENABLE_V1_MULTIPROCESSING 值重载 envs + 依赖它的模块。"""
    if value is None:
        monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    else:
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", value)
    import envs
    importlib.reload(envs)
    import core_client
    importlib.reload(core_client)
    import llm_engine
    importlib.reload(llm_engine)
    import llm as llm_mod
    importlib.reload(llm_mod)
    return llm_mod, core_client, llm_engine, envs


# --- 1 & 2 & 3: make_client 三分支 + env 决定默认 ---

def test_default_offline_uses_sync_mp_client(monkeypatch):
    _, core_client, _, envs = _reload_with_env(monkeypatch, "1")
    assert envs.VLLM_ENABLE_V1_MULTIPROCESSING is True
    client = core_client.EngineCoreClient.make_client(
        multiprocess_mode=True, asyncio_mode=False)
    assert isinstance(client, core_client.SyncMPClient)
    assert not isinstance(client, core_client.InprocClient)


def test_inproc_only_when_multiprocessing_disabled(monkeypatch):
    _, core_client, _, _ = _reload_with_env(monkeypatch, "0")
    client = core_client.EngineCoreClient.make_client(
        multiprocess_mode=False, asyncio_mode=False)
    assert isinstance(client, core_client.InprocClient)


def test_from_engine_args_forces_multiprocessing_when_env_true(monkeypatch):
    llm_mod, core_client, llm_engine, _ = _reload_with_env(monkeypatch, "1")
    eng = llm_engine.LLMEngine.from_engine_args(engine_args=object())
    # env 默认 True → 即便形参默认 enable_multiprocessing=False，也被强翻 → SyncMPClient。
    assert eng.multiprocess_mode is True
    assert isinstance(eng.engine_core, core_client.SyncMPClient)


def test_from_engine_args_falls_back_to_inproc_when_env_false(monkeypatch):
    llm_mod, core_client, llm_engine, _ = _reload_with_env(monkeypatch, "0")
    eng = llm_engine.LLMEngine.from_engine_args(engine_args=object())
    assert eng.multiprocess_mode is False
    assert isinstance(eng.engine_core, core_client.InprocClient)


def test_async_without_multiprocessing_raises(monkeypatch):
    _, core_client, _, _ = _reload_with_env(monkeypatch, "1")
    with pytest.raises(NotImplementedError):
        core_client.EngineCoreClient.make_client(
            multiprocess_mode=False, asyncio_mode=True)


# --- 4: SyncMPClient 后台线程喂阻塞队列 ---

def test_sync_mp_client_get_output_blocks_on_queue(monkeypatch):
    _, core_client, _, _ = _reload_with_env(monkeypatch, "1")
    from messages import SamplingParams, EngineCoreRequest
    client = core_client.SyncMPClient()
    # outputs_queue 是阻塞队列；后台 daemon 线程负责喂。
    assert isinstance(client.outputs_queue, queue.Queue)
    assert client.output_queue_thread.daemon is True
    assert client.output_queue_thread.is_alive()
    # 提交一个 3-token 请求，应能从 get_output 同步阻塞取到输出，最后一拍 finished。
    client.add_request(EngineCoreRequest(
        request_id="0", params=SamplingParams(max_tokens=3)))
    finished_seen = False
    for _ in range(3):
        outs = client.get_output()
        for o in outs.outputs:
            if o.finished:
                finished_seen = True
    assert finished_seen


# --- 5: _add_request 强设 FINAL_ONLY ---

def test_add_request_forces_final_only(monkeypatch):
    llm_mod, _, _, _ = _reload_with_env(monkeypatch, "1")
    from messages import RequestOutputKind, SamplingParams
    obj = llm_mod.LLM.__new__(llm_mod.LLM)
    obj.request_counter = llm_mod.Counter()
    captured = {}

    class _Eng:
        def add_request(self, request_id, prompt, params, **kw):
            captured["params"] = params
            return request_id

    obj.llm_engine = _Eng()
    sp = SamplingParams(output_kind=RequestOutputKind.CUMULATIVE)
    obj._add_request("p", sp)
    assert captured["params"].output_kind is RequestOutputKind.FINAL_ONLY


# --- 6: generate 端到端 + 排序还原 ---

def test_generate_returns_sorted_by_request_id(monkeypatch):
    llm_mod, _, _, _ = _reload_with_env(monkeypatch, "1")
    from messages import SamplingParams
    model = llm_mod.LLM(runner="generate")
    prompts = ["a", "b", "c", "d"]
    outs = model.generate(prompts, SamplingParams(max_tokens=2))
    assert len(outs) == 4
    ids = [int(o.request_id) for o in outs]
    assert ids == sorted(ids) == [0, 1, 2, 3]
    assert all(o.finished for o in outs)


def test_generate_sorted_even_if_completion_out_of_order(monkeypatch):
    """乱序完成也按 request_id 还原：让 id=2 比 id=0 先 finished。"""
    llm_mod, _, _, _ = _reload_with_env(monkeypatch, "1")
    from messages import SamplingParams
    model = llm_mod.LLM(runner="generate")
    # 不同 max_tokens 让完成顺序与提交序不同。
    params = [SamplingParams(max_tokens=5), SamplingParams(max_tokens=1),
              SamplingParams(max_tokens=3)]
    outs = model.generate(["x", "y", "z"], params)
    assert [int(o.request_id) for o in outs] == [0, 1, 2]


def test_generate_rejects_non_generate_runner(monkeypatch):
    llm_mod, _, _, _ = _reload_with_env(monkeypatch, "1")
    model = llm_mod.LLM(runner="pooling")
    with pytest.raises(ValueError):
        model.generate(["a"])


# --- 7: completion 不 warning，chat（物化 list）warning ---

def test_completion_does_not_warn(monkeypatch):
    llm_mod, _, _, _ = _reload_with_env(monkeypatch, "1")
    llm_mod._WARNED.clear()
    model = llm_mod.LLM(runner="generate")
    model.generate(["a", "b"])
    assert llm_mod._WARNED == set()


def test_chat_warns_when_prompts_materialized(monkeypatch):
    llm_mod, _, _, _ = _reload_with_env(monkeypatch, "1")
    llm_mod._WARNED.clear()
    model = llm_mod.LLM(runner="generate")
    # 直接调用 _render_and_run_requests 并传一个【已物化的 list】→ 触发 warning_once。
    model._render_and_run_requests(
        prompts=["a", "b"],  # list/tuple → warning
        params=[__import__("messages").SamplingParams(max_tokens=1)] * 2,
        output_type=__import__("messages").RequestOutput,
    )
    assert len(llm_mod._WARNED) == 1


def test_chat_generator_path_does_not_warn(monkeypatch):
    llm_mod, _, _, _ = _reload_with_env(monkeypatch, "1")
    llm_mod._WARNED.clear()
    model = llm_mod.LLM(runner="generate")
    # 经 chat() → _run_chat 传生成器 → 不打 warning。
    model.chat([[{"role": "user", "content": "hi"}]])
    assert llm_mod._WARNED == set()


# --- 8: embed = encode(embed) 薄封装 ---

def test_embed_wraps_encode(monkeypatch):
    llm_mod, _, _, _ = _reload_with_env(monkeypatch, "1")
    from messages import EmbeddingRequestOutput
    model = llm_mod.LLM(runner="pooling")
    outs = model.embed(["a", "b"])
    assert len(outs) == 2
    assert all(isinstance(o, EmbeddingRequestOutput) for o in outs)


def test_encode_returns_pooling_outputs(monkeypatch):
    llm_mod, _, _, _ = _reload_with_env(monkeypatch, "1")
    from messages import PoolingRequestOutput
    model = llm_mod.LLM(runner="pooling")
    outs = model.encode(["a"], pooling_task="encode")
    assert all(isinstance(o, PoolingRequestOutput) for o in outs)


# --- 9: n>1 在 add_request 处 ParentRequest 扇出 ---

def test_n_gt_1_fans_out_in_add_request(monkeypatch):
    _, core_client, llm_engine, _ = _reload_with_env(monkeypatch, "0")  # InprocClient 便于内省
    from messages import SamplingParams
    eng = llm_engine.LLMEngine.from_engine_args(engine_args=object())
    added = []
    orig = eng.engine_core.add_request

    def _spy(request):
        added.append(request.request_id)
        return orig(request)

    eng.engine_core.add_request = _spy
    eng.add_request("7", "prompt", SamplingParams(n=3, max_tokens=1))
    # 扇出 3 个子请求 → engine_core.add_request 被调 3 次。
    assert len(added) == 3


def test_inproc_get_output_steps_engine_directly(monkeypatch):
    """InprocClient.get_output 直接 step_fn()（无 ZMQ/无后台线程），体现进程内回退。"""
    _, core_client, _, _ = _reload_with_env(monkeypatch, "0")
    from messages import SamplingParams, EngineCoreRequest
    client = core_client.InprocClient()
    assert not hasattr(client, "output_queue_thread")  # 无后台线程
    client.add_request(EngineCoreRequest(
        request_id="0", params=SamplingParams(max_tokens=2)))
    outs = client.get_output()  # 直接 step 一拍
    assert outs.outputs and outs.outputs[0].request_id == "0"
