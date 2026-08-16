"""m2 driver — client 工厂二轴 (multiprocess_mode × asyncio_mode) 实测。

跑本章精简版 companion (implementation/engine_faces.py, 与 core_client.py:L89-L139
逐字一致的 make_client/make_async_mp_client/from_engine_args), 把 2×2 分发表、
NotImplementedError 拒绝、from_engine_args 的 envs 强翻 (envs.py:L149 默认 True)
逐项跑一遍并落盘 traces/m2_client_factory.json。
host 纯控制流, 无需容器。"""

import importlib.util
import json
import sys
import traceback
from pathlib import Path

_IMPL = Path(__file__).resolve().parent.parent.parent / "implementation" / "engine_faces.py"
_spec = importlib.util.spec_from_file_location("engine_faces", _IMPL)
ef = importlib.util.module_from_spec(_spec)
sys.modules["engine_faces"] = ef
_spec.loader.exec_module(ef)


class _EngineArgsStub:
    disable_log_stats = True

    def create_engine_config(self, usage_context=None):
        return ef.VllmConfig()


def cfg():
    return ef.VllmConfig()


def main():
    out = {
        "mechanism": "m2 client 工厂二轴",
        "companion": "implementation/engine_faces.py (pin vLLM v0.27.1 / 6e448d0ea)",
        "envs_defaults": {
            "VLLM_ENABLE_V1_MULTIPROCESSING": ef.envs.VLLM_ENABLE_V1_MULTIPROCESSING,
            "note": "vllm/envs.py:L149 默认值——离线面默认也多进程的开关",
        },
        "factory_table": [],
        "async_mp_client_identity": {},
        "from_engine_args_flip": {},
    }

    # ---- 2×2 分发表 (core_client.py:L89-L112) ----
    for mp in (True, False):
        for am in (True, False):
            entry = {"multiprocess_mode": mp, "asyncio_mode": am}
            try:
                client = ef.EngineCoreClient.make_client(
                    mp, am, cfg(), ef.Executor, False
                )
                entry["result"] = type(client).__name__
                if isinstance(client, ef.AsyncMPClient):
                    entry["constructed_via"] = "make_async_mp_client (L114-L139)"
            except NotImplementedError as e:
                entry["result"] = "NotImplementedError"
                entry["message"] = str(e)
            out["factory_table"].append(entry)

    # ---- make_async_mp_client 出生参数 (core_client.py:L116-L139) ----
    ident = ef.EngineCoreClient.make_async_mp_client(
        cfg(), ef.Executor, False, client_count=2, client_index=1
    )
    out["async_mp_client_identity"] = {
        "given": {"client_count": 2, "client_index": 1},
        "observed": {
            "type": type(ident).__name__,
            "client_count": ident.client_count,
            "client_index": ident.client_index,
        },
        "signature_defaults": {"client_count": 1, "client_index": 0},
    }
    # 默认构造 (不带出生参数) —— 单前端恒 0 的物证
    default_client = ef.EngineCoreClient.make_async_mp_client(cfg(), ef.Executor, False)
    out["async_mp_client_identity"]["default_construction"] = {
        "client_count": default_client.client_count,
        "client_index": default_client.client_index,
    }

    # ---- from_engine_args 的 envs 强翻 (llm_engine.py:L170-L186; envs.py:L149) ----
    try:
        ef.envs.VLLM_ENABLE_V1_MULTIPROCESSING = True
        eng = ef.LLMEngine.from_engine_args(_EngineArgsStub())
        out["from_engine_args_flip"]["envs_True"] = {
            "engine_core_type": type(eng.engine_core).__name__,
            "verdict": "离线默认 = SyncMPClient(跨进程), 不是 InprocClient",
        }
        ef.envs.VLLM_ENABLE_V1_MULTIPROCESSING = False
        eng2 = ef.LLMEngine.from_engine_args(_EngineArgsStub())
        out["from_engine_args_flip"]["envs_False"] = {
            "engine_core_type": type(eng2.engine_core).__name__,
            "verdict": "逃生舱 = InprocClient(V0-style, core_client.py:L306)",
        }
    finally:
        ef.envs.VLLM_ENABLE_V1_MULTIPROCESSING = True  # 恢复 pin 默认

    # ---- 汇总判据 (illustrator/writer 直接引用) ----
    out["summary"] = {
        "axis_count": 2,
        "leaf_count": 3,
        "rejected_combination": "asyncio_mode=True 且 multiprocess_mode=False",
        "offline_default_leaf": "SyncMPClient",
        "online_default_leaf": "AsyncMPClient",
        "escape_hatch_leaf": "InprocClient",
    }

    dest = Path(__file__).resolve().parent / "m2_client_factory.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
