# m02 取证：disaggregator 控制回路（m2）——回执信封 kv_transfer_params 的双跳搬运。
# 跳 1：proxy 发单 P（改写请求体）；跳 2：掏 P 回执原样附加转交 D；终：D 的 Request 挂载。
from _driver_common import KVCacheBlocksStub, dump, make_d_request, make_p_request
from _pd_harness import make_engine


def main() -> None:
    doc = {
        "mechanisms": ["m2"],
        "params": "16-token prompt；P 块 [0,1,2,3]；客户端原始请求 max_tokens=7、"
        "min_tokens=2、stream=True",
    }

    # ── 跳 1：proxy 发单 P（真 send_request_to_service + 假 HTTP 客户端） ──
    import asyncio

    from toy_proxy_server import send_request_to_service

    class _Resp:
        def raise_for_status(self):
            return None

        async def aread(self):
            return b"{}"

    class _Client:
        def __init__(self):
            self.sent = None

        async def post(self, endpoint, json=None, headers=None):
            self.sent = (endpoint, dict(json))  # 快照在 post 时刻
            return _Resp()

    client = _Client()
    req_data = {"model": "m", "prompt": "hi", "max_tokens": 7, "min_tokens": 2,
                "stream": True}
    asyncio.run(
        send_request_to_service(
            {"client": client}, "/completions", req_data, request_id="r1"
        )
    )
    sent = client.sent[1]
    doc["leg1_proxy_to_p"] = {
        "sent_max_tokens": sent["max_tokens"],
        "original_max_tokens": req_data["max_tokens"],
        "sent_stream": sent["stream"],
        "original_stream": req_data["stream"],
        "min_tokens_in_sent": "min_tokens" in sent,
        "min_tokens_restored_after": req_data["min_tokens"],
        "sent_params_do_remote_decode": sent["kv_transfer_params"]["do_remote_decode"],
        "sent_params_do_remote_prefill": sent["kv_transfer_params"]["do_remote_prefill"],
        "sent_params_keys": len(sent["kv_transfer_params"]),
    }

    # ── 跳 2 前半：P 引擎真终局产回执（request_finished 返回的字典） ──
    p = make_engine("p", kv_role="kv_producer")
    d = make_engine("d", kv_role="kv_consumer")
    try:
        from vllm.v1.request import RequestStatus

        p_req = make_p_request("req-1")
        p_req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        p_req.num_computed_tokens = 16
        delay_free, receipt = p.scheduler.request_finished(p_req, ([0, 1, 2, 3],))
        doc["p_terminal_receipt"] = {
            "delay_free_blocks": delay_free,
            "receipt": receipt,
            "receipt_keys": len(receipt),
            "remote_block_ids": receipt["remote_block_ids"],
            "remote_num_tokens": receipt["remote_num_tokens"],
            "remote_engine_id": receipt["remote_engine_id"],
            "remote_request_id": receipt["remote_request_id"],
            "tp_size": receipt["tp_size"],
            "do_remote_prefill": receipt["do_remote_prefill"],
            "do_remote_decode": receipt["do_remote_decode"],
            "remote_blocks_expiry_time": receipt["remote_blocks_expiry_time"],
        }

        # ── 跳 2 后半：proxy 掏回执转交 D（真 _handle_completions + 假双客户端） ──
        from fastapi import FastAPI
        from starlette.requests import Request as StarletteRequest

        from toy_proxy_server import _handle_completions

        class _JsonResp(_Resp):
            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

            async def aclose(self):
                return None

        class _StreamCtx:
            def __init__(self, chunks):
                self.chunks = chunks

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def raise_for_status(self):
                return None

            async def aiter_bytes(self):
                for chunk in self.chunks:
                    yield chunk

        class _DecodeClient:
            def __init__(self, chunks):
                self.chunks = chunks
                self.sent = None

            async def post(self, endpoint, json=None, headers=None):
                return _Resp()

            def stream(self, method, endpoint, json=None, headers=None):
                self.sent = (endpoint, json)
                return _StreamCtx(self.chunks)

        prefill_client = _Client()

        async def _p_post(endpoint, json=None, headers=None):
            prefill_client.sent = (endpoint, dict(json))
            return _JsonResp({"kv_transfer_params": receipt})

        prefill_client.post = _p_post  # P 腿返回体里带真回执
        decode_client = _DecodeClient(chunks=[b"data: {}\n\n"])

        app = FastAPI()
        app.state.prefill_clients = [
            {"client": prefill_client, "host": "hp", "port": 1, "id": 0}
        ]
        app.state.decode_clients = [
            {"client": decode_client, "host": "hd", "port": 2, "id": 0}
        ]
        import itertools

        app.state.prefill_iterator = itertools.cycle(range(1))
        app.state.decode_iterator = itertools.cycle(range(1))

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/completions",
            "headers": [],
            "app": app,
        }
        starlette_req = StarletteRequest(scope)

        async def _body():
            return {"model": "m", "prompt": "hi", "stream": True}

        starlette_req.json = _body

        async def run():
            resp = await _handle_completions("/completions", starlette_req)
            return b"".join([c async for c in resp.body_iterator])

        body = asyncio.run(run())
        d_sent = decode_client.sent[1]
        doc["leg2_proxy_to_d"] = {
            "streamed_body_back": body.decode(),
            "d_leg_params_identical_to_receipt": d_sent["kv_transfer_params"] == receipt,
            "d_leg_max_tokens": d_sent.get("max_tokens"),
            "d_leg_stream": d_sent.get("stream"),
            "d_leg_min_tokens": d_sent.get("min_tokens"),
            "d_leg_prompt": d_sent.get("prompt"),
        }

        # ── 终：D 引擎的 Request 挂载（extra_args → kv_transfer_params） ──
        d_req = make_d_request("req-1", receipt)
        mounted = d_req.kv_transfer_params
        coord_fields = (
            "remote_block_ids",
            "remote_engine_id",
            "remote_request_id",
            "remote_host",
            "remote_port",
            "tp_size",
            "remote_num_tokens",
        )
        doc["d_request_mount"] = {
            "mounted_keys": len(mounted),
            "coordinate_fields": len(coord_fields),
            "all_coordinates_equal": all(
                mounted[f] == receipt[f] for f in coord_fields
            ),
            "mounted_do_remote_prefill": mounted["do_remote_prefill"],
            "mounted_do_remote_decode": mounted["do_remote_decode"],
            "p_leg_flag_do_remote_decode": make_p_request("req-1")
            .kv_transfer_params["do_remote_decode"],
        }

        # 信封在 D 侧的下一步：查命中（m4 的入口，这里只登记跳数）
        doc["hops"] = {
            "http_hops": 2,
            "hop1": "client→proxy→P（proxy 改写：max_tokens=1、stream=False）",
            "hop2": "proxy→D（回执原样附加）",
            "engine_crossings": 2,
        }
    finally:
        p.close()
        d.close()

    dump("m02", doc)


if __name__ == "__main__":
    main()
