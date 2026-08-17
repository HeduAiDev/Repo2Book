"""ch05 m3 worked-example driver — 字节标签线格式：真 ROUTER/DEALER socket 抓帧。

捕获一条消息跨界的真实 ZMQ 帧（帧数/每帧字节数/标签字节），对照：
  - 无张量 ADD（主帧一条）
  - 小张量 ADD（<256B 内联进主帧，仍一条）
  - 大张量 ADD（≥256B aux 独立帧，多一条）
  - ABORT / UTILITY（同一首帧协议的另两类标签）
外加 m1 拓扑探针（HWM=0 ×4 socket、io_threads=2、linger=4000、identity 两字节、
bind 默认规则）与 256B 内联阈值边界（252B 内联 / 256B aux）。

Run (host, 纯 CPU):  python explainer/traces/run_m3_wire_format.py
Output:             explainer/traces/m3_wire_format.json
"""

import json
import sys
import uuid
from pathlib import Path

import msgpack
import torch
import zmq

IMPL_DIR = Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL_DIR))

import zmq_ipc  # noqa: E402


def hexs(b) -> str:
    return bytes(b).hex()


def main():
    out = {
        "driver": "run_m3_wire_format.py",
        "mechanism": "m3 字节标签线格式 (EngineCoreRequestType 单字节首帧 + (Identity, Type, *Payload) 布局)",
        "environment": {
            "platform": sys.platform,
            "pyzmq": zmq.pyzmq_version(),
            "libzmq": zmq.zmq_version(),
            "transport_note": (
                "host win32: zmq 无 ipc:// transport，按精简版 HOST SEAM 用回环 tcp://；"
                "bind/connect 流程与帧语义和 pin 的 Linux ipc:// 路径完全一致（impl-notes Seam 表）"
            ),
        },
        "config": {
            "engine_index": 0,
            "identity_bytes_len": 2,
            "identity_hex": (0).to_bytes(2, "little").hex(),
            "identity_note": "identity = engine_index.to_bytes(2, 'little') = b'\\x00\\x00'",
            "VLLM_MSGPACK_ZERO_COPY_THRESHOLD": zmq_ipc.envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD,
            "small_tensor": {"numel": 8, "dtype": "float32", "nbytes": 32},
            "big_tensor": {"numel": 2048, "dtype": "float32", "nbytes": 8192},
        },
        "topology_probe": {},
        "rounds": [],
    }

    ctx = zmq.Context(io_threads=2)

    # ---- m1 拓扑探针：make_zmq_socket 出的 socket 全对 HWM=0 + bind 默认规则 -------
    probe = out["topology_probe"]
    probe["context_io_threads"] = ctx.get(zmq.IO_THREADS)
    router = zmq_ipc.make_zmq_socket(ctx, "tcp://127.0.0.1:0", zmq.ROUTER, bind=True)
    probe["router"] = {
        "type": "ROUTER",
        "bind": True,
        "rcvhwm": router.getsockopt(zmq.RCVHWM),
        "sndhwm": router.getsockopt(zmq.SNDHWM),
        "last_endpoint": router.getsockopt(zmq.LAST_ENDPOINT).decode(),
    }
    probe["rcvhwm_note"] = "make_zmq_socket 对 PULL/DEALER/ROUTER 置 RCVHWM=0、对 PUSH/DEALER/ROUTER 置 SNDHWM=0（network_utils.py:L310-L316）"
    # PULL 默认 bind（不在 PUSH/SUB/XSUB connect 集里）、PUSH 默认 connect——bind 默认规则
    pull = zmq_ipc.make_zmq_socket(ctx, "tcp://127.0.0.1:0", zmq.PULL)
    pull_ep = pull.getsockopt(zmq.LAST_ENDPOINT).decode()
    probe["pull"] = {"type": "PULL", "bind": True, "rcvhwm": pull.getsockopt(zmq.RCVHWM)}
    push = zmq_ipc.make_zmq_socket(ctx, pull_ep, zmq.PUSH, linger=4000)
    probe["push"] = {
        "type": "PUSH",
        "bind": False,
        "sndhwm": push.getsockopt(zmq.SNDHWM),
        "linger": push.getsockopt(zmq.LINGER),
        "linger_note": "引擎输出 PUSH 恒传 linger=4000：保证 ENGINE_CORE_DEAD 死讯先于关 socket 发出（core.py:L1758-L1763）",
    }
    # 引擎输入侧 DEALER：identity 两字节小端（同 core.py:L1661-L1667 每前端一条）
    dealer_ep = router.getsockopt(zmq.LAST_ENDPOINT).decode()
    dealer = zmq_ipc.make_zmq_socket(
        ctx, dealer_ep, zmq.DEALER, identity=(0).to_bytes(2, "little"), bind=False
    )
    probe["dealer"] = {
        "type": "DEALER",
        "bind": False,
        "rcvhwm": dealer.getsockopt(zmq.RCVHWM),
        "sndhwm": dealer.getsockopt(zmq.SNDHWM),
        "identity_hex": (0).to_bytes(2, "little").hex(),
        "dealers_per_frontend": 1,
    }
    push.close(linger=0)
    pull.close(linger=0)

    # ---- 轮 0：DEALER 先发言认亲（m2 硬约束的线上物证）----------------------------
    ready_payload = msgpack.packb({"status": "READY", "max_model_len": 4096}, use_bin_type=True)
    dealer.send(ready_payload)
    frames = router.recv_multipart()
    out["rounds"].append({
        "round": 0,
        "name": "DEALER 先发言（认亲帧）",
        "who_sees": "client 侧 ROUTER recv_multipart 的视角",
        "frames": [
            {"i": 0, "what": "对端 identity 信封（ROUTER 收到才认识它）", "len": len(frames[0]), "hex": hexs(frames[0])},
            {"i": 1, "what": "EngineCoreReadyResponse msgpack 载荷", "len": len(frames[1])},
        ],
        "frame_count": len(frames),
        "sender_frame_count": 1,
        "note": "DEALER 发 1 帧、ROUTER 收 2 帧——信封是 ROUTER 投递时加的（反方向 ROUTER→DEALER 发送时信封被吃掉）",
    })

    enc = zmq_ipc.MsgpackEncoder()
    dec = zmq_ipc.MsgpackDecoder(zmq_ipc.EngineCoreRequest)
    T = zmq_ipc.EngineCoreRequestType

    def core_request(rid, embeds=None):
        return zmq_ipc.EngineCoreRequest(
            request_id=rid,
            prompt_token_ids=[1, 2, 3],
            mm_features=None,
            sampling_params=zmq_ipc.SamplingParams(max_tokens=8),
            pooling_params=None,
            arrival_time=1.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            prompt_embeds=embeds,
        )

    def send_and_capture(rid, name, request_type, payload_obj, embeds=None):
        """按 _send_input 的拼帧布局 (Identity, Type, *bufs) 从 ROUTER 发、DEALER 收。"""
        bufs = enc.encode(payload_obj)
        message = (request_type.value, *bufs)
        engine = (0).to_bytes(2, "little")
        router.send_multipart((engine,) + message, copy=False)
        got = dealer.recv_multipart(copy=False)
        rec = {
            "round": len(out["rounds"]),
            "name": name,
            "sender_frame_count": 1 + len(message),
            "sender_layout": f"(identity {len(engine)}B, type 1B, *{len(bufs)} payload 帧)",
            "receiver_frame_count": len(got),
            "receiver_view": "引擎输入线程 recv_multipart（core.py:L1705）——identity 信封已被投递吃掉",
            "frames": [
                {"i": 0, "what": "type_frame（字节标签）", "len": len(got[0].buffer), "hex": hexs(got[0].buffer)}
            ],
        }
        for i, f in enumerate(got[1:]):
            rec["frames"].append({"i": i + 1, "what": f"payload 帧 {i}", "len": len(f.buffer)})
        return rec, got

    # ---- 轮 1：无张量 ADD -----------------------------------------------------------
    rec, got = send_and_capture("req-plain", "ADD · 无张量", T.ADD, core_request("req-plain"))
    req = dec.decode(got[1:])
    rec.update({
        "decode": {"request_id": req.request_id, "prompt_token_ids": req.prompt_token_ids,
                   "roundtrip_ok": req.request_id == "req-plain"},
        "verdict": f"收 {len(got)} 帧 = [b'\\x00', 主帧 {len(got[1].buffer)}B]——无张量请求只有一条主帧",
    })
    out["rounds"].append(rec)

    # ---- 轮 2：小张量 ADD（32B < 256B → 内联主帧）------------------------------------
    small = torch.arange(8, dtype=torch.float32)  # 8 × 4B = 32B
    rec, got = send_and_capture("req-small", "ADD · 小张量 32B", T.ADD, core_request("req-small", embeds=small))
    req = dec.decode(got[1:])
    rec.update({
        "tensor_nbytes": small.numel() * 4,
        "main_frame_len": len(got[1].buffer),
        "main_frame_len_without_tensor": out["rounds"][1]["frames"][1]["len"],
        "inline_delta": len(got[1].buffer) - out["rounds"][1]["frames"][1]["len"],
        "decode": {"dtype": str(req.prompt_embeds.dtype), "numel": req.prompt_embeds.numel(),
                   "roundtrip_ok": bool(torch.equal(req.prompt_embeds, small))},
        "verdict": f"收 {len(got)} 帧——32B < 256B 阈值，张量走 CUSTOM_TYPE_RAW_VIEW 内联进主帧（主帧变长），不多出独立帧",
    })
    out["rounds"].append(rec)

    # ---- 轮 3：大张量 ADD（8192B ≥ 256B → aux 独立零拷贝帧）---------------------------
    big = torch.arange(2048, dtype=torch.float32)  # 2048 × 4B = 8192B
    rec, got = send_and_capture("req-big", "ADD · 大张量 8192B", T.ADD, core_request("req-big", embeds=big))
    req = dec.decode(got[1:])
    aux = got[2].buffer if len(got) > 2 else None
    rec.update({
        "tensor_nbytes": big.numel() * 4,
        "frames_note": "帧 2 = aux backing buffer：张量字节的 uint8 memoryview，不拷进主帧",
        "aux_frame": {"len": len(aux), "first_float_bits": hexs(bytes(aux)[:4])},
        "decode": {"dtype": str(req.prompt_embeds.dtype), "numel": req.prompt_embeds.numel(),
                   "first_val": float(req.prompt_embeds[0]), "roundtrip_ok": bool(torch.equal(req.prompt_embeds, big))},
        "verdict": f"发 {rec['sender_frame_count']} 帧、收 {len(got)} 帧 = [b'\\x00', 主帧 {len(got[1].buffer)}B, aux {len(got[2].buffer)}B]——大张量多出一条独立帧，主帧里只有 (dtype, shape, aux 索引) 三元组",
    })
    out["rounds"].append(rec)

    # ---- 轮 4：ABORT ------------------------------------------------------------------
    rec, got = send_and_capture(None, "ABORT", T.ABORT, ["req-big"])
    ids = msgpack.unpackb(bytes(got[1].buffer), use_list=True)
    rec.update({"decode": {"request_ids": ids}, "verdict": f"收 {len(got)} 帧 = [b'\\x01', msgpack ids {len(got[1].buffer)}B]——同一条 socket、换一个标签字节就是另一类消息"})
    out["rounds"].append(rec)

    # ---- 轮 5：UTILITY 薄 RPC -----------------------------------------------------------
    call_id = uuid.uuid1().int >> 64
    rec, got = send_and_capture(None, "UTILITY 薄 RPC", T.UTILITY, (0, call_id, "get_supported_tasks", ()))
    tup = msgpack.unpackb(bytes(got[1].buffer), use_list=True)
    rec.update({"decode": {"client_index": tup[0], "method": tup[2], "args": list(tup[3])},
                "verdict": f"收 {len(got)} 帧 = [b'\\x03', msgpack 元组 {len(got[1].buffer)}B]——RPC 四元组 (client_index, call_id, method, args) 与 ADD 走同一条线"})
    out["rounds"].append(rec)

    # ---- 阈值边界：252B 内联 vs 256B aux -----------------------------------------------
    enc2 = zmq_ipc.MsgpackEncoder()
    just_under = torch.zeros(63, dtype=torch.float32)  # 63 × 4B = 252B
    at_threshold = torch.zeros(64, dtype=torch.float32)  # 64 × 4B = 256B
    out["threshold_boundary"] = {
        "just_under": {"numel": 63, "nbytes": 252, "frames": len(enc2.encode(core_request("u", embeds=just_under)))},
        "at_threshold": {"numel": 64, "nbytes": 256, "frames": len(enc2.encode(core_request("t", embeds=at_threshold)))},
        "note": "判定条件 obj.nbytes < size_threshold：252 < 256 内联（1 帧）；256 不小于 256 → aux 独立帧（2 帧）",
    }

    dealer.close(linger=0)
    router.close(linger=0)
    ctx.destroy(linger=0)

    dest = Path(__file__).resolve().parent / "m3_wire_format.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dest}")
    print(json.dumps({k: out[k] for k in ("config", "topology_probe")}, indent=1, ensure_ascii=False))
    for r in out["rounds"]:
        print(r["round"], r["name"], "send", r.get("sender_frame_count"), "recv", r.get("receiver_frame_count"))


if __name__ == "__main__":
    main()
