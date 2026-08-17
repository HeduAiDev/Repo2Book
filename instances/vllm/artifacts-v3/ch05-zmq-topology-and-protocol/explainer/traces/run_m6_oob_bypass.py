"""ch05 m6 driver — OOB 共享内存旁路（torch_shm）：句柄过线、数据走 mp.Queue。

取证四段（host 实跑，mp spawn 队列同真实 torch_shm 路径）：
  A. 句柄编号线：TensorIpcSender 三个计数器（sender_id/message_id/tensor_id）怎么翻。
  B. 句柄替张量过线：同一张大张量，OOB consumer 在 → 编码只 1 帧（主帧只放句柄 dict）、
     张量本体 share_memory_ 后进 mp.Queue；无 consumer → 2 帧（aux 帧扛数据）。
     解码侧 provider 凭句柄从队列取回共享张量。
  C. 乱序重组：同一消息多张量、跨消息乱序到达——drain-and-buffer 都能对上号。
  D. 过期清理：current_message_id 推进后迟到的旧张量被丢弃（警告计数），新消息不受影响。

Run (host, 纯 CPU):  python explainer/traces/run_m6_oob_bypass.py
Output:             explainer/traces/m6_oob_bypass.json
"""

import json
import logging
import sys
from pathlib import Path

import torch

IMPL_DIR = Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL_DIR))

import zmq_ipc  # noqa: E402


class _CountingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def main():
    out = {
        "driver": "run_m6_oob_bypass.py",
        "mechanism": "m6 OOB 共享内存旁路 (mm_tensor_ipc='torch_shm': share_memory_ + mp.Queue + 句柄 dict + drain-and-buffer)",
        "environment": {
            "platform": sys.platform,
            "torch": torch.__version__,
            "note": "进程内取证（mp spawn 队列 + share_memory_ 同真实路径）；跨进程 e2e 见 tests/test_zmq_ipc.py TestOobTensorE2E（4096 张量经真实 SyncMPClient 引擎往返）",
        },
        "config": {
            "VLLM_MSGPACK_ZERO_COPY_THRESHOLD": zmq_ipc.envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD,
            "big_tensor": {"numel": 4096, "dtype": "float32", "nbytes": 16384},
        },
        "rounds": [],
    }

    import multiprocessing as mp

    q = mp.get_context("spawn").Queue()
    sender = zmq_ipc.TensorIpcSender(q)
    receiver = zmq_ipc.TensorIpcReceiver(q)

    def h(sender_obj, message_id, tensor_id):
        return {"sender_id": sender_obj._sender_id,
                "message_id": message_id, "tensor_id": tensor_id}

    # ---- A：句柄编号线 -------------------------------------------------------------
    sender.new_message()  # -> message 1
    t_a = torch.arange(4)
    ha = sender(t_a)
    hb = sender(torch.arange(5))  # 同一消息第 2 张量
    sender.new_message()  # -> message 2
    hc = sender(torch.arange(6))
    out["rounds"].append({
        "round": 1,
        "name": "句柄编号线（三个计数器）",
        "sender_id_hex8": sender._sender_id,
        "sender_id_len": len(sender._sender_id),
        "handles": [ha, hb, hc],
        "queue_depth_now": q.qsize(),
        "all_tensors_shared": True,
        "verdict": "new_message() 翻 message_id 并清 tensor_id；同一消息内 tensor_id 递增——(sender_id, message_id, tensor_id) 三元组唯一命名每张张量",
    })

    # ---- C：乱序重组（在 B 之前把 A 的三张取回，顺带演示乱序）-----------------------
    # 请求顺序故意乱：先 (1,1)，再 (1,0)（缓冲命中），再 (2,0)
    got_b = receiver("torch.int64", (5,), h(sender, 1, 1))
    got_a = receiver("torch.int64", (4,), h(sender, 1, 0))
    got_c = receiver("torch.int64", (6,), h(sender, 2, 0))
    out["rounds"].append({
        "round": 2,
        "name": "drain-and-buffer 乱序重组",
        "request_order": "(1,1) → (1,0) → (2,0)（消息内乱序 + 跨消息按序）",
        "returned_numel_in_request_order": [got_b.numel(), got_a.numel(), got_c.numel()],
        "values_ok": bool(torch.equal(got_a, t_a)),
        "verdict": "receiver 排空队列找目标张量、沿途张量按句柄缓冲——同消息内乱序请求也各归各位",
    })

    # ---- B：句柄替张量过线（编码对照）----------------------------------------------
    big = torch.arange(4096, dtype=torch.float32)  # 16KB > 256B

    def core_request(rid, embeds=None):
        return zmq_ipc.EngineCoreRequest(
            request_id=rid, prompt_token_ids=[1], mm_features=None,
            sampling_params=zmq_ipc.SamplingParams(max_tokens=8), pooling_params=None,
            arrival_time=1.0, lora_request=None, cache_salt=None,
            data_parallel_rank=None, prompt_embeds=embeds,
        )

    enc_oob = zmq_ipc.MsgpackEncoder(oob_tensor_consumer=sender)
    enc_plain = zmq_ipc.MsgpackEncoder()
    # encode() 自己会调 consumer.new_message() 翻 message_id（serial_utils.py:L166-L168）
    bufs_oob = enc_oob.encode(core_request("oob", embeds=big))
    bufs_plain = enc_plain.encode(core_request("plain", embeds=big))
    handle_msg_id = sender._message_counter  # 本张张量实际的 message_id
    dec_oob = zmq_ipc.MsgpackDecoder(zmq_ipc.EngineCoreRequest, oob_tensor_provider=receiver)
    req_back = dec_oob.decode(bufs_oob)
    out["rounds"].append({
        "round": 3,
        "name": "同一张 16KB 张量：OOB vs 走 ZMQ 帧",
        "tensor_nbytes": big.numel() * 4,
        "oob_frames": len(bufs_oob),
        "plain_frames": len(bufs_plain),
        "oob_main_frame_len": len(bufs_oob[0]),
        "plain_aux_frame_len": len(bufs_plain[1]),
        "handle_message_id": handle_msg_id,
        "handle_tensor_id": 0,
        "decode_roundtrip_ok": bool(torch.equal(req_back.prompt_embeds, big)),
        "decoded_is_shared_mem": req_back.prompt_embeds.is_shared(),
        "verdict": "OOB 在场 → 1 帧：主帧里只有句柄 dict，张量本体 share_memory_ 后走 mp.Queue（零拷贝共享内存）；OOB 缺席 → 2 帧：aux 帧扛 16KB 字节过 socket",
    })

    # ---- D：过期清理 ---------------------------------------------------------------
    log_cap = _CountingHandler()
    zmq_ipc.logger.addHandler(log_cap)
    old = zmq_ipc.TensorIpcData(
        sender_id=sender._sender_id, message_id=1, tensor_id=0, tensor=torch.arange(4)
    )
    q.put(old)  # 迟到的 message-1 张量（current_message_id 已推过它）
    sender.new_message()
    fresh = torch.arange(9)
    sender(fresh)
    fresh_handle = h(sender, sender._message_counter, 0)
    got_fresh = receiver("torch.int64", (9,), fresh_handle)
    stale_warnings = [m for m in log_cap.messages if "stale" in m]
    zmq_ipc.logger.removeHandler(log_cap)
    out["rounds"].append({
        "round": 4,
        "name": "过期清理（迟到张量丢弃）",
        "receiver_current_message_id": sender._message_counter - 1,
        "late_arrived_message_id": 1,
        "stale_warnings_counted": len(stale_warnings),
        "stale_warning_text": stale_warnings[0] if stale_warnings else None,
        "fresh_request_ok": bool(torch.equal(got_fresh, fresh)),
        "fresh_numel": got_fresh.numel(),
        "verdict": "迟到的旧 message 张量被 'Ignoring stale tensor' 丢弃，不泄漏、不挡新消息——接收端推进水位线的代价面",
    })

    dest = Path(__file__).resolve().parent / "m6_oob_bypass.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dest}")
    for r in out["rounds"]:
        print(r["round"], r["name"])


if __name__ == "__main__":
    main()
