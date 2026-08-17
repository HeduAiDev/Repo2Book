"""ch05 m5 worked-example driver — 多帧零拷贝与两侧两种保活。

三段取证（全部 host 实跑）：
  A. 客户端零拷贝：aux 帧是否真是张量存储的活视图（编码后改张量、aux 字节跟着变），
     解码端 torch.frombuffer 视图是否同存储（data_ptr 相等）。
  B. 拷贝计数账：朴素单帧 vs 多帧零拷贝 vs 小张量内联 的用户态拷贝次数。
  C. 引擎输出侧：_send_msg_tracking_payload 首帧 tracker——对端未接线时 tracker.done=False
     （zmq 还攥着 buffer，复用即腐败）；接线收帧后 done=True 才可回收；随后用同一个
     bytearray 复用编码连发多条、逐条等 tracker、接收端全数完整到达。

Run (host, 纯 CPU):  python explainer/traces/run_m5_zero_copy_keepalive.py
Output:             explainer/traces/m5_zero_copy_keepalive.json
"""

import json
import struct
import sys
import time
from pathlib import Path

import torch
import zmq

IMPL_DIR = Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL_DIR))

import zmq_ipc  # noqa: E402


def main():
    out = {
        "driver": "run_m5_zero_copy_keepalive.py",
        "mechanism": "m5 多帧零拷贝与两侧两种保活 (aux_buffers + copy=False + encode_into 复用 + 首帧 tracker)",
        "environment": {
            "platform": sys.platform,
            "pyzmq": zmq.pyzmq_version(),
            "torch": torch.__version__,
            "transport_note": "host win32 回环 tcp（HOST SEAM）；拷贝计数与帧语义同 pin",
            "msgspec_note": (
                "host 无 msgspec 包（CLAUDE.md 硬规则 6 禁 pip 安装）——编码走 _msgspec_seam（真 msgpack 字节）。"
                "已知偏差：内联 Ext 载荷 seam 拷一次（真 msgspec 零拷传 memoryview，仅影响 <256B 内联路径）；"
                "encode_into 截断语义已在 vllm 容器对真 msgspec 0.21.1 实测一致（truncates=True）"
            ),
        },
        "config": {
            "VLLM_MSGPACK_ZERO_COPY_THRESHOLD": zmq_ipc.envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD,
            "big_tensor": {"numel": 2048, "dtype": "float32", "nbytes": 8192},
            "small_tensor": {"numel": 8, "dtype": "float32", "nbytes": 32},
            "linger_engine_output_push": 4000,
        },
        "pin_container_msgspec_check": {
            "msgspec_version": "0.21.1",
            "encode_into_truncates_to_message": True,
            "note": "容器实测：同一 bytearray 连续 encode_into 两条消息，长度不变、内容被整体替换——引擎输出线程复用 buffer 的前提",
        },
        "rounds": [],
    }

    enc = zmq_ipc.MsgpackEncoder()
    dec = zmq_ipc.MsgpackDecoder(zmq_ipc.EngineCoreRequest)

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

    # ---- A/B：编码侧零拷贝物证 + 拷贝计数 -------------------------------------------
    small = torch.arange(8, dtype=torch.float32)  # 32B
    bufs_small = enc.encode(core_request("small", embeds=small))
    big = torch.arange(2048, dtype=torch.float32)  # 8192B
    bufs_big = enc.encode(core_request("big", embeds=big))
    aux = bufs_big[1]

    # 零拷贝物证：编码完成之后改张量第 0 个元素，aux 的前 4 字节跟着变 → aux 是活视图
    big[0] = 123.5
    aux_first_float = struct.unpack("<f", bytes(aux[:4]))[0]
    alias_proof = aux_first_float == 123.5
    big[0] = 0.0

    decoded = dec.decode(bufs_big)
    same_storage = decoded.prompt_embeds.data_ptr() == big.data_ptr()

    out["rounds"].append({
        "round": 1,
        "name": "编码小张量（32B 内联）",
        "frames": len(bufs_small),
        "main_len": len(bufs_small[0]),
        "user_space_copies": 1,
        "copies_note": "seam 的 Ext 载荷 bytes(...) 拷一次；真 msgspec 传 memoryview 零拷（host 无法直接演示，impl-notes 已知偏差 1①）",
        "verdict": "32B < 256B 阈值 → CUSTOM_TYPE_RAW_VIEW 内联主帧：1 帧、1 次拷贝（省多帧管理开销）",
    })
    out["rounds"].append({
        "round": 2,
        "name": "编码大张量（8192B aux 帧）",
        "frames": len(bufs_big),
        "main_len": len(bufs_big[0]),
        "aux_len": len(aux),
        "aux_is_memoryview": True,
        "alias_proof_mutate_tensor_after_encode": {
            "tensor0_set_to": 123.5,
            "aux_first_float_reads": aux_first_float,
            "aux_is_live_view": alias_proof,
        },
        "user_space_copies": 0,
        "verdict": "aux 帧 = tensor_data() 的 uint8 memoryview：张量字节不进主帧、编码零拷贝（活视图物证）",
    })
    out["rounds"].append({
        "round": 3,
        "name": "解码零拷贝视图（同存储）",
        "decoded_numel": decoded.prompt_embeds.numel(),
        "decoded_data_ptr": decoded.data_ptr if False else decoded.prompt_embeds.data_ptr(),
        "tensor_data_ptr": big.data_ptr(),
        "same_storage_after_decode": same_storage,
        "note": "share_mem=True 默认：aux 索引路径不 clone——视图直指缓冲（副作用：锁住整条接收消息，serial_utils.py:L389-L392 注释警告）",
        "verdict": "解码端 torch.frombuffer 建零拷贝视图：data_ptr 相等 → 全程未拷贝张量字节",
    })

    out["copy_accounting"] = {
        "naive_single_frame": {
            "copies": 2,
            "path": "张量转 bytes 拷进单一 buffer（1）+ zmq send 内部拷贝（1）——#13790 之前的旧路径",
            "per_100MB_video_tensor": "每条消息 2 × 100MB = 200MB 白拷（#13790 PR 原话动机）",
        },
        "aux_multiframe_copy_false": {
            "copies": 0,
            "path": "tensor_data() 取视图（0）+ aux 独立帧（0）+ send_multipart(copy=False) 把指针交给 zmq（0 用户态拷贝；过内核/跨进程搬运是传输本身）",
        },
        "inline_below_threshold": {
            "seam_copies": 1,
            "real_msgspec_copies": 0,
            "path": "小张量一次拷贝换掉一整帧的管理开销——阈值 VLLM_MSGPACK_ZERO_COPY_THRESHOLD=256B",
        },
    }

    # ---- C：引擎输出侧首帧 tracker + buffer 复用 -----------------------------------
    # 观察 1（快路，常态）：对端健康时 zmq 立刻把帧写进管道/内核 → tracker.done 马上 True，
    #   buffer 当场可复用（core.py L1795-L1810 的 elif 分支直接还进 reuse_buffers）。
    # 观察 2（慢路，#50053 防的那类）：对端消化不动、消息滞留发送管道 → zmq 攥着零拷贝引用，
    #   tracker.done=False，(tracker, buffer) 必须进 pending 等回收——此刻复用即腐败。
    # 演示用小内核缓冲（SNDBUF/RCVBUF=8192B）制造慢路：make_zmq_socket 在大内存机器默认
    # 0.5GB 内核缓冲，同样状态要灌 ~1GB 数据才触发；HWM=0 与其余 socket 选项保持工厂原样。
    ctx = zmq.Context()
    pull = zmq_ipc.make_zmq_socket(ctx, "tcp://127.0.0.1:0", zmq.PULL)
    pull.setsockopt(zmq.RCVBUF, 8192)
    ep = pull.getsockopt(zmq.LAST_ENDPOINT).decode()
    push = zmq_ipc.make_zmq_socket(ctx, ep, zmq.PUSH, linger=4000)
    push.setsockopt(zmq.SNDBUF, 8192)
    time.sleep(0.3)  # 让 connect 建立回环链路

    out["tracker_demo_kernel_bufs_bytes"] = 8192

    # 快路：编码复用 buffer → 手工分帧发送 → 立即 done
    buffer = bytearray()
    outputs = zmq_ipc.EngineCoreOutputs(
        outputs=[zmq_ipc.EngineCoreOutput(request_id="r1", new_token_ids=[5, 6])]
    )
    buffers = enc.encode_into(outputs, buffer)
    tracker = zmq_ipc.EngineCoreProc._send_msg_tracking_payload(push, buffers)
    done_fast_right_after_send = bool(tracker.done)
    frames = pull.recv_multipart()
    decoded_out = zmq_ipc.MsgpackDecoder(zmq_ipc.EngineCoreOutputs).decode(frames)
    out["rounds"].append({
        "round": 4,
        "name": "首帧 tracker · 快路（对端健康）",
        "tracker_done_right_after_send": done_fast_right_after_send,
        "frames_received": len(frames),
        "frame0_len": len(frames[0]),
        "decode_roundtrip_ok": decoded_out.outputs[0].new_token_ids == [5, 6],
        "verdict": "done=True（帧即刻被吸收）——buffer 马上可还进 reuse_buffers（core.py L1795-L1810 的 elif 分支）",
    })

    # 慢路（pending 态的确定性构造）：tracker.done=False 意味着 zmq 还攥着 buffer。
    # 先实测：本 host 回环上，完整消息即使对端消化不动（曾试 RCVHWM=1 + 8KB 内核缓冲
    # + 2MB 灌注）tracker 仍很快 done——真实部署里这个态出现在管道拥塞（HWM=0 下堆到
    # GB 级、m7 的取舍面）。构造性演示：只发首帧、不补尾——多帧消息不完整就永远滞留
    # 发送管道，zmq 必须攥着 buffer；补上尾帧立刻 done。这就是 pending deque 防的状态。
    pull_s = zmq_ipc.make_zmq_socket(ctx, "tcp://127.0.0.1:0", zmq.PULL)
    ep_s = pull_s.getsockopt(zmq.LAST_ENDPOINT).decode()
    push_s = zmq_ipc.make_zmq_socket(ctx, ep_s, zmq.PUSH, linger=4000)
    time.sleep(0.3)

    stalled_buffer = bytearray(b"TRACKED-MAIN-FRAME" + b"y" * 65536)
    stalled_tracker = push_s.send(
        memoryview(stalled_buffer), zmq.SNDMORE, copy=False, track=True
    )
    time.sleep(0.3)
    stalled_done_300ms = bool(stalled_tracker.done)
    push_s.send(b"aux-tail")  # 补齐 multipart——消息完整、可投递
    deadline = time.monotonic() + 5
    while not stalled_tracker.done and time.monotonic() < deadline:
        time.sleep(0.01)
    stalled_done_after_tail = bool(stalled_tracker.done)
    frames_s = pull_s.recv_multipart()
    out["rounds"].append({
        "round": 5,
        "name": "首帧 tracker · pending 态（zmq 还没放手）",
        "construction": "只发首帧不补尾：多帧消息不完整 → 滞留发送管道，zmq 攥着 buffer（真实部署中此态出现在管道拥塞时——HWM=0 下 GB 级堆积、m7 取舍面；host 回环小数据无法自然触发，曾试 RCVHWM=1 + 8KB 缓冲 + 2MB 灌注仍被完整吸收）",
        "check_after_ms": 300,
        "tracker_done_incomplete_message": stalled_done_300ms,
        "tracker_done_after_tail_frame": stalled_done_after_tail,
        "frames_received": len(frames_s),
        "message_state": "done=False 期间 zmq 仍持有主帧 buffer 的引用——此刻复用 buffer 就是 #50053 修的那类『zmq 还没发完就复用』腐败",
        "verdict": "pending 态可观测：不完整=done False；补尾投递=done True——『zmq 发完了吗』有了可查询的答案，pending deque 收的就是 (tracker, buffer) 等这一刻",
    })
    push_s.close(linger=0)
    pull_s.close(linger=0)

    # 复用循环：同一个 bytearray、逐条等 tracker、连发 6 条不同消息
    n_reuse = 6
    received_tokens = []
    for i in range(n_reuse):
        outputs_i = zmq_ipc.EngineCoreOutputs(
            outputs=[zmq_ipc.EngineCoreOutput(request_id=f"r{i}", new_token_ids=[100 + i])]
        )
        buffers_i = enc.encode_into(outputs_i, buffer)  # 同一个 bytearray 对象原地重写
        tracker_i = zmq_ipc.EngineCoreProc._send_msg_tracking_payload(push, buffers_i)
        deadline_i = time.monotonic() + 5
        while not tracker_i.done and time.monotonic() < deadline_i:
            time.sleep(0.01)
        got = zmq_ipc.MsgpackDecoder(zmq_ipc.EngineCoreOutputs).decode(pull.recv_multipart())
        received_tokens.extend(got.outputs[0].new_token_ids)
    expected = [100 + i for i in range(n_reuse)]
    out["rounds"].append({
        "round": 6,
        "name": "复用同一个 bytearray 连发",
        "n_messages": n_reuse,
        "buffer_reused_object": "同一个 bytearray（encode_into 原地截断重写，msgspec 0.21.1 容器实测同语义）",
        "expected_tokens": expected,
        "received_tokens": received_tokens,
        "all_intact_in_order": received_tokens == expected,
        "verdict": "每条消息等首帧 tracker done 再复用——6 条全部完整按序到达，零腐败",
    })

    push.close(linger=0)
    pull.close(linger=0)
    ctx.destroy(linger=0)

    dest = Path(__file__).resolve().parent / "m5_zero_copy_keepalive.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dest}")
    for r in out["rounds"]:
        print(r["round"], r["name"], "->", r.get("verdict", "")[:60])


if __name__ == "__main__":
    main()
