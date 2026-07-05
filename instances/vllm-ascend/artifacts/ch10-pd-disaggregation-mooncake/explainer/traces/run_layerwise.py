#!/usr/bin/env python3
"""ch10 layerwise-push worked example driver.

两部分,都确定性、可复现:

Part A —— **跑真实精简版控制流**:实例化 reduced `KVCacheSendingLayerThread`
(真的 `queue.Queue` 串行 FIFO 消费者),按 `save_kv_layer` 的做法「每层一 put」
入队 L 个 SendTask,再按 `run()` 的方式串行排空,证明:
  L 层 -> L 个 SendTask,按入队序 [0..L-1] 逐个流过真实 `_handle_request`,
  current_layer 从 0 递增到 L(fire-and-forget,不等整段算完)。
只 monkeypatch `_transfer_kv_cache`(host 无 NPU/mooncake engine——即 impl-notes 已
声明的边界),`_handle_request` 与 `queue.Queue` 均为真实精简版代码。

Part B —— **确定性流水线时序模型**:单串行发送线程(Part A 已证 FIFO),
第 i 层的传输只能在「第 i 层算完」且「第 i-1 层传输完」之后开始。取一组小到能心算
的参数(L=4, 每层 compute=10, 每层 transfer=6),逐层排出 compute/transfer 时间窗,
数出被后续层计算盖住的传输数,落成 (L-1)/L 的隐藏比例;再把 L=80 代入对照。

用法: python3 run_layerwise.py   (输出即 traces/layerwise.json 的原始来源)
"""
import json
import sys
import threading
from pathlib import Path

IMPL = Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))

from mooncake_layerwise_connector import KVCacheSendingLayerThread, SendTask  # noqa: E402


def part_a_run_real_send_queue(L: int):
    """跑真实 reduced 精简版的串行发送队列,观测逐层 fire-and-forget 入队/消费。"""
    t = KVCacheSendingLayerThread(
        engine=None, kv_cache_specs={}, layer_metadata={},
        total_layers=L, ready_event=threading.Event(),
    )
    processed = []
    # 只顶替 host 不可跑的 NPU/engine 搬运;_handle_request 与 send_queue 仍是真实精简版代码。
    t._transfer_kv_cache = lambda task: processed.append(task.layer_idx)

    enqueued = []
    current_layer = 0                       # 对应 reduced save_kv_layer 里的 self.current_layer
    for i in range(L):                      # 模型 forward 每算完一层回调一次 save_kv_layer
        t.send_queue.put(SendTask(layer_idx=i, layer_name=f"model.layers.{i}"))
        enqueued.append(i)
        current_layer += 1                  # self.current_layer += 1(逐层递增)

    # 按真实 run() 的方式串行排空队列(while: get -> _handle_request)
    while not t.send_queue.empty():
        task = t.send_queue.get()
        t._handle_request(task)             # 真实精简版方法

    return {
        "L": L,
        "enqueued_layers": enqueued,
        "processed_layers": processed,
        "num_send_tasks": len(processed),
        "current_layer_final": current_layer,
        "serial_fifo": enqueued == processed,   # send_queue 串行 FIFO,入队序=处理序
    }


def part_b_pipeline_timing(L: int, t_c: int, t_x: int):
    """确定性时序:单串行发送线程,transfer(i) 在 compute(i) 完成且 transfer(i-1) 完成后开始。"""
    rows = []
    prev_tx_end = 0
    hidden = 0
    exposed = 0
    for i in range(L):
        c_start = i * t_c
        c_end = c_start + t_c                       # 第 i 层算完
        tx_start = max(c_end, prev_tx_end)          # 串行发送线程:排在上一层传输之后
        tx_end = tx_start + t_x
        prev_tx_end = tx_end
        # 第 i 层的传输是否被「第 i+1 层的计算窗 [c_end, c_end+t_c]」盖住?
        next_compute_end = c_end + t_c if i + 1 < L else None
        is_hidden = next_compute_end is not None and tx_end <= next_compute_end
        if is_hidden:
            hidden += 1
            verdict = "hidden"
        else:
            exposed += 1
            verdict = "exposed"
        rows.append({
            "layer": i,
            "compute_window": [c_start, c_end],
            "transfer_window": [tx_start, tx_end],
            "next_layer": (i + 1) if i + 1 < L else None,
            "verdict": verdict,
        })

    seq_total = L * t_c + L * t_x                    # 无重叠:先全算完再全传完
    pipe_total = L * t_c + t_x                       # 重叠:算完全部 + 最后一层传输尾巴
    saved = seq_total - pipe_total                   # = (L-1)*t_x
    frac_hidden_num = L - 1
    frac_hidden = round(frac_hidden_num / L, 6)      # (L-1)/L
    return {
        "L": L, "t_c": t_c, "t_x": t_x,
        "rows": rows,
        "hidden_transfers": hidden,
        "exposed_transfers": exposed,
        "seq_total": seq_total,
        "pipe_total": pipe_total,
        "saved": saved,
        "frac_hidden_numer": frac_hidden_num,
        "frac_hidden": frac_hidden,
    }


def frac_for(L: int):
    return {"L": L, "numer": L - 1, "frac_hidden": round((L - 1) / L, 6)}


def main():
    out = {}
    out["part_a_real_send_queue"] = part_a_run_real_send_queue(L=4)
    out["part_b_timing_L4"] = part_b_pipeline_timing(L=4, t_c=10, t_x=6)
    # 量化对照:小例 L=4 vs 真实规模 L=80(Llama-70B 级层数,示教值,非源码常量)
    out["frac_hidden_by_L"] = [frac_for(4), frac_for(80)]
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
