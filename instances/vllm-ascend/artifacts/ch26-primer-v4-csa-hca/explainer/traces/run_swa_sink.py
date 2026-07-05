"""Driver: sliding-window-attn-sink —— 注意力 sink(Eq.27)让每头总注意力可 < 1;
滑窗支线取最近 n_win 个未压缩 token。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from attention_extras import (attention_sink_scores, sink_absorbed_mass,
                              sliding_window_recent_kv)

logits = np.array([2.0, 1.0, 0.0])   # 某头对 3 个候选的打分

# 两种 sink 强度
sink_rows = []
for label, sink_logit in (("weak", -10.0), ("strong", 3.0)):
    sc = attention_sink_scores(logits, sink_logit)
    sink_rows.append({
        "sink_case": label,
        "sink_logit": round(sink_logit, 1),
        "score_sum": round(float(np.sum(sc)), 3),
        "absorbed_mass": round(sink_absorbed_mass(sc), 3),
    })

# 滑窗支线:序列 10 个 token(每 token 一个 1 维 KV = token 序号)
token_kv = np.arange(10).reshape(10, 1).astype(float)
win_rows = []
for label, pos, n_win in (("mid", 7, 4), ("start", 1, 4)):
    w = sliding_window_recent_kv(token_kv, pos, n_win)
    win_rows.append({
        "window_case": label,
        "query_pos": pos,
        "n_win": n_win,
        "n_entries_taken": int(w.shape[0]),
        "first_token": int(w[0, 0]),
        "last_token": int(w[-1, 0]),
    })

out = {"logits": [2.0, 1.0, 0.0], "sink_rows": sink_rows, "window_rows": win_rows}
print(json.dumps(out, indent=2, ensure_ascii=False))
with open(os.path.join(os.path.dirname(__file__), "swa_sink.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
