"""Driver: position-weighted loss (Eq.4, w_k = exp(-(k-1)/gamma)) and the
"early error costs more" motivation. block_size=4, gamma=4. Dumps every
number used in explainer.json.
"""
from __future__ import annotations
import json
import torch

from position_weighted_loss import position_weights, position_weighted_cross_entropy

block_size = 4
gamma = 4

w = position_weights(block_size, gamma)
weights = [round(float(x), 4) for x in w]   # [1.0, 0.7788, 0.6065, 0.4724]

# Early-vs-late error: a block where the target tokens are all class 0. The
# draft is confident-correct everywhere EXCEPT one position, where it puts all
# mass on the wrong class. Compare weighted loss when that one wrong position
# is the FIRST (k=1) vs the LAST (k=4).
vocab = 5
def logits_with_wrong_at(pos):
    lg = torch.full((block_size, vocab), -5.0)
    for i in range(block_size):
        lg[i, 0] = 5.0            # correct = class 0, high logit
    lg[pos, 0] = -5.0             # wrong at `pos`
    lg[pos, 1] = 5.0
    return lg

targets = torch.zeros(block_size, dtype=torch.long)
loss_early = round(float(position_weighted_cross_entropy(logits_with_wrong_at(0), targets, gamma)), 4)
loss_late = round(float(position_weighted_cross_entropy(logits_with_wrong_at(3), targets, gamma)), 4)

out = {
    "params": {"block_size": block_size, "gamma": gamma, "vocab": vocab},
    "weights_k1_to_k4": weights,
    "early_error_weighted_loss_pos1": loss_early,
    "late_error_weighted_loss_pos4": loss_late,
    "early_over_late_ratio": round(loss_early / loss_late, 4),
    "block_position_indices_k": [1, 2, 3, 4],  # k labels used in the teaching table
}
print(json.dumps(out, indent=2))
with open("position_loss.json", "w") as f:
    json.dump(out, f, indent=2)
