"""Driver: draft-head training objective L = L_reg + w_cls * L_cls (w_cls=0.1).

EAGLE arXiv:2401.15077 §3.2: the Autoregression Head is trained with a
Smooth-L1 feature-regression loss L_reg plus a soft cross-entropy
classification loss L_cls between the target next-token distribution p and the
draft's predicted distribution p_hat, combined with a fixed weight w_cls=0.1
(chosen because "the classification loss is an order of magnitude larger than
the regression loss in numerical terms").

Two scenarios on toy tensors: (A) an ACCURATE draft feature close to the true
next feature -> small L_reg and L_cls; (B) an INACCURATE draft feature far off
-> both losses larger. Shows both terms, the fixed 0.1 weighting, and that
L_cls indeed dominates numerically (motivating w_cls<1). Training is NOT in
vLLM's inference path -- this is the paper's background objective, run purely
to make the numbers concrete.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from feature_autoregression import combined_loss  # noqa: E402

R = 3


def rnd(x):
    if torch.is_tensor(x):
        x = x.detach()
    return round(float(x), R)


torch.manual_seed(0)
HID, VOCAB = 4, 6
# Ground-truth next feature and a toy LM-head to turn features into logits.
target_feature = torch.tensor([1.0, -0.5, 0.5, 0.25])
lm_head = torch.nn.Linear(HID, VOCAB, bias=False)
with torch.no_grad():
    lm_head.weight.copy_(torch.arange(HID * VOCAB, dtype=torch.float32).reshape(VOCAB, HID) * 0.1 - 1.0)

target_logits = lm_head(target_feature)

scenarios = []
preds = {
    "accurate": target_feature + torch.tensor([0.05, -0.05, 0.05, 0.0]),
    "inaccurate": target_feature + torch.tensor([1.2, -0.9, 0.8, -0.7]),
}
for name, pred_feature in preds.items():
    pred_logits = lm_head(pred_feature)
    L, L_reg, L_cls = combined_loss(pred_feature, target_feature, pred_logits, target_logits, w_cls=0.1)
    scenarios.append({
        "scenario": name,
        "L_reg": rnd(L_reg),
        "L_cls": rnd(L_cls),
        "w_cls": 0.1,
        "w_cls_times_L_cls": rnd(0.1 * float(L_cls)),
        "L_total": rnd(L),
    })

out = {
    "params": {"hidden_dim": HID, "vocab_size": VOCAB, "w_cls": 0.1},
    "target_feature": target_feature.tolist(),
    "scenarios": scenarios,
}
print(json.dumps(out, indent=2))
Path(__file__).with_name("training_loss.json").write_text(json.dumps(out, indent=2))
