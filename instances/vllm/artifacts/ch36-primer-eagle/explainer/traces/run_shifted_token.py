"""Driver: feature uncertainty resolved by the "one-step-ahead" token.

EAGLE arXiv:2401.15077 §1/§3.1, Fig.3: the feature after f_I is *not*
determined by f_I alone -- if the target LLM sampled "am" the next feature is
f_am, if it sampled "always" it is f_always. EAGLE feeds the token advanced by
one time step (the sampling outcome) into the draft head so each branch is
individually determined.

We hold ONE feature f_I fixed and vary only the spliced last-slot token
(build_shifted_token_input on a length-1 request reduces to [t_x]); the draft
head's predicted next feature -- and the token the shared LM Head reads off it
-- must differ between branches, showing the shifted token *is* what resolves
the ambiguity. A feature-only draft (no token) would see the same f_I in both
branches and could not tell them apart.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from feature_autoregression import (AutoregressionHead, ToyTargetLLM,  # noqa: E402
                                     build_shifted_token_input)

R = 3


def rnd(x):
    if torch.is_tensor(x):
        x = x.detach()
    return round(float(x), R)


VOCAB, HID = 6, 4
target = ToyTargetLLM(vocab_size=VOCAB, hidden_dim=HID, seed=0)
draft = AutoregressionHead(hidden_dim=HID, seed=2)

# f_I: the feature of token t_I = 1 (a single-position "prefix").
t_I = 1
f_I = target.forward_prefix(torch.tensor([t_I]))  # shape (1, HID)

branches = []
for name, t_ahead in [("branch_A", 0), ("branch_B", 2)]:
    shifted = build_shifted_token_input(torch.tensor([t_I]), t_ahead)  # -> [t_ahead]
    embeds = target.embed_tokens(shifted)
    pred_feature = draft(embeds, f_I)[-1]
    probs = target.next_token_distribution(pred_feature)
    top = int(torch.argmax(probs).item())
    branches.append({
        "branch": name,
        "t_I": t_I,
        "shifted_last_token": int(t_ahead),
        "pred_feature_norm": rnd(torch.linalg.vector_norm(pred_feature)),
        "pred_feature_0": rnd(pred_feature[0]),
        "top_token": top,
        "top_conf": rnd(probs[top]),
    })

# Feature-only contrast: same f_I, no token distinction -> identical.
same_feature_norm = branches[0]["pred_feature_norm"] == branches[1]["pred_feature_norm"]
out = {
    "params": {"vocab_size": VOCAB, "hidden_dim": HID, "t_I": t_I},
    "f_I_norm": rnd(torch.linalg.vector_norm(f_I)),
    "branches": branches,
    "branches_differ": branches[0]["top_token"] != branches[1]["top_token"],
    "feature_norms_equal": same_feature_norm,
}
print(json.dumps(out, indent=2))
Path(__file__).with_name("shifted_token.json").write_text(json.dumps(out, indent=2))
