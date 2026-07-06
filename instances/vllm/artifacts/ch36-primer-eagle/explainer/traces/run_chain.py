"""Driver: feature-level autoregression + vLLM-style chain drafting.

Runs chain_drafting.propose_chain on the toy ToyTargetLLM/AutoregressionHead
(feature_autoregression.py) and records, per draft step, a scalar summary of
the predicted feature (its L2 norm), the greedy token read off the shared LM
Head, and the draft model's confidence c_j on that token. This is the
T->E->F->p->t pipeline (EAGLE arXiv:2401.15077 §2/§3.1) run feature-first:
each step autoregresses a *feature* through the Autoregression Head, THEN the
shared LM Head turns it into a token -- the paper's core "autoregress at the
feature level, not the token level" claim (§1, Fig.4).
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from chain_drafting import propose_chain  # noqa: E402
from feature_autoregression import AutoregressionHead, ToyTargetLLM  # noqa: E402

torch.manual_seed(0)
R = 3  # rounding


def rnd(x):
    if torch.is_tensor(x):
        x = x.detach()
    return round(float(x), R)


VOCAB, HID = 6, 4
target = ToyTargetLLM(vocab_size=VOCAB, hidden_dim=HID, seed=0)
draft = AutoregressionHead(hidden_dim=HID, seed=2)

prefix_tokens = torch.tensor([1, 3, 2])
next_token = 4
gamma = 4

prefix_feats = target.forward_prefix(prefix_tokens)

# Re-run propose_chain's internals step-by-step so we can log each feature.
from feature_autoregression import build_shifted_token_input  # noqa: E402

shifted = build_shifted_token_input(prefix_tokens, next_token)
embeds = target.embed_tokens(shifted)
fused = draft(embeds, prefix_feats)
feature = fused[-1]

steps = []
draft_tokens, confidences = [], []
tok = int(torch.argmax(target.next_token_distribution(feature)).item())
conf = float(target.next_token_distribution(feature)[tok].item())
draft_tokens.append(tok)
confidences.append(conf)
steps.append({
    "step": 1,
    "input_token": int(shifted[-1].item()),
    "feature_norm": rnd(torch.linalg.vector_norm(feature)),
    "draft_token": tok,
    "confidence": rnd(conf),
})
for s in range(2, gamma + 1):
    te = target.embed_tokens(torch.tensor(draft_tokens[-1]))
    feature = draft(te, feature)
    probs = target.next_token_distribution(feature)
    tok = int(torch.argmax(probs).item())
    conf = float(probs[tok].item())
    draft_tokens.append(tok)
    confidences.append(conf)
    steps.append({
        "step": s,
        "input_token": draft_tokens[-2],
        "feature_norm": rnd(torch.linalg.vector_norm(feature)),
        "draft_token": tok,
        "confidence": rnd(conf),
    })

# Cross-check against the library propose_chain (must match token-for-token).
lib_tokens, lib_conf = propose_chain(
    target, draft, prefix_tokens, prefix_feats, next_token, gamma
)
assert lib_tokens == draft_tokens, (lib_tokens, draft_tokens)

out = {
    "params": {"vocab_size": VOCAB, "hidden_dim": HID,
               "prefix_tokens": prefix_tokens.tolist(),
               "next_token": next_token, "num_speculative_tokens": gamma},
    "shifted_token_input": shifted.tolist(),
    "steps": steps,
    "draft_tokens": draft_tokens,
    "confidences_rounded": [rnd(c) for c in confidences],
}
print(json.dumps(out, indent=2))
Path(__file__).with_name("chain.json").write_text(json.dumps(out, indent=2))
