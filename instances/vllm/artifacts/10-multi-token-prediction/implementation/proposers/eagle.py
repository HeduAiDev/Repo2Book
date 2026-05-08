"""EagleProposer — pure inheritance from SpecDecodeBaseProposer.

# REFERENCE: vllm/v1/spec_decode/eagle.py:L1-L22 (the entire file)

The actual file is 22 lines. The interesting thing is what it DOESN'T do:
no propose() override, no _greedy_sample() override, no special init logic.
EAGLE's algorithm IS the base class — what makes EAGLE distinctive is its
architecture (the `Eagle3LlamaForCausalLM` model in `llama_eagle3.py`),
not its proposer logic.

The signal here: in vLLM, the spec-decode "method" name (eagle, eagle3,
mtp, dflash, ...) selects a model class. The proposer wraps the model;
the model class encodes the architectural choice (fc fusion vs eh_proj
fusion vs Medusa heads, etc.).
"""
# REFERENCE: vllm/v1/spec_decode/eagle.py:L1-L22
from .base import SpecDecodeBaseProposer


class EagleProposer(SpecDecodeBaseProposer):
    """EAGLE proposer.

    # REFERENCE: vllm/v1/spec_decode/eagle.py:L10-L22

    pass_hidden_states_to_model=True is the only configuration EAGLE specifies:
    EAGLE's architecture takes the target's hidden state as input, fused with
    the draft's previous embedding via an `fc` projection (see Eagle3 model).
    """

    def __init__(self, num_speculative_tokens: int, hidden_size: int):
        super().__init__(
            num_speculative_tokens=num_speculative_tokens,
            hidden_size=hidden_size,
            # REFERENCE: vllm/v1/spec_decode/eagle.py:L20
            pass_hidden_states_to_model=True,
        )
