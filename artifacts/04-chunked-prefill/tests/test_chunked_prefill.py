"""Tests — Chunked Prefill chapter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from implementation.chunked_prefill import (
    ChunkedPrefillConfig, SimRequest,
    ChunkedPrefillScheduler, ttft_vs_throughput_analysis,
)


class TestSimRequest:
    def test_num_new_tokens(self):
        req = SimRequest("r1", prompt_tokens=100, max_output_tokens=50)
        assert req.num_new_tokens == 100
        req.num_computed_tokens = 60
        assert req.num_new_tokens == 40

    def test_is_prefilling(self):
        req = SimRequest("r1", prompt_tokens=100, max_output_tokens=50)
        assert req.is_prefilling
        req.num_computed_tokens = 100
        assert not req.is_prefilling


class TestChunkedPrefillScheduler:
    def test_long_prompt_is_chunked(self):
        config = ChunkedPrefillConfig(
            enable_chunked_prefill=True,
            long_prefill_token_threshold=512,
            max_num_scheduled_tokens=1024,
        )
        sched = ChunkedPrefillScheduler(config)
        sched.add_request(SimRequest("r1", prompt_tokens=2000, max_output_tokens=50))

        # Step 1: should only get threshold (512), not full 2000
        out = sched.schedule()
        assert out["r1"] == 512
        sched.update_after_step(out)

        # Step 2: another 512 chunk
        out = sched.schedule()
        assert out["r1"] == 512

    def test_without_chunked_prefill_blocks_large_prompt(self):
        config = ChunkedPrefillConfig(
            enable_chunked_prefill=False,
            long_prefill_token_threshold=0,
            max_num_scheduled_tokens=512,
        )
        sched = ChunkedPrefillScheduler(config)
        sched.add_request(SimRequest("r1", prompt_tokens=2000, max_output_tokens=50))
        sched.add_request(SimRequest("r2", prompt_tokens=100, max_output_tokens=50))

        out = sched.schedule()
        # r1 can't fit (2000 > 512 budget) → should NOT be admitted
        assert "r1" not in out

    def test_multiple_requests_interleaved(self):
        config = ChunkedPrefillConfig(
            enable_chunked_prefill=True,
            long_prefill_token_threshold=512,
            max_num_scheduled_tokens=1024,
        )
        sched = ChunkedPrefillScheduler(config)
        sched.add_request(SimRequest("long", prompt_tokens=2000, max_output_tokens=20))
        sched.add_request(SimRequest("short", prompt_tokens=100, max_output_tokens=20))

        # Step 1: long gets 512 (capped), short gets 100
        out = sched.schedule()
        assert out.get("long", 0) == 512
        assert out.get("short", 0) == 100


class TestTTFTAnalysis:
    def test_chunked_improves_short_ttft(self):
        result = ttft_vs_throughput_analysis(
            long_prompt_len=128000, short_prompt_len=128,
            num_short_requests=8, max_tokens_per_step=2048,
            long_threshold=2048, output_len=256,
        )
        without = result["without_chunked_prefill"]["short_ttft_steps"]
        with_chunk = result["with_chunked_prefill"]["short_ttft_steps"]
        assert with_chunk < without

    def test_long_ttft_similar(self):
        """Long prompt TTFT is similar with/without chunked prefill."""
        result = ttft_vs_throughput_analysis(
            long_prompt_len=128000, short_prompt_len=128,
            num_short_requests=8, max_tokens_per_step=2048,
            long_threshold=2048, output_len=256,
        )
        without = result["without_chunked_prefill"]["long_prefill_steps"]
        with_chunk = result["with_chunked_prefill"]["long_prefill_steps"]
        assert abs(without - with_chunk) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
