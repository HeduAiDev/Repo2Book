# vllm_ascend/core/profiling_chunk_predictor.py —— subtract-only 精简版
#
# 二次延迟模型 f(l)=a·l²+b·l+c：拟合 profiling 采样，给定目标延迟 T 解增量 chunk x
# 使 f(L+x)−f(L)=T，即 a·x²+(2aL+b)·x−T=0 取正根，再平滑/对齐到 page。
#
# SUBTRACTED: history-aware 旁路（fit_chunk / predict_with_history /
#   get_time_with_history / record_batch_execution_time + with_history_ready 状态机）
#   默认 with_history_ready=False、调度路径走不到（见 predict_chunk_size 派发）；属需运行期
#   record_batch_execution_time 喂数据才启用的实验特性。原 profiling_chunk_predictor.py:
#   L130-L177,L210-L220,L260-L301,L365-L385。
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. (Apache-2.0)
import math

import numpy as np
from vllm.logger import logger


# SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L36
class ChunkSizePredictor:
    """Predictor for dynamic chunk size based on quadratic latency model.

    Models latency as: f(l) = a*l^2 + b*l + c.  Given a target latency T and
    current history length L, predicts next chunk size x such that
    f(L+x) - f(L) = T, i.e. the quadratic a*x^2 + (2aL+b)*x - T = 0.
    """

    # SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L47
    def __init__(self, smooth_factor: float = 0.8, min_chunk: int = 4096):
        self.quadratic_coeff_a: float = 0.0
        self.linear_coeff_b: float = 0.0
        self.constant_coeff_c: float = 0.0
        self.target_latency: float | None = None
        self.is_ready: bool = False
        self.with_history_ready: bool = False
        self.smooth_factor = smooth_factor
        self.min_chunk = min_chunk

    # SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L63
    def clamp_quadratic_and_linear_if_negative(self, fitted_a: float, fitted_b: float) -> tuple[float, float]:
        """For the Transformer structure of LLM, the fitted quadratic and linear
        terms should not be negative; zero-clamp inaccurate fits."""
        if fitted_a < 0:
            logger.warning("Fitted a=%.2e is not positive. Setting a=1e-9.", fitted_a)
            fitted_a = 1e-9
        if fitted_b < 0:
            logger.warning("Fitted b=%.2e is not positive. Setting b=0.0.", fitted_b)
            fitted_b = 1e-9
        return fitted_a, fitted_b

    # SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L76
    def fit(self, seq_lens: list[int], latencies: list[float]) -> bool:
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points."""
        L = np.array(seq_lens, dtype=np.float64)
        T = np.array(latencies, dtype=np.float64)
        MIN_FIT_POINTS_NO_CHUNK = 8

        if len(L) < MIN_FIT_POINTS_NO_CHUNK:
            logger.warning("Not enough data points for quadratic fitting (%d < 8)", len(L))
            return False

        X = np.column_stack([L * L, L, np.ones_like(L)])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, T, rcond=None)
            fitted_a = float(coeffs[0])
            fitted_b = float(coeffs[1])
            fitted_c = float(coeffs[2])
        except Exception as e:
            # Robust fallback for backends where least-squares may fail.
            try:
                poly = np.polyfit(L, T, 2)
                fitted_a = float(poly[0])
                fitted_b = float(poly[1])
                fitted_c = float(poly[2])
                logger.warning("Least-squares fitting failed (%s), fallback to polyfit succeeded.", e)
            except Exception as fallback_error:
                logger.warning("Failed to fit quadratic model: %s", fallback_error)
                return False

        fitted_a, fitted_b = self.clamp_quadratic_and_linear_if_negative(fitted_a, fitted_b)

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info("[ProfilingChunk] Fitted: a=%.2e, b=%.2e, c=%.2e", fitted_a, fitted_b, fitted_c)
        return True

    # SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L179
    def set_target_latency(self, base_chunk_size: int, elapsed_time: float = 0.0) -> None:
        """Set target latency based on base chunk size."""

        # SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L182
        def f(seq_lens: float) -> float:
            return self.quadratic_coeff_a * seq_lens * seq_lens + self.linear_coeff_b * seq_lens + self.constant_coeff_c

        if elapsed_time > 0:
            self.target_latency = elapsed_time
        else:
            self.target_latency = f(float(base_chunk_size)) - f(0.0)
        if self.target_latency <= 0:
            self.target_latency = 1.0

        logger.info(
            "[ProfilingChunk] Target latency: %.2f ms (base_chunk=%d)", self.target_latency, base_chunk_size
        )

    # SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L198
    def get_time(
        self,
        query_len: int,
        num_computed_tokens: int,
    ) -> float:
        """Get time T based on current seq_lens, f(l) = al^2 + bl + c, f(L+x) - f(L) = T."""

        # SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L205
        def f(seq_lens: float) -> float:
            return self.quadratic_coeff_a * seq_lens * seq_lens + self.linear_coeff_b * seq_lens + self.constant_coeff_c

        return f(query_len + num_computed_tokens) - f(num_computed_tokens)

    # SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L222
    def predict(
        self,
        num_computed_tokens: int,
        base_chunk_size: int,
        page_size: int,
    ) -> int | None:
        """Predict next chunk size x such that f(L+x) - f(L) = target_latency."""
        if not self.is_ready or self.target_latency is None:
            return None

        if self.quadratic_coeff_a <= 0:
            return None

        A = self.quadratic_coeff_a
        B = 2 * self.quadratic_coeff_a * num_computed_tokens + self.linear_coeff_b
        C = -self.target_latency

        discriminant = B * B - 4 * A * C
        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        x = (-B + sqrt_disc) / (2 * A)

        if x <= 0:
            return None

        smoothed = base_chunk_size + self.smooth_factor * (x - base_chunk_size)
        chunk_size = max(int(smoothed), self.min_chunk)

        align = max(page_size, 64)
        chunk_size = ((chunk_size + align - 1) // align) * align
        if chunk_size < align:
            chunk_size = align

        logger.debug("[ProfilingChunk] Predicted chunk_size=%d", chunk_size)
        return chunk_size if chunk_size >= align else None


# SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L304
class ProfilingChunkManager:
    """Manager for profiling-based dynamic chunk sizing.

    Handles the profiling process and maintains the ChunkSizePredictor.
    """

    # SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L310
    def __init__(
        self,
        base_chunk_size: int,
        page_size: int,
        smooth_factor: float = 0.8,
        min_chunk: int = 4096,
    ):
        self.base_chunk_size = base_chunk_size
        self.page_size = page_size
        self.predictor = ChunkSizePredictor(smooth_factor=smooth_factor, min_chunk=min_chunk)
        self._profiling_done = False

    @property
    # SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L326
    def is_ready(self) -> bool:
        return self._profiling_done and self.predictor.is_ready

    # SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L334
    def predict_chunk_size(self, num_computed_tokens: int, target_time: float) -> int | None:
        """Predict optimal chunk size for given history length."""
        if not self.is_ready:
            return None

        # NOTE(gjc): We found that the FIA operator has abnormal performance when
        # processing multiple request groups in a batch, so the target_latency
        # feature is temporarily fixed. It will be enabled again after the FIA
        # operator issues are resolved.
        # self.predictor.target_latency = target_time

        # SUBTRACTED: history_ready 派发到 predict_with_history（默认 False，走不到）。
        #   原 profiling_chunk_predictor.py:L346-L349
        return self.predictor.predict(
            num_computed_tokens=num_computed_tokens, base_chunk_size=self.base_chunk_size, page_size=self.page_size
        )

    # SOURCE: vllm_ascend/core/profiling_chunk_predictor.py:L354
    def predict_time(self, num_new_tokens: int, num_computed_tokens: int) -> float:
        """Get the consumed time of scheduled reqs for time_budget."""
        if not self.is_ready:
            return 0.0

        # SUBTRACTED: history_ready 派发到 get_time_with_history（默认 False，走不到）。
        #   原 profiling_chunk_predictor.py:L359-L362
        return self.predictor.get_time(query_len=num_new_tokens, num_computed_tokens=num_computed_tokens)
