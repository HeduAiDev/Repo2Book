# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/tracing.py —— HOST SEAM（最小承载）：本章消费面只有
# GenerateBaseServing._get_trace_headers 的三个符号与 setup_server 的
# @instrument 装饰器；真实为 OpenTelemetry 全量接入，host 精简环境取 no-op
# 退化（tracing 关闭时的真实行为：is_tracing_enabled=False → 返回 None）。
from functools import wraps


# SOURCE: vllm/tracing.py —— HOST SEAM：instrument 装饰器退化位（透传原函数）
def instrument(**kwargs):
    # SOURCE: vllm/tracing.py —— HOST SEAM：装饰器位
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kw):
            return fn(*args, **kw)

        return wrapper

    return deco


# SOURCE: vllm/tracing.py —— HOST SEAM：init_tracer no-op 位
def init_tracer(*args, **kwargs):
    return None


# SOURCE: vllm/tracing.py —— HOST SEAM：contains_trace_headers（真实按
# traceparent/tracestate 头判定；tracing 关闭路径只影响是否告警）
def contains_trace_headers(headers) -> bool:
    return any(k.lower() in ("traceparent", "tracestate") for k in headers)


# SOURCE: vllm/tracing.py —— HOST SEAM：extract_trace_headers 退化位
def extract_trace_headers(headers):
    return {
        k: v for k, v in headers.items() if k.lower() in ("traceparent", "tracestate")
    }


# SOURCE: vllm/tracing.py —— HOST SEAM：log_tracing_disabled_warning no-op 位
def log_tracing_disabled_warning():
    return None


# SUBTRACTED: vllm/tracing.py 其余（SpanAttributes/SpanKind/
# extract_trace_context/instrument_manual）——观测域。
