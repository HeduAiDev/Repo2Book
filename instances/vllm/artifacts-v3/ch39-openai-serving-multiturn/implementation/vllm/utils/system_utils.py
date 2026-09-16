# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/utils/system_utils.py —— HOST SEAM（最小承载）：本章消费面
# 只有 run_server 的 decorate_logs("APIServer")（日志前缀装饰）；
# set_ulimit（delete[10]：uvicorn 高并发丢请求的 workaround，部署可选项）
# 已从精简版 setup_server 删去。
# SUBTRACTED: get_cpu_count/terminate_process/内存与缓存路径工具族——归
# 各自域。


# SOURCE: vllm/utils/system_utils.py:L237 —— HOST SEAM：decorate_logs 退化位
# （真实为日志名前缀重写；no-op 与跳过语义一致）
def decorate_logs(*args, **kwargs):
    return None
