# SOURCE: vllm/pooling_params.py
# HOST SEAM：类型占位。真实 PoolingParams 是池化请求参数大类（ch04 域）——
# 本章 Request 切面只按名引用其类型（pooling 分支已随 HOST SEAM 减法删除：
# 本章只讲生成请求的语法门），不消费任何字段。
PoolingParams = object
