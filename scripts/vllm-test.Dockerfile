# 跑本书 vLLM 章测试用的镜像 = 官方 vllm-openai + pytest。
#
# 为什么要它:`scripts/vllm_docker.sh` 用的是 `docker run --rm`(每次新容器),
# 而官方镜像**不带 pytest** → 直接跑 `-m pytest` 会 `No module named pytest`,
# 在容器里 `pip install` 又会随 `--rm` 一起蒸发。故固化成一层薄镜像。
#
# 构建:
#   docker build -f scripts/vllm-test.Dockerfile -t repo2book/vllm-test:latest .
#
# ⚠️ 2026-07-21 实测:本机 `docker build` 走不通代理(pip ProxyError: Connection refused),
# 而**已在跑的**容器网络是通的。故本机改用 commit 法产出同一镜像:
#   docker exec vllm python3 -m pip install -q pytest
#   docker commit --change 'ENTRYPOINT ["/usr/bin/python3"]' vllm repo2book/vllm-test:latest
# 两条路等价;本 Dockerfile 保留为可复现的**意图声明**,换机器能 build 就 build。
# 使用(helper 已支持 VLLM_IMAGE 覆盖):
#   VLLM_IMAGE=repo2book/vllm-test:latest scripts/vllm_docker.sh \
#     -m pytest /work/instances/vllm/artifacts/ch18-model-runner/tests -q
#
# ⚠️ 版本口径:该镜像内的 vLLM 版本**未必等于本书钉死的 v0.21.0(ad7125a4)**
# (2026-07-21 实测镜像内为 vllm 0.15.1 / torch 2.9.1+cu129 / triton 3.5.1)。
# 故它只用于**取真实 GPU 行为与数值**,行号与 API 细节一律以 pin 源码为准;
# 若容器行为与 pin 源码不一致,正文必须挑明(exp-2026-07-18-01)。
FROM vllm/vllm-openai:latest
# pytest-asyncio 是必须的:ch04 的精简版有 14 个 @pytest.mark.asyncio 用例,
# 缺插件时 pytest 只发 PytestUnknownMarkWarning 然后把它们判 FAILED——
# 那是**环境缺件**,不是章节缺陷,别误读成回归。
RUN python3 -m pip install --no-cache-dir pytest pytest-asyncio
ENTRYPOINT ["/usr/bin/python3"]
