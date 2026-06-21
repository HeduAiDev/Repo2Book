#!/usr/bin/env bash
# Run vLLM-related code/tests inside the vLLM container (host has no CUDA/vLLM).
# Usage:
#   scripts/vllm_docker.sh -m pytest /work/instances/vllm/artifacts/ch04-async-llm/tests
#   scripts/vllm_docker.sh -c "import vllm; print(vllm.__version__)"
# The repo is mounted at /work; GPU is attached. Image override via VLLM_IMAGE.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
exec docker run --rm --gpus all --entrypoint /usr/bin/python3 \
  -v "$REPO":/work -w /work "$IMAGE" "$@"
