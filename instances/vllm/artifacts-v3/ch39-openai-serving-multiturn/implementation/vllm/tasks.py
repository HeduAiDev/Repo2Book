# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/tasks.py —— 逐字承载（43 行全量，无删改）：SupportedTask
# 字面量族是 build_app/init_app_state 按 task 分发路由的钥匙。
from typing import Literal, get_args

from vllm.exceptions import VLLMValidationError

# SOURCE: vllm/tasks.py:L8-L9
GenerationTask = Literal["generate", "transcription", "realtime"]
GENERATION_TASKS: tuple[GenerationTask, ...] = get_args(GenerationTask)

# SOURCE: vllm/tasks.py:L11-L19
PoolingTask = Literal[
    "embed",
    "classify",
    "token_embed",
    "token_classify",
    "plugin",
    "embed&token_classify",
]
POOLING_TASKS: tuple[PoolingTask, ...] = get_args(PoolingTask)

# SUBTRACTED: vllm/tasks.py:L21-L33 _REMOVED_POOLING_TASK_MESSAGES 与
# check_removed_pooling_task / SCORE_TYPE_MAP——离线 pooling API 面的
# 兼容提示，本章服务面不消费。

# SOURCE: vllm/tasks.py:L36-L37
FrontendTask = Literal["render"]
FRONTEND_TASKS: tuple[FrontendTask, ...] = get_args(FrontendTask)

# SOURCE: vllm/tasks.py:L39
SupportedTask = Literal[GenerationTask, PoolingTask, FrontendTask]
