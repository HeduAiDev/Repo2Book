"""LLM —— 离线批量推理门面（精简版，只做减法）。

与真实 vllm/entrypoints/llm.py 同名、同结构、同控制流：generate/chat/embed/encode 四个入口
最终都汇流到同一条脊——渲染(Renderer)→逐条 _add_request 入队(output_kind=FINAL_ONLY)→
_run_engine 的 while has_unfinished_requests(): step() 同步驱动→按 request_id 排序收集。

两条内部路径的关键对照原样保留：
  * completion（generate）走 _run_completion→_add_completion_requests→_render_and_add_requests，
    【不】打物化 warning。
  * chat 走 _run_chat→_render_and_run_requests，后者在 prompts 被物化成 list/tuple 时 warning_once。

把所有 # SUBTRACTED 分支删回去 ≈ 真实 LLM 的这条同步主干。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from llm_engine import LLMEngine
from messages import (
    EmbeddingRequestOutput,
    PoolingParams,
    PoolingRequestOutput,
    RequestOutput,
    RequestOutputKind,
    SamplingParams,
)

# SUBTRACTED: 真实 from tqdm import tqdm（进度条）。精简版用最小占位，不引第三方依赖，
#   语义（total/desc/update/close）保留以体现 _run_engine 的 tqdm 进度。原 vllm/entrypoints/llm.py 顶部 import。
try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    class tqdm:  # 最小占位：仅支持 _run_engine 用到的接口
        # SOURCE: tqdm.tqdm（第三方进度条）— stub，避免引入第三方依赖
        def __init__(self, total=0, desc="", dynamic_ncols=True, postfix=""):
            # SOURCE: tqdm.tqdm.__init__ — stub
            self.total = total
            self.n = 0
            self.postfix = postfix
            self.format_dict = {"elapsed": 1e-9}

        def update(self, k=1):
            # SOURCE: tqdm.tqdm.update — stub
            self.n += k

        def refresh(self):
            # SOURCE: tqdm.tqdm.refresh — stub
            pass

        def close(self):
            # SOURCE: tqdm.tqdm.close — stub
            pass


from itertools import count


class Counter:
    # SOURCE: vllm/utils 的 Counter（自增计数器）— stub
    def __init__(self) -> None:
        self._it = count()

    def __next__(self) -> int:
        # SOURCE: vllm/utils Counter.__next__ — stub
        return next(self._it)


# SOURCE: vllm/entrypoints/llm.py (prompt_to_seq 等归一化 helper) — stub
def _to_seq(x) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


# SOURCE: vllm/entrypoints/llm.py:L212 (class LLM)
class LLM:
    """An LLM for generating texts from given prompts and sampling parameters."""

    # SOURCE: vllm/entrypoints/llm.py:L212 (LLM.__init__)
    def __init__(self, model: str = "stub-model", *, runner: str = "generate", **kwargs):
        # SUBTRACTED: L259-L380 的参数归一化（swap_space 弃用、worker_cls cloudpickle、
        #   kv_transfer_config/compilation_config/structured_outputs 的 _make_config 转换、
        #   数十个构造形参展开成 EngineArgs）。本章保留'把参数拼成 EngineArgs 起引擎'这一句骨架。
        #   原 vllm/entrypoints/llm.py:L259-L380。
        engine_args = _EngineArgsStub(model=model, runner=runner)

        # SOURCE: vllm/entrypoints/llm.py:L381 (self.llm_engine = LLMEngine.from_engine_args(...))
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context="LLM_CLASS"
        )
        self.model_config = _ModelConfigStub(runner_type=runner)
        self.request_counter = Counter()
        self.default_sampling_params: dict[str, Any] | None = None

        supported_tasks = self.llm_engine.get_supported_tasks()
        self.supported_tasks = supported_tasks
        self.runner_type = runner

        # SUBTRACTED: pooling_task/renderer/chat_template/input_processor/chat_template_config/
        #   pooling_io_processors/_cached_repr 初始化（L390-L408）。pooling/chat 渲染装配与
        #   本章同步主干弱相关；encode 用最小 io_processor 顶替。原 vllm/entrypoints/llm.py:L390-L408。
        self.pooling_io_processors = {"embed": _IOProcessorStub(), "encode": _IOProcessorStub()}

    # SOURCE: vllm/entrypoints/llm.py (get_default_sampling_params) — stub
    def get_default_sampling_params(self) -> SamplingParams:
        return SamplingParams()

    # SOURCE: vllm/entrypoints/llm.py:L446 (generate)
    def generate(
        self,
        prompts,
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        *,
        use_tqdm=False,
        lora_request=None,
        priority: list[int] | None = None,
        tokenization_kwargs=None,
        mm_processor_kwargs=None,
    ) -> list[RequestOutput]:
        # SUBTRACTED: docstring（L457-L488）。原 vllm/entrypoints/llm.py:L457-L488。
        runner_type = self.model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.generate() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        return self._run_completion(
            prompts=prompts,
            params=sampling_params,
            output_type=RequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    # SOURCE: vllm/entrypoints/llm.py:L981 (chat)
    def chat(
        self,
        messages,
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm=False,
        lora_request=None,
        **kwargs,
    ) -> list[RequestOutput]:
        # SUBTRACTED: docstring + chat_template/content_format/tools/add_generation_prompt 等参数
        #   （L997-L1046 + 形参）。本章只需 runner 守卫 + 走 _run_chat。原 vllm/entrypoints/llm.py:L981-L1046。
        runner_type = self.model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.chat() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        return self._run_chat(
            messages=messages,
            params=sampling_params,
            output_type=RequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

    # SOURCE: vllm/entrypoints/llm.py:L1075 (encode)
    def encode(
        self,
        prompts,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        *,
        use_tqdm=False,
        lora_request=None,
        pooling_task: str | None = None,
        tokenization_kwargs=None,
    ) -> list[PoolingRequestOutput]:
        # SUBTRACTED: docstring + 'data' 字段/plugin task 校验、_verify_pooling_task 的 runner=='pooling'
        #   守卫、io_processor.pre_process_offline 的真实多模态预处理、task 校验细节（L1085-L1147）。
        #   原 vllm/entrypoints/llm.py:L1085-L1147。stub 保留 io_processor 预处理→渲染提交→运行→后处理骨架。
        if pooling_task is None:
            pooling_task = "encode"
        io_processor = self.pooling_io_processors[pooling_task]

        if pooling_params is None:
            pooling_params = PoolingParams()

        engine_inputs = io_processor.pre_process_offline(prompts)
        n_inputs = len(engine_inputs)
        params_seq = self._params_to_seq(pooling_params, n_inputs)
        for param in params_seq:
            if param.task is None:
                param.task = pooling_task

        self._render_and_add_requests(
            prompts=engine_inputs,
            params=params_seq,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm, output_type=PoolingRequestOutput)
        outputs = io_processor.post_process_offline(outputs)
        return outputs

    # SOURCE: vllm/entrypoints/llm.py:L1223 (embed)
    def embed(
        self,
        prompts,
        *,
        use_tqdm=False,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        lora_request=None,
        tokenization_kwargs=None,
    ) -> list[EmbeddingRequestOutput]:
        # SUBTRACTED: docstring（L1232-L1255）。原 vllm/entrypoints/llm.py:L1232-L1255。
        items = self.encode(
            prompts,
            use_tqdm=use_tqdm,
            pooling_params=pooling_params,
            lora_request=lora_request,
            pooling_task="embed",
            tokenization_kwargs=tokenization_kwargs,
        )

        return [EmbeddingRequestOutput.from_base(item) for item in items]

    # SUBTRACTED: beam_search（L691-L846）、classify/reward/score（L1268-L1465）等非主线任务方法。
    #   本章聚焦 generate/chat/embed/encode 四入口 + 同步 step 驱动。原 vllm/entrypoints/llm.py。

    # ---- 参数归一化 helper（stub） ----
    def _params_to_seq(self, params, n: int) -> list:
        # SOURCE: vllm/entrypoints/llm.py:_params_to_seq — stub
        if isinstance(params, (list, tuple)):
            return list(params)
        return [params for _ in range(n)]

    # SOURCE: vllm/entrypoints/llm.py:L1592 (_add_completion_requests)
    def _add_completion_requests(
        self,
        prompts,
        params,
        *,
        use_tqdm=False,
        lora_request=None,
        priority=None,
        tokenization_kwargs=None,
        mm_processor_kwargs=None,
    ) -> list[str]:
        seq_prompts = _to_seq(prompts)
        seq_params = self._params_to_seq(params, len(seq_prompts))
        seq_priority = None if priority is None else list(priority)

        # 【关键对照】completion 直接逐 prompt 渲染并 add（经生成器），【不】走 _render_and_run_requests，
        # 所以不打那条物化 warning（与 chat 路径区分的锚点）。
        return self._render_and_add_requests(
            prompts=(
                self._preprocess_cmpl_one(prompt, tokenization_kwargs)
                for prompt in seq_prompts
            ),
            params=seq_params,
            priorities=seq_priority,
        )

    def _preprocess_cmpl_one(self, prompt, tokenization_kwargs=None):
        # SOURCE: vllm/entrypoints/llm.py:_preprocess_cmpl_one — stub（真实经 renderer 渲染单 prompt）
        return prompt

    # SOURCE: vllm/entrypoints/llm.py:L1628 (_run_completion)
    def _run_completion(
        self,
        prompts,
        params,
        output_type,
        *,
        use_tqdm=False,
        lora_request=None,
        priority=None,
        tokenization_kwargs=None,
        mm_processor_kwargs=None,
    ):
        self._add_completion_requests(
            prompts=prompts,
            params=params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return self._run_engine(use_tqdm=use_tqdm, output_type=output_type)

    # SOURCE: vllm/entrypoints/llm.py:L1653 (_run_chat)
    def _run_chat(
        self,
        messages,
        params,
        output_type,
        *,
        use_tqdm=False,
        lora_request=None,
    ):
        seq_convs = _to_seq(messages)
        seq_params = self._params_to_seq(params, len(seq_convs))

        # SUBTRACTED: needs_parsing / _adjust_params_for_parsing 的 Gemma4 特判（L1677-L1686）。
        #   原 vllm/entrypoints/llm.py:L1677-L1686, L1713-L1758。边角 hook，删后 chat 主干完整。

        # chat 传入的是逐 conversation 渲染的【生成器】，故正常不触发 _render_and_run_requests 的
        # 物化 warning（warning 只在调用方把 prompts 提前物化成 list/tuple 时打）。
        return self._render_and_run_requests(
            prompts=(
                self._preprocess_chat_one(conversation)
                for conversation in seq_convs
            ),
            params=seq_params,
            output_type=output_type,
            use_tqdm=use_tqdm,
        )

    def _preprocess_chat_one(self, conversation):
        # SOURCE: vllm/entrypoints/llm.py:_preprocess_chat_one — stub（真实经 chat template 渲染）
        return conversation

    # SOURCE: vllm/entrypoints/llm.py:L1760 (_render_and_run_requests)
    def _render_and_run_requests(
        self,
        prompts: Iterable,
        params: Sequence,
        output_type,
        *,
        lora_requests=None,
        priorities=None,
        use_tqdm=False,
    ):
        # chat 路径专用 + 物化 warning_once：若调用方把 prompts 提前物化成 list/tuple（而非传生成器），
        # 提示'传生成器可让首个 prompt 的引擎执行与后续渲染重叠'。
        if isinstance(prompts, (list, tuple)):
            _warning_once(
                "Rendering all prompts before adding them to the engine "
                "is less efficient than performing both on the same prompt "
                "before processing the next prompt. You should instead pass "
                "a generator that renders one prompt per iteration, as that allows "
                "engine execution to begin for the first prompt while processing "
                "the next prompt."
            )

        self._render_and_add_requests(
            prompts=prompts,
            params=params,
            lora_requests=lora_requests,
            priorities=priorities,
        )

        return self._run_engine(output_type, use_tqdm=use_tqdm)

    # SOURCE: vllm/entrypoints/llm.py:L1789 (_render_and_add_requests)
    def _render_and_add_requests(
        self,
        prompts: Iterable,
        params: Sequence,
        *,
        lora_requests=None,
        priorities=None,
    ) -> list[str]:
        added_request_ids: list[str] = []

        try:
            for i, prompt in enumerate(prompts):
                request_id = self._add_request(
                    prompt,
                    params[i],
                    # SUBTRACTED: lora_request=self._resolve_mm_lora(prompt, ...)（多模态默认 LoRA 选择，
                    #   L594-L644）。边角 hook，对主控制流是旁路。原 vllm/entrypoints/llm.py:L1804-L1807。
                    lora_request=None if lora_requests is None else lora_requests[i],
                    priority=0 if priorities is None else priorities[i],
                )
                added_request_ids.append(request_id)
        except Exception as e:
            # 事务性：任一请求 add 失败 → abort 已加的全部（防半批请求悬挂）。
            if added_request_ids:
                self.llm_engine.abort_request(added_request_ids, internal=True)
            raise e

        return added_request_ids

    # SOURCE: vllm/entrypoints/llm.py:L1818 (_add_request)
    def _add_request(
        self,
        prompt,
        params,
        lora_request=None,
        priority: int = 0,
    ) -> str:
        if isinstance(params, SamplingParams):
            # We only care about the final output
            params.output_kind = RequestOutputKind.FINAL_ONLY

        request_id = str(next(self.request_counter))

        return self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            priority=priority,
        )

    # SOURCE: vllm/entrypoints/llm.py:L1839 (_run_engine)
    def _run_engine(
        self,
        output_type,
        *,
        use_tqdm=False,
    ) -> list:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: list = []
        total_in_toks = 0
        total_out_toks = 0
        # 【核心】同步阻塞主循环：while 还有未完成请求就 step() 拉动 EngineCore 一拍。
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                assert isinstance(output, output_type)
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        # SUBTRACTED: RequestOutput 分支的 toks/s 估算（in_spd/out_spd，
                        #   依赖 prompt_token_ids/CompletionOutput.token_ids，ch08-ch10）。
                        #   原 vllm/entrypoints/llm.py:L1867-L1881。保留进度推进主干。
                        pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))


# ---- 以下为本章不展开的最小占位类型（站位真实配置/EngineArgs/io_processor） ----

class _EngineArgsStub:
    # SOURCE: vllm/engine/arg_utils.py:EngineArgs — stub（本章一句带过'参数拼成 EngineArgs'）
    def __init__(self, model: str, runner: str):
        self.model = model
        self.runner = runner
        self.disable_log_stats = True


class _ModelConfigStub:
    # SOURCE: vllm/config:ModelConfig — stub（只暴露本章用到的 runner_type）
    def __init__(self, runner_type: str):
        self.runner_type = runner_type


class _IOProcessorStub:
    # SOURCE: vllm/entrypoints pooling io_processor — stub（pre/post_process_offline 透传）
    def pre_process_offline(self, prompts):
        # SOURCE: vllm pooling io_processor.pre_process_offline — stub（透传）
        return _to_seq(prompts)

    def post_process_offline(self, outputs):
        # SOURCE: vllm pooling io_processor.post_process_offline — stub（透传）
        return outputs


# SOURCE: vllm/logger.py:logger.warning_once — stub（去重 warning）
_WARNED: set[str] = set()


def _warning_once(msg: str) -> None:
    # SOURCE: vllm/logger.py:logger.warning_once — stub（去重 warning）
    if msg not in _WARNED:
        _WARNED.add(msg)
