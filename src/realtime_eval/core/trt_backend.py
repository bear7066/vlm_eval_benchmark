"""TensorRT-LLM serving backend for the real-time VLM sweep.

This mirrors :class:`vlm_eval.inference.gemma.HuggingFaceVLM`: it exposes a
``generate_from_frames(frames, prompt_text, max_new_tokens) -> dict`` method
returning the same metric keys, so :func:`realtime_eval.pipeline.runner.run_config`
can drive it unchanged.

``tensorrt_llm`` is imported lazily inside the methods so importing this module
(and running its self-check) never requires the heavy dependency. Install it on
the GPU/build box per ``PLAN.md``; note TensorRT-LLM needs an Ampere (sm_80+) GPU.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any


def _assemble_metrics(
    response: str,
    n_tokens: int,
    start_time: float,
    first_token_time: float | None,
    end_time: float,
) -> dict[str, Any]:
    """Build the metric dict shared with the HuggingFace backend.

    Pure function (no model, no clock) so the timing math is unit-testable.

    Args:
        response: Decoded generation text.
        n_tokens: Number of generated tokens.
        start_time: Wall-clock time just before generation (``time.time``).
        first_token_time: Wall-clock time of the first streamed token, or
            ``None`` if nothing was streamed.
        end_time: Wall-clock time just after generation completes.

    Returns:
        Dict with the keys :func:`runner.run_config` reads: ``response``,
        ``elapsed_sec``, ``elapsed_ms``, ``ttft_ms``, ``tokens``,
        ``throughput_tps``.
    """
    elapsed_sec = end_time - start_time
    ttft_sec = first_token_time - start_time if first_token_time is not None else None
    return {
        "response": response,
        "elapsed_sec": elapsed_sec,
        "elapsed_ms": elapsed_sec * 1000.0,
        "ttft_ms": ttft_sec * 1000.0 if ttft_sec is not None else None,
        "tokens": n_tokens,
        "throughput_tps": n_tokens / elapsed_sec if elapsed_sec > 0 else 0.0,
    }


class TensorRTVLM:
    """A TensorRT-LLM-served VLM with the ``HuggingFaceVLM`` interface."""

    def __init__(self, model_id: str, hf_token: str | None = None, **engine_kwargs: Any):
        """Build the TensorRT-LLM engine and load the matching processor.

        Args:
            model_id: HuggingFace model ID (also the TensorRT-LLM checkpoint).
            hf_token: Optional HuggingFace access token (for the processor).
            **engine_kwargs: Extra args forwarded to ``tensorrt_llm.LLM``
                (e.g. ``tensor_parallel_size``, ``dtype``).
        """
        from tensorrt_llm import LLM
        from transformers import AutoProcessor

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        # ponytail: defaults (bf16, tensor_parallel_size=1) come from the engine;
        # pass dtype/tensor_parallel_size via engine_kwargs when the box needs them.
        self.llm = LLM(model=model_id, **engine_kwargs)

    def _format_prompt(self, num_images: int, prompt_text: str) -> str:
        """Format the chat prompt identically to the HuggingFace backend."""
        content_items = [{"type": "image"} for _ in range(num_images)]
        content_items.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content_items}]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def generate_from_frames(
        self,
        frames: list[Any],
        prompt_text: str,
        max_new_tokens: int = 150,
    ) -> dict[str, Any]:
        """Generate a response from sampled frames, measuring latency and TTFT.

        Args:
            frames: Sampled PIL frames.
            prompt_text: Instruction text sent with the frames.
            max_new_tokens: Generation cap.

        Returns:
            The metric dict produced by :func:`_assemble_metrics`.
        """
        from tensorrt_llm import SamplingParams

        prompt = self._format_prompt(len(frames), prompt_text)
        inputs = {"prompt": prompt, "multi_modal_data": {"image": frames}}
        sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

        first_token_time: float | None = None
        response = ""

        async def _stream() -> tuple[float | None, str]:
            nonlocal first_token_time
            text = ""
            async for output in self.llm.generate_async(
                inputs, sampling, streaming=True
            ):
                diff = output.outputs[0].text_diff
                if diff:
                    if first_token_time is None:
                        first_token_time = time.time()
                    text += diff
            return first_token_time, text

        start_time = time.time()
        first_token_time, response = asyncio.run(_stream())
        end_time = time.time()

        response = response.strip()
        n_tokens = len(
            self.processor.tokenizer.encode(response, add_special_tokens=False)
        )
        return _assemble_metrics(
            response, n_tokens, start_time, first_token_time, end_time
        )


def _self_check() -> None:
    """Verify the pure metric math; no GPU or heavy deps required."""
    m = _assemble_metrics(
        response="hello world",
        n_tokens=3,
        start_time=10.0,
        first_token_time=10.2,
        end_time=11.0,
    )
    assert abs(m["elapsed_ms"] - 1000.0) < 1e-6, m["elapsed_ms"]
    assert abs(m["ttft_ms"] - 200.0) < 1e-6, m["ttft_ms"]
    assert abs(m["throughput_tps"] - 3.0) < 1e-9, m["throughput_tps"]
    assert m["tokens"] == 3

    # No tokens streamed -> ttft is None, throughput is 0 when no time elapsed.
    z = _assemble_metrics("", 0, 5.0, None, 5.0)
    assert z["ttft_ms"] is None
    assert z["throughput_tps"] == 0.0
    print("trt_backend self-check passed")


if __name__ == "__main__":
    _self_check()
