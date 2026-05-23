from __future__ import annotations

import time
from threading import Thread
from typing import Any

from vlm_eval.hardware import get_gpu_power_watts


class HuggingFaceVLM:
    def __init__(self, model_id: str, hf_token: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate_from_frames(
        self,
        frames: list[Any],
        prompt_text: str,
        max_new_tokens: int = 150,
    ) -> dict[str, Any]:
        from transformers import TextIteratorStreamer

        content_items = [{"type": "image"} for _ in range(len(frames))]
        content_items.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content_items}]

        formatted_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.processor(
            text=formatted_prompt,
            images=frames,
            return_tensors="pt",
        ).to(self.model.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.torch.bfloat16)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
        )

        start_power = get_gpu_power_watts()
        start_time = time.time()

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        first_chunk_time = None
        response_chunks: list[str] = []
        for new_text in streamer:
            now = time.time()
            if first_chunk_time is None and new_text:
                first_chunk_time = now
            response_chunks.append(new_text)

        thread.join()
        end_time = time.time()
        end_power = get_gpu_power_watts()

        response = "".join(response_chunks).strip()
        elapsed_sec = end_time - start_time

        generated_ids = self.processor.tokenizer.encode(
            response,
            add_special_tokens=False,
        )

        power_reading = None
        if start_power is not None and end_power is not None:
            power_reading = (start_power + end_power) / 2.0
        elif start_power is not None:
            power_reading = start_power
        elif end_power is not None:
            power_reading = end_power

        ttft_sec = first_chunk_time - start_time if first_chunk_time is not None else None

        return {
            "response": response,
            "elapsed_sec": elapsed_sec,
            "elapsed_ms": elapsed_sec * 1000.0,
            "ttft_ms": ttft_sec * 1000.0 if ttft_sec is not None else None,
            "tokens": len(generated_ids),
            "throughput_tps": len(generated_ids) / elapsed_sec if elapsed_sec > 0 else 0.0,
            "average_power_watts": power_reading,
        }
