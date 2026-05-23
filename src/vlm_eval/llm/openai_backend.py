from __future__ import annotations

import os

from vlm_eval.llm.base import LLM


class OpenAILLM(LLM):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        super().__init__(model=model)
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
        except ImportError:
            pass

        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key == "":
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print("Connected to OpenAI")

    def __str__(self) -> str:
        return f"OpenAILLM(model={self.model})"

    def change_model(self, model: str) -> None:
        self.model = model

    def generate(self, prompt: str, sys_prompt: str | None = None) -> tuple[str, int, int]:
        request_msg = []
        if sys_prompt:
            request_msg.append({"role": "system", "content": sys_prompt})
        request_msg.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=request_msg,
        )

        prompt_tokens = 0
        completion_tokens = 0
        if completion.usage:
            prompt_tokens = completion.usage.prompt_tokens or 0
            completion_tokens = completion.usage.completion_tokens or 0
            self.cum_prompt_tokens += prompt_tokens
            self.cum_completion_tokens += completion_tokens

        return str(completion.choices[0].message.content), prompt_tokens, completion_tokens
