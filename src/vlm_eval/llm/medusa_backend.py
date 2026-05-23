from __future__ import annotations

import http.client
import json
import os
from urllib.parse import urlparse

from vlm_eval.llm.base import LLM


class OuterMedusaLLM(LLM):
    def __init__(self, model: str = "gpt-oss-20b"):
        super().__init__(model=model)
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        self.outer_medusa_endpoint = os.environ.get("OUTER_MEDUSA_ENDPOINT", "")
        self.outer_medusa_api_key = os.environ.get("OUTER_MEDUSA_API_KEY", "")

        if self.outer_medusa_endpoint == "":
            raise ValueError("OUTER_MEDUSA_ENDPOINT is not set")
        if self.outer_medusa_api_key == "":
            raise ValueError("OUTER_MEDUSA_API_KEY is not set")

        parsed_endpoint = urlparse(
            self.outer_medusa_endpoint
            if self.outer_medusa_endpoint.startswith("http")
            else "https://" + self.outer_medusa_endpoint
        )
        host = parsed_endpoint.netloc if parsed_endpoint.netloc else parsed_endpoint.path
        self.host = host.rstrip("/")

        conn = self._get_connection()
        try:
            response, response_data = self._get_response(conn, "GET", "/v1/models")
            if response.status != 200:
                raise ValueError(
                    f"Request failed: {response.status} {response.reason} - {response_data}"
                )
        finally:
            conn.close()

        print(f"Connected to Outer Medusa client at {self.outer_medusa_endpoint}")

    def __str__(self) -> str:
        return f"OuterMedusaLLM(model={self.model})"

    def _get_connection(self):
        return http.client.HTTPSConnection(self.host)

    def _get_response(self, conn, method, path, headers=None, body=None):
        if headers is None:
            headers = {
                "Authorization": f"Bearer {self.outer_medusa_api_key}",
                "Content-Type": "application/json",
            }
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        response_data = response.read().decode()
        return response, response_data

    def change_model(self, model: str) -> None:
        conn = self._get_connection()
        try:
            response, response_data = self._get_response(conn, "GET", "/v1/models")
            if response.status != 200:
                raise ValueError(
                    f"Request failed: {response.status} {response.reason} - {response_data}"
                )
        finally:
            conn.close()

        data = json.loads(response_data)
        models = data.get("data", [])
        if model in [item["id"] for item in models]:
            self.model = model
            return
        raise ValueError(f"Model {model} not found in available models")

    def generate(self, prompt: str, sys_prompt: str | None = None) -> tuple[str, int, int]:
        payload = {"model": self.model, "messages": []}
        if sys_prompt:
            payload["messages"].append({"role": "system", "content": sys_prompt})
        payload["messages"].append({"role": "user", "content": prompt})

        conn = self._get_connection()
        try:
            response, response_data = self._get_response(
                conn,
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload),
            )
            if response.status != 200:
                raise ValueError(
                    f"Request failed: {response.status} {response.reason} - {response_data}"
                )
            data = json.loads(response_data)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = data.get("usage", {}).get("completion_tokens", 0)

            self.cum_prompt_tokens += prompt_tokens
            self.cum_completion_tokens += completion_tokens

            return content, prompt_tokens, completion_tokens
        finally:
            conn.close()
