from __future__ import annotations

MEDUSA_MODELS = {
    "gpt-oss-20b",
    "gpt-oss-120b",
    "Google-Gemma-3-27B",
    "Llama-3.1-70B",
    "Llama-3.1-405B-Instruct-FP8",
}
OPENAI_MODELS = {"gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-5"}


def get_llm_instance(model: str, backend: str | None = None):
    resolved = backend
    if resolved is None:
        if model in MEDUSA_MODELS:
            resolved = "medusa"
        elif model in OPENAI_MODELS:
            resolved = "openai"
        else:
            raise ValueError(f"Unknown model '{model}'. Specify --backend explicitly.")

    if resolved == "medusa":
        from vlm_eval.llm.medusa_backend import OuterMedusaLLM

        return OuterMedusaLLM(model=model)
    if resolved == "openai":
        from vlm_eval.llm.openai_backend import OpenAILLM

        return OpenAILLM(model=model)
    raise ValueError(f"Unknown backend: {resolved}. Choose 'openai' or 'medusa'.")
