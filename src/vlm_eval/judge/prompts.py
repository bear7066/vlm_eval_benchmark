from __future__ import annotations


def build_judge_prompt(answer: str, label: str) -> str:
    return f"""
You are an expert and impartial evaluator for Vision-Language Models (VLMs).

A VLM was asked:
"These are uniformly sampled frames from a video. Analyze what action is happening."

Ground Truth Action:
{label}

VLM Output:
{answer}

Evaluate the output using these criteria:
1. Does it correctly identify the core action?
2. Is it concise and relevant?
3. Does it avoid hallucinations or unrelated details?
4. Does it reply with English?

Return exactly in this format:
Score: [0-10]
Reason: [brief explanation in under 20 words]
"""
