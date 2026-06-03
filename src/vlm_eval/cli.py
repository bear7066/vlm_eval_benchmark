from __future__ import annotations

import argparse
from pathlib import Path

from vlm_eval.config import BenchmarkConfig, JudgeConfig
from vlm_eval.hf_dataset import DEFAULT_DATASET_REPO


def benchmark_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run VLM inference benchmark on videos.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="HF dataset config name, e.g. 'climbing_ladder'")
    parser.add_argument("--dataset_repo", type=str, default=DEFAULT_DATASET_REPO,
                        help="HuggingFace dataset repo ID (default: $HF_DATASET_REPO or built-in default)")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--output_root", type=Path, default=Path("runs"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    args = parser.parse_args(argv)

    config = BenchmarkConfig(
        dataset=args.dataset,
        dataset_repo=args.dataset_repo,
        model_id=args.model_id,
        num_frames=args.num_frames,
        sample_size=args.sample_size,
        output_root=args.output_root,
        run_id=args.run_id,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        **({"prompt": args.prompt} if args.prompt is not None else {}),
    )
    from vlm_eval.inference.runner import run_benchmark

    run_dir = run_benchmark(config)
    return 0 if run_dir is not None else 1


def judge_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Use an LLM to judge VLM benchmark predictions.")
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--legacy_log", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=Path("runs"))
    parser.add_argument("--dataset", type=str, default=None,
                        help="HF dataset config name used to find the latest matching run")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    parser.add_argument("--backend", choices=["openai", "medusa"], default=None)
    parser.add_argument(
        "--skip_llm_judge",
        action="store_true",
        help="Only compute BLEU/ROUGE/CIDEr text metrics.",
    )
    args = parser.parse_args(argv)

    config = JudgeConfig(
        judge_model=args.judge_model,
        backend=args.backend,
        run_dir=args.run_dir,
        predictions_path=args.predictions,
        legacy_log_path=args.legacy_log,
        output_root=args.output_root,
        dataset=args.dataset,
        model_id=args.model_id,
        num_frames=args.num_frames,
        skip_llm_judge=args.skip_llm_judge,
    )
    from vlm_eval.judge.runner import run_judge

    run_dir = run_judge(config)
    return 0 if run_dir is not None else 1


def batch_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run benchmark and judge across multiple datasets/models.")
    parser.add_argument("--datasets", nargs="+", default=["falling_off_chair"],
                        help="HF dataset config names to evaluate")
    parser.add_argument("--dataset_repo", type=str, default=DEFAULT_DATASET_REPO,
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--model_ids", nargs="+", default=["google/gemma-4-E4B-it"])
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--output_root", type=Path, default=Path("runs"))
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    parser.add_argument("--backend", choices=["openai", "medusa"], default=None)
    args = parser.parse_args(argv)

    for dataset_name in args.datasets:
        from vlm_eval.inference.runner import run_benchmark
        from vlm_eval.judge.runner import run_judge

        for model_id in args.model_ids:
            benchmark_config = BenchmarkConfig(
                dataset=dataset_name,
                dataset_repo=args.dataset_repo,
                model_id=model_id,
                num_frames=args.num_frames,
                sample_size=args.sample_size,
                output_root=args.output_root,
            )
            run_dir = run_benchmark(benchmark_config)
            if run_dir is None:
                return 1

            judge_config = JudgeConfig(
                judge_model=args.judge_model,
                backend=args.backend,
                run_dir=run_dir,
            )
            judged_dir = run_judge(judge_config)
            if judged_dir is None:
                return 1

    return 0


def gpu_test_main() -> int:
    import time

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(2000, 2000, device=device)
    b = torch.randn(2000, 2000, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    c = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    print("device =", device)
    print("elapsed =", end - start)
    print("sum =", c.sum().item())
    return 0
