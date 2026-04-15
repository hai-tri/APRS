"""
Utility benchmark evaluation via lm-evaluation-harness.

Runs GSM8k, MATH500, and MMLU on a saved model checkpoint using the
lm-evaluation-harness CLI as a subprocess.  The defended model must be
saved to disk before calling this (Stage 8 in the pipeline already does
this via model.save_pretrained()).

Each benchmark is evaluated on 100 randomly sampled examples (seeded),
selected via lm_eval's --samples flag which accepts explicit doc indices.
This ensures reproducibility across runs and fair comparison across defense
configurations.

Benchmarks:
    - GSM8k      : grade-school math word problems (8-shot), 1319 test examples
    - MATH500    : competition math, 500-problem subset (4-shot)
    - MMLU       : massive multitask language understanding (5-shot), ~14k test examples

Reference: Gao et al. 2021, "A Framework for Few-Shot Language Model Evaluation"
https://github.com/EleutherAI/lm-evaluation-harness
"""

import json
import os
import random
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

_PYTHON = sys.executable

# Task name → (lm_eval task string, num_fewshot, hf_dataset_args)
# hf_dataset_args: (path, name, split) for size detection
TASKS = {
    "gsm8k":   ("gsm8k",            8, ("openai/gsm8k",      "main",    "test")),
    "math500": ("hendrycks_math500", 4, ("HuggingFaceH4/MATH-500", "default", "test")),
    "mmlu":    ("mmlu",              5, ("hails/mmlu_no_train", None,    "test")),
}

# Fallback dataset sizes if HF download fails
_FALLBACK_SIZES = {
    "gsm8k":   1319,
    "math500": 500,
    "mmlu":    14042,
}


def _get_dataset_size(task_key: str) -> int:
    """Return number of examples in the test split for a task."""
    _, _, (path, name, split) = TASKS[task_key]
    try:
        from datasets import load_dataset
        kwargs = {"split": split, "streaming": True}
        if name:
            kwargs["name"] = name
        ds = load_dataset(path, **kwargs)
        # Streaming dataset — count via info if available
        try:
            return ds.info.splits[split].num_examples
        except Exception:
            pass
    except Exception:
        pass
    return _FALLBACK_SIZES[task_key]


def _sample_indices(task_key: str, n: int, seed: int) -> List[int]:
    """
    Return n randomly sampled doc indices for a task, seeded for reproducibility.
    For math500 (exactly 500 examples), cap n at 500.
    """
    size = _get_dataset_size(task_key)
    n = min(n, size)
    rng = random.Random(seed)
    return sorted(rng.sample(range(size), n))


def run_lm_harness(
    model_path: str,
    tasks: Optional[list] = None,
    n_samples: int = 100,
    batch_size: int = 4,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """
    Run lm-evaluation-harness benchmarks on a saved model.

    Parameters
    ----------
    model_path : path to saved HF model directory
    tasks      : list of task keys (default: all — gsm8k, math500, mmlu)
    n_samples  : number of examples to evaluate per task (default: 100)
    batch_size : per-device batch size
    device     : "cuda", "mps", or "cpu" (auto-detected if None)
    output_dir : directory to write lm_eval JSON results
    seed       : random seed for example selection AND fewshot sampling

    Returns
    -------
    dict with per-task metrics, e.g.:
        {
          "gsm8k":   {"exact_match": 0.42, ...},
          "math500": {"exact_match": 0.18, ...},
          "mmlu":    {"acc": 0.61, ...},
        }
    """
    import torch

    if tasks is None:
        tasks = list(TASKS.keys())

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="lm_eval_")
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for task_key in tasks:
        if task_key not in TASKS:
            print(f"[lm-harness] Unknown task '{task_key}', skipping.")
            continue

        lm_task, num_fewshot, _ = TASKS[task_key]

        # Use --limit to cap examples per task. --samples requires exact subtask
        # keys (e.g. "mmlu_anatomy") and silently does nothing when passed the
        # group name "mmlu", so --limit is the only reliable approach.
        print(f"[lm-harness] {task_key}: limiting to {n_samples} examples (seed={seed})")

        output_path = os.path.join(output_dir, f"{task_key}_results.json")

        cmd = [
            _PYTHON, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True",
            "--tasks", lm_task,
            "--num_fewshot", str(num_fewshot),
            "--batch_size", str(batch_size),
            "--device", device,
            "--output_path", output_path,
            "--seed", str(seed),
            "--limit", str(n_samples),
            "--trust_remote_code",
            "--apply_chat_template",
        ]

        print(f"[lm-harness] Running {task_key} ({lm_task}, {num_fewshot}-shot, "
              f"n={n_samples}) …")

        proc = subprocess.run(cmd, capture_output=False, text=True)

        if proc.returncode != 0:
            print(f"[lm-harness] WARNING: {task_key} exited with code {proc.returncode}")
            results[task_key] = {"error": f"exit code {proc.returncode}"}
            continue

        # Parse output JSON — lm_eval may nest under a timestamped subdir
        result_file = output_path
        if not os.path.exists(result_file):
            candidates = []
            for root, _, files in os.walk(output_dir):
                for fname in files:
                    if fname.endswith(".json") and task_key in root:
                        candidates.append(os.path.join(root, fname))
            if candidates:
                result_file = sorted(candidates)[-1]

        if os.path.exists(result_file):
            with open(result_file) as f:
                raw = json.load(f)
            task_results = raw.get("results", {}).get(lm_task, {})
            results[task_key] = _extract_metrics(task_key, task_results)
            results[task_key]["n_samples"] = n_samples
            results[task_key]["seed"] = seed
            print(f"[lm-harness] {task_key}: {results[task_key]}")
        else:
            print(f"[lm-harness] WARNING: result file not found for {task_key}")
            results[task_key] = {}

    # Save combined results
    combined_path = os.path.join(output_dir, "utility_benchmarks.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[lm-harness] Combined results saved to {combined_path}")

    return results


def _extract_metrics(task_key: str, raw: dict) -> dict:
    """Pull the primary accuracy metric from lm_eval's result dict."""
    if task_key == "gsm8k":
        return {
            "exact_match": raw.get("exact_match,strict-match",
                                   raw.get("exact_match", None)),
            "exact_match_flexible": raw.get("exact_match,flexible-extract", None),
        }
    elif task_key == "math500":
        return {
            "exact_match": raw.get("exact_match,get-answer",
                                   raw.get("exact_match", None)),
        }
    elif task_key == "mmlu":
        return {
            "acc": raw.get("acc,none", raw.get("acc", None)),
            "acc_norm": raw.get("acc_norm,none", raw.get("acc_norm", None)),
        }
    return raw
