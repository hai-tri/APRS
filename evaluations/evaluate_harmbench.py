"""
HarmBench ASR evaluation for the obfuscation defense pipeline.

Samples 100 behaviors from HarmBench, generates completions with the
(possibly defended) model, then scores them with the HarmBench
Llama-2-13B classifier (cais/HarmBench-Llama-2-13b-cls).

Reference: Mazeika et al. 2024, "HarmBench: A Standardized Evaluation
Framework for Automated Red Teaming and Robust Refusal"
https://arxiv.org/abs/2402.04249
"""

import json
import os
import sys
import torch
from typing import Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFUSAL_DIR = os.path.join(_ROOT, "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)

from pipeline.submodules.evaluate_jailbreak import harmbench_judge_fn, load_harmbench_behaviors

_DEFAULT_CSV = os.path.join(_ROOT, "data", "harmbench_behaviors_text_test.csv")


def evaluate_harmbench_asr(
    model,
    tokenize_fn,
    fwd_pre_hooks: list = [],
    fwd_hooks: list = [],
    behaviors_csv: str = _DEFAULT_CSV,
    n_behaviors: int = 100,
    seed: int = 42,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    artifact_dir: Optional[str] = None,
    classifier_model_id: str = "cais/HarmBench-Llama-2-13b-cls",
) -> Dict:
    """
    Full HarmBench evaluation: generation + classification.

    Parameters
    ----------
    model : the (possibly defended) causal LM
    tokenize_fn : callable(instructions=[...]) → BatchEncoding
    fwd_pre_hooks / fwd_hooks : optional inference-time hooks (e.g. ablation)
    behaviors_csv : path to HarmBench behaviors CSV
    n_behaviors : number of behaviors to sample (default 100)
    seed : random seed for behavior sampling
    max_new_tokens : generation budget per completion
    batch_size : generation batch size
    artifact_dir : if set, save completions + evaluations here

    Returns
    -------
    dict with:
        ``asr``             — fraction of behaviors where model complied
        ``n_harmful``       — number of harmful completions
        ``n_total``         — total behaviors evaluated
        ``completions``     — list of {behavior, behavior_id, response, is_harmful}
        ``per_category_asr``— ASR broken down by SemanticCategory
    """
    from pipeline.utils.hook_utils import add_hooks

    device = next(model.parameters()).device

    # ----------------------------------------------------------------
    # Step 1: Load behaviors
    # ----------------------------------------------------------------
    print(f"[HarmBench] Loading {n_behaviors} behaviors from {behaviors_csv}")
    behaviors = load_harmbench_behaviors(behaviors_csv, n=n_behaviors, seed=seed)
    prompts = [b["behavior"] for b in behaviors]

    # ----------------------------------------------------------------
    # Step 2: Generate completions
    # ----------------------------------------------------------------
    print(f"[HarmBench] Generating completions ({len(prompts)} prompts) …")
    responses = []
    model.eval()

    with add_hooks(fwd_pre_hooks, fwd_hooks):
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = tokenize_fn(instructions=batch)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=input_ids[0, -1].item(),
                )

            # Decode only the newly generated tokens
            for j, out in enumerate(output_ids):
                prompt_len = input_ids.shape[1]
                new_tokens = out[prompt_len:]
                # get tokenizer from tokenize_fn's closure — fall back to model's config
                response_text = _decode(model, new_tokens)
                responses.append(response_text)

            print(f"[HarmBench] Generated {min(i + batch_size, len(prompts))}/{len(prompts)}")

    # ----------------------------------------------------------------
    # Step 3: Offload target model to free memory for classifier
    # ----------------------------------------------------------------
    print("[HarmBench] Offloading target model before running classifier …")
    original_device = next(model.parameters()).device
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ----------------------------------------------------------------
    # Step 4: Classify
    # ----------------------------------------------------------------
    print("[HarmBench] Running HarmBench classifier …")
    classifications = harmbench_judge_fn(
        prompts=prompts,
        responses=responses,
        classifier_model_id=classifier_model_id,
    )

    # Restore target model
    model.to(original_device)

    # ----------------------------------------------------------------
    # Step 5: Aggregate
    # ----------------------------------------------------------------
    completions = []
    for b, response, is_harmful in zip(behaviors, responses, classifications):
        completions.append({
            "behavior": b["behavior"],
            "behavior_id": b["behavior_id"],
            "semantic_category": b["semantic_category"],
            "response": response,
            "is_harmful": is_harmful,
        })

    asr = sum(classifications) / len(classifications)

    per_category = {}
    for c in completions:
        cat = c["semantic_category"] or "unknown"
        per_category.setdefault(cat, []).append(c["is_harmful"])
    per_category_asr = {cat: sum(vals) / len(vals) for cat, vals in per_category.items()}

    print(f"[HarmBench] ASR: {asr:.4f} ({sum(classifications)}/{len(classifications)})")

    result = {
        "asr": asr,
        "n_harmful": sum(classifications),
        "n_total": len(classifications),
        "completions": completions,
        "per_category_asr": per_category_asr,
    }

    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "harmbench_evaluation.json"), "w") as f:
            json.dump(result, f, indent=4)
        print(f"[HarmBench] Results saved to {artifact_dir}/harmbench_evaluation.json")

    return result


def _decode(model, token_ids: torch.Tensor) -> str:
    """Best-effort decode using the model's tied tokenizer if available."""
    # Try common attribute names
    for attr in ("tokenizer", "_tokenizer"):
        tok = getattr(model, attr, None)
        if tok is not None:
            return tok.decode(token_ids, skip_special_tokens=True)
    # Fall back to the config's vocab — shouldn't normally happen
    return str(token_ids.tolist())
