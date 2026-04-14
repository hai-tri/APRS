"""
CAST (Conditional Activation Steering) defense — Shi et al. 2024 (ICLR 2025).

Extracts two vectors from contrastive activation pairs:

  1. **Condition vector** — separates harmful from harmless inputs (via PCA
     on the difference of hidden states).  Used to detect whether the current
     input looks dangerous.

  2. **Behavior vector** — the direction to steer toward (refusal) when the
     condition is met.

At inference, for each instrumented layer:
  - Compute cosine similarity of the hidden state with the condition vector.
  - If similarity exceeds a threshold, add strength × behavior_vector.
  - Optionally re-normalise the hidden state to preserve its original scale.

The key difference from Surgical: CAST is *conditional* — it only intervenes
when the input resembles a harmful prompt, leaving benign inputs unmodified.
This makes it more conservative and avoids the utility degradation seen in
unconditional ablation.

Reference: "Programming Refusal with Conditional Activation Steering"
— Shi et al., 2024
Paper: https://arxiv.org/abs/2409.05907
Code:  https://github.com/IBM/activation-steering

Hook-based defense — no weight modification.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFUSAL_DIR = os.path.join(_ROOT, "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)

from pipeline.submodules.generate_directions import get_mean_activations


# ---------------------------------------------------------------------------
# Vector extraction via PCA on contrastive pairs
# ---------------------------------------------------------------------------

def _pca_direction(activations_pos: torch.Tensor,
                   activations_neg: torch.Tensor) -> torch.Tensor:
    """
    Extract the principal component of (pos - neg) activation differences.

    Parameters
    ----------
    activations_pos : (n, d_model) — activations for the "positive" class
    activations_neg : (n, d_model) — activations for the "negative" class

    Returns
    -------
    (d_model,) unit vector — first principal component of the differences
    """
    diffs = (activations_pos - activations_neg).float()  # (n, d)
    # Centre
    diffs = diffs - diffs.mean(dim=0, keepdim=True)
    # SVD — take the first right singular vector
    try:
        _, _, Vt = torch.linalg.svd(diffs, full_matrices=False)
        direction = Vt[0]
    except Exception:
        # Fallback to mean diff if SVD fails
        direction = diffs.mean(dim=0)

    direction = direction / (direction.norm() + 1e-8)

    # Ensure direction points from neg → pos (flip if needed)
    mean_diff = (activations_pos - activations_neg).float().mean(dim=0)
    if (direction @ mean_diff).item() < 0:
        direction = -direction

    return direction  # (d_model,)


def extract_cast_vectors(
    model,
    tokenizer,
    tokenize_fn,
    block_modules,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    positions: List[int] = [-1],
    batch_size: int = 16,
    condition_layers: Optional[List[int]] = None,
    behavior_layers: Optional[List[int]] = None,
) -> Dict:
    """
    Extract per-layer condition and behavior vectors.

    Both use PCA on the harmful/harmless activation difference; in the CAST
    paper the condition and behavior vectors can differ, but for refusal control
    it is standard to use the same contrastive set for both.

    Parameters
    ----------
    condition_layers : layers where condition check is applied
                       (default: middle third of the network)
    behavior_layers  : layers where behavior steering is applied
                       (default: same as condition_layers)

    Returns
    -------
    dict with:
        ``condition_vectors`` — {layer_idx: (d_model,) tensor}
        ``behavior_vectors``  — {layer_idx: (d_model,) tensor}
        ``condition_layers``  — list of layer indices
        ``behavior_layers``   — list of layer indices
    """
    n_layers = len(block_modules)
    device   = next(model.parameters()).device

    # Default layer ranges: middle third for condition, same for behavior
    if condition_layers is None:
        start = n_layers // 3
        end   = 2 * n_layers // 3
        condition_layers = list(range(start, end))
    if behavior_layers is None:
        behavior_layers = list(condition_layers)

    all_layers = sorted(set(condition_layers) | set(behavior_layers))

    print(f"[CAST] Extracting activations for {len(harmful_prompts)} harmful "
          f"and {len(harmless_prompts)} harmless prompts …")

    mean_harmful  = get_mean_activations(
        model, tokenizer, harmful_prompts, tokenize_fn,
        block_modules, batch_size=batch_size, positions=positions,
    )  # (n_positions, n_layers, d_model)
    mean_harmless = get_mean_activations(
        model, tokenizer, harmless_prompts, tokenize_fn,
        block_modules, batch_size=batch_size, positions=positions,
    )

    # Use last-token position
    pos_acts = mean_harmful[-1].float().to(device)   # (n_layers, d_model)
    neg_acts = mean_harmless[-1].float().to(device)

    # Extract per-layer vectors
    # We use mean diff directly as the direction (sufficient for our use-case).
    # Full CAST uses per-sample activations for PCA; we approximate with the
    # layer-wise mean difference since we only have means from get_mean_activations.
    # For more accurate PCA, a future extension could collect full activation matrices.
    diffs = pos_acts - neg_acts  # (n_layers, d_model)
    directions = F.normalize(diffs, dim=-1)  # (n_layers, d_model)

    condition_vectors = {ell: directions[ell] for ell in condition_layers}
    behavior_vectors  = {ell: directions[ell] for ell in behavior_layers}

    print(f"[CAST] Condition layers : {condition_layers}")
    print(f"[CAST] Behavior  layers : {behavior_layers}")

    return {
        "condition_vectors": condition_vectors,
        "behavior_vectors":  behavior_vectors,
        "condition_layers":  condition_layers,
        "behavior_layers":   behavior_layers,
        "directions":        directions,
    }


# ---------------------------------------------------------------------------
# Hook builder
# ---------------------------------------------------------------------------

def _make_cast_hook(
    condition_vector: torch.Tensor,
    behavior_vector: torch.Tensor,
    strength: float,
    threshold: float,
    preserve_norm: bool,
):
    """
    Return a forward pre-hook implementing CAST conditional steering.

    For each token in the batch:
      similarity = cos(hidden_state, condition_vector)
      if similarity > threshold:
          h = h + strength * behavior_vector
          if preserve_norm: h = h * ||h_original|| / ||h||
    """
    cond = condition_vector.float()
    behav = behavior_vector.float()

    def hook(module, input):
        if isinstance(input, tuple):
            x = input[0].float()      # (batch, seq, d_model)
        else:
            x = input.float()

        x_norm   = F.normalize(x, dim=-1)                       # (B, T, d)
        cond_n   = F.normalize(cond.unsqueeze(0).unsqueeze(0), dim=-1)
        sim      = (x_norm * cond_n).sum(dim=-1, keepdim=True)  # (B, T, 1)

        mask = (sim > threshold).float()                         # (B, T, 1)
        if mask.any():
            orig_norm = x.norm(dim=-1, keepdim=True)
            x = x + mask * strength * behav
            if preserve_norm:
                new_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                x = x * (orig_norm / new_norm)

        out = x.to(input[0].dtype if isinstance(input, tuple) else input.dtype)
        if isinstance(input, tuple):
            return (out,) + input[1:]
        return out

    return hook


# ---------------------------------------------------------------------------
# Main: build defense hooks
# ---------------------------------------------------------------------------

def apply_cast(
    model,
    tokenizer,
    tokenize_fn,
    block_modules,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    behavior_strength: float = 1.5,
    condition_threshold: float = 0.02,
    preserve_norm: bool = True,
    condition_layers: Optional[List[int]] = None,
    behavior_layers: Optional[List[int]] = None,
    positions: List[int] = [-1],
    batch_size: int = 16,
    artifact_dir: Optional[str] = None,
) -> Dict:
    """
    Build the CAST defense: extract vectors and return inference-time hooks.

    The returned hooks detect harmful input (via condition vector similarity)
    and steer the hidden state toward the refusal behavior (via behavior vector
    addition) whenever the condition is met.

    Parameters
    ----------
    behavior_strength    : multiplier for behavior vector addition (default 1.5)
    condition_threshold  : cosine similarity threshold for triggering steering
                           (lower = more aggressive; default 0.02 per paper)
    preserve_norm        : re-scale hidden state after steering (default True)
    condition_layers     : layers where similarity check is performed
    behavior_layers      : layers where behavior vector is added
    positions            : token positions used for direction extraction

    Returns
    -------
    dict with:
        ``fwd_pre_hooks`` — list of (module, hook_fn) pairs
        ``fwd_hooks``     — [] (CAST uses pre-hooks only)
        ``vectors``       — extracted condition/behavior vectors
        ``n_hooks``       — number of hooks registered
    """
    print("[CAST] Extracting condition and behavior vectors …")
    vector_info = extract_cast_vectors(
        model, tokenizer, tokenize_fn, block_modules,
        harmful_prompts, harmless_prompts,
        positions=positions, batch_size=batch_size,
        condition_layers=condition_layers,
        behavior_layers=behavior_layers,
    )

    cond_vecs  = vector_info["condition_vectors"]
    behav_vecs = vector_info["behavior_vectors"]
    all_layers = sorted(set(cond_vecs.keys()) | set(behav_vecs.keys()))

    fwd_pre_hooks = []

    for ell in all_layers:
        cv = cond_vecs.get(ell)
        bv = behav_vecs.get(ell)
        if cv is None or bv is None:
            continue
        hook = _make_cast_hook(
            condition_vector=cv,
            behavior_vector=bv,
            strength=behavior_strength,
            threshold=condition_threshold,
            preserve_norm=preserve_norm,
        )
        fwd_pre_hooks.append((block_modules[ell], hook))

    print(f"[CAST] Registered {len(fwd_pre_hooks)} hooks on "
          f"{len(all_layers)} layers "
          f"(threshold={condition_threshold}, strength={behavior_strength}).")

    result = {
        "fwd_pre_hooks":        fwd_pre_hooks,
        "fwd_hooks":            [],
        "vectors":              vector_info,
        "n_hooks":              len(fwd_pre_hooks),
        "behavior_strength":    behavior_strength,
        "condition_threshold":  condition_threshold,
        "preserve_norm":        preserve_norm,
        "condition_layers":     vector_info["condition_layers"],
        "behavior_layers":      vector_info["behavior_layers"],
    }

    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        summary = {
            "n_hooks":             result["n_hooks"],
            "condition_layers":    result["condition_layers"],
            "behavior_layers":     result["behavior_layers"],
            "behavior_strength":   behavior_strength,
            "condition_threshold": condition_threshold,
            "preserve_norm":       preserve_norm,
        }
        with open(os.path.join(artifact_dir, "cast_defense.json"), "w") as f:
            json.dump(summary, f, indent=2)

    return result
