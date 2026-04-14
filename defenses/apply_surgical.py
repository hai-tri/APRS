"""
Surgical defense — Zheng et al. 2024 (ICLR 2025).

Mitigates false refusal by identifying the refusal direction via
difference-in-means and ablating it from the residual stream at inference
time.  Unlike our obfuscation approach (which bakes changes into weights),
Surgical applies the intervention as forward-pass hooks — the model weights
are unchanged.

The method is a calibration defense: it reduces over-refusal on benign
prompts while ideally preserving refusal on genuinely harmful ones.  It does
NOT hide the direction from an attacker, making it vulnerable to Arditi-style
white-box attacks (but useful as a baseline).

Reference: "Surgical, Cheap, and Flexible: Mitigating False Refusal in
Language Models via Single Vector Ablation" — Zheng et al., 2024
Paper: https://arxiv.org/abs/2410.03415
Code:  https://github.com/mainlp/False-Refusal-Mitigation

Hook-based defense — no weight modification.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFUSAL_DIR = os.path.join(_ROOT, "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)

from pipeline.submodules.generate_directions import get_mean_activations


# ---------------------------------------------------------------------------
# Direction extraction
# ---------------------------------------------------------------------------

def extract_surgical_directions(
    model,
    tokenizer,
    tokenize_fn,
    block_modules,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    positions: List[int] = [-1],
    batch_size: int = 16,
    top_n: int = 1,
    kl_threshold: Optional[float] = None,
) -> Dict:
    """
    Compute per-layer refusal directions via difference-in-means.

    Parameters
    ----------
    top_n         : number of layers to select (by magnitude of mean diff)
    kl_threshold  : if set, discard directions where KL divergence of the
                    ablated model exceeds this value (not yet implemented —
                    left as a future extension)

    Returns
    -------
    dict with:
        ``directions``     — (n_layers, d_model) tensor, unit-normed per layer
        ``magnitudes``     — (n_layers,) norms of the raw mean diffs
        ``selected_layers``— list of layer indices chosen by top_n
        ``mean_diffs``     — (n_positions, n_layers, d_model) raw diffs
    """
    device = next(model.parameters()).device

    mean_harmful  = get_mean_activations(
        model, tokenizer, harmful_prompts, tokenize_fn,
        block_modules, batch_size=batch_size, positions=positions,
    )
    mean_harmless = get_mean_activations(
        model, tokenizer, harmless_prompts, tokenize_fn,
        block_modules, batch_size=batch_size, positions=positions,
    )

    # mean_diffs: (n_positions, n_layers, d_model)
    mean_diffs = (mean_harmful - mean_harmless).float()

    # Use last-token position (index -1) for direction selection
    diffs_last = mean_diffs[-1]  # (n_layers, d_model)
    magnitudes = diffs_last.norm(dim=-1)  # (n_layers,)

    # Unit-normalise each layer's direction
    directions = diffs_last / (magnitudes.unsqueeze(-1) + 1e-8)  # (n_layers, d_model)

    # Select top-n layers by magnitude
    selected_layers = magnitudes.topk(min(top_n, len(magnitudes))).indices.tolist()

    print(f"[Surgical] Top-{top_n} layers by refusal direction magnitude: "
          f"{selected_layers}")
    for ell in selected_layers:
        print(f"  layer {ell}: ||diff|| = {magnitudes[ell].item():.4f}")

    return {
        "directions":      directions.to(device),
        "magnitudes":      magnitudes.to(device),
        "selected_layers": selected_layers,
        "mean_diffs":      mean_diffs.to(device),
    }


# ---------------------------------------------------------------------------
# Hook builders
# ---------------------------------------------------------------------------

def _make_ablation_hook(direction: torch.Tensor, coeff: float = 1.0):
    """
    Return a pre-hook that projects out `direction` from the input hidden state.

    x' = x - coeff * (x · d̂) * d̂
    """
    d = direction.float()
    d = d / (d.norm() + 1e-8)

    def hook(module, input):
        if isinstance(input, tuple):
            x = input[0].float()
            x = x - coeff * (x @ d) .unsqueeze(-1) * d
            return (x.to(input[0].dtype),) + input[1:]
        else:
            x = input.float()
            x = x - coeff * (x @ d).unsqueeze(-1) * d
            return x.to(input.dtype)

    return hook


def _make_actadd_hook(direction: torch.Tensor, coeff: float = 1.0):
    """
    Return a pre-hook that adds `coeff * direction` to the hidden state.

    Used to steer the model *away* from harmful intent at a specific layer.
    """
    d = direction.float()

    def hook(module, input):
        if isinstance(input, tuple):
            x = input[0].float()
            x = x + coeff * d
            return (x.to(input[0].dtype),) + input[1:]
        else:
            x = input.float()
            x = x + coeff * d
            return x.to(input.dtype)

    return hook


# ---------------------------------------------------------------------------
# Main: build defense hooks
# ---------------------------------------------------------------------------

def apply_surgical(
    model,
    tokenizer,
    tokenize_fn,
    block_modules,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    ablation_coeff: float = 1.0,
    actadd_coeff: float = 0.0,
    top_n: int = 1,
    apply_all_layers: bool = True,
    positions: List[int] = [-1],
    batch_size: int = 16,
    artifact_dir: Optional[str] = None,
) -> Dict:
    """
    Build the Surgical defense: extract refusal directions and return hooks.

    The returned hooks should be registered on the model for all subsequent
    forward passes (both evaluation and attack evaluation).

    Parameters
    ----------
    ablation_coeff   : strength of direction ablation (1.0 = full projection)
    actadd_coeff     : optional activation addition coefficient (0.0 = off)
    top_n            : layers to apply ablation to (by magnitude); ignored if
                       apply_all_layers=True
    apply_all_layers : if True, apply ablation at every layer (recommended)
    positions        : token positions used for direction extraction

    Returns
    -------
    dict with:
        ``fwd_pre_hooks``  — list of (module, hook_fn) for torch hook registration
        ``directions``     — extracted per-layer directions
        ``selected_layers``— layers where hooks are attached
        ``n_hooks``        — total hooks registered
    """
    print("[Surgical] Extracting refusal directions …")
    direction_info = extract_surgical_directions(
        model, tokenizer, tokenize_fn, block_modules,
        harmful_prompts, harmless_prompts,
        positions=positions, batch_size=batch_size,
        top_n=top_n if not apply_all_layers else len(block_modules),
    )

    directions      = direction_info["directions"]   # (n_layers, d_model)
    selected_layers = (
        list(range(len(block_modules)))
        if apply_all_layers
        else direction_info["selected_layers"]
    )

    fwd_pre_hooks = []

    for ell in selected_layers:
        d = directions[ell]

        # Ablation hook: project out refusal direction
        if ablation_coeff != 0.0:
            fwd_pre_hooks.append(
                (block_modules[ell], _make_ablation_hook(d, coeff=ablation_coeff))
            )

        # Optional activation addition (steer toward harmlessness)
        if actadd_coeff != 0.0:
            # Add the *negated* direction to push activations toward harmless
            fwd_pre_hooks.append(
                (block_modules[ell], _make_actadd_hook(-d, coeff=actadd_coeff))
            )

    print(f"[Surgical] Registered {len(fwd_pre_hooks)} hooks on "
          f"{len(selected_layers)} layers.")

    result = {
        "fwd_pre_hooks":   fwd_pre_hooks,
        "fwd_hooks":       [],
        "directions":      directions,
        "selected_layers": selected_layers,
        "n_hooks":         len(fwd_pre_hooks),
        "ablation_coeff":  ablation_coeff,
        "actadd_coeff":    actadd_coeff,
        "apply_all_layers": apply_all_layers,
    }

    if artifact_dir:
        import json, os
        os.makedirs(artifact_dir, exist_ok=True)
        summary = {
            "n_hooks":         result["n_hooks"],
            "selected_layers": result["selected_layers"],
            "ablation_coeff":  ablation_coeff,
            "actadd_coeff":    actadd_coeff,
            "apply_all_layers": apply_all_layers,
            "magnitudes": direction_info["magnitudes"].cpu().tolist(),
        }
        with open(os.path.join(artifact_dir, "surgical_defense.json"), "w") as f:
            json.dump(summary, f, indent=2)

    return result
