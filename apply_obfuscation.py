"""
Representational Obfuscation of Safety Features via Per-Layer Rank-One Aliasing
===============================================================================

Core defense implementation (Steps 0–6 from the technical specification).

The algorithm:
  0. Extract the refusal direction and identify pertinent layers.
  1. Generate per-layer random alias vectors (zero-mean, ε-scaled).
  2. Patch writer matrices (W_O, W_down) at pertinent layers so their output
     at the calibration operating point becomes a random alias instead of the
     original value.
  3. Track cumulative pollution (net change to the residual stream).
  4. Patch reader matrices (Q, K, V, gate, up) at every downstream layer so
     they compensate for the LayerNorm-transformed pollution and produce the
     same activations as the undefended model.
  5. (Implicit — handled by Step 4 for attention output projections at
     non-pertinent layers.)
  6. Patch the unembedding matrix (LM head) against cumulative pollution
     through the final LayerNorm.

No training is required.  This is a one-time, post-hoc weight edit.
"""

import torch
import os
import json
from typing import Dict, List, Optional, Set, Tuple

from obfuscation_config import ObfuscationConfig
from obfuscation_utils import (
    ModelComponents,
    collect_calibration_activations,
    generate_random_alias,
    rank_one_update,
)


# ------------------------------------------------------------------
# Pertinent layer selection
# ------------------------------------------------------------------

def select_pertinent_layers(
    mean_diffs: torch.Tensor,
    pos: int,
    k: Optional[int] = None,
    ablation_scores: Optional[Dict] = None,
    ablation_score_threshold: float = 0.0,
) -> List[int]:
    """
    Select layers that are causally responsible for refusal.

    The preferred method uses ablation refusal scores from ``select_direction``
    — the causal test of how much refusal drops when a direction is removed at
    each layer.  Layers where ablating the direction pushes the refusal score
    below ``ablation_score_threshold`` are selected (lower score = stronger
    causal effect, negative = model no longer refuses).

    Falls back to top-k by raw magnitude when ablation scores are unavailable.

    Parameters
    ----------
    mean_diffs : (n_positions, n_layers, d_model)
        From ``generate_directions``.
    pos : int
        Token position selected by ``select_direction``.
    k : int or None
        Manual override: take exactly the top-k layers by ablation score
        (or magnitude if scores unavailable).  Useful for ablation sweeps.
    ablation_scores : dict or None
        List of dicts from ``select_direction/direction_evaluations.json``.
        Each entry has ``layer``, ``position``, ``refusal_score``.
    ablation_score_threshold : float
        Select layers whose ablation refusal score is below this value.
        Default 0.0 — only layers where ablation causes the model to stop
        refusing (score < 0) are selected.
    """
    n_layers = mean_diffs.shape[1]

    if ablation_scores is not None:
        # Use causal ablation scores — the ground truth for which layers matter
        # Aggregate across token positions: take the minimum score per layer
        # (most causally impactful position at that layer)
        per_layer_best = {}
        for entry in ablation_scores:
            ell = entry["layer"]
            score = entry["refusal_score"]
            if ell not in per_layer_best or score < per_layer_best[ell]:
                per_layer_best[ell] = score

        if k is not None:
            sorted_layers = sorted(per_layer_best.items(), key=lambda x: x[1])
            selected = sorted(ell for ell, _ in sorted_layers[:k])
        else:
            selected = sorted(
                ell for ell, score in per_layer_best.items()
                if score < ablation_score_threshold
            )
    else:
        # Fallback: top-k by raw magnitude
        magnitudes = mean_diffs[pos].norm(dim=-1)
        if k is not None:
            _, top_indices = magnitudes.topk(min(k, n_layers))
            selected = sorted(top_indices.tolist())
        else:
            # Top-5 by default when no ablation scores available
            _, top_indices = magnitudes.topk(min(5, n_layers))
            selected = sorted(top_indices.tolist())

    return selected


# ------------------------------------------------------------------
# Core defense
# ------------------------------------------------------------------

def apply_obfuscation(
    model,
    tokenize_fn,
    harmful_prompts: List[str],
    mean_diffs: torch.Tensor,
    selected_pos: int,
    selected_layer: int,
    direction: torch.Tensor,
    cfg: ObfuscationConfig = ObfuscationConfig(),
    ablation_scores: Optional[List[Dict]] = None,
    explicit_layers: Optional[List[int]] = None,
    harmless_prompts: Optional[List[str]] = None,
    harmless_ratio: float = 0.5,
) -> Dict:
    """
    Apply the representational-obfuscation defense to *model* **in-place**.

    Returns a diagnostics dict with keys:
        * ``pertinent_layers``  — list of patched layer indices
        * ``z_sum_norm``        — L2 norm of total residual-stream pollution
        * ``num_writers_patched`` / ``num_readers_patched``
    """
    device = next(model.parameters()).device
    components = ModelComponents(model)
    d = components.d_model
    num_layers = components.num_layers

    # ----------------------------------------------------------------
    # Step 0: Identify pertinent layers
    # ----------------------------------------------------------------
    if explicit_layers is not None:
        pertinent_layers: Set[int] = set(explicit_layers)
    else:
        pertinent_layers: Set[int] = set(
            select_pertinent_layers(
                mean_diffs, selected_pos,
                k=cfg.num_pertinent_layers,
                ablation_scores=ablation_scores,
            )
        )
    print(f"[obfuscation] Pertinent layers ({len(pertinent_layers)}): "
          f"{sorted(pertinent_layers)}")

    # ----------------------------------------------------------------
    # Collect calibration activations — mixed harmful + harmless
    # ----------------------------------------------------------------
    print("[obfuscation] Collecting calibration activations …")
    activations = collect_calibration_activations(
        model=model,
        components=components,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        harmless_ratio=harmless_ratio,
        tokenize_fn=tokenize_fn,
        num_prompts=cfg.num_calibration_prompts,
    )

    # ----------------------------------------------------------------
    # Step 1: Generate per-layer random aliases
    # ----------------------------------------------------------------
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    # Normalised refusal direction(s)
    r_hat_global = (direction / direction.norm()).float().to(device)

    # Per-layer direction dict: maps layer_idx → unit vector
    if cfg.per_layer_direction:
        r_hat_map: Dict[int, torch.Tensor] = {}
        for ell in sorted(pertinent_layers):
            layer_dir = mean_diffs[selected_pos, ell].float().to(device)
            norm = layer_dir.norm()
            if norm > 1e-8:
                r_hat_map[ell] = layer_dir / norm
            else:
                # Fallback to global if layer direction is degenerate
                r_hat_map[ell] = r_hat_global
        print(f"[obfuscation] Using per-layer refusal directions")
    else:
        r_hat_map = {ell: r_hat_global for ell in sorted(pertinent_layers)}

    # Keep a global r_hat reference for compatibility
    r_hat = r_hat_global

    mode = cfg.projection_mode
    assert mode in ("hadamard", "binary", "mask", "scalar_projection", "full"), \
        f"Unknown projection_mode: {mode}"

    # Per-layer noise containers
    attn_noise: Dict[int, object] = {}
    mlp_noise: Dict[int, object] = {}

    def _rademacher(n):
        """Generate a random ±1 vector of length n."""
        return (torch.randint(0, 2, (n,), device=device, generator=generator).float() * 2 - 1)

    def _binary_mask(n):
        """Generate a random {0, 1} vector of length n."""
        return torch.randint(0, 2, (n,), device=device, generator=generator).float()

    if mode == "hadamard":
        # Hadamard mode: r̂ℓ ⊙ ξ where ξ ~ N(0, ε²I).
        for ell in sorted(pertinent_layers):
            r_l = r_hat_map[ell]
            attn_noise[ell] = r_l * torch.randn(d, device=device, generator=generator) * cfg.epsilon
            if cfg.separate_attn_mlp_aliases:
                mlp_noise[ell] = r_l * torch.randn(d, device=device, generator=generator) * cfg.epsilon
            else:
                mlp_noise[ell] = attn_noise[ell].clone()
    elif mode == "binary":
        # Binary mode: r̂ℓ ⊙ s where s_i ∈ {-1, +1} (Rademacher).
        for ell in sorted(pertinent_layers):
            r_l = r_hat_map[ell]
            attn_noise[ell] = r_l * _rademacher(d)
            if cfg.separate_attn_mlp_aliases:
                mlp_noise[ell] = r_l * _rademacher(d)
            else:
                mlp_noise[ell] = attn_noise[ell].clone()
    elif mode == "mask":
        # Mask mode: r̂ℓ ⊙ m where m_i ∈ {0, 1}.
        for ell in sorted(pertinent_layers):
            r_l = r_hat_map[ell]
            attn_noise[ell] = r_l * _binary_mask(d)
            if cfg.separate_attn_mlp_aliases:
                mlp_noise[ell] = r_l * _binary_mask(d)
            else:
                mlp_noise[ell] = attn_noise[ell].clone()
    elif mode == "scalar_projection":
        # Surgical mode: η · r̂ (single random scalar per writer).
        for ell in sorted(pertinent_layers):
            eta = torch.randn(1, device=device, generator=generator).item() * cfg.epsilon
            attn_noise[ell] = eta
            if cfg.separate_attn_mlp_aliases:
                mlp_noise[ell] = torch.randn(1, device=device, generator=generator).item() * cfg.epsilon
            else:
                mlp_noise[ell] = eta
    else:  # full
        # Full-alias mode: complete d-dimensional random vector.
        for ell in sorted(pertinent_layers):
            attn_noise[ell] = generate_random_alias(d, cfg.epsilon, device, generator)
            if cfg.separate_attn_mlp_aliases:
                mlp_noise[ell] = generate_random_alias(d, cfg.epsilon, device, generator)
            else:
                mlp_noise[ell] = attn_noise[ell].clone()

    # ----------------------------------------------------------------
    # Steps 2–5: Patch writers and readers, tracking pollution
    # ----------------------------------------------------------------
    cumulative_pollution = torch.zeros(d, device=device, dtype=torch.float32)
    num_writers_patched = 0
    num_readers_patched = 0

    for ell in range(num_layers):

        # ==============================================================
        # ATTENTION SUBLAYER
        # ==============================================================

        # --- Step 4a: Patch attention readers (Q, K, V) if pollution exists ---
        if cumulative_pollution.norm() > 1e-8:
            ln_module = components.get_attn_layernorm(ell)
            x_clean = activations[f"layer_{ell}_attn_ln_input"].float()
            x_polluted = x_clean + cumulative_pollution

            # Empirical LayerNorm correction: compute LN on both clean and
            # polluted residual streams, then patch readers so that
            # W_read @ LN(x_polluted) == W_read @ LN(x_clean).
            with torch.no_grad():
                ln_clean = ln_module(
                    x_clean.unsqueeze(0).unsqueeze(0)
                ).squeeze().float()
                ln_polluted = ln_module(
                    x_polluted.unsqueeze(0).unsqueeze(0)
                ).squeeze().float()

            for proj_name, proj_module in components.get_attn_reader_projs(ell):
                W = proj_module.weight.data
                W_new = rank_one_update(W, ln_polluted, W.float() @ ln_clean)
                proj_module.weight.data = W_new
                num_readers_patched += 1

        # --- Step 2a: Patch W_O (attention writer) if pertinent ---
        if ell in pertinent_layers and cfg.patch_writers in ("both", "attn_only"):
            o_proj = components.get_attn_output_proj(ell)
            x_attn = activations[f"layer_{ell}_attn_o_input"].float()
            r_l = r_hat_map[ell]

            current_output = o_proj.weight.data.float() @ x_attn

            if mode in ("hadamard", "binary", "mask"):
                proj_scalar = (current_output @ r_l).item()
                target = current_output - proj_scalar * r_l + attn_noise[ell]
            elif mode == "scalar_projection":
                proj_scalar = (current_output @ r_l).item()
                target = current_output - proj_scalar * r_l + attn_noise[ell] * r_l
            else:
                target = attn_noise[ell].float()

            o_proj.weight.data = rank_one_update(o_proj.weight.data, x_attn, target)

            # Track net change to the residual stream
            cumulative_pollution = cumulative_pollution + (target - current_output)
            num_writers_patched += 1

        # ==============================================================
        # MLP SUBLAYER
        # ==============================================================

        # --- Step 4b: Patch MLP readers (gate, up) if pollution exists ---
        if cumulative_pollution.norm() > 1e-8:
            ln_module = components.get_mlp_layernorm(ell)
            x_clean = activations[f"layer_{ell}_mlp_ln_input"].float()
            x_polluted = x_clean + cumulative_pollution

            with torch.no_grad():
                ln_clean = ln_module(
                    x_clean.unsqueeze(0).unsqueeze(0)
                ).squeeze().float()
                ln_polluted = ln_module(
                    x_polluted.unsqueeze(0).unsqueeze(0)
                ).squeeze().float()

            for proj_name, proj_module in components.get_mlp_reader_projs(ell):
                W = proj_module.weight.data
                W_new = rank_one_update(W, ln_polluted, W.float() @ ln_clean)
                proj_module.weight.data = W_new
                num_readers_patched += 1

        # --- Step 2b: Patch W_down (MLP writer) if pertinent ---
        if ell in pertinent_layers and cfg.patch_writers in ("both", "mlp_only"):
            down_proj = components.get_mlp_output_proj(ell)
            x_mlp = activations[f"layer_{ell}_mlp_down_input"].float()
            r_l = r_hat_map[ell]

            current_output = down_proj.weight.data.float() @ x_mlp

            if mode in ("hadamard", "binary", "mask"):
                proj_scalar = (current_output @ r_l).item()
                target = current_output - proj_scalar * r_l + mlp_noise[ell]
            elif mode == "scalar_projection":
                proj_scalar = (current_output @ r_l).item()
                target = current_output - proj_scalar * r_l + mlp_noise[ell] * r_l
            else:
                target = mlp_noise[ell].float()

            down_proj.weight.data = rank_one_update(
                down_proj.weight.data, x_mlp, target
            )

            cumulative_pollution = cumulative_pollution + (target - current_output)
            num_writers_patched += 1

    # ----------------------------------------------------------------
    # Step 6: Patch the unembedding matrix (LM head)
    # ----------------------------------------------------------------
    if cumulative_pollution.norm() > 1e-8:
        z_sum = cumulative_pollution
        final_ln = components.final_norm
        x_clean_final = activations["final_ln_input"].float()
        x_polluted_final = x_clean_final + z_sum

        with torch.no_grad():
            ln_clean = final_ln(
                x_clean_final.unsqueeze(0).unsqueeze(0)
            ).squeeze().float()
            ln_polluted = final_ln(
                x_polluted_final.unsqueeze(0).unsqueeze(0)
            ).squeeze().float()

        W_unembed = components.lm_head.weight.data
        target_logits = W_unembed.float() @ ln_clean
        W_unembed_new = rank_one_update(W_unembed, ln_polluted, target_logits)
        components.lm_head.weight.data = W_unembed_new

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------
    z_sum_norm = cumulative_pollution.norm().item()
    print(f"[obfuscation] Defense applied ({mode} mode).")
    print(f"  Writers patched : {num_writers_patched}")
    print(f"  Readers patched : {num_readers_patched}")
    print(f"  z_sum norm      : {z_sum_norm:.4f}")

    return {
        "pertinent_layers": sorted(pertinent_layers),
        "z_sum_norm": z_sum_norm,
        "num_writers_patched": num_writers_patched,
        "num_readers_patched": num_readers_patched,
    }


# ------------------------------------------------------------------
# Convenience: load artifacts produced by the existing pipeline and
# apply the defense in one call.
# ------------------------------------------------------------------

def apply_obfuscation_from_artifacts(
    model,
    tokenize_fn,
    harmful_prompts: List[str],
    artifact_dir: str,
    cfg: ObfuscationConfig = ObfuscationConfig(),
) -> Dict:
    """
    Load ``direction.pt``, ``mean_diffs.pt``, and ``direction_metadata.json``
    from *artifact_dir* (produced by the upstream ``generate_directions`` /
    ``select_direction`` pipeline), then apply the defense.
    """
    direction = torch.load(
        os.path.join(artifact_dir, "direction.pt"), map_location="cpu"
    )
    mean_diffs = torch.load(
        os.path.join(artifact_dir, "generate_directions", "mean_diffs.pt"),
        map_location="cpu",
    )
    with open(os.path.join(artifact_dir, "direction_metadata.json")) as f:
        meta = json.load(f)

    return apply_obfuscation(
        model=model,
        tokenize_fn=tokenize_fn,
        harmful_prompts=harmful_prompts,
        mean_diffs=mean_diffs,
        selected_pos=meta["pos"],
        selected_layer=meta["layer"],
        direction=direction,
        cfg=cfg,
    )
