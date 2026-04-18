"""
AlphaSteer defense — Hu et al., ICLR 2026.

Learns per-layer steering matrices that redirect harmful inputs toward refusal
while preserving benign inputs via a null-space constraint.

Algorithm (per layer ℓ):
  1. Collect hidden states H_harmful (n_h, d) and H_benign (n_b, d).
  2. Compute null-space projection P_ℓ from H_benign via SVD so that
     P_ℓ @ h ≈ 0 for benign inputs.
  3. Compute refusal direction r_ℓ (difference-in-means, unit-normalised).
  4. Solve regularised least-squares for Δ_ℓ:
       H_harmful @ P_ℓ @ Δ_ℓ ≈ r_ℓ
     closed form: Δ_ℓ = (A^T A + λ P_ℓ^T P_ℓ)^{-1} A^T r_ℓ
     where A = H_harmful.
  5. Steering matrix M_ℓ = P_ℓ @ Δ_ℓ  (shape d × d)

At inference (via forward pre-hook on each block):
  last_h = hidden_states[:, last_token_idx, :]     # (B, d)
  delta  = last_h @ M_ℓ * strength                 # (B, d)
  hidden_states += delta.unsqueeze(1)               # broadcast to all tokens

Reference: "AlphaSteer: Learning Refusal Steering with Principled Null-Space
Constraint" — Hu et al., ICLR 2026
https://github.com/AlphaLab-USTC/AlphaSteer
"""

import json
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFUSAL_DIR = os.path.join(_ROOT, "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)

from pipeline.submodules.generate_directions import get_mean_activations


# ---------------------------------------------------------------------------
# Step 1: Collect full activation matrices (not just means)
# ---------------------------------------------------------------------------

def _collect_activations(
    model,
    tokenizer,
    prompts: List[str],
    tokenize_fn,
    block_modules,
    batch_size: int = 16,
    position: int = -1,
) -> torch.Tensor:
    """
    Return activations at a single token position for all layers.

    Returns
    -------
    (n_prompts, n_layers, d_model) float32 on CPU
    """
    n_layers = len(block_modules)
    device = next(model.parameters()).device
    all_acts = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenize_fn(instructions=batch).to(device)

        handles = []
        layer_acts = [None] * n_layers

        def _make_hook(idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                # Grab the requested position
                if position == -1:
                    # last non-padding token per sample
                    attn_mask = inputs.get("attention_mask")
                    if attn_mask is not None:
                        last_idx = attn_mask.sum(dim=1).long() - 1  # (B,)
                        h_pos = h[torch.arange(h.size(0), device=h.device), last_idx]
                    else:
                        h_pos = h[:, -1, :]
                else:
                    h_pos = h[:, position, :]
                layer_acts[idx] = h_pos.detach().float().cpu()
            return hook

        for idx, mod in enumerate(block_modules):
            handles.append(mod.register_forward_hook(_make_hook(idx)))

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        # Stack: (B, n_layers, d)
        stacked = torch.stack(layer_acts, dim=1)
        all_acts.append(stacked)

    return torch.cat(all_acts, dim=0)  # (n_prompts, n_layers, d_model)


# ---------------------------------------------------------------------------
# Step 2: Compute null-space projection matrix from benign activations
# ---------------------------------------------------------------------------

def _null_space_projection(H: torch.Tensor, null_ratio: float = 0.5) -> torch.Tensor:
    """
    Compute projection matrix onto the null space of H.

    H : (n, d) — benign hidden states at one layer
    null_ratio : fraction of singular vectors to keep as null-space basis

    Returns
    -------
    P : (d, d) — projection onto null space of H
    """
    H = H.float()
    n, d = H.shape
    # Gram matrix A = H^T H  (d × d)
    A = H.T @ H  # (d, d)
    try:
        _, S, Vt = torch.linalg.svd(A, full_matrices=True)
    except Exception:
        return torch.eye(d)

    # Null space = right singular vectors with near-zero singular values
    threshold = S.max() * 1e-6
    null_mask = S < threshold
    # Also include the smallest (null_ratio * d) vectors to capture soft null-space
    k_null = max(1, int(null_ratio * d))
    # Sort by ascending singular value and take the bottom k
    sorted_idx = S.argsort()
    null_idx = sorted_idx[:k_null]

    Q = Vt[null_idx].T  # (d, k_null) — null space basis
    P = Q @ Q.T          # (d, d)    — projection matrix
    return P


# ---------------------------------------------------------------------------
# Step 3: Compute per-layer steering matrices
# ---------------------------------------------------------------------------

def compute_steering_matrices(
    H_harmful: torch.Tensor,
    H_benign: torch.Tensor,
    refusal_directions: torch.Tensor,
    null_ratio: float = 0.5,
    lambda_reg: float = 10.0,
    target_layers: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor]:
    """
    Compute per-layer AlphaSteer steering matrices.

    Parameters
    ----------
    H_harmful          : (n_h, n_layers, d) harmful activations
    H_benign           : (n_b, n_layers, d) benign activations
    refusal_directions : (n_layers, d) unit refusal direction per layer
    null_ratio         : fraction of singular vectors defining the null space
    lambda_reg         : regularisation coefficient (default 10.0)
    target_layers      : which layers to compute matrices for (default: all)

    Returns
    -------
    dict mapping layer_idx → steering_matrix (d × d)
    """
    n_layers = H_harmful.shape[1]
    d = H_harmful.shape[2]

    if target_layers is None:
        target_layers = list(range(n_layers))

    matrices = {}
    for ell in target_layers:
        X_h = H_harmful[:, ell, :].float()  # (n_h, d)
        X_b = H_benign[:, ell, :].float()   # (n_b, d)
        r   = refusal_directions[ell].float()  # (d,)
        r   = r / (r.norm() + 1e-8)

        # Null-space projection from benign activations
        P = _null_space_projection(X_b, null_ratio=null_ratio)  # (d, d)

        # Regularised least squares: find Δ such that X_h @ P @ Δ ≈ r
        # Normal equations: (A^T A + λ P^T P) Δ = A^T r
        # where A = X_h
        A = X_h  # (n_h, d)
        AtA = A.T @ A            # (d, d)
        PtP = P.T @ P            # (d, d)
        lhs = AtA + lambda_reg * PtP  # (d, d)
        rhs = A.T @ (A @ r)  # (d,)

        try:
            delta = torch.linalg.solve(lhs, rhs)  # (d,)
        except Exception:
            delta = torch.linalg.lstsq(lhs, rhs.unsqueeze(-1)).solution.squeeze(-1)

        M = P @ torch.outer(delta, r)  # (d, d)

        matrices[ell] = M.float()
        print(f"  [AlphaSteer] Layer {ell}: ||M|| = {M.norm().item():.4f}")

    return matrices


# ---------------------------------------------------------------------------
# Step 4: Build forward hooks
# ---------------------------------------------------------------------------

def _make_alphasteer_hook(
    steering_matrix: torch.Tensor,
    strength: float,
):
    """
    Return a forward pre-hook that applies AlphaSteer steering.

    At each layer, for the last valid token position:
        steering_vector = last_hidden @ M * strength
        hidden_states  += steering_vector  (broadcast to all sequence positions)

    Only activates during prefill (seq_len > 1) to avoid repeated steering
    during token-by-token decoding.
    """
    M = steering_matrix.float()

    def hook(module, input):
        if isinstance(input, tuple):
            x = input[0]
        else:
            x = input

        # Only steer during prefill
        if x.shape[1] <= 1:
            return input if isinstance(input, tuple) else x

        dtype = x.dtype
        x_f = x.float()  # (B, T, d)
        B, T, d = x_f.shape

        # Use last token of each sequence
        last_h = x_f[:, -1, :]          # (B, d)
        M_dev  = M.to(x_f.device)

        sv = (last_h @ M_dev) * strength  # (B, d)
        x_f = x_f + sv.unsqueeze(1)       # broadcast to (B, T, d)

        out = x_f.to(dtype)
        if isinstance(input, tuple):
            return (out,) + input[1:]
        return out

    return hook


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_alphasteer(
    model,
    tokenizer,
    tokenize_fn,
    block_modules,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    refusal_direction: torch.Tensor,
    mean_diffs: Optional[torch.Tensor] = None,
    target_layers: Optional[List[int]] = None,
    null_ratio: float = 0.5,
    lambda_reg: float = 10.0,
    strength: float = 0.4,
    batch_size: int = 16,
    artifact_dir: Optional[str] = None,
) -> Dict:
    """
    Compute AlphaSteer steering matrices and return inference-time hooks.

    Parameters
    ----------
    refusal_direction : (d_model,) global refusal direction (from pipeline)
    mean_diffs        : (n_pos, n_layers, d_model) optional per-layer diffs;
                        if None, the global direction is broadcast to all layers
    target_layers     : layers to steer (default: middle third, layers 8-19 for 32L)
    null_ratio        : fraction of benign singular space to project out
    lambda_reg        : regularisation coefficient
    strength          : steering intensity (default 0.4, range −0.5 to +0.5)
    batch_size        : batch size for activation collection

    Returns
    -------
    dict with:
        ``fwd_pre_hooks``    — list of (module, hook_fn) pairs
        ``fwd_hooks``        — []
        ``steering_matrices``— {layer_idx: (d, d) tensor}
        ``target_layers``    — list of layer indices
        ``strength``         — steering strength used
    """
    n_layers = len(block_modules)
    device   = next(model.parameters()).device

    if target_layers is None:
        # Default: layers 8–19 for a 32-layer model (following AlphaSteer paper)
        start = max(0, n_layers // 4)
        end   = min(n_layers, 3 * n_layers // 5)
        target_layers = list(range(start, end))

    print(f"[AlphaSteer] Target layers: {target_layers}")
    print(f"[AlphaSteer] Collecting activations for {len(harmful_prompts)} harmful prompts …")
    H_harmful = _collect_activations(
        model, tokenizer, harmful_prompts, tokenize_fn,
        block_modules, batch_size=batch_size,
    )

    print(f"[AlphaSteer] Collecting activations for {len(harmless_prompts)} benign prompts …")
    H_benign = _collect_activations(
        model, tokenizer, harmless_prompts, tokenize_fn,
        block_modules, batch_size=batch_size,
    )

    # Build per-layer refusal directions
    if mean_diffs is not None:
        # mean_diffs: (n_pos, n_layers, d) — use last position
        layer_dirs = mean_diffs[-1].float()  # (n_layers, d)
        layer_dirs = F.normalize(layer_dirs, dim=-1)
    else:
        # Broadcast global direction to all layers
        r = refusal_direction.float()
        r = r / (r.norm() + 1e-8)
        layer_dirs = r.unsqueeze(0).expand(n_layers, -1)  # (n_layers, d)

    print(f"[AlphaSteer] Computing steering matrices …")
    matrices = compute_steering_matrices(
        H_harmful=H_harmful,
        H_benign=H_benign,
        refusal_directions=layer_dirs,
        null_ratio=null_ratio,
        lambda_reg=lambda_reg,
        target_layers=target_layers,
    )

    # Register hooks
    fwd_pre_hooks = []
    for ell in target_layers:
        if ell not in matrices:
            continue
        M = matrices[ell].to(device)
        hook = _make_alphasteer_hook(M, strength=strength)
        fwd_pre_hooks.append((block_modules[ell], hook))

    print(f"[AlphaSteer] Registered {len(fwd_pre_hooks)} hooks "
          f"(strength={strength}, null_ratio={null_ratio}, λ={lambda_reg}).")

    result = {
        "fwd_pre_hooks":     fwd_pre_hooks,
        "fwd_hooks":         [],
        "steering_matrices": matrices,
        "target_layers":     target_layers,
        "strength":          strength,
        "null_ratio":        null_ratio,
        "lambda_reg":        lambda_reg,
    }

    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        summary = {
            "target_layers": target_layers,
            "strength":      strength,
            "null_ratio":    null_ratio,
            "lambda_reg":    lambda_reg,
            "n_hooks":       len(fwd_pre_hooks),
        }
        with open(os.path.join(artifact_dir, "alphasteer_defense.json"), "w") as f:
            json.dump(summary, f, indent=2)

    return result
