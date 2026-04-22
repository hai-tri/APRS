import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class ModelComponents:
    """
    Architecture-agnostic accessor for sublayer components needed by the
    obfuscation defense.  Supports Llama-2/3, Gemma, and Mistral (all share
    the HuggingFace ``model.model.layers`` layout).  Qwen-1 uses a different
    layout and is detected separately.
    """

    def __init__(self, model):
        self.model = model
        self.num_layers = model.config.num_hidden_layers
        self.d_model = model.config.hidden_size
        self._detect_architecture()

    # ------------------------------------------------------------------
    # Architecture detection
    # ------------------------------------------------------------------
    def _detect_architecture(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.arch = "llama"
            self.layers = self.model.model.layers
            self.final_norm = self.model.model.norm
            self.lm_head = self.model.lm_head
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            self.arch = "qwen"
            self.layers = self.model.transformer.h
            self.final_norm = self.model.transformer.ln_f
            self.lm_head = self.model.lm_head
        else:
            raise ValueError(
                f"Unsupported architecture: {type(self.model).__name__}. "
                "Expected Llama/Gemma/Mistral or Qwen-style model."
            )

    # ------------------------------------------------------------------
    # Sublayer accessors
    # ------------------------------------------------------------------
    def get_attn_layernorm(self, layer_idx: int) -> nn.Module:
        layer = self.layers[layer_idx]
        if self.arch == "llama":
            return layer.input_layernorm
        return layer.ln_1

    def get_mlp_layernorm(self, layer_idx: int) -> nn.Module:
        layer = self.layers[layer_idx]
        if self.arch == "llama":
            return layer.post_attention_layernorm
        return layer.ln_2

    def get_attn_reader_projs(self, layer_idx: int) -> List[Tuple[str, nn.Module]]:
        """Return [(name, Linear)] for attention reader projections (Q, K, V)."""
        layer = self.layers[layer_idx]
        if self.arch == "llama":
            attn = layer.self_attn
            return [
                ("q_proj", attn.q_proj),
                ("k_proj", attn.k_proj),
                ("v_proj", attn.v_proj),
            ]
        raise NotImplementedError(f"Attention reader patching not implemented for {self.arch}")

    def get_attn_output_proj(self, layer_idx: int) -> nn.Module:
        layer = self.layers[layer_idx]
        if self.arch == "llama":
            return layer.self_attn.o_proj
        return layer.attn.c_proj

    def get_mlp_reader_projs(self, layer_idx: int) -> List[Tuple[str, nn.Module]]:
        """Return [(name, Linear)] for MLP reader projections (gate, up)."""
        layer = self.layers[layer_idx]
        if self.arch == "llama":
            return [
                ("gate_proj", layer.mlp.gate_proj),
                ("up_proj", layer.mlp.up_proj),
            ]
        return [
            ("w1", layer.mlp.w1),
            ("w2", layer.mlp.w2),
        ]

    def get_mlp_output_proj(self, layer_idx: int) -> nn.Module:
        layer = self.layers[layer_idx]
        if self.arch == "llama":
            return layer.mlp.down_proj
        return layer.mlp.c_proj


# ----------------------------------------------------------------------
# Rank-one update
# ----------------------------------------------------------------------

def rank_one_update(
    W: torch.Tensor,
    x: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a rank-one update to *W* so that ``W_new @ x == target`` (exactly
    for this specific *x*).

    Formula:  W_new = W + outer(target - W @ x, x) / (x @ x)

    All arithmetic is done in float32 for numerical stability; the result is
    cast back to *W*'s original dtype.
    """
    orig_dtype = W.dtype
    W_f = W.float()
    x_f = x.float()
    target_f = target.float()

    current = W_f @ x_f
    delta = target_f - current
    W_f = W_f + torch.outer(delta, x_f) / (x_f @ x_f)
    return W_f.to(orig_dtype)


# ----------------------------------------------------------------------
# Calibration activation collection
# ----------------------------------------------------------------------

def collect_calibration_activations(
    model: nn.Module,
    components: ModelComponents,
    harmful_prompts: List[str],
    tokenize_fn,
    num_prompts: int = 32,
    harmless_prompts: Optional[List[str]] = None,
    harmless_ratio: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Run forward passes and return **averaged** sublayer activations at the
    last token position, used to anchor the rank-one weight edits.

    By default uses a 50/50 mix of harmful and harmless prompts so the
    calibration point sits near the model's average operating point rather
    than purely in the refusal regime.  This reduces reader-patch drift for
    harmless inputs while keeping the writer patches well-anchored.

    Parameters
    ----------
    harmful_prompts : prompts that trigger refusal.
    harmless_prompts : benign prompts.  If None, calibrate on harmful only
        (original behaviour).
    harmless_ratio : fraction of the calibration batch to draw from
        harmless_prompts (default 0.5 = equal mix).

    Collected keys per layer ``ℓ``:
        * ``layer_{ℓ}_attn_ln_input``  — residual stream before attention
        * ``layer_{ℓ}_attn_o_input``   — attention output before W_O projection
        * ``layer_{ℓ}_mlp_ln_input``   — residual stream before MLP
        * ``layer_{ℓ}_mlp_down_input`` — MLP hidden state before W_down projection

    Plus ``final_ln_input`` — residual stream at the end of the model.
    """
    device = next(model.parameters()).device
    num_layers = components.num_layers

    # Build the mixed prompt list
    if harmless_prompts is not None and harmless_ratio > 0:
        n_harmless = int(num_prompts * harmless_ratio)
        n_harmful = num_prompts - n_harmless
        prompts = (
            harmful_prompts[:n_harmful] +
            harmless_prompts[:n_harmless]
        )
    else:
        prompts = harmful_prompts[:num_prompts]

    # Accumulator (float64 for numerical stability when averaging)
    accum: Dict[str, torch.Tensor] = {}
    count = 0

    def _make_hook(key: str):
        """Create a hook that accumulates the module's *input* at the last-token position."""
        def hook_fn(module, inp, output):
            x = inp[0] if isinstance(inp, tuple) else inp
            # x: (batch=1, seq_len, d)
            vec = x[0, -1, :].detach().float().cpu()
            if key not in accum:
                accum[key] = torch.zeros_like(vec, dtype=torch.float64)
            accum[key] += vec.double()
        return hook_fn

    # Register hooks on every layer
    hooks: List[torch.utils.hooks.RemovableHook] = []
    for ell in range(num_layers):
        layer = components.layers[ell]

        # Attention LayerNorm input  →  residual stream before attention
        hooks.append(
            components.get_attn_layernorm(ell).register_forward_hook(
                _make_hook(f"layer_{ell}_attn_ln_input")
            )
        )
        # W_O input  →  attention mechanism output before projection
        hooks.append(
            components.get_attn_output_proj(ell).register_forward_hook(
                _make_hook(f"layer_{ell}_attn_o_input")
            )
        )
        # MLP LayerNorm input  →  residual stream after attention
        hooks.append(
            components.get_mlp_layernorm(ell).register_forward_hook(
                _make_hook(f"layer_{ell}_mlp_ln_input")
            )
        )
        # W_down input  →  MLP hidden state (gate * up)
        hooks.append(
            components.get_mlp_output_proj(ell).register_forward_hook(
                _make_hook(f"layer_{ell}_mlp_down_input")
            )
        )

    # Final LayerNorm input  →  residual stream at end
    hooks.append(
        components.final_norm.register_forward_hook(
            _make_hook("final_ln_input")
        )
    )

    # Forward passes (one prompt at a time to avoid padding artefacts)
    n_harmful_used = len(harmful_prompts[:num_prompts if harmless_prompts is None else num_prompts - int(num_prompts * harmless_ratio)])
    n_harmless_used = len(prompts) - n_harmful_used
    print(f"[calibration] {len(prompts)} prompts: {n_harmful_used} harmful + {n_harmless_used} harmless")
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Calibration fwd passes"):
            inputs = tokenize_fn(instructions=[prompt])
            model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
            )
            count += 1

    # Remove hooks
    for h in hooks:
        h.remove()

    # Average, cast to float32 (MPS doesn't support float64), move to model device
    for key in accum:
        accum[key] = (accum[key] / count).float().to(device)

    return accum


# ----------------------------------------------------------------------
# Random alias generation
# ----------------------------------------------------------------------

def generate_random_alias(
    d: int,
    epsilon: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Generate a zero-mean random vector scaled to standard deviation *epsilon*.

    Zero-mean minimises LayerNorm centering distortion; small std keeps
    the pollution negligible relative to typical residual-stream magnitudes.
    """
    z = torch.randn(d, device=device, generator=generator)
    z = z - z.mean()
    z = z / z.std() * epsilon
    return z
