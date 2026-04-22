from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ObfuscationConfig:
    # Random alias vector standard deviation. Controls pollution magnitude.
    # Smaller = less LayerNorm distortion, larger = harder for attacker to filter out.
    epsilon: float = 0.1

    # Number of pertinent layers to patch.
    # None = auto-detect from data (layers whose refusal magnitude > 20% of peak).
    # Set to an int to override with a fixed top-k (useful for ablation sweeps).
    num_pertinent_layers: Optional[int] = None

    # Number of harmful prompts used for calibration forward passes.
    # More = better generalization of rank-one patches.
    num_calibration_prompts: int = 32

    # Whether to use separate random alias vectors for W_O and W_down at each layer.
    # True = more obfuscation (2 random vectors per layer), False = shared alias.
    separate_attn_mlp_aliases: bool = True

    # Random seed for reproducible alias generation.
    seed: int = 42

    # Which writer matrices to patch at pertinent layers.
    # Options: "both", "attn_only", "mlp_only"
    patch_writers: str = "both"

    # Projection mode for writer patches.
    # "hadamard"  = replace r̂ component with r̂ ⊙ ξ, ξ ~ N(0, ε²I).
    #               Element-wise Gaussian noise weighted by r̂.
    # "binary"    = replace r̂ component with r̂ ⊙ s, s_i ∈ {-1, +1}.
    #               Rademacher sign flips. Magnitude = ||r̂|| = 1. ε not used.
    # "mask"      = replace r̂ component with r̂ ⊙ m, m_i ∈ {0, 1}.
    #               Random dropout mask. Magnitude ≈ ||r̂||/√2. ε not used.
    # "scalar_projection"  = replace r̂ component with η · r̂ (single random scalar).
    #               Pollution purely along r̂.
    # "full"      = replace entire writer output with a full random alias vector
    #               (original behaviour).  Highest pollution, worst utility.
    projection_mode: str = "hadamard"

    # Use per-layer refusal directions instead of the global r̂.
    # When True, each writer patch uses mean_diffs[pos, layer] as r̂ for that
    # layer, so a different direction is obfuscated at each layer.  The attacker
    # must then recover a different direction per layer rather than one global one.
    per_layer_direction: bool = False
