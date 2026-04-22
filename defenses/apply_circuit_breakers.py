"""
Circuit Breakers defense — Zou et al. 2024.

Fine-tunes the model with LoRA using a two-component objective:

  1. **RR loss** (Representation Rerouting): applied to harmful prompts.
     Minimises the cosine similarity between the LoRA model's hidden states
     and the frozen base model's hidden states on the same harmful input.
     ReLU-clipped so the loss is zero once the representations are already
     orthogonal.  Drives harmful activations to a completely different region
     of representation space.

     L_RR = mean_over_layers(mean_over_tokens(
                ReLU( norm(h_lora) · norm(h_base) )))

  2. **Retention loss**: applied to harmless prompts.
     Minimises the L2 distance between LoRA and base hidden states, so the
     model's behaviour on benign inputs is preserved.

     L_retain = mean_over_layers(||h_lora - h_base||₂)

  Combined with a linear schedule that transitions from emphasising RR early
  in training to emphasising retention later:

     loss = cb_coeff(t) × L_RR  +  retain_coeff(t) × L_retain

  where cb_coeff(t) decreases linearly from cb_coeff_max → 0 and
  retain_coeff(t) increases linearly from 0 → retain_coeff_max.

The LoRA adapter can be merged into the base weights after training so that
no PEFT dependency is required at inference time.

Reference: "Improving Alignment and Robustness with Circuit Breakers"
— Zou et al., 2024
Paper: https://arxiv.org/abs/2406.04313
Code:  https://github.com/GraySwanAI/circuit-breakers

Requires: peft >= 0.10  (pip install peft)
Weight-modifying defense — modifies model weights via LoRA fine-tuning.
"""

import json
import math
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFUSAL_DIR = os.path.join(_ROOT, "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def _get_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: List[int],
) -> torch.Tensor:
    """
    Return stacked hidden states for the requested layers.

    Returns (len(layers), batch, seq_len, d_model).
    Padding positions (attention_mask == 0) are zeroed out.
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # hidden_states: tuple of (n_layers+1,) each (batch, seq, d)
    # index 0 = embedding layer, index l+1 = after layer l
    hs = torch.stack([out.hidden_states[l + 1] for l in layers], dim=0)
    # Zero out padding
    mask = attention_mask.unsqueeze(0).unsqueeze(-1).float()  # (1, B, T, 1)
    hs = hs * mask
    return hs  # (n_layers, B, T, d)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _rr_loss(
    h_lora: torch.Tensor,
    h_base: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Representation Rerouting loss.

    Minimise ReLU(cos_sim(h_lora, h_base)) — drives representations apart.
    h_lora, h_base: (n_layers, B, T, d)
    """
    h_l = F.normalize(h_lora.float(), dim=-1)
    h_b = F.normalize(h_base.float(), dim=-1)
    cos_sim = (h_l * h_b).sum(dim=-1)          # (n_layers, B, T)
    loss = F.relu(cos_sim)                      # clip at 0

    # Mask padding tokens
    mask = attention_mask.unsqueeze(0).float()  # (1, B, T)
    n_tokens = mask.sum().clamp(min=1)
    return (loss * mask).sum() / n_tokens


def _retention_loss(
    h_lora: torch.Tensor,
    h_base: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Retention loss: L2 distance between LoRA and base hidden states.

    h_lora, h_base: (n_layers, B, T, d)
    """
    diff = (h_lora.float() - h_base.float()).norm(dim=-1)  # (n_layers, B, T)
    mask = attention_mask.unsqueeze(0).float()
    n_tokens = mask.sum().clamp(min=1)
    return (diff * mask).sum() / n_tokens


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def apply_circuit_breakers(
    model,
    tokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    cb_layers: Optional[List[int]] = None,
    max_steps: int = 150,
    lr: float = 1e-4,
    batch_size: int = 4,
    max_length: int = 256,
    cb_coeff_max: float = 1.0,
    retain_coeff_max: float = 1.0,
    merge_weights: bool = True,
    seed: int = 42,
    artifact_dir: Optional[str] = None,
) -> Dict:
    """
    Apply the Circuit Breakers defense via LoRA fine-tuning.

    Parameters
    ----------
    model / tokenizer    : target causal LM (modified in-place)
    harmful_prompts      : list of harmful instructions for RR loss
    harmless_prompts     : list of benign instructions for retention loss
    lora_rank            : LoRA rank (default 16)
    lora_alpha           : LoRA alpha (default 16)
    lora_dropout         : LoRA dropout (default 0.05)
    lora_target_modules  : projection modules to apply LoRA to; default is
                           all 7 attention + MLP projections
    cb_layers            : layer indices for RR/retention loss; default is
                           the first two-thirds of the network
    max_steps            : training steps (default 150)
    lr                   : AdamW learning rate (default 1e-4)
    batch_size           : prompts per gradient step (default 4)
    max_length           : tokenisation max length (default 256)
    cb_coeff_max         : peak RR loss coefficient (default 1.0)
    retain_coeff_max     : peak retention loss coefficient (default 1.0)
    merge_weights        : if True, merge LoRA into base weights after training
                           so no PEFT dependency is needed at inference
    seed                 : random seed
    artifact_dir         : if set, saves training log and config here

    Returns
    -------
    dict with:
        ``fwd_pre_hooks`` — [] (weight-modifying defense; no hooks needed)
        ``fwd_hooks``     — []
        ``train_log``     — list of per-step loss dicts
        ``lora_config``   — LoRA hyperparameters used
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError(
            "Circuit Breakers requires the 'peft' library. "
            "Install it with: pip install peft"
        )

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers

    # Default: circuit-break the first 2/3 of layers
    if cb_layers is None:
        cb_layers = list(range(int(n_layers * 2 / 3)))

    # Default LoRA target modules (all projection layers)
    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    print(f"[CB] Applying LoRA (rank={lora_rank}, alpha={lora_alpha}) …")
    print(f"[CB] CB layers: {cb_layers[:5]}…{cb_layers[-1]} "
          f"({len(cb_layers)} layers)")

    # Apply LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )

    # Tokenise prompts
    def _tokenise(prompts: List[str]) -> Dict[str, torch.Tensor]:
        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_list
        ]
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    # Build batches by cycling through both sets
    import random
    rng = random.Random(seed)

    harmful_idx  = list(range(len(harmful_prompts)))
    harmless_idx = list(range(len(harmless_prompts)))
    rng.shuffle(harmful_idx)
    rng.shuffle(harmless_idx)

    train_log = []

    print(f"[CB] Training for {max_steps} steps …")

    for step in range(max_steps):
        model.train()
        progress = step / max(max_steps - 1, 1)

        # Linear schedule
        cb_coeff     = cb_coeff_max     * (1.0 - progress)
        retain_coeff = retain_coeff_max * progress

        # Sample a batch of harmful prompts
        h_indices = [harmful_idx[(step * batch_size + i) % len(harmful_idx)]
                     for i in range(batch_size)]
        r_indices = [harmless_idx[(step * batch_size + i) % len(harmless_idx)]
                     for i in range(batch_size)]

        h_batch = _tokenise([harmful_prompts[i]  for i in h_indices])
        r_batch = _tokenise([harmless_prompts[i] for i in r_indices])

        total_loss = torch.tensor(0.0, device=device)
        rr_loss_val = 0.0
        ret_loss_val = 0.0

        # ── RR loss on harmful prompts ──────────────────────────────────
        if cb_coeff > 0:
            # LoRA forward
            h_lora = _get_hidden_states(
                model, h_batch["input_ids"], h_batch["attention_mask"],
                cb_layers,
            )
            # Base (frozen) forward — disable LoRA adapters
            with model.disable_adapter():
                with torch.no_grad():
                    h_base = _get_hidden_states(
                        model, h_batch["input_ids"], h_batch["attention_mask"],
                        cb_layers,
                    )
            loss_rr = _rr_loss(h_lora, h_base.detach(), h_batch["attention_mask"])
            total_loss = total_loss + cb_coeff * loss_rr
            rr_loss_val = loss_rr.item()

        # ── Retention loss on harmless prompts ──────────────────────────
        if retain_coeff > 0:
            r_lora = _get_hidden_states(
                model, r_batch["input_ids"], r_batch["attention_mask"],
                cb_layers,
            )
            with model.disable_adapter():
                with torch.no_grad():
                    r_base = _get_hidden_states(
                        model, r_batch["input_ids"], r_batch["attention_mask"],
                        cb_layers,
                    )
            loss_ret = _retention_loss(r_lora, r_base.detach(), r_batch["attention_mask"])
            total_loss = total_loss + retain_coeff * loss_ret
            ret_loss_val = loss_ret.item()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        log_entry = {
            "step":         step + 1,
            "total_loss":   total_loss.item(),
            "rr_loss":      rr_loss_val,
            "retain_loss":  ret_loss_val,
            "cb_coeff":     cb_coeff,
            "retain_coeff": retain_coeff,
        }
        train_log.append(log_entry)

        if (step + 1) % 25 == 0 or step == 0:
            print(f"[CB] step {step+1:3d}/{max_steps}  "
                  f"total={total_loss.item():.4f}  "
                  f"rr={rr_loss_val:.4f}  "
                  f"retain={ret_loss_val:.4f}  "
                  f"(cb_c={cb_coeff:.2f}, ret_c={retain_coeff:.2f})")

    # ── Merge LoRA into base weights ────────────────────────────────────
    if merge_weights:
        print("[CB] Merging LoRA weights into base model …")
        model = model.merge_and_unload()
        print("[CB] Merge complete.")

    print(f"[CB] Training done. Final loss: {train_log[-1]['total_loss']:.4f}")

    lora_config_dict = {
        "rank":            lora_rank,
        "alpha":           lora_alpha,
        "dropout":         lora_dropout,
        "target_modules":  lora_target_modules,
        "cb_layers":       cb_layers,
        "max_steps":       max_steps,
        "lr":              lr,
        "batch_size":      batch_size,
        "cb_coeff_max":    cb_coeff_max,
        "retain_coeff_max": retain_coeff_max,
        "merge_weights":   merge_weights,
    }

    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "circuit_breakers_log.json"), "w") as f:
            json.dump({"config": lora_config_dict, "train_log": train_log}, f, indent=2)

    return {
        "fwd_pre_hooks": [],
        "fwd_hooks":     [],
        "model":         model,
        "train_log":     train_log,
        "lora_config":   lora_config_dict,
    }
