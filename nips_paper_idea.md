# Technical Specification: Representational Obfuscation of Safety Features via Per-Layer Rank-One Aliasing

## Overview

This document describes a defense against "abliteration" — a technique that removes safety alignment from LLMs by identifying and projecting out a single "refusal direction" from the model's residual stream. Our defense makes the refusal direction undetectable in the residual stream by replacing it with per-layer random aliases, then patching all reader/writer matrices so the model's behavior is functionally identical.

**Key insight**: Abliteration works because refusal is encoded as a single, globally coherent direction r̂ in the residual stream. If we replace r̂ with different random vectors at each layer — while ensuring each layer can still locally decode the refusal signal — an attacker running difference-in-means on the residual stream finds random noise instead of r̂.

**No training is required.** This is a one-time weight edit applied post-hoc.

---

## Background: How Abliteration Works

1. Collect hidden states from the residual stream for harmful prompts H and benign prompts B
2. Compute mean activations at each layer ℓ and token position p:
   - μ_ℓ,p = (1/|H|) Σ_{x∈H} h_ℓ,p(x)
   - ν_ℓ,p = (1/|B|) Σ_{x∈B} h_ℓ,p(x)
3. Compute candidate refusal directions: r_ℓ,p = μ_ℓ,p - ν_ℓ,p
4. Select the best refusal direction r̂ (the one that maximally reduces refusal when removed)
5. Project r̂ out of all output matrices: W̃_out^(ℓ) = (I - r̂ r̂ᵀ) W_out^(ℓ)

This works because r̂ is the same vector everywhere in the residual stream.

---

## Architecture Context: Transformer Block Structure

A standard decoder-only transformer (e.g., Llama) has this structure per layer:

```
residual_stream
    │
    ├──→ LayerNorm ──→ Multi-Head Attention ──→ add back to residual
    │         │              │
    │     (RMSNorm)     Q = W_Q · x_norm
    │                   K = W_K · x_norm
    │                   V = W_V · x_norm
    │                   attn_out = W_O · attention(Q,K,V)
    │
    ├──→ LayerNorm ──→ FFN (MLP) ──→ add back to residual
    │         │           │
    │     (RMSNorm)   gate = W_gate · x_norm
    │                 up = W_up · x_norm
    │                 down_out = W_down · (gate * up)
    │
    └──→ continues to next layer
```

**Matrices that WRITE to the residual stream** (output matrices):
- W_O (attention output projection) — per layer
- W_down (MLP down projection) — per layer

**Matrices that READ from the residual stream** (input matrices, but they read from LayerNorm output):
- W_Q, W_K, W_V (attention input projections) — per layer, per head
- W_gate, W_up (MLP input projections) — per layer

**Final reader:**
- W_unembed (language model head / unembedding matrix)

**LayerNorm/RMSNorm** sits between the residual stream and every reader. This is critical because it's nonlinear.

---

## The Defense: Step by Step

### Step 0: Extract the Refusal Direction and Identify Pertinent Layers

Use the standard Arditi et al. (2024) method:

1. Collect a dataset of harmful prompts H and benign prompts B (use existing datasets like AdvBench, CatQA, etc.)
2. Run forward passes, collect residual stream activations at each layer
3. Compute the mean difference r_ℓ = mean(h_ℓ(harmful)) - mean(h_ℓ(benign)) at each layer
4. Identify the "pertinent layers" — the layers where ||r_ℓ|| is largest or where ablating r_ℓ causes the greatest drop in refusal rate
5. Select the global refusal direction r̂ (normalized) — typically the direction at the most pertinent layer, or the first principal component across layers

**Output of this step:**
- r̂ ∈ R^d (unit vector, the refusal direction)
- A set of pertinent layer indices L = {ℓ_1, ℓ_2, ..., ℓ_k}
- Clean residual stream activations x_clean at each layer (from a representative calibration prompt — use the mean across several prompts for robustness)

**CRITICAL: Calibration activations must come from refusal-triggering prompts.** The rank-one patches are calibrated to specific activation vectors. To patch the matrices at the points where refusal actually occurs, the calibration forward pass must use harmful prompts that trigger the refusal pathway. Collect residual stream values at each sublayer boundary during these refusal-triggering forward passes. These are the activations where the refusal direction is actually being written/read, so these are the activations the patches need to be exact for. Using benign prompts for calibration would patch the matrices at irrelevant operating points.

### Step 1: Generate Per-Layer Random Alias Vectors

For each pertinent layer ℓ ∈ L, generate a zero-mean random vector scaled to small variance ε:

**Random vector properties (LayerNorm mitigation):** The random alias vector should be **zero-mean** with a **small variance** controlled by a hyperparameter ε ∈ (0, 1):
- **Zero-mean**: LayerNorm/RMSNorm subtracts the mean. A zero-mean random vector doesn't shift LayerNorm's centering, minimizing nonlinear distortion to downstream layers.
- **Small variance (ε)**: The random vector's standard deviation is set to a small ε. This keeps the pollution added to the residual stream small relative to the typical activation magnitudes, which means LayerNorm's variance normalization (division by σ) is minimally perturbed. Smaller ε = less LayerNorm distortion but potentially easier for an attacker to filter out. This is a key hyperparameter to ablate.

```python
epsilon = 0.1  # hyperparameter: std of the random alias vectors, ε ∈ (0, 1)

z = torch.randn(d, device=model.device)
z = z - z.mean()  # force zero mean
z = z / z.std() * epsilon  # scale to small variance
random_aliases[ell] = z
```

### Step 2: Patch Output Matrices (Writers)

For each pertinent layer ℓ, modify the output matrix (W_O for attention sublayers, W_down for MLP sublayers) so that when its penultimate activation x would produce a component along r̂, it instead writes z_ℓ.

**Rank-one update formula:**

```
W̃_out^(ℓ) = W_out^(ℓ) + (v_random - W_out^(ℓ) · x) · xᵀ / (xᵀ · x)
```

Where:
- W_out^(ℓ) is the original output matrix at layer ℓ
- x is the penultimate activation (the input to this output matrix) from the calibration forward pass
- v_random = z_ℓ (the random alias for this layer, appropriately scaled)
- The update ensures: W̃_out^(ℓ) · x = z_ℓ (exactly, for this specific x)

**In code:**

```python
def patch_output_matrix(W_out, x, z_ell):
    """
    Modify W_out so that W_out @ x produces z_ell instead of the original output.
    
    Args:
        W_out: (d_out, d_in) output weight matrix
        x: (d_in,) penultimate activation vector (input to this matrix)
        z_ell: (d_out,) target random alias vector
    
    Returns:
        Modified W_out
    """
    current_output = W_out @ x  # what it currently produces
    # Rank-one update: change the output for input x from current_output to z_ell
    delta = z_ell - current_output  # (d_out,)
    W_out_new = W_out + torch.outer(delta, x) / (x @ x)
    return W_out_new
```

**Apply this to BOTH output matrices at pertinent layers:**
- The attention output projection W_O at layer ℓ
- The MLP down projection W_down at layer ℓ

**Note:** You need the actual activations from a calibration forward pass to get x for each matrix. Run a forward pass on a calibration prompt (or average of several), saving intermediate activations at each relevant point.

### Step 3: Compute the Polluted Residual Stream

After patching the output matrices, the residual stream is now "polluted" — it contains z_ℓ vectors instead of r̂ components. Track this pollution:

```python
# After patching layer ℓ's output matrix:
# The residual stream at layer ℓ+1 now has z_ℓ added instead of the original r̂ component
# x_polluted = x_clean - r_component + z_ℓ  (approximately)

# More precisely: run a forward pass through the patched model to get actual polluted activations
# OR track the cumulative difference:
pollution = {}  # layer -> cumulative random vectors added
pollution[first_pertinent_layer] = z_{first_layer}
pollution[next_layer] = pollution[prev_layer] + z_{current_layer}  # accumulates
```

### Step 4: Patch Input Matrices (Readers) via LayerNorm Empirical Correction

This is the key step. Every matrix that reads from the residual stream does so through a LayerNorm. We need each reader to interpret the polluted stream the same way it would interpret the clean stream.

**The empirical LayerNorm correction:**

Instead of analytically decomposing LayerNorm (which is nonlinear), just compute the difference empirically:

```python
def compute_layernorm_correction(layernorm, x_clean, x_polluted):
    """
    Compute the artifact introduced by LayerNorm when the residual stream is polluted.
    
    Args:
        layernorm: the LayerNorm/RMSNorm module
        x_clean: (d,) clean residual stream activation
        x_polluted: (d,) polluted residual stream activation (with random vectors added)
    
    Returns:
        v: (d,) the correction vector. The next reader needs to be fortified against this.
    """
    ln_clean = layernorm(x_clean)
    ln_polluted = layernorm(x_polluted)
    v = ln_clean - ln_polluted  # what the reader "misses" due to pollution
    return v
```

**Then patch each reader matrix so it maps the polluted LayerNorm output to the same result as the clean one:**

For each reader matrix W_read (which is W_Q, W_K, W_V for attention, or W_gate, W_up for MLP):

```python
def patch_reader_matrix(W_read, ln_clean, ln_polluted):
    """
    Modify W_read so that W_read @ ln_polluted == W_read @ ln_clean.
    
    Args:
        W_read: (d_out, d_in) reader weight matrix
        ln_clean: (d_in,) LayerNorm output on clean residual stream
        ln_polluted: (d_in,) LayerNorm output on polluted residual stream
    
    Returns:
        Modified W_read
    """
    target = W_read @ ln_clean     # what it should produce
    current = W_read @ ln_polluted  # what it currently produces with pollution
    delta = target - current        # (d_out,)
    
    W_read_new = W_read + torch.outer(delta, ln_polluted) / (ln_polluted @ ln_polluted)
    return W_read_new
```

**Apply this to ALL reader matrices at every layer DOWNSTREAM of each pertinent layer:**
- For attention sublayer at layer ℓ': patch W_Q^(ℓ'), W_K^(ℓ'), W_V^(ℓ') for all heads
- For MLP sublayer at layer ℓ': patch W_gate^(ℓ'), W_up^(ℓ')
- The pollution is cumulative, so downstream layers see the sum of all previous random vectors

### Step 5: Patch the Attention Output Projection (Attn → Residual)

The attention output projection W_O also writes back to the residual stream. If the attention sublayer at layer ℓ is pertinent, its W_O needs to be patched (done in Step 2). But even non-pertinent layers' W_O may need patching if the attention computation was affected by the pollution propagating through Q, K, V.

This is handled implicitly: if you patch the Q, K, V readers (Step 4), the attention computation produces the correct output, and W_O maps it back correctly. No additional W_O patch is needed for non-pertinent layers unless you want to add additional aliasing there too.

### Step 6: Patch the Unembedding Matrix (LM Head)

The LM head must be fortified against the **combined effect** of two things:

1. **The sum of every random vector** $z_i$ that patched "writing" matrices output to the residual stream (one per patched $W_O$ and $W_{\text{down}}$), minus the original outputs they replaced
2. **The LayerNorm residual** — the nonlinear distortion from the final LayerNorm processing the polluted stream instead of the clean stream

Let $z_{\text{sum}}$ be the cumulative net change to the residual stream, tracked during patching:

$$z_{\text{sum}} = \sum_{i \in \text{patched writers}} (z_i - W_{\text{out},i} \cdot x_i)$$

where $W_{\text{out},i} \cdot x_i$ is what writer $i$ originally produced (already part of $x_{\text{clean}}$) and $z_i$ is the random alias that replaced it. The polluted residual stream at the final layer is:

$$x_{\text{polluted}} = x_{\text{clean}} + z_{\text{sum}}$$

The final LayerNorm transforms this before the LM head reads it. We fortify the LM head so that it produces the same logits on the polluted LN output as it would on the clean LN output. This single rank-one update accounts for BOTH the random vectors AND the LayerNorm distortion in one shot:

$$\widetilde{W}_{\text{unembed}} = W_{\text{unembed}} + \frac{(W_{\text{unembed}} \cdot \text{LN}(x_{\text{clean}}) - W_{\text{unembed}} \cdot \text{LN}(x_{\text{polluted}})) \cdot \text{LN}(x_{\text{polluted}})^\top}{\text{LN}(x_{\text{polluted}})^\top \cdot \text{LN}(x_{\text{polluted}})}$$

```python
def patch_unembedding(W_unembed, final_layernorm, x_clean_final, z_sum):
    """
    Fortify the unembedding matrix against cumulative pollution + LayerNorm residual.
    
    The LM head is fortified against the COMBINED effect of:
    - z_sum: net change to residual stream from all patched writers
    - LayerNorm distortion: LN(x_clean + z_sum) ≠ LN(x_clean) + LN(z_sum)
    
    By computing LN(x_polluted) empirically, we handle both in one rank-one update.
    
    Args:
        W_unembed: (vocab_size, d) unembedding matrix
        final_layernorm: the final LayerNorm before the LM head
        x_clean_final: (d,) clean residual stream at final layer
        z_sum: (d,) cumulative net change = Σ(z_i - original_output_i)
    
    Returns:
        Modified W_unembed
    """
    x_polluted_final = x_clean_final + z_sum
    
    ln_clean = final_layernorm(x_clean_final)
    ln_polluted = final_layernorm(x_polluted_final)
    
    # The LM head sees ln_polluted but should produce the same logits as on ln_clean
    target = W_unembed @ ln_clean
    current = W_unembed @ ln_polluted
    delta = target - current  # (vocab_size,)
    
    W_unembed_new = W_unembed + torch.outer(delta, ln_polluted) / (ln_polluted @ ln_polluted)
    return W_unembed_new
```

The key insight: $z_{\text{sum}}$ is a known quantity — we constructed every $z_i$ ourselves and tracked the cumulative pollution. So the total effect on the LM head is fully deterministic and we correct for it exactly via a single empirical computation through the final LayerNorm.

---

## Complete Algorithm (Pseudocode)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def apply_defense(model, harmful_prompts, benign_prompts, num_calibration_prompts=32):
    """
    Apply representational obfuscation defense to a model.
    
    Args:
        model: HuggingFace model (e.g., LlamaForCausalLM)
        harmful_prompts: list of harmful prompt strings
        benign_prompts: list of benign prompt strings
        num_calibration_prompts: number of prompts to use for calibration
    """
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    d = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    
    # ================================================================
    # STEP 0: Extract refusal direction and identify pertinent layers
    # ================================================================
    
    # Collect residual stream activations for harmful and benign prompts
    harmful_activations = {}  # layer -> list of activation tensors
    benign_activations = {}
    
    for layer_idx in range(num_layers):
        harmful_activations[layer_idx] = []
        benign_activations[layer_idx] = []
    
    # Hook to capture residual stream activations
    def make_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            # For Llama-style models, the residual stream is the output of the layer
            # We want the hidden states at each layer's output
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Take the last token position
            storage[layer_idx].append(hidden[:, -1, :].detach().cpu())
        return hook_fn
    
    # Register hooks on each transformer layer
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(harmful_activations, layer_idx))
        hooks.append(h)
    
    # Forward pass on harmful prompts
    with torch.no_grad():
        for prompt in harmful_prompts[:num_calibration_prompts]:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            model(**inputs)
    
    # Remove hooks, re-register for benign
    for h in hooks:
        h.remove()
    
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(benign_activations, layer_idx))
        hooks.append(h)
    
    with torch.no_grad():
        for prompt in benign_prompts[:num_calibration_prompts]:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            model(**inputs)
    
    for h in hooks:
        h.remove()
    
    # Compute refusal direction per layer
    refusal_directions = {}
    refusal_magnitudes = {}
    for layer_idx in range(num_layers):
        mean_harmful = torch.stack(harmful_activations[layer_idx]).mean(dim=0).squeeze()
        mean_benign = torch.stack(benign_activations[layer_idx]).mean(dim=0).squeeze()
        r = mean_harmful - mean_benign
        refusal_magnitudes[layer_idx] = r.norm().item()
        refusal_directions[layer_idx] = r / r.norm()
    
    # Select global refusal direction (from the layer with highest magnitude)
    best_layer = max(refusal_magnitudes, key=refusal_magnitudes.get)
    r_hat = refusal_directions[best_layer].to(model.device)
    
    # Select pertinent layers (top-k by refusal magnitude)
    k = 10  # number of pertinent layers to patch (hyperparameter)
    sorted_layers = sorted(refusal_magnitudes.items(), key=lambda x: x[1], reverse=True)
    pertinent_layers = set([idx for idx, mag in sorted_layers[:k]])
    
    print(f"Global refusal direction from layer {best_layer}")
    print(f"Pertinent layers: {sorted(pertinent_layers)}")
    
    # ================================================================
    # STEP 1: Generate per-layer random alias vectors
    # ================================================================
    
    # We need the typical output each pertinent layer writes during refusal.
    # This requires a calibration forward pass on harmful prompts with sublayer hooks.
    # For now, generate aliases; we'll scale them during the patching step
    # once we have the per-sublayer activations.
    
    random_aliases = {}
    epsilon = 0.1  # hyperparameter: std of random alias vectors, ε ∈ (0, 1)
    for ell in pertinent_layers:
        z = torch.randn(d, device=model.device)
        z = z - z.mean()  # force zero mean
        z = z / z.std() * epsilon  # scale to small variance
        random_aliases[ell] = z  # will be used directly in Step 2
    
    # ================================================================
    # STEP 2-6: Apply patches using calibration forward pass
    # ================================================================
    
    # Run a calibration forward pass to collect all intermediate activations.
    # CRITICAL: Use HARMFUL prompts that trigger the refusal pathway.
    # The rank-one patches must be calibrated at the operating point where
    # the refusal direction is actually being written/read. Using benign
    # prompts would patch at irrelevant operating points.
    # Average over multiple harmful prompts for robustness.
    calibration_prompt = harmful_prompts[0]  # Use harmful prompt that triggers refusal
    cal_inputs = tokenizer(calibration_prompt, return_tensors="pt").to(model.device)
    
    # We need fine-grained activations at each sublayer.
    # Collect:
    # - Input to each attention sublayer's Q/K/V projections (= LayerNorm output)
    # - Input to each attention W_O (= attention output before projection)
    # - Input to each MLP's gate/up projections (= LayerNorm output)
    # - Input to each MLP's W_down (= MLP hidden state)
    # - The residual stream at each point
    
    # This requires hooking into sublayer components.
    # The exact hook points depend on the model architecture.
    
    # For Llama-style models:
    # model.model.layers[ℓ].input_layernorm  → feeds attention
    # model.model.layers[ℓ].self_attn.q_proj, k_proj, v_proj  → read from LN output
    # model.model.layers[ℓ].self_attn.o_proj  → writes to residual
    # model.model.layers[ℓ].post_attention_layernorm  → feeds MLP
    # model.model.layers[ℓ].mlp.gate_proj, up_proj  → read from LN output
    # model.model.layers[ℓ].mlp.down_proj  → writes to residual
    
    # Collect all needed activations:
    activations = {}
    
    def save_activation(name):
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                activations[name + "_input"] = input[0].detach()[:, -1, :]
            else:
                activations[name + "_input"] = input.detach()[:, -1, :]
            if isinstance(output, tuple):
                activations[name + "_output"] = output[0].detach()[:, -1, :]
            else:
                activations[name + "_output"] = output.detach()[:, -1, :]
        return hook_fn
    
    hooks = []
    for ell in range(num_layers):
        layer = model.model.layers[ell]
        hooks.append(layer.input_layernorm.register_forward_hook(save_activation(f"layer_{ell}_attn_ln")))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(save_activation(f"layer_{ell}_attn_o")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(save_activation(f"layer_{ell}_mlp_ln")))
        hooks.append(layer.mlp.down_proj.register_forward_hook(save_activation(f"layer_{ell}_mlp_down")))
    
    # Also hook the final layernorm
    hooks.append(model.model.norm.register_forward_hook(save_activation("final_ln")))
    
    with torch.no_grad():
        model(**cal_inputs)
    
    for h in hooks:
        h.remove()
    
    # Now we have clean activations. We'll track pollution as we patch.
    
    # ================================================================
    # Apply patches layer by layer, tracking pollution
    # ================================================================
    
    cumulative_pollution = torch.zeros(d, device=model.device)
    
    for ell in range(num_layers):
        layer = model.model.layers[ell]
        
        # --- ATTENTION SUBLAYER ---
        
        # 1. Patch attention readers (Q, K, V) if there's existing pollution
        if cumulative_pollution.norm() > 1e-8:
            # Get clean and polluted inputs to the attention LayerNorm
            # Clean: what the LN would output on the clean residual stream
            # Polluted: what the LN outputs on the polluted residual stream
            # We stored the clean LN output; now compute polluted version
            
            ln_module = layer.input_layernorm
            # x_clean for this LN = the clean residual stream at this point
            x_clean_here = activations[f"layer_{ell}_attn_ln_input"].squeeze()
            x_polluted_here = x_clean_here + cumulative_pollution
            
            ln_clean = ln_module(x_clean_here.unsqueeze(0).unsqueeze(0)).squeeze()
            ln_polluted = ln_module(x_polluted_here.unsqueeze(0).unsqueeze(0)).squeeze()
            
            # Patch Q, K, V projections
            for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                W = getattr(layer.self_attn, proj_name).weight.data  # (d_out, d_in)
                
                target = W @ ln_clean
                current = W @ ln_polluted
                delta = target - current
                
                W_new = W + torch.outer(delta, ln_polluted) / (ln_polluted @ ln_polluted)
                getattr(layer.self_attn, proj_name).weight.data = W_new
        
        # 2. Patch attention output projection (W_O) if this is a pertinent layer
        if ell in pertinent_layers:
            # Get the input to W_O (the attention output before projection)
            x_attn = activations[f"layer_{ell}_attn_o_input"].squeeze()
            z_ell = random_aliases[ell]  # already zero-mean, ε-scaled
            
            W_O = layer.self_attn.o_proj.weight.data  # (d, d)
            current_output = W_O @ x_attn
            
            delta = z_ell - current_output
            W_O_new = W_O + torch.outer(delta, x_attn) / (x_attn @ x_attn)
            layer.self_attn.o_proj.weight.data = W_O_new
            
            # Update cumulative pollution: track ONLY the random vectors added
            # The original current_output is already part of x_clean, so we only
            # need to track what's NEW (z_ell) minus what was REMOVED (current_output)
            # Net change to residual stream = z_ell - current_output
            cumulative_pollution = cumulative_pollution + z_ell - current_output
        
        # --- MLP SUBLAYER ---
        
        # 3. Patch MLP readers (gate, up) if there's pollution
        if cumulative_pollution.norm() > 1e-8:
            ln_module = layer.post_attention_layernorm
            x_clean_here = activations[f"layer_{ell}_mlp_ln_input"].squeeze()
            x_polluted_here = x_clean_here + cumulative_pollution
            
            ln_clean = ln_module(x_clean_here.unsqueeze(0).unsqueeze(0)).squeeze()
            ln_polluted = ln_module(x_polluted_here.unsqueeze(0).unsqueeze(0)).squeeze()
            
            for proj_name in ['gate_proj', 'up_proj']:
                W = getattr(layer.mlp, proj_name).weight.data
                
                target = W @ ln_clean
                current = W @ ln_polluted
                delta = target - current
                
                W_new = W + torch.outer(delta, ln_polluted) / (ln_polluted @ ln_polluted)
                getattr(layer.mlp, proj_name).weight.data = W_new
        
        # 4. Patch MLP output projection (W_down) if this is a pertinent layer
        if ell in pertinent_layers:
            x_mlp = activations[f"layer_{ell}_mlp_down_input"].squeeze()
            # Generate a SEPARATE zero-mean, ε-scaled random vector for the MLP
            z_ell_mlp = torch.randn(d, device=model.device)
            z_ell_mlp = z_ell_mlp - z_ell_mlp.mean()  # force zero mean
            z_ell_mlp = z_ell_mlp / z_ell_mlp.std() * epsilon  # scale to small variance
            
            W_down = layer.mlp.down_proj.weight.data
            current_output = W_down @ x_mlp
            
            delta = z_ell_mlp - current_output
            W_down_new = W_down + torch.outer(delta, x_mlp) / (x_mlp @ x_mlp)
            layer.mlp.down_proj.weight.data = W_down_new
            
            cumulative_pollution = cumulative_pollution + z_ell_mlp - current_output
    
    # ================================================================
    # STEP 6: Patch the unembedding matrix
    # ================================================================
    
    # cumulative_pollution IS z_sum: the sum of every random vector that
    # patched "writing" matrices output to the residual stream.
    # Fortify the LM head against this known vector.
    if cumulative_pollution.norm() > 1e-8:
        z_sum = cumulative_pollution  # explicitly name it
        final_ln = model.model.norm
        x_clean_final = activations["final_ln_input"].squeeze()
        x_polluted_final = x_clean_final + z_sum
        
        ln_clean = final_ln(x_clean_final.unsqueeze(0).unsqueeze(0)).squeeze()
        ln_polluted = final_ln(x_polluted_final.unsqueeze(0).unsqueeze(0)).squeeze()
        
        W_unembed = model.lm_head.weight.data  # (vocab_size, d)
        
        target = W_unembed @ ln_clean
        current = W_unembed @ ln_polluted
        delta = target - current
        
        W_unembed_new = W_unembed + torch.outer(delta, ln_polluted) / (ln_polluted @ ln_polluted)
        model.lm_head.weight.data = W_unembed_new
    
    print(f"Defense applied. Patched {len(pertinent_layers)} pertinent layers.")
    print(f"z_sum norm (total pollution): {cumulative_pollution.norm().item():.4f}")
    
    return model

```

---

## Important Implementation Notes

### 1. Activation Collection
The code above collects activations at the last token position. For more robust calibration, average over:
- Multiple harmful/refusal-triggering prompts (32+)
- Multiple token positions (e.g., last 5 tokens)
- **CRITICAL**: Calibration activations MUST come from harmful prompts that trigger the refusal pathway. The patches need to be exact at the operating point where r̂ is actually being written/read. Benign prompts don't activate the refusal direction, so patching at those activations is useless.

### 2. Model Architecture Differences
The code assumes Llama-style architecture. For other architectures:
- **Qwen**: Similar structure, may have different attribute names
- **Gemma**: Uses different LayerNorm placement
- **Mistral**: Uses Grouped Query Attention (GQA), so K and V projections are smaller
- Always inspect `model.model.layers[0]` to find correct attribute names

### 3. Separate Random Vectors for Attn and MLP
Each pertinent layer has TWO write matrices (W_O and W_down). You can:
- Use the same random vector for both (simpler, less obfuscation)
- Use different random vectors for each (more obfuscation, more pollution)
The current code uses different vectors. This is a hyperparameter to ablate.

### 4. Random Vector Scaling
All random alias vectors must be:
- **Zero-mean**: So they don't shift LayerNorm's centering operation (x - μ). A non-zero-mean addition would change μ and create nonlinear distortion.
- **Small variance (ε)**: Scaled to have std = ε, where ε ∈ (0, 1) is a hyperparameter. Small ε keeps the pollution negligible relative to typical residual stream magnitudes, minimizing LayerNorm distortion. Trade-off: smaller ε = better utility preservation but potentially easier for an attacker to filter out as noise; larger ε = more robust obfuscation but more LayerNorm distortion.

### 5. Calibration Prompt Selection
The rank-one patches are exact for the calibration activations but approximate for other inputs. For best generalization:
- Use the mean activation across 32+ diverse harmful prompts
- Verify the model still refuses harmful prompts after patching
- Verify the model still answers benign prompts correctly

### 6. The Non-Linearity Concern
The MLP contains nonlinear activations (SiLU/GeLU). Patching gate_proj and up_proj ensures their inputs are correct, but the nonlinearity means small errors in the input can amplify. Monitor the model's output quality carefully.

### 7. Bias Terms
If the model has bias terms (most modern LLMs don't), you may need to adjust them too. Llama, Qwen, and Gemma do not use bias in attention/MLP projections.

### 8. Memory Considerations
- The rank-one updates modify weights in-place, so no additional memory at inference
- The calibration forward pass requires storing intermediate activations (~d floats per hook point)
- For a 7B model with d=4096, this is negligible

---

## Evaluation Plan

### Safety Evaluation
1. **Pre-defense refusal rate**: Run CatQA (550 harmful prompts), measure refusal rate via Llama-Guard-3
2. **Post-defense refusal rate**: Should be identical (defense preserves behavior)
3. **Post-abliteration refusal rate**: Run standard abliteration on the defended model, measure refusal rate
   - Vanilla model: expect ~20% refusal (abliteration works)
   - ROSI-defended model: expect ~20% refusal (abliteration still works, just needs slightly more effort)
   - Our defended model: expect ~90%+ refusal (abliteration fails because r̂ is not in the residual stream)

### Utility Evaluation
- MMLU, HellaSwag, ARC, BoolQ, TruthfulQA
- Perplexity on C4 (1000 passages)
- Target: <1% degradation from original model

### Abliteration Resistance
- Standard difference-in-means attack
- PCA-based multi-direction attack (remove top-k directions)
- Heretic automated abliteration
- **Adaptive attack**: attacker hooks into sublayer internals and tries to find refusal direction within each layer's local computation

### Key Diagnostic
After applying the defense, compute:
```python
# Run difference-in-means on the DEFENDED model's residual stream
r_hat_defended = mean(h_defended(harmful)) - mean(h_defended(benign))
# Compute cosine similarity with original r̂
cos_sim = torch.cosine_similarity(r_hat_defended, r_hat_original, dim=0)
# This should be near 0 (random), not near 1
```

If cos_sim ≈ 0 at each layer, the defense is working — the refusal direction has been successfully obfuscated in the residual stream.

---

## File Structure

Built on top of `github.com/andyrdt/refusal_direction`. Fork the repo and add the new files:

```
refusal_direction/              # forked from andyrdt/refusal_direction
├── pipeline/
│   ├── run_pipeline.py              # MODIFY — add obfuscation stage after select_direction
│   ├── generate_directions.py       # EXISTING — extracts candidate refusal directions
│   ├── select_direction.py          # EXISTING — picks best r̂, saves direction.pt
│   ├── apply_obfuscation.py         # NEW — Steps 1-6: generate aliases, patch writers/readers/LM head
│   ├── completions.py               # EXISTING — evaluate refusal on harmful/harmless prompts
│   ├── loss_evals.py                # EXISTING — evaluate CE loss / utility
│   ├── evaluate_abliteration.py     # NEW — run abliteration on defended model, measure survival
│   ├── evaluate_adaptive_attack.py  # NEW — per-layer adaptive attack, PCA top-k, Heretic
│   ├── utils.py                     # NEW — hook helpers, activation collection, rank-one update functions
│   └── config.py                    # NEW — hyperparameters (ε, k, calibration size, etc.)
├── dataset/
│   ├── harmful.json                 # EXISTING — harmful prompts
│   └── harmless.json                # EXISTING — benign prompts
├── setup.sh                         # EXISTING
├── requirements.txt                 # EXISTING
└── README.md                        # MODIFY — document the defense
```

**Pipeline order** (modified `run_pipeline.py`):
1. `generate_directions` — extract candidate refusal directions (existing)
2. `select_direction` — pick best r̂, save `direction.pt` (existing)
3. `apply_obfuscation` — **NEW**: load `direction.pt`, apply defense, save defended model
4. `completions` — evaluate refusal on harmful/harmless prompts (existing, run on defended model)
5. `loss_evals` — evaluate utility (existing, run on defended model)
6. `evaluate_abliteration` — **NEW**: run standard abliteration on defended model
7. `evaluate_adaptive_attack` — **NEW**: run adaptive attacks on defended model

---

## Hyperparameters to Tune

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| ε (random vector std) | 0.1 | (0, 1) | Controls pollution magnitude. Smaller = less LN distortion, larger = harder to filter out. KEY hyperparameter to ablate. |
| k (num pertinent layers) | 10 | 5-20 | More = more obfuscation, more pollution |
| Calibration prompts | 32 | 16-128 | More = better generalization of rank-one patches. Must be harmful/refusal-triggering. |
| Separate attn/MLP aliases | True | True/False | True = more obfuscation |
| Random seed | 42 | any | Different seeds = different aliases |
| Which output matrices to patch | Both W_O and W_down | W_O only, W_down only, both | Ablate this |
