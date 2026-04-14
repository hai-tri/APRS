#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot environment setup for APRS on a Lambda Labs GH200
#
# Usage:
#   bash setup.sh
#
# What it does:
#   1. Clones the repo with submodules
#   2. Installs Python dependencies
#   3. Downloads data files (HarmBench, XSTest)
#   4. Authenticates with HuggingFace (prompts for token)
#   5. Pre-downloads the target model weights
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/hai-tri/APRS.git"
REPO_DIR="$HOME/APRS"
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"

echo "================================================================"
echo " APRS Setup Script"
echo "================================================================"

# ── 1. Clone repo ────────────────────────────────────────────────────
if [ -d "$REPO_DIR" ]; then
    echo "[setup] Repo already exists at $REPO_DIR — pulling latest …"
    cd "$REPO_DIR"
    git pull
    git submodule update --init --recursive
else
    echo "[setup] Cloning $REPO_URL …"
    git clone --recurse-submodules "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# ── 2. Install Python dependencies ───────────────────────────────────
echo "[setup] Installing Python dependencies …"
pip install --quiet --upgrade pip
pip install --quiet \
    transformers>=4.40.0 \
    datasets \
    accelerate \
    peft \
    optuna \
    tqdm \
    scipy \
    scikit-learn \
    pandas \
    sentencepiece \
    protobuf \
    lm-eval \
    jaxtyping

# Install refusal_direction submodule deps if present
if [ -f "$REPO_DIR/refusal_direction/requirements.txt" ]; then
    echo "[setup] Installing refusal_direction requirements …"
    pip install --quiet -r "$REPO_DIR/refusal_direction/requirements.txt"
fi

echo "[setup] Dependencies installed."

# ── 3. HuggingFace authentication ────────────────────────────────────
echo ""
echo "[setup] HuggingFace login required for $MODEL_ID"
echo "        Get your token at: https://huggingface.co/settings/tokens"
echo ""
huggingface-cli login

# ── 4. Pre-download model weights ────────────────────────────────────
echo "[setup] Pre-downloading $MODEL_ID …"
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print('Downloading tokenizer …')
AutoTokenizer.from_pretrained('$MODEL_ID')
print('Downloading model weights …')
AutoModelForCausalLM.from_pretrained('$MODEL_ID', torch_dtype=torch.bfloat16)
print('Model downloaded successfully.')
"

# ── 5. Pre-download HarmBench classifier ─────────────────────────────
echo "[setup] Pre-downloading HarmBench classifier …"
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_id = 'cais/HarmBench-Llama-2-13b-cls'
print('Downloading HarmBench classifier …')
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
print('HarmBench classifier downloaded.')
"

echo ""
echo "================================================================"
echo " Setup complete. Run the sweep with:"
echo "   bash $REPO_DIR/scripts/run_sweep.sh"
echo "================================================================"
