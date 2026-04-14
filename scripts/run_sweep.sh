#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh — Hyperparameter sweep for the APRS obfuscation defense
#
# Usage:
#   bash run_sweep.sh [--model MODEL_ID] [--output_dir DIR] [--attacks]
#
# Flags:
#   --model      HuggingFace model ID (default: meta-llama/Meta-Llama-3-8B-Instruct)
#   --output_dir Base directory for per-run CSVs & logs (default: $HOME/aprs_sweep)
#   --attacks    If set, also runs GCG/AutoDAN/CipherChat/PAIR/ReNeLLM per run
#
# Sweep design:
#   Sweep 0  — undefended baseline (--undefended_only)
#   Sweep 1  — projection_mode ∈ {scalar_projection, hadamard, full}      (ε=0.1)
#   Sweep 2a — epsilon ∈ {0.3,0.2,0.15,0.05,0.025,0.01}  scalar_projection
#   Sweep 2b — epsilon ∈ {0.3,0.2,0.15,0.05,0.025,0.01}  hadamard
#   Sweep 2c — epsilon ∈ {0.3,0.2,0.15,0.05,0.025,0.01}  full
#   Sweep 3  — n_calibration ∈ {32,64,256,512}             (ε=0.1, scalar)
#   Sweep 4  — per_layer_direction on/off                  (ε=0.1, scalar)
#   Sweep 5  — baselines: surgical / cast / circuit_breakers
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_BASE="$HOME/aprs_sweep"
RUN_ATTACKS=0

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL_ID="$2";    shift 2 ;;
        --output_dir) OUTPUT_BASE="$2"; shift 2 ;;
        --attacks)    RUN_ATTACKS=1;    shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_BASE"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$OUTPUT_BASE/sweep_${TIMESTAMP}.log"

echo "================================================================" | tee "$MASTER_LOG"
echo " APRS Hyperparameter Sweep — $(date)"                             | tee -a "$MASTER_LOG"
echo " Model      : $MODEL_ID"                                          | tee -a "$MASTER_LOG"
echo " Output dir : $OUTPUT_BASE"                                       | tee -a "$MASTER_LOG"
echo " Attacks    : $RUN_ATTACKS"                                       | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"

# Attack flags (appended when --attacks is set)
_ATTACK_FLAGS=()
if [[ $RUN_ATTACKS -eq 1 ]]; then
    _ATTACK_FLAGS=(
        --gcg   --gcg_n_behaviors 25
        --autodan --autodan_n_behaviors 25
        --cipherchat
        --pair  --pair_n_behaviors 25
        --renellm
    )
fi

# ── Helper: run one pipeline configuration ───────────────────────────────────
# run_config <tag> [pipeline_args...]
run_config() {
    local tag="$1"; shift
    local extra_args=("$@")

    local csv_path="$OUTPUT_BASE/${tag}.csv"
    local log_path="$OUTPUT_BASE/${tag}.log"

    echo "" | tee -a "$MASTER_LOG"
    echo "── $tag" | tee -a "$MASTER_LOG"
    echo "   args : ${extra_args[*]}" | tee -a "$MASTER_LOG"

    set +e
    python3 "$REPO_DIR/run_obfuscation_pipeline.py" \
        --model_path "$MODEL_ID" \
        --save_csv   "$csv_path" \
        "${extra_args[@]}" \
        "${_ATTACK_FLAGS[@]}" \
        2>&1 | tee "$log_path"
    local rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
        echo "   [OK] → $csv_path" | tee -a "$MASTER_LOG"
    else
        echo "   [WARN] exited $rc — see $log_path" | tee -a "$MASTER_LOG"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# Sweep 0 — Undefended baseline
# ═══════════════════════════════════════════════════════════════════════════
run_config "sweep0_undefended" \
    --undefended_only

# ═══════════════════════════════════════════════════════════════════════════
# Sweep 1 — Projection mode  (ε=0.1, n_cal=128, per_layer)
# ═══════════════════════════════════════════════════════════════════════════
for mode in scalar_projection hadamard full; do
    run_config "sweep1_mode_${mode}" \
        --projection_mode "$mode" \
        --epsilon 0.1 \
        --num_calibration_prompts 128 \
        --per_layer_direction
done

# ═══════════════════════════════════════════════════════════════════════════
# Sweep 2 — Epsilon × projection mode
# (ε=0.1 already covered by Sweep 1, skip it here)
# ═══════════════════════════════════════════════════════════════════════════
for mode in scalar_projection hadamard full; do
    for eps in 0.3 0.2 0.15 0.05 0.025 0.01; do
        eps_tag="${eps//./_}"
        run_config "sweep2_${mode}_eps${eps_tag}" \
            --projection_mode "$mode" \
            --epsilon "$eps" \
            --num_calibration_prompts 128 \
            --per_layer_direction
    done
done

# ═══════════════════════════════════════════════════════════════════════════
# Sweep 3 — Calibration set size  (scalar, ε=0.1, per_layer)
# (n_cal=128 already covered by Sweep 1)
# ═══════════════════════════════════════════════════════════════════════════
for ncal in 32 64 256 512; do
    run_config "sweep3_ncal${ncal}" \
        --projection_mode scalar_projection \
        --epsilon 0.1 \
        --num_calibration_prompts "$ncal" \
        --per_layer_direction
done

# ═══════════════════════════════════════════════════════════════════════════
# Sweep 4 — Per-layer vs global direction  (scalar, ε=0.1, n_cal=128)
# (per_layer already covered by Sweep 1)
# ═══════════════════════════════════════════════════════════════════════════
run_config "sweep4_global_direction" \
    --projection_mode scalar_projection \
    --epsilon 0.1 \
    --num_calibration_prompts 128

# ═══════════════════════════════════════════════════════════════════════════
# Sweep 5 — Baseline defenses
# ═══════════════════════════════════════════════════════════════════════════
run_config "sweep5_surgical" \
    --defense_type surgical

run_config "sweep5_cast" \
    --defense_type cast

run_config "sweep5_circuit_breakers" \
    --defense_type circuit_breakers

# ═══════════════════════════════════════════════════════════════════════════
# Aggregate all per-run CSVs → all_results.csv
# ═══════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$MASTER_LOG"
echo "── Aggregating results ──────────────────────────────────────" | tee -a "$MASTER_LOG"

OUTPUT_BASE="$OUTPUT_BASE" python3 - <<'PYEOF' 2>&1 | tee -a "$MASTER_LOG"
import csv, glob, os

output_base = os.environ["OUTPUT_BASE"]
all_rows, header = [], None

for csv_path in sorted(glob.glob(os.path.join(output_base, "sweep*.csv"))):
    sweep_tag = os.path.splitext(os.path.basename(csv_path))[0]
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        if header is None:
            header = list(rows[0].keys()) + ["sweep_tag"]
        for r in rows:
            r["sweep_tag"] = sweep_tag
            all_rows.append(r)
    except Exception as e:
        print(f"[WARN] {csv_path}: {e}")

if all_rows:
    out = os.path.join(output_base, "all_results.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows → {out}")
else:
    print("[WARN] No result CSVs found.")
PYEOF

echo "" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo " Sweep complete — $(date)" | tee -a "$MASTER_LOG"
echo " Log     : $MASTER_LOG" | tee -a "$MASTER_LOG"
echo " Results : $OUTPUT_BASE/all_results.csv" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
