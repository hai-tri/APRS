#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh — Per-mode hyperparameter sweep for the APRS obfuscation defense
#
# Usage:
#   bash run_sweep.sh [--model MODEL_ID] [--output_dir DIR] [--attacks]
#
# Flags:
#   --model      HuggingFace model ID (default: meta-llama/Meta-Llama-3-8B-Instruct)
#   --output_dir Base directory for per-run CSVs & logs (default: $HOME/aprs_sweep)
#   --attacks    If set, also runs GCG/AutoDAN/CipherChat/PAIR/ReNeLLM per run
# Env:
#   APRS_EXTRA_FLAGS Additional space-separated flags forwarded to
#                    run_obfuscation_pipeline.py for every run
#
# Sweep design (40 runs) — each projection mode is tuned independently:
#
#   Sweep 0   — undefended baseline
#
#   Sweep 1   — epsilon × mode  (7 ε values × 3 modes = 21 runs)
#               ε ∈ {0.3, 0.2, 0.15, 0.1, 0.05, 0.025, 0.01}
#               mode ∈ {scalar_projection, hadamard, full}
#               Fixed: n_cal=128, per_layer=True
#
#   Sweep 2   — n_calibration × mode  (4 n_cal values × 3 modes = 12 runs)
#               n_cal ∈ {32, 64, 256, 512}  (128 covered by Sweep 1)
#               Fixed: best ε per mode (approximated as 0.1 anchor), per_layer=True
#
#   Sweep 3   — per_layer vs global × mode  (1 × 3 = 3 runs)
#               Fixed: ε=0.1, n_cal=128  (per_layer covered by Sweep 1)
#
#   Sweep 4   — baseline defenses  (3 runs)
#               defense_type ∈ {surgical, cast, circuit_breakers}
#
# NOTE: Attacks (GCG/AutoDAN/CipherChat/PAIR/ReNeLLM) should be run separately
# on the best config per mode after inspecting all_results.csv — use --attacks
# only for targeted final-table runs to avoid 40× attack overhead.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_BASE="$HOME/aprs_sweep"
RUN_ATTACKS=0
EXTRA_FLAGS_STR="${APRS_EXTRA_FLAGS:-}"
_EXTRA_PIPELINE_FLAGS=()
if [[ -n "$EXTRA_FLAGS_STR" ]]; then
    read -r -a _EXTRA_PIPELINE_FLAGS <<< "$EXTRA_FLAGS_STR"
fi

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
echo " Extra args : ${EXTRA_FLAGS_STR:-<none>}"                         | tee -a "$MASTER_LOG"
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

# Direction extraction is keyed only by model alias inside refusal_direction,
# so after the first successful run we can safely reuse the cached direction.
_DIRECTION_READY=0

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

    # Resume: skip if CSV already exists and is non-empty
    if [[ -s "$csv_path" ]]; then
        echo "   [SKIP] already complete → $csv_path" | tee -a "$MASTER_LOG"
        _DIRECTION_READY=1
        return
    fi

    local cache_args=()
    if [[ $_DIRECTION_READY -eq 1 ]]; then
        cache_args=(--skip_direction_extraction)
    fi

    set +e
    python3 -u "$REPO_DIR/run_obfuscation_pipeline.py" \
        --model_path "$MODEL_ID" \
        --save_csv   "$csv_path" \
        --skip_heretic \
        "${cache_args[@]}" \
        "${extra_args[@]}" \
        "${_EXTRA_PIPELINE_FLAGS[@]}" \
        "${_ATTACK_FLAGS[@]}" \
        2>&1 | tee "$log_path"
    local rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
        _DIRECTION_READY=1
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
# Sweep 1 — Epsilon × projection mode  (7 × 3 = 21 runs)
# ═══════════════════════════════════════════════════════════════════════════
for mode in scalar_projection hadamard full; do
    for eps in 0.3 0.2 0.15 0.1 0.05 0.025 0.01; do
        eps_tag="${eps//./_}"
        run_config "sweep1_${mode}_eps${eps_tag}" \
            --projection_mode "$mode" \
            --epsilon "$eps" \
            --num_calibration_prompts 128 \
            --per_layer_direction
    done
done

# ═══════════════════════════════════════════════════════════════════════════
# Sweep 2 — Calibration set size × projection mode  (4 × 3 = 12 runs)
# n_cal=128 already covered by Sweep 1; use ε=0.1 as anchor
# ═══════════════════════════════════════════════════════════════════════════
for mode in scalar_projection hadamard full; do
    for ncal in 32 64 256 512; do
        run_config "sweep2_${mode}_ncal${ncal}" \
            --projection_mode "$mode" \
            --epsilon 0.1 \
            --num_calibration_prompts "$ncal" \
            --per_layer_direction
    done
done

# ═══════════════════════════════════════════════════════════════════════════
# Sweep 3 — Per-layer vs global direction × projection mode  (1 × 3 = 3 runs)
# per_layer=True already covered by Sweep 1; ε=0.1, n_cal=128 as anchor
# ═══════════════════════════════════════════════════════════════════════════
for mode in scalar_projection hadamard full; do
    run_config "sweep3_${mode}_global" \
        --projection_mode "$mode" \
        --epsilon 0.1 \
        --num_calibration_prompts 128
done

# ═══════════════════════════════════════════════════════════════════════════
# Sweep 4 — Baseline defenses  (3 runs)
# ═══════════════════════════════════════════════════════════════════════════
run_config "sweep4_surgical" \
    --defense_type surgical

run_config "sweep4_cast" \
    --defense_type cast

run_config "sweep4_circuit_breakers" \
    --defense_type circuit_breakers

# ═══════════════════════════════════════════════════════════════════════════
# Aggregate all per-run CSVs → all_results.csv
# ═══════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$MASTER_LOG"
echo "── Aggregating results ──────────────────────────────────────" | tee -a "$MASTER_LOG"

OUTPUT_BASE="$OUTPUT_BASE" python3 - <<'PYEOF' 2>&1 | tee -a "$MASTER_LOG"
import csv, glob, os

output_base = os.environ["OUTPUT_BASE"]
all_rows, all_keys = [], []

for csv_path in sorted(glob.glob(os.path.join(output_base, "sweep*.csv"))):
    sweep_tag = os.path.splitext(os.path.basename(csv_path))[0]
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        for k in rows[0].keys():
            if k not in all_keys:
                all_keys.append(k)
        for r in rows:
            r["sweep_tag"] = sweep_tag
            all_rows.append(r)
    except Exception as e:
        print(f"[WARN] {csv_path}: {e}")

if "sweep_tag" not in all_keys:
    all_keys.append("sweep_tag")

if all_rows:
    out = os.path.join(output_base, "all_results.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
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
