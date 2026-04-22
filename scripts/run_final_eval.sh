#!/usr/bin/env bash
# =============================================================================
# run_final_eval.sh — Final APRS comparison run for the selected configs
#
# Usage:
#   bash run_final_eval.sh [--model MODEL_ID] [--output_dir DIR]
#
# Runs the fixed final comparison set:
#   - APRS scalar_projection  (epsilon=0.3,   n_cal=128, per_layer=True)
#   - APRS hadamard           (epsilon=0.3,   n_cal=128, per_layer=True)
#   - APRS full               (epsilon=0.025, n_cal=128, per_layer=True)
#   - Baselines: surgical / cast / circuit_breakers / alphasteer
#
# For each run, the full benchmark stack is enabled, including:
#   - HarmBench before attacks
#   - GCG / AutoDAN / CipherChat / PAIR / ReNeLLM
#   - HarmBench re-scoring on post-attack outputs
#   - XSTest
#   - utility benchmarks where supported
#
# Notes:
#   - Hook-based defenses are evaluated correctly for local generation-based
#     benchmarks and attacks. External subprocess benchmarks that require a
#     saved model checkpoint (lm-harness, Heretic) are skipped for hook-based
#     defenses because their runtime hooks cannot be serialized into weights.
#   - Additional pipeline flags can be forwarded via APRS_EXTRA_FLAGS.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_BASE="$HOME/aprs_final"
EXTRA_FLAGS_STR="${APRS_EXTRA_FLAGS:-}"
_EXTRA_PIPELINE_FLAGS=()
if [[ -n "$EXTRA_FLAGS_STR" ]]; then
    read -r -a _EXTRA_PIPELINE_FLAGS <<< "$EXTRA_FLAGS_STR"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL_ID="$2";    shift 2 ;;
        --output_dir) OUTPUT_BASE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_BASE"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$OUTPUT_BASE/final_${TIMESTAMP}.log"

echo "================================================================" | tee "$MASTER_LOG"
echo " APRS Final Evaluation — $(date)"                                | tee -a "$MASTER_LOG"
echo " Model      : $MODEL_ID"                                         | tee -a "$MASTER_LOG"
echo " Output dir : $OUTPUT_BASE"                                      | tee -a "$MASTER_LOG"
echo " Extra args : ${EXTRA_FLAGS_STR:-<none>}"                        | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"

_ATTACK_FLAGS=(
    --gcg --gcg_n_behaviors 25
    --autodan --autodan_n_behaviors 25
    --cipherchat
    --pair --pair_n_behaviors 25
    --renellm
    --softopt --softopt_limit 25
)

_DIRECTION_READY=0

run_config() {
    local tag="$1"; shift
    local extra_args=("$@")

    local csv_path="$OUTPUT_BASE/${tag}.csv"
    local log_path="$OUTPUT_BASE/${tag}.log"

    echo "" | tee -a "$MASTER_LOG"
    echo "── $tag" | tee -a "$MASTER_LOG"
    echo "   args : ${extra_args[*]}" | tee -a "$MASTER_LOG"

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
        --save_csv "$csv_path" \
        --llamaguard \
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

run_config "final_scalar_eps0_3" \
    --projection_mode scalar_projection \
    --epsilon 0.3 \
    --num_calibration_prompts 128 \
    --per_layer_direction

run_config "final_hadamard_eps0_3" \
    --projection_mode hadamard \
    --epsilon 0.3 \
    --num_calibration_prompts 128 \
    --per_layer_direction

run_config "final_full_eps0_025" \
    --projection_mode full \
    --epsilon 0.025 \
    --num_calibration_prompts 128 \
    --per_layer_direction

run_config "final_full_eps0_025_writer_only" \
    --projection_mode full \
    --epsilon 0.025 \
    --num_calibration_prompts 128 \
    --per_layer_direction \
    --obfuscation_writer_only

run_config "final_surgical" \
    --defense_type surgical

run_config "final_cast" \
    --defense_type cast

run_config "final_circuit_breakers" \
    --defense_type circuit_breakers

run_config "final_alphasteer" \
    --defense_type alphasteer

echo "" | tee -a "$MASTER_LOG"
echo "── Aggregating results ──────────────────────────────────────" | tee -a "$MASTER_LOG"

OUTPUT_BASE="$OUTPUT_BASE" python3 - <<'PYEOF' 2>&1 | tee -a "$MASTER_LOG"
import csv, glob, os

output_base = os.environ["OUTPUT_BASE"]
all_rows, all_keys = [], []

for csv_path in sorted(glob.glob(os.path.join(output_base, "final_*.csv"))):
    run_tag = os.path.splitext(os.path.basename(csv_path))[0]
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        for k in rows[0].keys():
            if k not in all_keys:
                all_keys.append(k)
        for r in rows:
            r["run_tag"] = run_tag
            all_rows.append(r)
    except Exception as e:
        print(f"[WARN] {csv_path}: {e}")

if "run_tag" not in all_keys:
    all_keys.append("run_tag")

if all_rows:
    out = os.path.join(output_base, "final_results.csv")
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
echo " Final eval complete — $(date)" | tee -a "$MASTER_LOG"
echo " Log     : $MASTER_LOG" | tee -a "$MASTER_LOG"
echo " Results : $OUTPUT_BASE/final_results.csv" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
