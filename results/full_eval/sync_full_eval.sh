#!/usr/bin/env bash
set -euo pipefail
REMOTE='ubuntu@192.222.59.142'
REMOTE_DIR='/home/ubuntu/aprs_sweep_full_eval'
LOCAL_DIR='/Users/nathan/Documents/Development/APRS/results/full_eval'
mkdir -p "$LOCAL_DIR"
while true; do
  scp "$REMOTE:$REMOTE_DIR/launcher.log" "$LOCAL_DIR/launcher.log.tmp" 2>/dev/null && mv "$LOCAL_DIR/launcher.log.tmp" "$LOCAL_DIR/launcher.log" || true
  scp "$REMOTE:$REMOTE_DIR/all_results.csv" "$LOCAL_DIR/all_results.csv.tmp" 2>/dev/null && mv "$LOCAL_DIR/all_results.csv.tmp" "$LOCAL_DIR/all_results.csv" || true
  date '+%Y-%m-%d %H:%M:%S %Z' > "$LOCAL_DIR/last_sync.txt"
  sleep 300
 done
