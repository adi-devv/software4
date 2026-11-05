#!/usr/bin/env bash
# Push a Kaggle kernel folder, wait for it, and pull its outputs.
# Usage: scripts/kaggle_run.sh <folder> [accelerator] [timeout_s]
set -euo pipefail
DIR=${1:?folder with kernel-metadata.json}
ACC=${2:-NvidiaTeslaT4}
TO=${3:-10800}
ID=$(python -c "import json,sys;print(json.load(open(sys.argv[1]+'/kernel-metadata.json'))['id'])" "$DIR")
kaggle kernels push -p "$DIR" --accelerator "$ACC" -t "$TO"
echo "pushed $ID on $ACC; waiting..."
sleep 30
while true; do
  S=$(kaggle kernels status "$ID" 2>&1 | tail -1)
  echo "$(date +%H:%M:%S) $S"
  case "$S" in
    *COMPLETE*|*ERROR*|*CANCEL*) break ;;
  esac
  sleep 60
done
mkdir -p "$DIR/output"
kaggle kernels output "$ID" -p "$DIR/output" --force >/dev/null 2>&1 || kaggle kernels output "$ID" -p "$DIR/output"
ls -la "$DIR/output"
