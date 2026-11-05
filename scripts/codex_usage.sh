#!/usr/bin/env bash
# Prints Codex usage % per account, read from each CODEX_HOME's newest session log.
# 5h = 5-hour window, wk = weekly window. Numbers are as of that account's last run.
for entry in 7859:C:/Users/aadit/.codex-7859 w:C:/Users/aadit/.codex-w amit:C:/Users/aadit/.codex-amit default:C:/Users/aadit/.codex; do
  name=${entry%%:*}; home=${entry#*:}
  line=""
  for f in $(ls -t "$home"/sessions/*/*/*/*.jsonl 2>/dev/null | head -200); do
    line=$(grep -o '"rate_limits":{"limit_id":"codex"[^}]*}[^}]*}' "$f" | tail -1)
    [ -n "$line" ] && break
  done
  if [ -z "$line" ]; then echo "$name: no data"; continue; fi
  p=$(echo "$line" | grep -o '"primary":{[^}]*}'); s=$(echo "$line" | grep -o '"secondary":{[^}]*}')
  pu=$(echo "$p" | grep -o '"used_percent":[0-9.]*' | cut -d: -f2); pr=$(echo "$p" | grep -o '"resets_at":[0-9]*' | cut -d: -f2)
  su=$(echo "$s" | grep -o '"used_percent":[0-9.]*' | cut -d: -f2); sr=$(echo "$s" | grep -o '"resets_at":[0-9]*' | cut -d: -f2)
  now=$(date +%s)
  [ -n "$pr" ] && [ "$pr" -lt "$now" ] && pu="0 (reset)"
  [ -n "$sr" ] && [ "$sr" -lt "$now" ] && su="0 (reset)"
  echo "$name: 5h ${pu}% (resets $(date -d @$pr '+%a %H:%M' 2>/dev/null)) | wk ${su}% (resets $(date -d @$sr '+%a %H:%M' 2>/dev/null))"
done
