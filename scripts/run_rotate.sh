#!/usr/bin/env bash
# Runs Codex web-research jobs from a queue, N at a time, rotating Codex accounts on usage limits.
# Queue line: slug|prompt file|output file
# Accounts are CODEX_HOME folders tried in order; one that hits its limit gets an EXHAUSTED_<name>
# marker in $STATE and the job retries on the next one. When all are exhausted, ALL_LIMIT is written
# and remaining jobs are skipped (finish them with Sonnet subagents).
# SANDBOX=workspace-write lets Codex edit files (coding jobs); default read-only.
# Usage: run_rotate.sh <queue> [parallel=4]
cd "$(dirname "$0")"
QUEUE="$1"; PAR="${2:-4}"
export STATE="$PWD/rotate-state"; mkdir -p "$STATE"
export HOMES="${HOMES_OVERRIDE:-7859:C:/Users/aadit/.codex-7859 w:C:/Users/aadit/.codex-w amit:C:/Users/aadit/.codex-amit}"
export PATH="/c/Users/aadit/.codex/plugins/.plugin-appserver:$PATH"
export MODEL="${MODEL:-gpt-6.1-terra}" EFFORT="${EFFORT:-medium}"

one() {
  IFS='|' read -r slug prompt out <<< "$1"
  [ -s "$out" ] && { echo "$slug done-before"; return; }
  for entry in $HOMES; do
    name=${entry%%:*}; home=${entry#*:}
    [ -f "$STATE/ALL_LIMIT" ] && break
    [ -f "$STATE/EXHAUSTED_$name" ] && continue
    [ -f "$home/auth.json" ] || continue
    CODEX_HOME="$home" codex --search exec -m "$MODEL" -c model_reasoning_effort="\"$EFFORT\"" -s "${SANDBOX:-read-only}" \
      --ephemeral --skip-git-repo-check -C "D:/Aadit/claudez/software4" -o "$out" - < "$prompt" > "$STATE/$slug.log" 2>&1
    rc=$?
    if grep -qiE "usage limit|rate limit|quota exceeded|hit your limit" "$STATE/$slug.log"; then
      date -u +%FT%TZ > "$STATE/EXHAUSTED_$name"; rm -f "$out"; echo "$slug LIMIT on $name, rotating"; continue
    fi
    echo "$slug exit $rc on $name $( [ -s "$out" ] && echo ok || echo EMPTY )"; return
  done
  avail=0
  for entry in $HOMES; do name=${entry%%:*}; home=${entry#*:}
    [ -f "$home/auth.json" ] && [ ! -f "$STATE/EXHAUSTED_$name" ] && avail=1; done
  [ "$avail" = 0 ] && touch "$STATE/ALL_LIMIT"
  echo "$slug SKIPPED (all accounts at limit)"
}
export -f one
grep -v '^\s*$' "$QUEUE" | xargs -d '\n' -P "$PAR" -I{} bash -c 'one "$@"' _ {}
