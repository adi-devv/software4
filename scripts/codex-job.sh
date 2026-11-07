#!/usr/bin/env bash
# Run one Codex job in the foreground and record its status under .codex-jobs/.
# Usage: scripts/codex-job.sh <label> <model> <effort> <sandbox> <brief.md>
# Launch it as a background Bash call; read only .codex-jobs/<label>.out.md when it finishes.
set -u
if [ "$#" -ne 5 ]; then
  echo "usage: $0 <label> <model> <effort> <sandbox> <brief.md>" >&2
  exit 2
fi
label=$1 model=$2 effort=$3 sandbox=$4 brief=$5
case "$label" in
  ''|*[!A-Za-z0-9._-]*|.*) echo "label must match [A-Za-z0-9][A-Za-z0-9._-]*" >&2; exit 2 ;;
esac
[ -r "$brief" ] || { echo "brief not readable: $brief" >&2; exit 2; }

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dir=$root/.codex-jobs
mkdir -p "$dir"
status=$dir/$label.status
start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
write_status() { # state end exit
  printf 'label=%s\nmodel=%s\neffort=%s\nsandbox=%s\nstate=%s\nstart=%s\nend=%s\nexit=%s\n' \
    "$label" "$model" "$effort" "$sandbox" "$1" "$start" "$2" "$3" > "$status"
}

write_status running "" ""
export NODE_OPTIONS="--require $(cygpath -m "$root")/scripts/node-userinfo-shim.cjs ${NODE_OPTIONS:-}"
network_config=()
if [ "${CODEX_JOB_NETWORK_ACCESS:-0}" = "1" ]; then
  network_config=(-c "sandbox_workspace_write.network_access=true")
fi
# shellcheck disable=SC2086
codex exec ${CODEX_JOB_EXTRA:-} --skip-git-repo-check -m "$model" -c "model_reasoning_effort=$effort" "${network_config[@]}" -s "$sandbox" -C "$root" \
  -o "$dir/$label.out.md" - < "$brief" > "$dir/$label.log" 2>&1
code=$?
write_status "$([ "$code" -eq 0 ] && echo done || echo failed)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code"
exit "$code"
