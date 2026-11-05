# Software4

Research project: a brain-inspired LLM system that keeps learning after deployment. The current plan is in `PLAN.md`. Papers, critiques and reviews are in `research/`.

## How to talk to the user
- Keep answers short: a few bullets or a small table. Expand only when asked.
- **Status = percentages.** Whenever the user asks for status (including a bare "status" or "%"), report:
  - **Each running task:** an estimated % done, plus one line on what it's doing. Base the estimate on steps finished out of the total, or on output so far against what the task needs. Say that it's an estimate.
  - **Claude 5-hour limit:** % used, if the usage tool is available.
  - **Codex accounts:** run `bash scripts/codex_usage.sh` and give the 5h and weekly % per account.
  - Never answer just "still running".

## Codex (OpenAI) — three accounts
- The CLI is `codex`, on PATH through shims in `C:\Users\aadit\bin`. Models are `gpt-6.1-sol` (strongest) and `gpt-6.1-terra` (cheaper).
- Account homes (CODEX_HOME):
  - 7859: `C:/Users/aadit/.codex-7859`
  - w: `C:/Users/aadit/.codex-w`
  - amit: `C:/Users/aadit/.codex-amit`
  - `~/.codex` is the app's own default (currently w), so don't use it for rotation.
- **Run one job:**
  ```bash
  CODEX_HOME=C:/Users/aadit/.codex-7859 codex --search exec -m gpt-6.1-sol -c model_reasoning_effort='"high"' -s read-only --skip-git-repo-check -C D:/Aadit/claudez/software4 -o out.md - < prompt.md
  ```
  - Always feed stdin (`- < prompt.md`, or `< /dev/null`). Otherwise it can hang on "Reading additional input from stdin".
  - Launch through Bash `run_in_background`, with a description like "Codex (gpt-6.1-sol): <job>", so the run shows up as a background task.
  - Check its log within a few minutes.
- **Run many jobs with automatic rotation:** use `scripts/run_rotate.sh <queue> [parallel]`.
  - Each queue line is `slug|prompt file|output file`.
  - An account that hits its limit is marked exhausted, and the job moves to the next account.
  - When every account is exhausted, the remaining jobs are skipped. Finish those on Sonnet subagents.
  - `MODEL` / `EFFORT` env vars override the default terra/medium.
- **Usage %:** `bash scripts/codex_usage.sh` reads each home's newest session log. The figure is from that account's last run.
- **Spread the load:** a Plus account lasts only about 15–30 minutes under 3–4 parallel high-effort runs. Spread jobs over all three, and keep at most one Codex run per account once it's past ~50%.
- **Switching or re-signing an account** (if it shows logged out):
  1. Run `CODEX_HOME=<home> codex login` in the background.
  2. Open the printed URL in the browser, pick the account, and click Continue.
  3. Resume with `codex exec resume --last`, or relaunch the brief.
  - Never type passwords. If a password prompt appears, stop and ask the user.
  - The current account is the `email` claim in the id_token in `<home>/auth.json`.
- Codex usage doesn't count against Claude's limits, so prefer Codex for heavy research and critic runs. Use Sonnet/Haiku subagents for cheap checks.

## Working rules
- **Hardware:** the user's GTX 1650 (4 GB VRAM, no bf16, Windows). Use fp16, and benchmark VRAM before you assume anything fits. Run heavier training on free Kaggle.
- **Citations:** verify every claim before relying on it, and mark unverified ones.
