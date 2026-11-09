# Software4

Research project: a brain-inspired LLM system that keeps learning after deployment. The current plan is in `PLAN.md`; **read its Direction section first** (keep the frozen core, original work goes into a grounded value function, stages T → V1 → V2 → V3 in order). Papers, critiques and reviews are in `research/`. Diagrams, design choices and milestones are in `docs/architecture.html`.

## How to talk to the user
- Keep answers short: a few bullets or a small table. Expand only when asked.
- **Status = percentages.** Whenever the user asks for status (including a bare "status" or "%"), report:
  - **Each running task:** an estimated % done, plus one line on what it's doing. Base the estimate on steps finished out of the total, or on output so far against what the task needs. Say that it's an estimate.
  - **Claude 5-hour limit:** % used, if the usage tool is available.
  - **Codex accounts:** run `python scripts/codex-usage.py --once` and give the 5h and weekly % left per account (live, no tokens used).
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
- **Usage %:** `python scripts/codex-usage.py` asks ChatGPT's usage page for each home, one short line per account (5h left, week left, 5h reset in IST). It uses no model tokens and no Claude limits. `scripts/codex_usage.sh` is the old log-based version and can be stale.
- **Live usage terminal:** whenever Codex jobs are running or queued, keep `python D:\Aadit\claudez\software4\scripts\codex-usage.py` open in a Terminal-panel tab titled "Codex usage" (open one if none exists, reuse it if it does). Close it when no Codex work is left.
- **Parallel, one job per account:** run up to 3 Codex jobs at once, one on each account. Don't use `run_rotate.sh <queue> 3`: it tries accounts in the same order, so every job lands on the first free account, and a Plus account lasts only about 15–30 minutes under 3–4 parallel high-effort runs. Instead start one queue per account, each with parallel 1 and a different account first, so a job moves on only when its account hits the limit:
  ```bash
  A7859=7859:C:/Users/aadit/.codex-7859; AW=w:C:/Users/aadit/.codex-w; AAMIT=amit:C:/Users/aadit/.codex-amit
  HOMES_OVERRIDE="$A7859 $AW $AAMIT" scripts/run_rotate.sh <queue_1> 1
  HOMES_OVERRIDE="$AW $AAMIT $A7859" scripts/run_rotate.sh <queue_2> 1
  HOMES_OVERRIDE="$AAMIT $A7859 $AW" scripts/run_rotate.sh <queue_3> 1
  ```
  Launch each through Bash `run_in_background`. The script always runs in the main checkout, so jobs that write code must either add only their own named new files, or work in a separate `git worktree` named in the brief. Never switch branches in the main checkout during a job, and never let two jobs write the same file.
- **Switching or re-signing an account** (if it shows logged out):
  1. Run `CODEX_HOME=<home> codex login` in the background.
  2. Open the printed URL in the browser, pick the account, and click Continue.
  3. Resume with `codex exec resume --last`, or relaunch the brief.
  - Never type passwords. If a password prompt appears, stop and ask the user.
  - The current account is the `email` claim in the id_token in `<home>/auth.json`.
- Codex usage doesn't count against Claude's limits, so prefer Codex for heavy research and critic runs. Use Sonnet/Haiku subagents for cheap checks.

## Kaggle: don't waste GPU hours
Budget is about 30 GPU-hours a week per Kaggle account; the user may use more than one account (friends' accounts included, signed in by the user, never by typing passwords). Run jobs with `scripts/kaggle_run.sh <folder> [accelerator] [timeout_s]`.

**Before pushing:**
1. **Test the code without a GPU.** Run the script locally on a tiny random-weight model, so bugs and library errors show up for free.
2. **Add a smoke mode (`S4_SMOKE=1`) and run it first on Kaggle.** It does 2 training steps on the *longest* batch, plus a short eval, in under 5 minutes. This catches out-of-memory errors. Memory depends on the longest sequence × batch × 248K vocab, so don't guess.
3. **Get a review before any run over 30 minutes.** Have Codex read the script (see below).
4. **Pin versions that have already bitten us:** uninstall Kaggle's old `torchao`, because it breaks peft.

**During the run:**
5. **Use both T4s.** Treat the second GPU as wasted unless it's working (we assume Kaggle charges session time, not per GPU; unverified). Run independent tests as one process per GPU with `CUDA_VISIBLE_DEVICES`. Run in series only what really depends on an earlier result.
6. **Save expensive work and reuse it.** Save baselines, replay sets and trained adapters to `/kaggle/working`. Attach the earlier run through `kernel_sources` so reruns skip them.
7. **Stop on failure.** If a test fails, skip every test that depends on it, because its numbers would be meaningless. Don't rerun tests that already finished.

**After the run:**
8. **Report only real timings.** Kaggle's `lastRunTime` is in UTC (`kaggle kernels list --mine`). Check it before saying how long a run has taken, and never guess why it's slow.
9. **Log every run** in that step's `RESULTS.md`: time used, result, and running GPU-hours for the week.
10. **Results only appear when a run ends.** Watch with a `run_in_background` loop. If the laptop is closed the job continues, but nobody is notified.

## Who does what
| Work | Who |
|---|---|
| Decisions, pushing to Kaggle, applying patches, final numbers | Claude (main chat) |
| Experiment design review, deep paper reads, hard bugs | Codex `gpt-6.1-sol` |
| Reviewing Kaggle scripts, writing routine test code as a patch, summarising logs | Codex `gpt-6.1-terra` |

- **Use Codex first** for any of the Codex work above, up to one job per account at once (see "Parallel, one job per account"). Use Claude subagents only when all 3 accounts are at their limit.
- Codex runs read-only and returns patches or reports. Claude checks them, applies them and runs them.
- Check a Codex claim before relying on it.
- Quick checks stay in the main chat.

## Working rules
- **Milestone page:** when a build step starts or finishes, or a design choice changes, update `docs/architecture.html` in the same change (steps in the comment at its top).
- **Pages:** the README links to the Claude artifacts. Keep an HTML copy of every page in the repo, because the Claude account may not renew. When a page changes, update the copy and republish the artifact together.
- **Hardware:** the user's GTX 1650 (4 GB VRAM, no bf16, Windows). Use fp16, and benchmark VRAM before you assume anything fits. Run heavier training on free Kaggle, following the rules above.
- **Citations:** verify every claim before relying on it, and mark unverified ones.
