# Task: prune stale entries from the preconfigured model catalog

## Problem

`src/swiss_ai_model_launch/assets/models.json` (the preconfigured model
catalog, ~31 entries) contains entries whose model weights no longer exist on
Clariden, so launching them fails at vllm/sglang startup with
`OSError: Can't load the configuration of '<path>' ... containing a config.json`.

Confirmed example: `swiss-ai/Apertus-1.5-8B-Instruct` has
`model_path: /capstor/store/cscs/swissai/infra01/models/apertus-8b-sft-1.5--lr8e-5-MaxMin_4096-Filtered-dpo-lr1e-06-beta25.0-lenNormTrue-ebs128-ep1`
— that directory was moved (the models dir was reorganised into
`Alignment/`, `GRPO/`, `SFT/` subfolders around July; this checkpoint now
lives under `Alignment/trials/<same dirname>`). Others may be stale too.

## What to do

Check EVERY entry in `models.json`; delete every entry whose weights don't
exist. Do not fix paths or dedupe — delete stale, keep the rest untouched.

1. For each entry, the path that gets loaded is:
   - `model_path` if the entry has one, otherwise
   - `/capstor/store/cscs/swissai/infra01/hf_models/models/<model>`
     (the default registry — see `_REMOTE_MODEL_REGISTRY` /
     `resolve_model_path` in `src/swiss_ai_model_launch/launchers/`).

2. Check existence over SSH. `ssh clariden` works from this machine (alias in
   `~/.ssh/config`, jump via ela). A path counts as VALID only if the
   directory exists AND contains `config.json`. Do all paths in ONE ssh
   invocation, e.g. generate a script:

   ```bash
   ssh clariden 'while read -r p; do
     if [ -f "$p/config.json" ]; then echo "OK $p"; else echo "MISSING $p"; fi
   done' < paths.txt
   ```

3. Delete every `MISSING` entry from `models.json`. Notes:
   - `swiss-ai/Apertus-8B-Instruct-2509` appears twice (sglang and vllm) —
     intentional, keep both if valid.
   - Some entries use `nodes_per_worker` instead of `nodes_per_replica` —
     leave as-is, it's not your concern.
   - Do not touch `environment` / `framework_args` / anything else.

4. Validate: `python -c "import json; json.load(open('src/swiss_ai_model_launch/assets/models.json'))"`
   and run the repo's tests (`ModelCatalogEntry` must still parse every
   remaining entry — `get_preconfigured_models` has coverage).

5. Report which entries you deleted and why (the missing path for each), then
   open a PR against swiss-ai/model-launch.

## Known-good already (spot-checked 2026-08-20, no need to re-verify but fine if you do)

All gemma entries, Apertus-70B/8B-Instruct-2509, Apertus-1.5-8B-gbs512/cooldown
entries were verified present. The only known-stale one is
`swiss-ai/Apertus-1.5-8B-Instruct`.

## Why this matters downstream

evals-svc (github.com/swiss-ai/evals-svc) serves this catalog verbatim to its
frontend via `GET /v1/evals/preconfigured-models`, read from the installed
package — stale entries become one-click broken launches for eval users. After
the PR merges, evals-svc picks it up on its next image build (git dependency).
