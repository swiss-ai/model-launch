# Agent instructions

Rules for coding agents (and humans) working in this repository.

## Git workflow

- Make every change on a new, dedicated branch created from an up-to-date `origin/main`, named with a `feat/`, `fix/`, `chore/`, `docs/` or `ci/` prefix.
- Never commit to `main`, and never commit to a branch you did not create for the task at hand.
- Do not switch a checkout that is on another branch or has uncommitted changes; other people or agents may be using it. Create a worktree instead:

  ```bash
  git fetch origin
  git worktree add ../model-launch-<topic> -b <prefix>/<topic> origin/main
  ```

- Keep commits atomic (one logical change each, checks passing) and land changes through a pull request to `main`.
