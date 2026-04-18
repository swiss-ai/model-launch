# CI/CD

This repository has several CI/CD workflows to ensure consistency and reliability of the services. The pipeline runs in three sequential stages:

> Static Checks → Docker Image Builds → Integration Tests

Each stage starts only after the previous one passes successfully.

## Static Checks

**Trigger**: Called by the CI workflow on every push to `main` or pull request targeting `main`, or manual dispatch.

The codebase is screened for common issues and style inconsistencies via static analysis tools. All checks run in parallel:

1. Python Lint and Format (using `ruff`)
2. Python Type Checking (using `mypy`)
3. Shell Scripts Lint (using `shellcheck`)
4. Docker Linting (using `hadolint`)
5. Markdown Linting (using `markdownlint`)
6. TOML Format (using `taplo`)
7. JSON & YAML Format (using `prettier`)
