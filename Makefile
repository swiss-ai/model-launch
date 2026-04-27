.PHONY: install-dev lint format check mypy dmypy shellcheck markdownlint dockerlint tomlfmt prettier static _test-lightweight _test-medium _test-comprehensive test-lightweight test-medium test-comprehensive docs docs-build clean-cache clean-dev

install-dev:
	uv venv --python 3.12
	uv pip install -e ".[dev]"
	uv run pre-commit install

lint:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

format:
	uv run --frozen ruff format .
	uv run --frozen ruff check --fix .
	find . -name "*.toml" -not -path "./legacy/*" -not -path "./.venv/*" | xargs npx --yes @taplo/cli fmt
	find . \( -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) -not -path "./legacy/*" -not -path "./.venv/*" | xargs npx prettier --write
	-npx markdownlint-cli2 --config .markdownlint.yaml --fix "**/*.md" "!legacy/**/*.md" "!.venv/**/*.md"

check: static

mypy:
	uv run --frozen mypy src

dmypy:
	uv run --frozen dmypy run -- src

shellcheck:
	find . -name "*.sh" -not -path "./legacy/*" -not -path "./.venv/*" | xargs uv run --frozen shellcheck

markdownlint:
	npx markdownlint-cli2 --config .markdownlint.yaml "**/*.md" "!legacy/**/*.md" "!.venv/**/*.md"

dockerlint:
	find images/ -name "Dockerfile*" | xargs uv run --frozen hadolint

tomlfmt:
	find . -name "*.toml" -not -path "./legacy/*" -not -path "./.venv/*" | xargs npx --yes @taplo/cli fmt --check

prettier:
	find . \( -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) -not -path "./legacy/*" -not -path "./.venv/*" | xargs npx prettier --check

static: lint mypy shellcheck markdownlint dockerlint tomlfmt prettier

_test-lightweight:
	uv run --frozen pytest -m lightweight --cov --cov-report=term-missing -n auto

_test-medium:
	uv run --frozen pytest -m medium --cov --cov-report=term-missing -n auto

_test-comprehensive:
	uv run --frozen pytest -m full --cov --cov-report=term-missing -n auto

test-lightweight:
	. ./.test.sh && $(MAKE) _test-lightweight

test-medium:
	. ./.test.sh && $(MAKE) _test-medium

test-comprehensive:
	. ./.test.sh && $(MAKE) _test-comprehensive

docs:
	uv pip install -e ".[docs]" --quiet
	uv run mkdocs serve

docs-build:
	uv pip install -e ".[docs]" --quiet
	uv run mkdocs build --strict

clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	-uv run --frozen dmypy stop 2>/dev/null; rm -f .dmypy.json

clean-dev: clean-cache
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".venv" -exec rm -rf {} +
	rm -f .coverage
	rm -f coverage.xml
