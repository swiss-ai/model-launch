.PHONY: lint format mypy dmypy shellcheck markdownlint dockerlint tomlfmt prettier static _test-lightweight _test-medium _test-comprehensive test-lightweight test-medium test-comprehensive clean-cache clean-dev

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .
	find . -name "*.toml" -not -path "./legacy/*" -not -path "./.venv/*" | xargs taplo fmt
	find . \( -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) -not -path "./legacy/*" -not -path "./.venv/*" | xargs npx prettier --write

mypy:
	uv run --frozen mypy

dmypy:
	uv run --frozen dmypy run -- src

shellcheck:
	find . -name "*.sh" -not -path "./legacy/*" -not -path "./.venv/*" | xargs shellcheck

markdownlint:
	npx markdownlint-cli2 --config .markdownlint.yaml "**/*.md" "!legacy/**/*.md" "!.venv/**/*.md"

dockerlint:
	find images/ -name "Dockerfile*" | while read -r f; do echo "--- $$f"; docker run --rm -i docker.io/hadolint/hadolint < "$$f"; done

tomlfmt:
	find . -name "*.toml" -not -path "./legacy/*" -not -path "./.venv/*" | xargs taplo fmt --check

prettier:
	find . \( -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) -not -path "./legacy/*" -not -path "./.venv/*" | xargs npx prettier --check

static: lint mypy shellcheck markdownlint dockerlint tomlfmt prettier

_test-lightweight:
	pytest -m lightweight --cov --cov-report=term-missing -n auto

_test-medium:
	pytest -m medium --cov --cov-report=term-missing -n auto

_test-comprehensive:
	pytest -m full --cov --cov-report=term-missing -n auto

test-lightweight:
	. ./.test.sh && $(MAKE) _test-lightweight

test-medium:
	. ./.test.sh && $(MAKE) _test-medium

test-comprehensive:
	. ./.test.sh && $(MAKE) _test-comprehensive

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
