.PHONY: lint format mypy shellcheck markdownlint dockerlint static _test-lightweight _test-medium _test-comprehensive test-lightweight test-medium test-comprehensive clean-cache clean-dev

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

mypy:
	mypy

shellcheck:
	find . -name "*.sh" -not -path "./legacy/*" -not -path "./.venv/*" | xargs shellcheck

markdownlint:
	npx markdownlint-cli2 --config .markdownlint.yaml "**/*.md" "!legacy/**/*.md" "!.venv/**/*.md"

dockerlint:
	find images/ -name "Dockerfile*" | while read -r f; do echo "--- $$f"; docker run --rm -i docker.io/hadolint/hadolint < "$$f"; done

static: lint mypy shellcheck markdownlint dockerlint

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

clean-dev: clean-cache
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".venv" -exec rm -rf {} +
	rm -f .coverage
	rm -f coverage.xml
