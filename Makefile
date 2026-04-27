.PHONY: install-dev lint format check mypy dmypy shellcheck markdownlint dockerlint tomlfmt prettier static test _test-lightweight _test-medium _test-comprehensive test-lightweight test-medium test-comprehensive docs docs-build demo clean-cache clean-dev

install-dev:
	uv venv --python 3.12
	uv pip install -e ".[dev]"
	uv run pre-commit install
	@command -v taplo >/dev/null 2>&1 || { \
		echo ""; \
		echo "WARNING: 'taplo' not found on PATH."; \
		echo "  Install it for 'make format' / pre-commit to work:"; \
		echo "    macOS:  brew install taplo"; \
		echo "    Linux:  curl -fsSL https://github.com/tamasfe/taplo/releases/download/0.9.3/taplo-full-linux-x86_64.gz | gzip -d > /usr/local/bin/taplo && chmod +x /usr/local/bin/taplo"; \
		echo ""; \
	}

lint:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

format:
	uv run --frozen ruff format .
	uv run --frozen ruff check --fix .
	find . -name "*.toml" -not -path "./legacy/*" -not -path "./.venv/*" | xargs taplo fmt
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
	find . -name "*.toml" -not -path "./legacy/*" -not -path "./.venv/*" | xargs taplo fmt --check

prettier:
	find . \( -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) -not -path "./legacy/*" -not -path "./.venv/*" | xargs npx prettier --check

static: lint mypy shellcheck markdownlint dockerlint tomlfmt prettier

test:
	uv run --frozen pytest tests/unit/ --cov=src --cov-report=term-missing

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

# Timestamp (in seconds) where the TUI / log viewer starts in the raw VHS render.
# Eyeball this once after the first `make demo-raw`, then set DEMO_TUI_START
# accordingly. Everything after this point is sped up DEMO_SPEEDUP times.
DEMO_TUI_START ?= 15
DEMO_SPEEDUP ?= 5

demo-raw:
	@command -v vhs >/dev/null 2>&1 || { echo "vhs not found. Install with: brew install vhs"; exit 1; }
	mkdir -p docs/assets
	uv run vhs tapes/launch-apertus.tape

demo: demo-raw
	@command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found. Install with: brew install ffmpeg"; exit 1; }
	ffmpeg -y -i docs/assets/launch-apertus.mp4 -filter_complex \
		"[0:v]trim=0:$(DEMO_TUI_START),setpts=PTS-STARTPTS[wiz]; \
		 [0:v]trim=$(DEMO_TUI_START),setpts=(PTS-STARTPTS)/$(DEMO_SPEEDUP)[tui]; \
		 [wiz][tui]concat=n=2:v=1[out]" \
		-map "[out]" docs/assets/launch-apertus-fast.mp4
	ffmpeg -y -i docs/assets/launch-apertus-fast.mp4 \
		-vf "fps=15,scale=1200:-1:flags=lanczos" \
		docs/assets/launch-apertus.gif

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
