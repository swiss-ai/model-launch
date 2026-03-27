.PHONY: format _test-lightweight _test-comprehensive test-lightweight test-comprehensive clean-cache clean-dev

format:
	ruff format .
	ruff check --fix .

_test-lightweight:
	pytest -m lightweight --cov --cov-report=term-missing -n auto

_test-comprehensive:
	pytest -m full --cov --cov-report=term-missing -n auto

test-lightweight:
	. ./.test.sh && $(MAKE) _test-lightweight

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
