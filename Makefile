.PHONY: format test clean-cache clean-dev

format:
	ruff format .
	ruff check --fix .

test:
	. ./.test.sh && pytest --cov --cov-report=term-missing -n auto

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
