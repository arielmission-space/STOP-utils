.PHONY: help install test check format clean

help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies using poetry"
	@echo "  make test      - Run tests"
	@echo "  make check     - Run type checking with mypy"
	@echo "  make format    - Format code with black and isort"
	@echo "  make clean     - Remove Python cache files"

install:
	poetry install

version:
	@echo "Current version: $(shell poetry version --short)"

test:
	poetry run pytest

check:
	poetry run mypy .

format:
	poetry run isort .  # Run isort first
	poetry run black .  # Then black to ensure final formatting consistency

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +