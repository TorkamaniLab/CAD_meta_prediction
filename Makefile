# Makefile

format:
	ruff format .

lint:
	ruff check .

check: format lint
