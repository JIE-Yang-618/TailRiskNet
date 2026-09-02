.PHONY: install test lint demo clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest --cov=tailrisknet --cov-report=term-missing --cov-fail-under=70

lint:
	ruff check src tests examples
	ruff format --check src tests examples

demo:
	python -m tailrisknet run --config configs/demo.yaml

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('.').rglob('*.pyc')]"
