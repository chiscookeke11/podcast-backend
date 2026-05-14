install:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --port 8000

test-pipeline:
	python test_pipeline.py

lint:
	ruff check app/

format:
	ruff format app/