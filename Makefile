.PHONY: install data pipeline api dashboard test docker clean

install:
	pip install -e ".[dl,llm,dashboard,dev]"

data:
	python -m aeroforesight.data.generate

pipeline:
	python -m aeroforesight.mlops.pipeline

api:
	uvicorn aeroforesight.serving.api:app --reload --port 8000

dashboard:
	streamlit run dashboard/streamlit_app.py

test:
	pytest -q

docker:
	docker compose up --build

clean:
	rm -rf artifacts data/*.parquet data/*.csv .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
