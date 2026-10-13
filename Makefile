.PHONY: test run up up-local
test:
	python -m pytest -q
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000
up:
	docker compose up --build
up-local:
	docker compose --profile local-llm up --build
