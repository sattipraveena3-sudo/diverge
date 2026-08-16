install:
	pip install -e '.[dev]'
test:
	pytest
lint:
	ruff check .
evaluate:
	python scripts/evaluate.py
serve:
	uvicorn api.main:app --reload
dashboard:
	streamlit run dashboard/app.py
