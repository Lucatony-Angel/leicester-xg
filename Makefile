# Rebuild everything with `make all`, or run one stage. Activate the virtual environment first.
PYTHON ?= python

.PHONY: all data db model figures test test-slow lint app

all: data db model figures

data:     ## Download StatsBomb events (about 15 minutes, once)
	$(PYTHON) src/download.py

db:       ## Build DuckDB tables, features and the processed files the app reads
	$(PYTHON) src/build_db.py

model:    ## Train the xG models, then the shot funnel stages used for the defensive split
	$(PYTHON) src/train.py
	$(PYTHON) src/funnel.py

figures:  ## Metrics, charts and results tables
	$(PYTHON) src/evaluate.py
	$(PYTHON) src/analysis.py
	$(PYTHON) src/champions.py
	$(PYTHON) src/formations.py

test:
	ruff check src app tests
	pytest

test-slow:  ## Full rebuild and retrain from the raw download, compared with the committed data
	pytest -m slow

lint:
	ruff check src app tests

app:
	streamlit run app/streamlit_app.py
