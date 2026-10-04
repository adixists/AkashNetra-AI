# Thin wrapper over scripts/tasks.py so the same targets work on Linux/macOS (make)
# and Windows (python scripts/tasks.py <target>). Override interpreter: make test PY=.venv/bin/python
PY ?= python

.PHONY: demo data train api dashboard test lint format

demo:       ## Synthetic end-to-end demo (M1+)
	$(PY) scripts/tasks.py demo

data:       ## Download real data (M6)
	$(PY) scripts/tasks.py data

train:      ## Build dataset and train models
	$(PY) scripts/tasks.py train

api:        ## FastAPI on :8000
	$(PY) scripts/tasks.py api

dashboard:  ## Streamlit on :8501
	$(PY) scripts/tasks.py dashboard

test:       ## pytest
	$(PY) scripts/tasks.py test

lint:       ## ruff + black --check
	$(PY) scripts/tasks.py lint

format:     ## ruff --fix + black
	$(PY) scripts/tasks.py format
