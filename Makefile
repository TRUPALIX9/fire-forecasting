PY=python
VENV=.venv

venv:
	python3 -m venv $(VENV); . $(VENV)/bin/activate; pip install --upgrade pip

install:
	. $(VENV)/bin/activate; pip install -r requirements.txt

run-train:
	. $(VENV)/bin/activate; $(PY) -m src.experiments.run_pipeline --config config.yaml

run-backend:
	. $(VENV)/bin/activate; uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

learning-curve:
	. $(VENV)/bin/activate; $(PY) -m src.experiments.learning_curve --config config.yaml

threshold:
	. $(VENV)/bin/activate; $(PY) -m src.experiments.threshold_sweep --config config.yaml

.PHONY: fetch-firms fetch-frap fetch-raws run-train run-backend

fetch-firms:
	. $(VENV)/bin/activate; $(PY) -m scripts.fetch_firms_data

fetch-frap:
	. $(VENV)/bin/activate; $(PY) -m scripts.fetch_frap_bbox

fetch-raws:
	. $(VENV)/bin/activate; $(PY) -m scripts.fetch_raws_data
