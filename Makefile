PYTHON=.venv/bin/python
PIP=.venv/bin/pip

.PHONY: setup test run

setup:
	python3 -m venv .venv
	$(PIP) install -e ".[dev]"

test:
	PYTHONPATH=src $(PYTHON) -m pytest

run:
	MPLCONFIGDIR=/tmp/mpl LOKY_MAX_CPU_COUNT=8 PYTHONPATH=src $(PYTHON) -m promo_uplift.cli --config configs/default.yaml

