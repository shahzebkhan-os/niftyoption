# --- Variables ---
PYTHON := ./.venv/bin/python3
PIP := ./.venv/bin/pip
STREAMLIT := ./.venv/bin/streamlit
PYTEST := ./.venv/bin/pytest
RUFF := ./.venv/bin/ruff
BLACK := ./.venv/bin/black

# --- Phony Targets ---
.PHONY: install lint format test run backtest verify-bs dashboard clean

# --- Installation ---
install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install ruff black pytest pytest-mock pydantic-settings

# --- Quality Control ---
lint:
	$(RUFF) check .

format:
	$(BLACK) .

# --- Testing ---
test:
	$(PYTEST) tests/

verify-bs:
	$(PYTHON) verify_bs_engine.py

# --- Execution ---
run:
	PYTHONPATH=. $(PYTHON) -m src.main

dashboard:
	PYTHONPATH=. $(STREAMLIT) run src/dashboard.py

backtest:
	PYTHONPATH=. $(PYTHON) -m src.main --mode backtest

# --- Cleaning ---
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
