# One-command setup + run:  make run   (then open http://localhost:8500)

PYTHON ?= python3
VENV   ?= .venv
PORT   ?= 8500
BIN     = $(VENV)/bin

.PHONY: run test clean help

help:
	@echo "make run    create the venv, install dependencies, start the app on :$(PORT)"
	@echo "make test   run the test suite"
	@echo "make clean  remove the venv (keeps data/ caches)"

# Stamp files make reinstalls happen only when the requirements change.
$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.txt
	touch $@

$(VENV)/.installed-dev: requirements-dev.txt $(VENV)/.installed
	$(BIN)/python -m pip install -r requirements-dev.txt
	touch $@

run: $(VENV)/.installed
	$(BIN)/python -m uvicorn server:app --port $(PORT)

test: $(VENV)/.installed-dev
	$(BIN)/python -W ignore -m pytest tests -q

clean:
	rm -rf $(VENV)
