PYTHON ?= python
RESOLVED_PYTHON := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))

.PHONY: verify test lint typecheck

verify: test lint typecheck

test:
	$(RESOLVED_PYTHON) -m pytest

lint:
	$(RESOLVED_PYTHON) -m ruff check .

typecheck:
	cd modules/visual-perception && $(RESOLVED_PYTHON) -m mypy
