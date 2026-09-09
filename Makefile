PYTHON ?= python
RESOLVED_PYTHON := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
M1_ARTIFACT ?= artifacts/m1-demo.json
M1_PYTHONPATH := $(CURDIR)/contracts:$(CURDIR)/modules/state-estimation/src:$(CURDIR)/modules/geometric-map/src:$(CURDIR)/modules/sensor-association/src:$(CURDIR)/apps/mapping-runtime/src

.PHONY: verify test lint typecheck m1-demo m1-test map-explorer-install map-explorer-build

verify: test lint typecheck

test:
	$(RESOLVED_PYTHON) -m pytest

lint:
	$(RESOLVED_PYTHON) -m ruff check .

typecheck:
	cd modules/visual-perception && $(RESOLVED_PYTHON) -m mypy

m1-demo:
	PYTHONPATH="$(M1_PYTHONPATH)" $(RESOLVED_PYTHON) -m mapping_runtime demo --output "$(M1_ARTIFACT)"

m1-test:
	PYTHONPATH="$(M1_PYTHONPATH)" $(RESOLVED_PYTHON) -m pytest \
		contracts/tests \
		modules/state-estimation/tests \
		modules/geometric-map/tests \
		modules/sensor-association/tests \
		apps/mapping-runtime/tests

map-explorer-install:
	cd apps/map-explorer/web && npm ci

map-explorer-build:
	cd apps/map-explorer/web && npm run build
