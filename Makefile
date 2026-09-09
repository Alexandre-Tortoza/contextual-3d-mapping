PYTHON ?= python
RESOLVED_PYTHON := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
M1_ARTIFACT ?= artifacts/m1-demo.json
M1_PCD_SOURCE ?= artifacts/corridor-02-fastlio-20s.pcd
M1_PCD_ARTIFACT ?= artifacts/corridor-02-fastlio-20s.json
M1_CONTEXT_ARTIFACT ?= artifacts/corridor-02-fastlio-20s-context.json
M1_VISUAL_RUN ?= modules/visual-perception/benchmarks/results/samples/20260909T135428Z
MAP_EXPLORER_ARTIFACT ?= $(M1_CONTEXT_ARTIFACT)
MAP_EXPLORER_ASSETS := $(basename $(MAP_EXPLORER_ARTIFACT))-assets
FASTLIO_SEGMENT_SECONDS ?= 20
M1_PYTHONPATH := $(CURDIR)/contracts:$(CURDIR)/modules/state-estimation/src:$(CURDIR)/modules/geometric-map/src:$(CURDIR)/modules/sensor-association/src:$(CURDIR)/apps/mapping-runtime/src

.PHONY: verify test lint typecheck corridor-02-map corridor-02-context m1-demo m1-pcd-slice m1-test map-explorer-install map-explorer-test map-explorer-build map-explorer-serve

verify: test lint typecheck

test:
	$(RESOLVED_PYTHON) -m pytest

lint:
	$(RESOLVED_PYTHON) -m ruff check .

typecheck:
	cd modules/visual-perception && $(RESOLVED_PYTHON) -m mypy

m1-demo:
	PYTHONPATH="$(M1_PYTHONPATH)" $(RESOLVED_PYTHON) -m mapping_runtime demo --output "$(M1_ARTIFACT)"

m1-pcd-slice:
	PYTHONPATH="$(M1_PYTHONPATH)" $(RESOLVED_PYTHON) -m mapping_runtime pcd-slice \
		"$(M1_PCD_SOURCE)" \
		--output "$(M1_PCD_ARTIFACT)" \
		--map-id corridor-02-fastlio-20s \
		--map-frame map

corridor-02-map:
	docker compose --profile ros1 run --rm fastlio-ros1 bash \
		/workspace/apps/mapping-runtime/scripts/run_fastlio_ros1_segment.sh \
		/workspace/datasets/raw/corridor-02/corridor-02.bag \
		/workspace/modules/state-estimation/configs/corridor-02-velodyne.yaml \
		"$(FASTLIO_SEGMENT_SECONDS)" \
		/workspace/$(M1_PCD_SOURCE)
	$(MAKE) m1-pcd-slice PYTHON="$(PYTHON)"

corridor-02-context:
	PYTHONPATH="$(M1_PYTHONPATH)" $(RESOLVED_PYTHON) -m mapping_runtime corridor-02-context \
		--geometric-slice "$(M1_PCD_ARTIFACT)" \
		--bag datasets/raw/corridor-02/corridor-02.bag \
		--intrinsics datasets/raw/corridor-02/corridor-02-Intrinsics.yaml \
		--extrinsics datasets/raw/corridor-02/corridor-02-extrinsics.yaml \
		--ground-truth datasets/raw/corridor-02/corridor-02-gt.txt \
		--visual-observation "$(M1_VISUAL_RUN)/frames/corridor-02-000/observation.json" \
		--raw-image "$(M1_VISUAL_RUN)/frames/corridor-02-000/raw.png" \
		--overlay-image "$(M1_VISUAL_RUN)/frames/corridor-02-000/regions-overlay.png" \
		--valid-area-mask "$(M1_VISUAL_RUN)/frames/corridor-02-000/valid-area-mask.png" \
		--output "$(M1_CONTEXT_ARTIFACT)"

m1-test:
	PYTHONPATH="$(M1_PYTHONPATH)" $(RESOLVED_PYTHON) -m pytest \
		contracts/tests \
		modules/state-estimation/tests \
		modules/geometric-map/tests \
		modules/sensor-association/tests \
		apps/mapping-runtime/tests

map-explorer-install:
	cd apps/map-explorer/web && npm ci

map-explorer-test:
	cd apps/map-explorer/web && npm test

map-explorer-build:
	cd apps/map-explorer/web && npm run build

map-explorer-serve:
	mkdir -p apps/map-explorer/web/public
	cp "$(MAP_EXPLORER_ARTIFACT)" apps/map-explorer/web/public/current-map.json
	if [ -d "$(MAP_EXPLORER_ASSETS)" ]; then cp -R "$(MAP_EXPLORER_ASSETS)" apps/map-explorer/web/public/; fi
	cd apps/map-explorer/web && npm run dev -- --host 0.0.0.0
