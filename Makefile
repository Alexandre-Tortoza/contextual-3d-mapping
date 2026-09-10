PYTHON ?= python
RESOLVED_PYTHON := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
M1_ARTIFACT ?= artifacts/m1-demo.json
# Janela do trecho, em segundos após o primeiro frame RGB do bag. O default
# equivale a t0+180 s do ground-truth do corridor-02, escolhido por ser um
# trecho de movimento estável (34,1 m percorridos, razão deslocamento/caminho
# 0,91). Ao mudar SEGMENT_START_S, mude também SEGMENT_ID.
SEGMENT_START_S ?= 176.3
SEGMENT_SECONDS ?= 30
SEGMENT_KEYFRAME_INTERVAL_S ?= 2
SEGMENT_LEAD_S ?= 3
SEGMENT_ID ?= corridor-02-fastlio-176s-30s
M1_PCD_SOURCE ?= artifacts/$(SEGMENT_ID).pcd
M1_PCD_ARTIFACT ?= artifacts/$(SEGMENT_ID).json
M1_CONTEXT_ARTIFACT ?= artifacts/$(SEGMENT_ID)-context.json
M1_SEGMENT_WINDOW ?= artifacts/$(SEGMENT_ID)-window.json
M1_ODOMETRY ?= artifacts/$(SEGMENT_ID)-odometry.csv
M1_MAX_POINTS ?= 150000
M1_VISUAL_RUN ?= modules/visual-perception/benchmarks/results/samples/20260909T135428Z
MAP_EXPLORER_ARTIFACT ?= $(M1_CONTEXT_ARTIFACT)
MAP_EXPLORER_ASSETS := $(basename $(MAP_EXPLORER_ARTIFACT))-assets
M1_PYTHONPATH := $(CURDIR)/contracts:$(CURDIR)/modules/state-estimation/src:$(CURDIR)/modules/geometric-map/src:$(CURDIR)/modules/sensor-association/src:$(CURDIR)/modules/semantic-fusion/src:$(CURDIR)/apps/mapping-runtime/src

.PHONY: verify test lint typecheck corridor-02-window corridor-02-map corridor-02-context m1-demo m1-pcd-slice m1-test map-explorer-install map-explorer-test map-explorer-build map-explorer-serve

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
		--map-id "$(SEGMENT_ID)" \
		--map-frame map \
		--max-points $(M1_MAX_POINTS)

corridor-02-window:
	PYTHONPATH="$(M1_PYTHONPATH)" $(RESOLVED_PYTHON) -m mapping_runtime bag-window \
		--bag datasets/raw/corridor-02/corridor-02.bag \
		--start-s $(SEGMENT_START_S) \
		--duration-s $(SEGMENT_SECONDS) \
		--keyframe-interval-s $(SEGMENT_KEYFRAME_INTERVAL_S) \
		--lead-s $(SEGMENT_LEAD_S) \
		--output "$(M1_SEGMENT_WINDOW)"

corridor-02-map: corridor-02-window
	offset=$$($(RESOLVED_PYTHON) -c "import json;print(json.load(open('$(M1_SEGMENT_WINDOW)'))['play_offset_s'])"); \
	duration=$$($(RESOLVED_PYTHON) -c "import json;print(json.load(open('$(M1_SEGMENT_WINDOW)'))['play_duration_s'])"); \
	docker compose --profile ros1 run --rm fastlio-ros1 bash \
		/workspace/apps/mapping-runtime/scripts/run_fastlio_ros1_segment.sh \
		/workspace/datasets/raw/corridor-02/corridor-02.bag \
		/workspace/modules/state-estimation/configs/corridor-02-velodyne.yaml \
		"$$duration" \
		/workspace/$(M1_PCD_SOURCE) \
		"$$offset"
	$(MAKE) m1-pcd-slice PYTHON="$(PYTHON)"

corridor-02-context:
	PYTHONPATH="$(M1_PYTHONPATH)" $(RESOLVED_PYTHON) -m mapping_runtime corridor-02-context \
		--geometric-slice "$(M1_PCD_ARTIFACT)" \
		--bag datasets/raw/corridor-02/corridor-02.bag \
		--intrinsics datasets/raw/corridor-02/corridor-02-Intrinsics.yaml \
		--extrinsics datasets/raw/corridor-02/corridor-02-extrinsics.yaml \
		--window "$(M1_SEGMENT_WINDOW)" \
		--visual-run "$(M1_VISUAL_RUN)" \
		--odometry "$(M1_ODOMETRY)" \
		--ground-truth datasets/raw/corridor-02/corridor-02-gt.txt \
		--output "$(M1_CONTEXT_ARTIFACT)"

m1-test:
	PYTHONPATH="$(M1_PYTHONPATH)" $(RESOLVED_PYTHON) -m pytest \
		contracts/tests \
		modules/state-estimation/tests \
		modules/geometric-map/tests \
		modules/sensor-association/tests \
		modules/semantic-fusion/tests \
		apps/mapping-runtime/tests

map-explorer-install:
	cd apps/map-explorer/web && npm ci

map-explorer-test:
	cd apps/map-explorer/web && npm test

map-explorer-build:
	cd apps/map-explorer/web && npm run build

map-explorer-serve:
	mkdir -p apps/map-explorer/web/public
	mkdir -p apps/map-explorer/web/public/maps
	cp "$(MAP_EXPLORER_ARTIFACT)" apps/map-explorer/web/public/current-map.json
	cp "$(M1_PCD_ARTIFACT)" "apps/map-explorer/web/public/maps/$(SEGMENT_ID)-geometry.json"
	if [ -d "$(MAP_EXPLORER_ASSETS)" ]; then cp -R "$(MAP_EXPLORER_ASSETS)" apps/map-explorer/web/public/; fi
	$(RESOLVED_PYTHON) apps/map-explorer/scripts/publish_map_index.py apps/map-explorer/web/public
	cd apps/map-explorer/web && npm run dev -- --host 0.0.0.0
