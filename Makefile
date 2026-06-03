# Makefile — VisionServe
# Requires Go >= 1.22. The runtime needs libonnxruntime.so (set ORT_DYLIB_PATH).

BINARY      := visionserve
PKG         := ./cmd/visionserve
BIN_DIR     := bin
VERSION     := $(shell git describe --tags --always 2>/dev/null || echo "0.1.0-dev")
LDFLAGS     := -s -w -X visionserve/internal/cli.Version=$(VERSION)
GOFLAGS     ?=

# Install destination ($GOBIN, else $GOPATH/bin)
INSTALL_DIR := $(shell go env GOBIN 2>/dev/null)
ifeq ($(INSTALL_DIR),)
INSTALL_DIR := $(shell go env GOPATH)/bin
endif

# Defaults for run / serve / pull
MODEL  ?= rf-detr
IMAGE  ?= test/testdata/sample.jpg
ADDR   ?= :11435
MODELS ?= ./models
# Use the GPU (CUDA EP) by default; falls back to CPU automatically if no CUDA-enabled
# ORT lib is found. Override with `make run GPU=0` to force CPU.
GPU    ?= 1

# Python interpreter used to build the client package for PyPI.
PYTHON ?= python3

# The runtime needs libonnxruntime.so. If the user has not exported ORT_DYLIB_PATH, auto-detect:
# skip node_modules (avoids other-arch builds), prefer a full ORT build (onnxruntime/capi).
ORT_DYLIB_PATH ?= $(shell find $(HOME) /usr/local/lib /usr/lib -name 'libonnxruntime.so*' 2>/dev/null | grep -v node_modules | grep -E 'onnxruntime/capi.*\.so\.[0-9]' | head -1)

.PHONY: all build install run serve list ps rm pull demo terminate test fmt vet tidy lint clean \
        build-linux-arm64 docker docker-edge pypi help

all: build ## Default target: build

## --- Build & install ---

build: ## Compile the binary into ./bin/visionserve
	@mkdir -p $(BIN_DIR)
	go build $(GOFLAGS) -ldflags '$(LDFLAGS)' -o $(BIN_DIR)/$(BINARY) $(PKG)
	@echo "→ $(BIN_DIR)/$(BINARY) ($(VERSION))"

install: ## Install the binary into GOBIN/GOPATH bin (use `visionserve` globally)
	go install $(GOFLAGS) -ldflags '$(LDFLAGS)' $(PKG)
	@echo "→ installed to $(INSTALL_DIR)/$(BINARY)"

## --- Run ---
# All of run/serve/demo use the GPU (CUDA EP) by default. They source scripts/gpu-env.sh
# to auto-detect a CUDA-enabled ORT lib + the cuDNN/CUDA libs, and fall back to the
# auto-detected CPU ORT lib if none is found. Add GPU=0 to force CPU.

run: build ## Run on 1 image: make run MODEL=rf-detr IMAGE=path.jpg [OUT=r.png] [BOX=x,y,w,h] [PROMPT="cat."] [POINT=x,y] [MIN_SIZE=px²] [MAX_SIZE=px²] [GPU=0]
	@bash -c 'if [ "$(GPU)" = "1" ] && source scripts/gpu-env.sh; then :; else export ORT_DYLIB_PATH="$(ORT_DYLIB_PATH)"; fi; \
		"$(BIN_DIR)/$(BINARY)" run --models "$(MODELS)" $(MODEL) "$(IMAGE)" \
		$(if $(OUT),--out "$(OUT)") $(if $(BOX),--box "$(BOX)") $(if $(PROMPT),--prompt "$(PROMPT)") $(if $(POINT),--point "$(POINT)") \
		$(if $(MIN_SIZE),--min-size "$(MIN_SIZE)") $(if $(MAX_SIZE),--max-size "$(MAX_SIZE)")'

serve: build ## Start the HTTP server: make serve [ADDR=:11435] [GPU=0]
	@bash -c 'if [ "$(GPU)" = "1" ] && source scripts/gpu-env.sh; then :; else export ORT_DYLIB_PATH="$(ORT_DYLIB_PATH)"; fi; \
		"$(BIN_DIR)/$(BINARY)" serve --models "$(MODELS)" --addr "$(ADDR)"'

list: build ## List models in the registry (+ pullable models)
	$(BIN_DIR)/$(BINARY) list --models $(MODELS)

ps: build ## Show models loaded in a running server: make ps [ADDR=:11435]
	$(BIN_DIR)/$(BINARY) ps --addr $(ADDR)

rm: build ## Unload a model from a running server: make rm MODEL=rf-detr [ADDR=:11435]
	$(BIN_DIR)/$(BINARY) rm $(MODEL) --addr $(ADDR)

pull: build ## Download a model from HuggingFace: make pull MODEL=rf-detr [MODELS=./models]
	$(BIN_DIR)/$(BINARY) pull $(MODEL) --models $(MODELS)

terminate: ## Kill any process listening on :11435 (make terminate [ADDR=:11435])
	@port=$$(echo "$(ADDR)" | sed 's/.*://'); \
	pids=$$(lsof -ti tcp:$$port 2>/dev/null || true); \
	if [ -z "$$pids" ]; then \
		echo "terminate: no process listening on :$$port" >&2; \
	else \
		echo "terminate: killing PID(s) $$pids on :$$port" >&2; \
		kill $$pids; \
	fi

demo: build ## Demo on real COCO images (boxes/masks -> demo/out/) [GPU=0]
	@bash -c 'if [ "$(GPU)" = "1" ] && source scripts/gpu-env.sh; then :; fi; bash scripts/demo.sh'

## --- Quality ---

test: ## Run unit tests (pre/postprocess, etc.)
	go test $(GOFLAGS) ./...

fmt: ## gofmt the whole tree
	gofmt -w .

vet: ## go vet
	go vet ./...

tidy: ## go mod tidy
	go mod tidy

lint: vet ## Alias for vet (also runs golangci-lint if installed)
	@command -v golangci-lint >/dev/null 2>&1 && golangci-lint run || echo "golangci-lint not installed — ran go vet only"

## --- Cross build / Docker ---

build-linux-arm64: ## Build for Jetson/arm64
	@mkdir -p $(BIN_DIR)
	GOOS=linux GOARCH=arm64 go build $(GOFLAGS) -ldflags '$(LDFLAGS)' -o $(BIN_DIR)/$(BINARY)-linux-arm64 $(PKG)

docker: ## Build the server image (CPU; add ORT_VARIANT=gpu for a GPU image)
	cp deploy/.dockerignore .dockerignore
	docker build -f deploy/Dockerfile $(if $(filter gpu,$(ORT_VARIANT)),--build-arg ORT_VARIANT=gpu) -t visionserve:$(VERSION) -t visionserve:latest .

docker-edge: ## Build the edge image (arm64/Jetson)
	cp deploy/.dockerignore .dockerignore
	docker buildx build --platform linux/arm64 -f deploy/Dockerfile.edge -t visionserve:$(VERSION)-edge .

DOCKER_HUB_USER ?= mtbui2010
# PUSH_VERSION is the tag of the already-built local image (e.g. v0.1.0).
# Override at the command line if needed: make push-docker PUSH_VERSION=v0.2.0
PUSH_VERSION    ?= v0.1.0

push-docker: ## Tag and push CPU + GPU images to Docker Hub (DOCKER_HUB_USER=mtbui2010)
	@echo "=== Tagging images for Docker Hub ($(DOCKER_HUB_USER)) ==="
	docker tag visionserve:$(PUSH_VERSION)-cpu   $(DOCKER_HUB_USER)/visionserve:$(PUSH_VERSION)-cpu
	docker tag visionserve:$(PUSH_VERSION)-cpu   $(DOCKER_HUB_USER)/visionserve:$(PUSH_VERSION)
	docker tag visionserve:$(PUSH_VERSION)-cpu   $(DOCKER_HUB_USER)/visionserve:latest
	docker tag visionserve:$(PUSH_VERSION)-gpu   $(DOCKER_HUB_USER)/visionserve:$(PUSH_VERSION)-gpu
	@echo "=== Pushing to Docker Hub ==="
	docker push $(DOCKER_HUB_USER)/visionserve:$(PUSH_VERSION)-cpu
	docker push $(DOCKER_HUB_USER)/visionserve:$(PUSH_VERSION)
	docker push $(DOCKER_HUB_USER)/visionserve:latest
	docker push $(DOCKER_HUB_USER)/visionserve:$(PUSH_VERSION)-gpu
	@echo "=== Done. Images pushed: ==="
	@echo "  $(DOCKER_HUB_USER)/visionserve:$(PUSH_VERSION)"
	@echo "  $(DOCKER_HUB_USER)/visionserve:$(PUSH_VERSION)-cpu"
	@echo "  $(DOCKER_HUB_USER)/visionserve:$(PUSH_VERSION)-gpu"
	@echo "  $(DOCKER_HUB_USER)/visionserve:latest"

## --- Python client (PyPI) ---

pypi: ## Build + validate the Python client package (clients/python -> dist/). Publish via CI.
	$(PYTHON) -m pip install --quiet --upgrade build twine
	cd clients/python && rm -rf dist build *.egg-info && $(PYTHON) -m build && $(PYTHON) -m twine check dist/*
	@echo "→ built clients/python/dist/"
	@echo "  Publish to PyPI:      git tag vX.Y.Z && git push origin vX.Y.Z   (CI Trusted Publishing)"
	@echo "  Publish to TestPyPI:  run the 'Publish Python client' workflow manually (workflow_dispatch)"

clean: ## Remove build artifacts
	rm -rf $(BIN_DIR) clients/python/dist clients/python/build

help: ## Print the list of targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
