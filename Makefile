# Makefile — VisionServe
# Yêu cầu: Go >= 1.22. Runtime cần libonnxruntime.so (đặt ORT_DYLIB_PATH).

BINARY      := visionserve
PKG         := ./cmd/visionserve
BIN_DIR     := bin
VERSION     := $(shell git describe --tags --always 2>/dev/null || echo "0.1.0-dev")
LDFLAGS     := -s -w -X visionserve/internal/cli.Version=$(VERSION)
GOFLAGS     ?=

# Cài đặt mặc định vào $GOBIN hoặc $GOPATH/bin
INSTALL_DIR := $(shell go env GOBIN 2>/dev/null)
ifeq ($(INSTALL_DIR),)
INSTALL_DIR := $(shell go env GOPATH)/bin
endif

# Tham số mặc định cho `make run` / `make serve`
MODEL  ?= rf-detr
IMAGE  ?= test/testdata/sample.jpg
ADDR   ?= :11435
MODELS ?= ./models

# Runtime cần libonnxruntime.so. Nếu user chưa export ORT_DYLIB_PATH, tự dò:
# loại node_modules (tránh bản arch khác), ưu tiên bản ORT đầy đủ (onnxruntime/capi).
ORT_DYLIB_PATH ?= $(shell find $(HOME) /usr/local/lib /usr/lib -name 'libonnxruntime.so*' 2>/dev/null | grep -v node_modules | grep -E 'onnxruntime/capi.*\.so\.[0-9]' | head -1)

.PHONY: all build install run serve list demo test fmt vet tidy lint clean \
        build-linux-arm64 docker docker-edge help

all: build ## Mặc định: build

## --- Build & cài đặt ---

build: ## Biên dịch binary vào ./bin/visionserve
	@mkdir -p $(BIN_DIR)
	go build $(GOFLAGS) -ldflags '$(LDFLAGS)' -o $(BIN_DIR)/$(BINARY) $(PKG)
	@echo "→ $(BIN_DIR)/$(BINARY) ($(VERSION))"

install: ## Cài binary vào GOBIN/GOPATH bin (dùng được lệnh `visionserve` toàn cục)
	go install $(GOFLAGS) -ldflags '$(LDFLAGS)' $(PKG)
	@echo "→ đã cài vào $(INSTALL_DIR)/$(BINARY)"

## --- Chạy ---

run: build ## Run on 1 image: make run MODEL=rf-detr IMAGE=path.jpg [OUT=r.png] [BOX=x,y,w,h] [PROMPT="cat."] [POINT=x,y]
	ORT_DYLIB_PATH="$(ORT_DYLIB_PATH)" $(BIN_DIR)/$(BINARY) run --models $(MODELS) $(MODEL) $(IMAGE) \
		$(if $(OUT),--out $(OUT)) $(if $(BOX),--box "$(BOX)") $(if $(PROMPT),--prompt "$(PROMPT)") $(if $(POINT),--point "$(POINT)")

serve: build ## Khởi động HTTP server: make serve ADDR=:11435
	ORT_DYLIB_PATH="$(ORT_DYLIB_PATH)" $(BIN_DIR)/$(BINARY) serve --models $(MODELS) --addr $(ADDR)

list: build ## Liệt kê model trong registry
	$(BIN_DIR)/$(BINARY) list --models $(MODELS)

demo: build ## Demo: detection trên ảnh COCO thật, xuất ảnh có bbox vào demo/out/
	@ORT_DYLIB_PATH="$(ORT_DYLIB_PATH)" bash scripts/demo.sh

## --- Chất lượng ---

test: ## Chạy unit test (pre/postprocess...)
	go test $(GOFLAGS) ./...

fmt: ## gofmt toàn bộ
	gofmt -w .

vet: ## go vet
	go vet ./...

tidy: ## go mod tidy
	go mod tidy

lint: vet ## Alias chạy vet (thêm golangci-lint nếu có)
	@command -v golangci-lint >/dev/null 2>&1 && golangci-lint run || echo "golangci-lint chưa cài — chỉ chạy go vet"

## --- Cross build / Docker ---

build-linux-arm64: ## Build cho Jetson/arm64
	@mkdir -p $(BIN_DIR)
	GOOS=linux GOARCH=arm64 go build $(GOFLAGS) -ldflags '$(LDFLAGS)' -o $(BIN_DIR)/$(BINARY)-linux-arm64 $(PKG)

docker: ## Build image GPU (x86_64)
	docker build -f deploy/Dockerfile -t visionserve:$(VERSION) .

docker-edge: ## Build image edge (arm64/Jetson)
	docker build -f deploy/Dockerfile.edge -t visionserve:$(VERSION)-edge .

clean: ## Xoá artifacts
	rm -rf $(BIN_DIR)

help: ## In danh sách target
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
