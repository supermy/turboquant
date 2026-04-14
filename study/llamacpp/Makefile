# llama.cpp TurboQuant Makefile

.PHONY: help build test clean benchmark

CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

help:
	@echo "llama.cpp TurboQuant - Makefile 帮助"
	@echo "======================================"
	@echo ""
	@echo "使用方法: make [target]"
	@echo ""
	@echo "可用目标:"
	@echo "  help        显示此帮助信息"
	@echo "  build       编译 CPU 实现"
	@echo "  test        运行测试"
	@echo "  benchmark   运行基准测试"
	@echo "  clean       清理编译文件"
	@echo "  cuda        编译 CUDA 实现"
	@echo "  download    下载最新代码"
	@echo "  update      更新到最新版本"
	@echo ""

build: tq/tq_quants.o
	@echo "编译完成"

tq/tq_quants.o: tq/tq_quants.c tq/tq_quants.h
	$(CC) $(CFLAGS) -c tq/tq_quants.c -o tq/tq_quants.o

test: build
	@echo "运行测试..."
	$(CC) $(CFLAGS) -o tq/test_tq tq/test_tq.c tq/tq_quants.o $(LDFLAGS)
	./tq/test_tq

benchmark: build
	@echo "运行基准测试..."
	@echo "维度 128, 100 个向量"
	$(CC) $(CFLAGS) -o tq/test_tq tq/test_tq.c tq/tq_quants.o $(LDFLAGS)
	./tq/test_tq

cuda:
	@echo "编译 CUDA 实现..."
	@if command -v nvcc >/dev/null 2>&1; then \
		nvcc -O2 -o cuda/tq4_rotate cuda/tq4-rotate.cu; \
		nvcc -O2 -o cuda/tq4_set_rows cuda/tq4-set-rows.cu; \
		echo "CUDA 编译完成"; \
	else \
		echo "错误: nvcc 未找到，请安装 CUDA Toolkit"; \
	fi

clean:
	@echo "清理编译文件..."
	rm -f tq/*.o tq/test_tq
	rm -f cuda/*.o cuda/tq4_rotate cuda/tq4_set_rows
	find . -type d -name ".dSYM" -exec rm -rf {} + 2>/dev/null || true

download:
	@echo "下载最新代码..."
	curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/e8c90c02d1989d4c2ad9b66c2e67a8d394acc1c4/ggml/src/tq/tq_quants.h" -o tq/tq_quants.h
	curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/e8c90c02d1989d4c2ad9b66c2e67a8d394acc1c4/ggml/src/tq/tq_quants.c" -o tq/tq_quants.c
	curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/e8c90c02d1989d4c2ad9b66c2e67a8d394acc1c4/ggml/src/tq/test_tq.c" -o tq/test_tq.c
	curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/e8c90c02d1989d4c2ad9b66c2e67a8d394acc1c4/ggml/src/ggml-cuda/tq4-rotate.cu" -o cuda/tq4-rotate.cu
	curl -sL "https://raw.githubusercontent.com/ggml-org/llama.cpp/e8c90c02d1989d4c2ad9b66c2e67a8d394acc1c4/ggml/src/ggml-cuda/tq4-set-rows.cu" -o cuda/tq4-set-rows.cu
	@echo "下载完成"

update: download
	@echo "代码已更新到最新版本"

all: build test
