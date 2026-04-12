# TurboQuant 主项目 Makefile

.PHONY: help all clean test install update

help:
	@echo "TurboQuant - 高维向量压缩工具集"
	@echo "====================================="
	@echo ""
	@echo "使用方法: make [target]"
	@echo ""
	@echo "可用目标:"
	@echo "  help        显示此帮助信息"
	@echo "  all         构建所有子项目"
	@echo "  test        运行所有测试"
	@echo "  clean       清理所有项目"
	@echo "  install     安装所有依赖"
	@echo "  update      更新所有子项目代码"
	@echo "  benchmark   运行所有基准测试"
	@echo ""
	@echo "子项目目标:"
	@echo "  google      Google 官方实现"
	@echo "  llamacpp    llama.cpp 实现"
	@echo "  faiss       FAISS 实现"
	@echo "  glm51       Rust 实现"
	@echo ""

all: google llamacpp faiss glm51
	@echo "所有子项目构建完成"

google:
	@echo "构建 Google 官方实现..."
	$(MAKE) -C google build

llamacpp:
	@echo "构建 llama.cpp 实现..."
	$(MAKE) -C llamacpp build

faiss:
	@echo "构建 FAISS 实现..."
	$(MAKE) -C faiss install

glm51:
	@echo "构建 Rust 实现..."
	cd glm51 && cargo build --release

test: test-google test-llamacpp test-faiss test-glm51
	@echo "所有测试完成"

test-google:
	$(MAKE) -C google test

test-llamacpp:
	$(MAKE) -C llamacpp test

test-faiss:
	$(MAKE) -C faiss test

test-glm51:
	cd glm51 && cargo test

benchmark: benchmark-google benchmark-llamacpp benchmark-faiss benchmark-glm51
	@echo "所有基准测试完成"

benchmark-google:
	$(MAKE) -C google benchmark

benchmark-llamacpp:
	$(MAKE) -C llamacpp benchmark

benchmark-faiss:
	$(MAKE) -C faiss benchmark

benchmark-glm51:
	cd glm51 && cargo run --example kv_cache_demo

clean: clean-google clean-llamacpp clean-faiss clean-glm51
	@echo "所有项目清理完成"

clean-google:
	$(MAKE) -C google clean

clean-llamacpp:
	$(MAKE) -C llamacpp clean

clean-faiss:
	$(MAKE) -C faiss clean

clean-glm51:
	cd glm51 && cargo clean

install: install-google install-faiss install-glm51
	@echo "所有依赖安装完成"

install-google:
	$(MAKE) -C google install

install-faiss:
	$(MAKE) -C faiss install

install-glm51:
	cd glm51 && cargo fetch

update: update-google update-llamacpp update-faiss
	@echo "所有子项目代码已更新"

update-google:
	$(MAKE) -C google update

update-llamacpp:
	$(MAKE) -C llamacpp update

update-faiss:
	$(MAKE) -C faiss update

demo:
	@echo "运行演示程序..."
	@echo ""
	@echo "1. Rust 实现:"
	cd glm51 && cargo run --example kv_cache_demo
	@echo ""
	@echo "2. FAISS 实现:"
	$(MAKE) -C faiss demo

sync:
	@echo "同步代码到 GitHub..."
	git add -A
	git status
	@echo ""
	@echo "请确认要提交的更改，然后运行:"
	@echo "  git commit -m '更新代码'"
	@echo "  git push origin main"
