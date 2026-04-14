# Makefile for PolarQuant Project
# PolarQuant: 基于极坐标变换的无损量化

.PHONY: help install test test-verbose test-coverage lint format clean docs benchmark

# 默认 Python 解释器
PYTHON := python3
PIP := pip3

# 项目目录
SRC_DIR := polarquant
TEST_DIR := tests
EXAMPLES_DIR := examples
DOCS_DIR := docs

# 帮助目标
help:
	@echo "PolarQuant - 基于极坐标变换的无损量化"
	@echo ""
	@echo "可用目标:"
	@echo "  安装相关:"
	@echo "    install         - 安装包和依赖"
	@echo "    install-dev     - 开发模式安装（含开发依赖）"
	@echo ""
	@echo "  测试相关:"
	@echo "    test            - 运行所有测试"
	@echo "    test-verbose    - 详细输出运行测试"
	@echo "    test-coverage   - 运行测试并生成覆盖率报告"
	@echo "    test-unit       - 仅运行单元测试"
	@echo "    test-integration- 仅运行集成测试"
	@echo ""
	@echo "  代码质量:"
	@echo "    lint            - 运行代码检查"
	@echo "    format          - 使用 black 和 isort 格式化代码"
	@echo "    type-check      - 使用 mypy 运行类型检查"
	@echo ""
	@echo "  文档和示例:"
	@echo "    docs            - 生成文档"
	@echo "    docs-diagram    - 生成时序图"
	@echo "    example         - 运行基本示例"
	@echo "    example-kv      - 运行 KV Cache 示例"
	@echo "    example-qwen    - 运行 Qwen 模型示例"
	@echo ""
	@echo "  构建和发布:"
	@echo "    build           - 构建包分发"
	@echo "    publish         - 发布到 PyPI（需要凭证）"
	@echo "    clean           - 清理构建产物"
	@echo ""
	@echo "  其他:"
	@echo "    benchmark       - 运行性能基准测试"
	@echo "    check-all       - 提交前运行所有检查"
	@echo "    quick-check     - 快速检查（仅单元测试）"

# 安装
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	$(PIP) install pytest pytest-cov black isort mypy flake8

# 测试
test:
	$(PYTHON) -m pytest $(TEST_DIR) -v

test-verbose:
	$(PYTHON) -m pytest $(TEST_DIR) -vv --tb=short

test-coverage:
	$(PYTHON) -m pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

test-unit:
	$(PYTHON) -m pytest $(TEST_DIR)/test_utils.py $(TEST_DIR)/test_core.py -v

test-integration:
	$(PYTHON) -m pytest $(TEST_DIR)/test_integration.py -v

# 代码质量
lint:
	$(PYTHON) -m flake8 $(SRC_DIR) $(TEST_DIR) --max-line-length=100 --ignore=E203,W503

format:
	$(PYTHON) -m black $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR) --line-length=100
	$(PYTHON) -m isort $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR) --profile=black

type-check:
	$(PYTHON) -m mypy $(SRC_DIR) --ignore-missing-imports

# 清理
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# 文档
docs:
	@echo "生成文档..."
	$(PYTHON) -m pydoc -w $(SRC_DIR)
	@echo "文档生成完成。"

docs-diagram:
	@echo "生成时序图..."
	$(PYTHON) $(DOCS_DIR)/generate_sequence_diagram.py
	@echo "时序图生成完成。"

# 基准测试
benchmark:
	$(PYTHON) -m pytest $(TEST_DIR)/test_integration.py::TestPerformanceBenchmark -v

# 示例
example:
	$(PYTHON) $(EXAMPLES_DIR)/basic_usage.py

example-kv:
	$(PYTHON) $(EXAMPLES_DIR)/kv_cache_demo.py

example-qwen:
	$(PYTHON) $(EXAMPLES_DIR)/qwen_quantization_demo.py

# 构建
build:
	$(PYTHON) -m build

# 发布（需要 PyPI 凭证）
publish:
	$(PYTHON) -m twine upload dist/*

# 开发环境
dev:
	@echo "启动开发环境..."
	$(PYTHON) -c "import polarquant; print('PolarQuant version:', polarquant.__version__)"

# 提交前运行所有检查
check-all: lint type-check test
	@echo "所有检查通过！"

# 快速检查
quick-check: test-unit
	@echo "快速检查通过！"

# GitHub 同步相关
git-status:
	git status

git-add:
	git add .

git-commit:
	git commit -m "$(msg)"

git-push:
	git push origin main

# 一键提交（使用示例: make git-submit msg="更新README"）
git-submit: git-add
	@if [ -z "$(msg)" ]; then \
		echo "错误: 请提供提交信息，例如: make git-submit msg='更新README'"; \
		exit 1; \
	fi
	git commit -m "$(msg)" || true
	git push origin main
