# Google PolarQuant Makefile

.PHONY: help install test clean benchmark

help:
	@echo "Google PolarQuant - Makefile 帮助"
	@echo "=================================="
	@echo ""
	@echo "使用方法: make [target]"
	@echo ""
	@echo "可用目标:"
	@echo "  help        显示此帮助信息"
	@echo "  install     安装 Python 依赖"
	@echo "  test        运行测试"
	@echo "  benchmark   运行基准测试"
	@echo "  clean       清理缓存文件"
	@echo "  demo        运行演示程序"
	@echo "  download    下载最新代码"
	@echo ""

install:
	@echo "安装 Python 依赖..."
	pip install torch transformers accelerate
	pip install numpy scipy

test:
	@echo "运行测试..."
	python -m pytest tests/ -v

benchmark:
	@echo "运行基准测试..."
	python test4gsm8k.py --model llama-2-7b --bits 4
	python test4long.py --model llama-2-7b --context 4096

demo:
	@echo "运行演示..."
	python -c "from models.kernel4group import PolarQuantizer; print('PolarQuant 已加载')"

clean:
	@echo "清理缓存文件..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

download:
	@echo "下载最新代码..."
	curl -sL "https://raw.githubusercontent.com/ericshwu/PolarQuant/main/models/kernel4group.py" -o models/kernel4group.py
	curl -sL "https://raw.githubusercontent.com/ericshwu/PolarQuant/main/models/modeling_llama_polar.py" -o models/modeling_llama_polar.py
	curl -sL "https://raw.githubusercontent.com/ericshwu/PolarQuant/main/models/modeling_llama_qjl.py" -o models/modeling_llama_qjl.py
	@echo "下载完成"

update: download
	@echo "代码已更新到最新版本"
