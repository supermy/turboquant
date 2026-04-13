# FAISS TurboQuant Makefile

.PHONY: help install test clean benchmark download update demo

help:
	@echo "FAISS TurboQuant - Makefile 帮助"
	@echo "=================================="
	@echo ""
	@echo "使用方法: make [target]"
	@echo ""
	@echo "可用目标:"
	@echo "  help        显示此帮助信息"
	@echo "  install     安装 FAISS 依赖"
	@echo "  test        运行 Python 测试"
	@echo "  benchmark   运行基准测试"
	@echo "  clean       清理缓存文件"
	@echo "  download    下载最新代码"
	@echo "  update      更新到最新版本"
	@echo "  demo        运行演示"
	@echo ""

install:
	@echo "安装 FAISS..."
	pip install faiss-cpu
	@echo "或安装 GPU 版本: pip install faiss-gpu"

test:
	@echo "运行 Python 测试..."
	python -m pytest tests/ -v -k "tqmse"

benchmark:
	@echo "运行基准测试..."
	python -c "\
import faiss; \
import numpy as np; \
d = 128; n = 10000; \
xb = np.random.randn(n, d).astype('float32'); \
faiss.normalize_L2(xb); \
index = faiss.index_factory(d, 'L2norm,HR,SQtqmse4'); \
index.train(xb); \
index.add(xb); \
print(f'索引大小: {index.ntotal} 向量'); \
print('基准测试完成')"

clean:
	@echo "清理缓存文件..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

download:
	@echo "下载最新代码..."
	@mkdir -p scalar_quantizer vector_transform
	@echo "下载 scalar_quantizer..."
	curl -sL "https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/impl/scalar_quantizer/quantizers.h" -o scalar_quantizer/quantizers.h
	curl -sL "https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/impl/scalar_quantizer/training.h" -o scalar_quantizer/training.h
	curl -sL "https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/impl/scalar_quantizer/training.cpp" -o scalar_quantizer/training.cpp
	curl -sL "https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/impl/scalar_quantizer/sq-avx2.cpp" -o scalar_quantizer/sq-avx2.cpp
	curl -sL "https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/impl/scalar_quantizer/sq-avx512.cpp" -o scalar_quantizer/sq-avx512.cpp
	curl -sL "https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/impl/scalar_quantizer/sq-neon.cpp" -o scalar_quantizer/sq-neon.cpp
	@echo "下载 vector_transform (RHT)..."
	curl -sL "https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/VectorTransform.h" -o vector_transform/VectorTransform.h
	curl -sL "https://raw.githubusercontent.com/facebookresearch/faiss/main/faiss/VectorTransform.cpp" -o vector_transform/VectorTransform.cpp
	@echo "下载完成"

update: download
	@echo "代码已更新到最新版本"

demo:
	@echo "运行演示..."
	python -c "\
import faiss; \
import numpy as np; \
print('FAISS TurboQuant 演示'); \
print(''); \
d = 32; \
print('=== 方式 1: 仅 Beta 分布量化 (无旋转) ==='); \
index1 = faiss.index_factory(d, 'SQtqmse4'); \
print(f'创建索引: SQtqmse4, 维度={d}'); \
xb = np.random.randn(100, d).astype('float32'); \
faiss.normalize_L2(xb); \
index1.train(xb); \
index1.add(xb); \
print(f'添加 {index1.ntotal} 个向量'); \
print(''); \
print('=== 方式 2: L2归一化 + Hadamard旋转 + Beta分布量化 ==='); \
index2 = faiss.index_factory(d, 'L2norm,HR,SQtqmse4'); \
print(f'创建索引: L2norm,HR,SQtqmse4, 维度={d}'); \
index2.train(xb); \
index2.add(xb); \
print(f'添加 {index2.ntotal} 个向量'); \
print(''); \
xq = np.random.randn(1, d).astype('float32'); \
faiss.normalize_L2(xq); \
D, I = index2.search(xq, 5); \
print(f'搜索结果: Top-5 索引={I[0]}')"
