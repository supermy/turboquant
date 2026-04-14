# TurboQuant C++ 测试 Makefile

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
LDFLAGS = -lm

# 目标文件
TARGET = test_turboquant_pipeline

.PHONY: all clean run

all: $(TARGET)

$(TARGET): test_turboquant_pipeline.cpp
	@echo "编译 TurboQuant 测试..."
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo "编译完成: ./$(TARGET)"

run: $(TARGET)
	@echo "运行测试..."
	./$(TARGET)

clean:
	@echo "清理..."
	rm -f $(TARGET)

help:
	@echo "TurboQuant C++ 测试"
	@echo "==================="
	@echo ""
	@echo "可用目标:"
	@echo "  make        - 编译测试程序"
	@echo "  make run    - 编译并运行测试"
	@echo "  make clean  - 清理编译文件"
	@echo ""
	@echo "测试内容:"
	@echo "  1. 完整流程: 输入 -> L2归一化 -> Hadamard旋转 -> Beta量化 -> 反量化 -> 反旋转 -> 输出"
	@echo "  2. 仅量化: 输入 -> Beta量化 -> 反量化 -> 输出"
	@echo "  3. 不同维度: 128, 256, 512"
	@echo "  4. 不同位宽: 1, 2, 3, 4, 8 bit"
