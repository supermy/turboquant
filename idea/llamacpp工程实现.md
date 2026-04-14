# TurboQuant - 极致 KV 缓存量化 #20969

2026-03-25

谷歌研究院刚发布了一篇博客和论文，介绍了一种全新算法，可将 **KV 缓存量化至 3 比特以下** ，且精度损失几乎为 0。

博客：[https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

论文：[https://arxiv.org/pdf/2504.19874](https://arxiv.org/pdf/2504.19874)

如果他们的说法属实，这将是颠覆性的突破。MLX 的开发者已经开始跟进适配。

我把这个消息分享到这里，想看看 llama.cpp 的开发者是否有兴趣添加这项功能。

---

## 其他厂商相关技术

英伟达也在主推类似技术（KTVC）：

文章：[https://venturebeat.com/orchestration/nvidia-shrinks-llm-memory-20x-without-changing-model-weights](https://venturebeat.com/orchestration/nvidia-shrinks-llm-memory-20x-without-changing-model-weights)

非常期待开发者们分享这类功能的后续规划！

---

## 修正后的基准测试数据（2026-04-01）

此前数据因测量方式错误（RSS 统计、请求静默失败）导致结论失真，现已修正。

测试环境：llama.cpp build 8399，Nemotron-3-Nano-30B-A3B Q4_K_XL，128K 上下文，通过 nvidia-smi+llama.cpp 内部 KV 缓存统计

### 内存占用

表格

| 缓存类型 | KV 缓存大小 | 总 GPU 显存 | 节省空间 |
| -------- | ----------- | ----------- | -------- |
| f16      | 768 MiB     | 23092 MiB   | 基准     |
| q8_0     | 408 MiB     | 22732 MiB   | -47%     |
| q4_0     | 216 MiB     | 22540 MiB   | -72%     |

### 提示词处理速度（tok/s）

所有缓存类型、所有上下文长度下 **速度无差异** 。

### 生成速度（tok/s）

长上下文下因逐 token 反量化开销出现性能下降：

表格

| 上下文长度 | f16  | q4_0 | 下降幅度 |
| ---------- | ---- | ---- | -------- |
| ~6K        | 44.7 | 45.0 | +0.7%    |
| ~24K       | 44.6 | 39.3 | -11.9%   |
| ~110K      | 38.0 | 24.0 | -36.8%   |

 **核心结论** ：q4_0 在 110K 上下文时生成速度下降 37%，这正是 TurboQuant 要解决的瓶颈 —— 它支持 **直接在量化值上计算** ，无需反量化。

---

## 已实现的 TurboQuant 适配进展

### 1. TheTom 分支（llama-cpp-turboquant）

* 支持**turbo3（3.25 比特）、turbo4（4.25 比特）** 两种新类型
* 完整支持 Apple Silicon Metal GPU
* 实现 SET_ROWS、反量化、Flash Attention 内核
* 压缩比达标：turbo3 相比 FP16 压缩**4.9 倍**
* 修复上下文缩放回退问题，32K 上下文内速度保持 FP16 的 98.7%~99.5%

### 2. CPU 纯实现（veritatisquaesitoressumus）

* 无依赖 C 语言实现，量化 / 反量化 / 旋转矩阵生成 / 比特打包完整
* 测试通过：MSE 与论文误差 < 1%
* 实际效果：70B 模型下，FP16 支持约 109K 上下文，TQ3 可支持**536K 上下文**

### 3. CUDA 实现（Madreag/spiritbuun）

* 适配 RTX 5090/3090，完整支持 Flash Attention
* KV 缓存每 token 仅约 14KB（FP16 为 64KB），压缩**4.6 倍**
* 数学、代码生成、事实性任务全部通过，700K 上下文无内存压力

### 4. AMD HIP/ROCm 适配（domvox）

* 支持 RX 7900 XTX，实现 SWA 缓存类型独立配置
* 全局层用 turbo3、SWA 层保留 f16，兼顾压缩与精度

---

## 关键工程发现

1. **K 与 V 范数差异极大**
   现代 LLM 的 Key 与 Value 向量范数相差可达 **100 倍以上** ，K 需要比 V 更高的比特数。
2. **MSE 量化优于 QJL**
   论文中的 QJL 残差校正会 **放大方差** ，破坏 softmax 的 Top-K 排序，实际使用 MSE 最优量化即可。
3. **WHT（沃尔什 - 阿达玛变换）至关重要**
   相比随机旋转，WHT 能均匀分散能量，大幅降低量化误差，低比特下效果提升 **数十倍** 。
4. **动态 α 缩放优化**
   针对量化权重模型（如 Q4_K_M），将反量化范数乘以 1.02，可降低 **10%~17%** 的 KLD 散度。

---

## 使用方式

bash

运行

```
# 启用turbo3 KV缓存
./main -m 模型.gguf --cache-type-k turbo3 --cache-type-v turbo3 -fa on
```

---

## 已知问题与修复

* MSVC 编译报错 `M_PI`未定义：需添加 `#define _USE_MATH_DEFINES`
* CUDA 13.1 存在 MMQ 内核段错误，建议使用 12.8
* 早期版本长上下文速度退化，已通过字节批量读取优化修复
