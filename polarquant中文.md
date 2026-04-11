这是论文《Optimal Gaussian Weight Quantization via Hadamard Rotation for LLM Compression》（通过 Hadamard 旋转实现最优高斯权重量化用于 LLM 压缩）的完整翻译：

---

# 通过 Hadamard 旋转实现最优高斯权重量化用于 LLM 压缩

## 摘要

我们提出了 PolarQuant，一种用于大型语言模型（LLM）的后训练权重量化方法，该方法利用神经网络权重的分布结构来实现接近无损的压缩。PolarQuant 分三个阶段运行：(1) 分块归一化到单位超球面，(2) Walsh-Hadamard 旋转变换坐标为近似高斯随机变量，(3) 使用与高斯分布匹配的质心进行量化。我们的消融实验表明，Hadamard 旋转单独贡献了 98% 的质量提升，将 Qwen3.5-9B 的困惑度从 6.90（absmax Q5）降低到 6.40（与 FP16 相比 Δ=+0.03），实现了 **无需任何校准数据即可达到近乎无损的效果** 。此外，PolarQuant 可作为下游 INT4 量化器的有效预处理步骤：PolarQuant Q5 反量化后由 torchao INT4 重新量化，困惑度达到 6.56，而直接 absmax INT4 的困惑度为 6.68，同时在 6.5 GB VRAM 下保持 43.1 tok/s 的吞吐量。代码和模型已公开发布。

## 1 引言

在消费级和边缘设备上部署大型语言模型需要激进的权重压缩。一个 90 亿参数的 FP16 模型需要约 18 GB GPU 显存，超过了大多数消费级 GPU 的容量。量化到 4 位可将显存降至约 5-6 GB，使推理能够在从 NVIDIA RTX 桌面 GPU 到 Apple Silicon 笔记本电脑等设备上运行。

最简单且最广泛部署的量化方案是  **absmax** （绝对最大值）量化 [10]，它将 [−α,α] 范围内的值线性映射到整数码，其中 α=max|wᵢ|。虽然计算上很简单，但当权重分布非均匀时，absmax 被证明是次优的：它在很少出现的异常值幅度上浪费了宝贵的码本条目，并将量化误差集中在高密度中心区域。

更复杂的方法，如 GPTQ [8] 和 AWQ [11]，分别通过基于校准的 Hessian 修正或激活感知缩放来缓解这个问题。NormalFloat (NF4) [7] 采取分布方法，设计针对正态分布权重的最优码本。QuIP# [6] 应用随机非相干处理来改进量化界限。最近，TurboQuant [3] 证明通过 Hadamard 变换归一化和旋转 KV 缓存向量可产生适合最优量化的高斯坐标。

在这项工作中，我们将极坐标量化思想从 KV 缓存压缩 **适配并扩展到权重压缩** 。我们的核心洞察是，经过分块归一化和 Hadamard 旋转后，LLM 权重块可以被很好地近似为独立同分布的标准正态随机变量，对于这种分布，通过 Lloyd-Max 算法 [13, 14] 可以解析地获得 MSE 最优量化器。这产生了一种量化方案，其特点是：

1. **接近无损** 。Hadamard 旋转单独将量化误差减少 98%，使 Q5 困惑度达到 FP16 的 +0.03 范围内（6.40 对比 6.37），无需任何校准数据。
2. **简单** 。核心算法是每个块进行一次矩阵乘法（H₁₂₈b̂ᵢ），无需梯度计算、无需迭代优化、无需校准数据。Hadamard 矩阵是确定性的且自逆的。
3. **可组合** 。PolarQuant 可作为与任何下游量化器兼容的预处理步骤。我们证明 PolarQuant Q5 反量化权重由 torchao INT4 [16] 重新量化时，困惑度比直接 absmax INT4 低 0.12 个点，速度和显存相同。
4. **高效** 。Hadamard 矩阵是其自身的逆（HHᵀ=I），无需额外存储。反量化在模型加载时增加约 8 秒，运行时零开销。

我们在 Qwen3.5-9B [17] 上评估 PolarQuant，并展示了最先进的困惑度-压缩权衡。与 AWQ 结合，PolarQuant 在 WikiText-2 [15] 上达到 6.43 的困惑度，仅比 FP16 高 +0.06，压缩比为 3.6×。在 Apple Silicon（Mac mini M4）上，PolarQuant 在 4.8 GB 显存下达到 19.7 tok/s，困惑度为 6.90。

## 2 背景与相关工作

### 2.1 Absmax 量化

给定权重张量 W ∈ ℝᵐˣⁿ 被划分为大小为 d 的块 {bᵢ}，absmax 量化计算尺度 sᵢ=maxⱼ|bᵢ,ⱼ| 并将每个元素映射到 {−2ᵇ⁻¹, ..., 2ᵇ⁻¹−1} 中最接近的整数：

**q_{i,j} = \text{round}\!\left(\frac{b_{i,j}}{s_i} \cdot (2^{b-1}-1)\right) \tag{1}**

这种方法假设在 [−sᵢ, sᵢ] 上均匀分布，这与经验观察到的近高斯权重分布 [4] 匹配不佳。

### 2.2 GPTQ

GPTQ [8] 使用近似二阶（Hessian）信息逐层进行量化，迭代量化列并通过最优脑外科框架在剩余列中补偿误差。GPTQ 取得了很好的结果，但需要校准数据且对大型模型计算昂贵。

### 2.3 AWQ：激活感知权重量化

AWQ [11] 观察到一小部分权重通道对输出质量影响不成比例，因为它们对应于大激活幅度。AWQ 从校准激活计算每通道缩放因子，保护重要通道免受量化误差。我们证明 AWQ 与 PolarQuant 互补：AWQ 在**通道**上操作，而 PolarQuant 在**块**上操作，它们的结合实现了接近无损的压缩。

### 2.4 NormalFloat (NF4)

Dettmers 等人 [7] 引入了 NormalFloat，一种在标准正态分布的分位数域中均匀间隔的数据类型。当每个量化区间具有相等概率时，NF4 在信息论上是最优的。PolarQuant 在两个关键方面不同：(1) 我们使用**最小化 MSE** 的 Lloyd-Max 质心而非最大化熵，(2) 我们通过 Hadamard 旋转**显式**将权重变换为高斯分布，而非先验假设高斯性。

### 2.5 QuIP 和 QuIP#

QuIP [5] 为权重量化引入了非相干处理，证明随机正交旋转改进了量化界限。QuIP# [6] 通过随机 Hadamard 变换 (RHT) 和格码本扩展到 2 位量化。虽然 PolarQuant 共享 Hadamard 变换的使用，但我们的方法在两个关键方面不同：(1) QuIP# 将旋转应用于**整个**权重矩阵列（块间），而 PolarQuant 将旋转应用于每 128 个元素的 **每个块内** （块内）；(2) QuIP# 使用非相干性来限制最坏情况误差，而 PolarQuant 使用归一化和旋转变换分布为高斯分布。

### 2.6 QuaRot 和 SpinQuant

QuaRot [2] 证明将 Hadamard 旋转应用于隐藏状态、激活和 KV 缓存可移除异常值而不改变模型输出，实现无异常值的 4 位推理。SpinQuant [12] 进一步证明**学习**的旋转矩阵在零样本任务上比固定 Hadamard 旋转高出 16 个点。两种方法都在层**之间**应用旋转（需要图手术将旋转吸收到相邻线性层），而 PolarQuant 独立地将旋转应用于每个权重张量的 **块内** ，无需修改模型图，可作为简单的预处理步骤使用。

### 2.7 TurboQuant

TurboQuant [3] 将极坐标量化框架应用于推理期间的 KV 缓存压缩，归一化 KV 向量，应用随机旋转，并使用最优质心量化产生的近高斯坐标。TurboQuant 证明了信息论下界并实现了接近最优的失真率。我们的工作将此方法从 KV 缓存 **适配到权重压缩** ，证明其作为下游 INT4 量化器预处理步骤的有效性，并提供了首次消融实验，量化了旋转（98%）与最优质心（2%）在 Q5 上的各自贡献。

### 2.8 Lloyd-Max 量化

Lloyd-Max 算法 [13, 14] 计算给定源分布的 MSE 最优标量量化器。对于具有密度 f(x) 的源 X，具有 L 级的最优量化器满足两个条件：(1) 每个质心 cᵢ 等于条件期望 E[X | X ∈ Rᵢ]，其中 Rᵢ 是 cᵢ 的 Voronoi 区域；(2) 区域之间的边界是相邻质心的中点。对于标准正态分布，这些条件产生具有唯一解的方程组 [9]，我们在第 3 节中利用这一点。

## 3 方法

### 3.1 概述

PolarQuant 分四个阶段量化权重张量 W ∈ ℝᵐˣⁿ：

1. **分块分解和归一化** 。展平 W 并划分为大小为 d 的块 {bᵢ}ᵢ₌₁ᴺ（我们使用 d=128）。提取 ℓ₂ 范数 rᵢ=‖bᵢ‖₂ 并归一化：b̂ᵢ = bᵢ/rᵢ。
2. **Hadamard 旋转** 。应用 d×d 归一化 Walsh-Hadamard 矩阵：b̃ᵢ = H_d b̂ᵢ。旋转后，每个坐标近似为 N(0, 1/d)。
3. **缩放和量化** 。缩放为单位方差：zᵢ = √d · b̃ᵢ，因此 zᵢ,ⱼ ∼ N(0,1)。将每个元素量化到最近的 Lloyd-Max 质心 c* = arg min_{cₖ} ‖zᵢ,ⱼ − cₖ‖。
4. **存储** 。存储量化码（int8）、每块范数（fp16）和质心表（fp32，全局共享）。

反量化是精确的逆过程：从码查找质心，缩放 1/√d，应用逆 Hadamard 旋转（因为 H_d⁻¹ = H_d），并按存储的范数 rᵢ 缩放。

### 3.2 Hadamard 旋转为何产生高斯坐标

我们现在为旋转坐标的正态性提供理论依据。

 **定义 3.1（归一化 Walsh-Hadamard 矩阵）** 。阶数为 d=2ᵏ 的 Walsh-Hadamard 矩阵 H_d 递归定义：

**\mathbf{H}_1 = [1], \quad \mathbf{H}_{2d} = \frac{1}{\sqrt{2}}\begin{bmatrix}\mathbf{H}_d & \mathbf{H}_d \\ \mathbf{H}_d & -\mathbf{H}_d\end{bmatrix} \tag{2}**

该矩阵是正交的（H_d H_dᵀ = I_d）且对称的（H_d = H_dᵀ），因此是自逆的（H_d⁻¹ = H_d）。

 **命题 3.2（旋转坐标的正态性）** 。设 b̂ ∈ ℝᵈ 为在单位球面 Sᵈ⁻¹ 上均匀分布的随机向量。设 H_d 为归一化 Walsh-Hadamard 矩阵。则对于每个坐标 j，旋转元素 b̃ⱼ = (H_d b̂)ⱼ 满足：

**\sqrt{d} \cdot \tilde{b}_j \xrightarrow{d} \mathcal{N}(0,1) \quad \text{当 } d \to \infty \tag{3}**

 **证明概要** 。由于 H_d 是正交的，b̃ = H_d b̂ 也在 Sᵈ⁻¹ 上均匀分布。单位球面上均匀点的每个坐标具有边缘分布 [9]：

**\tilde{b}_j \sim \frac{1}{\sqrt{d}} \cdot \text{Beta}\!\left(\tfrac{1}{2}, \tfrac{d-1}{2}\right)^{1/2} \cdot \text{Rademacher} \tag{4}**

根据球面投影的中心极限定理，当 d→∞ 时收敛于 N(0, 1/d)。对于 d=128，近似非常好：旋转 LLM 权重坐标的经验分布与 N(0, 1/d) 之间的 Kolmogorov-Smirnov 统计量通常低于 0.01。

### 3.3 标准正态分布的 Lloyd-Max 质心

鉴于旋转和缩放后的坐标 zᵢ,ⱼ ∼ N(0,1)，具有 L=2ᵇ 级的 MSE 最优标量量化器通过 Lloyd-Max 算法获得。

 **定理 3.4（Lloyd-Max 最优性条件）** 。对于具有密度 f(x) 的连续源 X，具有级 {c₁, ..., c_L} 和决策边界 {t₀, t₁, ..., t_L}（其中 t₀=−∞, t_L=+∞）的 MSE 最优量化器满足：

**c_i = \mathbb{E}[X \mid t_{i-1} < X \leq t_i] = \frac{\int_{t_{i-1}}^{t_i} x\,f(x)\,dx}{\int_{t_{i-1}}^{t_i} f(x)\,dx} \tag{5}**

**t_i = \frac{c_i + c_{i+1}}{2}, \quad i=1,\ldots,L-1 \tag{6}**

对于标准正态分布 f(x) = φ(x) = (1/√(2π))e^(−x²/2)，质心条件 (5) 简化为闭式表达式：

**c_i = \frac{\phi(t_{i-1}) - \phi(t_i)}{\Phi(t_i) - \Phi(t_{i-1})} \tag{7}**

其中 φ(·) 和 Φ(·) 分别是标准正态 PDF 和 CDF。这遵循恒等式 ∫ₐᵇ x φ(x) dx = φ(a) − φ(b)。

Lloyd-Max 算法在方程 (7) 和 (6) 之间迭代，从均匀间隔的边界开始，并单调收敛于 MSE。在实践中，机器精度的收敛在 50 次迭代内实现；我们使用 100 次迭代以确保安全。

 **命题 3.5（对称性）** 。对于标准正态分布，具有 L=2ᵇ 级的 Lloyd-Max 量化器关于零对称：对所有 i 有 cᵢ = −c_{L+1−i}。

 **证明** 。标准正态密度是对称的（φ(x) = φ(−x)），Lloyd-Max 不动点方程保持这种对称性。根据不动点的唯一性，解必须是对称的。

命题 3.5 将质心表的存储需求减半并简化了实现。

 **命题 3.6（相对于 Absmax 的 MSE 优势）** 。设 MSE_LM 和 MSE_abs 分别表示 N(0,1) 源在 b 位时 Lloyd-Max 和 absmax 量化器的均方量化误差。则：

**\frac{\text{MSE}_{\text{LM}}}{\text{MSE}_{\text{abs}}} \leq 0.46 \quad \text{在 } b=3 \tag{8}**

这种 54% 的 MSE 降低是 PolarQuant 相对于 absmax 的核心优势。改进源于 absmax 在 [−α, α] 上均匀放置量化级，在高斯密度低的尾部浪费分辨率，而 Lloyd-Max 将级集中在高密度中心区域。

### 3.4 算法

我们展示了完整的 PolarQuant 和 PolarDequant 算法。

**算法 1 PolarQuant：权重量化**

输入：权重张量 W ∈ ℝᵐˣⁿ，位数 b，块大小 d
输出：码 q，范数 r

1. 预计算 N(0,1) 的 Lloyd-Max 质心 c = {c₁, ..., c_{2ᵇ}}
2. 通过 (2) 构造归一化 Hadamard 矩阵 H_d
3. w ← flatten(W)
4. 将 w 划分为大小为 d 的块 {b₁, ..., b_N}
5. 对于 i = 1 到 N 执行：
   6. rᵢ ← ‖bᵢ‖₂  {提取块范数}
   7. b̂ᵢ ← bᵢ/rᵢ  {归一化到单位球面}
   8. b̃ᵢ ← H_d b̂ᵢ  {Hadamard 旋转}
   9. zᵢ ← √d · b̃ᵢ  {缩放到 N(0,1)}
   10. 对于 j = 1 到 d 执行：
   11. qᵢ,ⱼ ← arg minₖ ‖zᵢ,ⱼ − cₖ‖  {最近质心}
   12. 结束
   13. 结束
6. 返回 q, r

**算法 2 PolarDequant：权重重建**

输入：码 q，范数 r，质心 c，块大小 d，原始形状 (m,n)
输出：重建的权重张量 Ŵ ∈ ℝᵐˣⁿ

1. 构造归一化 Hadamard 矩阵 H_d
2. 对于 i = 1 到 N 执行：
   3. 对于 j = 1 到 d 执行：
   4. ẑᵢ,ⱼ ← c_{qᵢ,ⱼ}  {质心查找}
   5. 结束
   6. b̃̂ᵢ ← ẑᵢ/√d  {缩放回来}
   7. b̂̂ᵢ ← H_d b̃̂ᵢ  {逆 Hadamard (H_d⁻¹ = H_d)}
   8. b̂ᵢ ← rᵢ · b̂̂ᵢ  {恢复块范数}
3. 结束
4. Ŵ ← reshape([b̂₁, ..., b̂_N], (m,n))
5. 返回 Ŵ

 **复杂度** 。Walsh-Hadamard 变换允许 O(d log d) 快速实现（类似于 FFT），使 PolarQuant 与权重数量成线性关系。对于 d=128，通过 torch.matmul 的矩阵乘法 H_d bᵢ 在现代 GPU 上非常高效，在约 4 秒内完成 90 亿参数模型的整个反量化。

 **存储开销** 。PolarQuant 每 d 个元素存储一个 fp16 范数，对于 d=128 增加 16/d = 0.125 位每权重。质心表（2ᵇ 个 fp32 值）全局共享且可忽略。

### 3.5 PolarQuant + AWQ

AWQ [11] 和 PolarQuant 在正交轴上操作：AWQ 重新缩放**通道**以保护激活敏感权重，而 PolarQuant 归一化和旋转**块**以实现最优量化。它们的组合很简单：

1. 从校准数据计算 AWQ 每通道尺度 s。
2. 将权重乘以尺度：W' = W · diag(s)。
3. 对 W' 应用 PolarQuant。
4. 在反量化时，应用逆 AWQ：Ŵ = Ŵ' · diag(s⁻¹)。

这种组合将困惑度增量从 +0.52（仅 PolarQuant Q5）降低到仅 +0.06（PolarQuant Q5 + AWQ），如第 4 节所示。

### 3.6 作为 INT4 推理预处理的 PolarQuant

本工作的一个关键贡献是证明 PolarQuant 可作为**预处理**步骤来改善下游 INT4 量化的质量。流程是：

**\mathbf{W} \xrightarrow{\text{PolarQuant Q5}} \hat{\mathbf{W}}_{\text{PQ}} \xrightarrow{\text{dequant BF16}} \hat{\mathbf{W}}_{\text{BF16}} \xrightarrow{\text{torchao INT4}} \hat{\mathbf{W}}_{\text{INT4}} \tag{9}**

直觉如下。具有 Lloyd-Max 质心的 PolarQuant Q5 产生的反量化权重 Ŵ_BF16 与原始 W 的 MSE 低于 absmax Q5 反量化。当 torchao 随后将此中间表示重新量化为 INT4 时，它从 W 的**更好**近似开始。在 Ŵ_BF16 上计算的组级 absmax 尺度因子更能代表真实权重分布，因为 PolarQuant 已经移除了异常值引起的失真。

这与传统的双重量化不同（其中两个量化步骤都有损且误差累积）。相反，PolarQuant 充当**去噪**步骤：Q5 量化和反量化有效地将权重投影到更适合后续 absmax INT4 量化的表示上。

## 4 实验

### 4.1 设置

 **模型** 。我们在 Qwen3.5-9B [17] 上评估，这是一个具有混合 DeltaNet + MoE 架构的近期 90 亿参数语言模型，因其强大的基线质量和架构多样性而被选中。

 **硬件** 。主要实验在具有 96 GB VRAM 的 NVIDIA RTX PRO 6000 Blackwell GPU 上进行。跨平台实验使用具有 16 GB 统一内存的 Apple Mac mini M4。

 **评估** 。我们在 WikiText-2 [15] 上使用 2048 个 token 的滑动窗口和 512 的步长报告困惑度，在每个窗口中掩蔽前 1536 个上下文 token。此方法遵循先前量化工作 [8, 11] 建立的标准。所有困惑度数字都是确定性和可复现的。

 **速度测量** 。吞吐量测量为 100 个生成 token 的 3 次运行的平均值，在预热运行之后，以每秒 token 数（tok/s）表示。

 **基线** 。我们与以下方法比较：(1) FP16，全精度参考；(2) torchao INT4 [16]，具有组级 absmax 量化（组大小 128）；(3) BitsAndBytes NF4 [7]，NormalFloat 4 位格式。

### 4.2 主要结果

表 1 呈现了 Qwen3.5-9B 上的主要结果。反量化到 FP16 的 PolarQuant Q5 达到 6.39 的困惑度（Δ=+0.02），这 **无需任何校准数据即可达到近乎无损** 。当与 torchao INT4 结合进行高效推理时，PolarQuant 在所有 INT4 方法中达到最佳困惑度（6.56），与纯 torchao absmax INT4 相比，将到 FP16 的差距减少了 39%（从 +0.31 到 +0.19）。

**表 1：Qwen3.5-9B 上的主要结果（RTX PRO 6000 Blackwell）。PolarQuant Q5 + torchao INT4 在 INT4 方法中达到最佳质量，同时保持相当的速度和显存。**

值得注意的是，没有 AWQ 的 PolarQuant Q5 已经达到 Δ=+0.02，优于使用混合位分配测量的 PolarQuant+AWQ（Δ=+0.06）。这证明 **具有 Hadamard 旋转的统一 Q5 优于具有 AWQ 校准的混合位分配** ，这是一个简化推荐流程的反直觉发现。在反量化 FP16 范围内，PolarQuant Q5 以全 FP16 速度运行（45.9 tok/s），同时需要与 FP16 相同的显存（18.1 GB），使其适合作为高保真压缩存储和分发格式。

### 4.3 跨平台结果

为了证明可移植性，我们使用 MLX 框架 [1] 在 Apple Silicon 上评估 PolarQuant。

**表 2：Apple Mac mini M4（16 GB）上的跨平台结果**

PolarQuant 使在 16 GB 消费级设备上运行 90 亿参数模型成为可能，速度接近每秒 20 个 token，证明了最优高斯量化在边缘硬件上的实际适用性。

### 4.4 消融研究

表 3 分解了 PolarQuant 两个组件的贡献。结果揭示了一个惊人的发现： **Hadamard 旋转单独贡献了 98% 的改进** ，将困惑度从 6.90 降低到 6.40（Δ=−0.50）。Lloyd-Max 质心在 Q5 仅提供边际额外增益（Δ=−0.01），将困惑度带到 6.39。

这个结果有一个直观的解释。在 b=5 位（32 个量化级）时，级的密度足以用均匀间隔的质心很好地近似高斯分布。误差的主要来源不是质心放置，而是 **权重分布与量化器设计分布之间的不匹配** 。Hadamard 旋转通过将权重分布变换为高斯分布来消除这种不匹配，在高斯分布中，即使是简单的均匀量化器也表现接近最优。

在较低位宽（例如 b=2 或 b=3）时，少数可用质心必须仔细放置，Lloyd-Max 质心将贡献更大比例的改进。这与命题 3.6 一致，后者显示在 Q3 有 54% 的 MSE 降低。

**表 3：PolarQuant 组件的消融（Qwen3.5-9B，Q5）。Hadamard 旋转单独实现了总改进的 98%，将困惑度带到 FP16 的 +0.03 范围内。**

一个特别值得注意的结果是， **没有 AWQ 的 PolarQuant Q5 已经达到 6.39 的困惑度** ，低于在早期使用混合位分配实验的 PolarQuant+AWQ（6.43）。这是因为统一 Q5 在所有层中保留了比混合位（Q3-Q6）更多的信息，证实了 **具有旋转的统一位分配优于没有旋转的非统一分配** 。

### 4.5 版本演进

表 4 显示了我们量化方法的渐进发展，说明了每种技术的累积影响。

**表 4：跨开发版本的量化质量演进（Qwen3.5-9B）。**

从 v1（absmax）到 v3（PolarQuant + AWQ）的过渡将困惑度增量从 +0.89 降低到 +0.06，量化引起的质量损失减少了 93%。

## 5 分析

### 5.1 PolarQuant 为何改进下游 INT4

本工作的核心发现是，PolarQuant Q5 反量化权重由 torchao INT4 重新量化时，比直接 absmax INT4 产生更好的困惑度。我们提供以下解释。

Torchao 使用组大小为 128 的组级 absmax INT4 量化。对于每组 128 个权重，它计算 s=max|wⱼ| 并使用此尺度将所有值映射到 4 位整数。这种量化的质量关键取决于每组的 **动态范围** ：具有异常值的组具有大的 s 和大多数值的差分辨率。

如我们的消融（第 4.4 节）所示，主导机制是 Hadamard 旋转，贡献了 98% 的质量改进。旋转将权重分布从重尾、非均匀形状变换为近似高斯分布。在高斯范围内，每组 128 个元素的动态范围紧密集中在 2 ln 128 ≈ 3.1 个标准差周围，因此组级 absmax 尺度因子密切跟踪真实数据范围。没有旋转时，异常值权重膨胀 absmax 尺度，在尾部浪费分辨率。

PolarQuant Q5 反量化权重已经经历了这种分布归一化。当 torchao 随后重新量化为 INT4 时，它遇到跨组更均匀的权重分布，异常值更少，动态范围更一致。结果是每组 INT4 更好地利用其 16 个可用级。

本质上，PolarQuant 充当 **分布正则化器** ：Hadamard 旋转使权重分布均匀化，产生更适合均匀（absmax）量化的组。

### 5.2 双重量化悖论

人们可能期望在 torchao INT4 之前应用**较低**位宽（例如 Q3）的 PolarQuant 会产生更好的结果，因为分布正则化会更强。令人惊讶的是，情况正好相反：

* 混合位 PolarQuant（门/上投影的 Q3）+ torchao INT4：PPL 7.25（ **更差** ）
* 统一 PolarQuant Q5 + torchao INT4：PPL 6.56（ **更好** ）

解释是 Q3 反量化 **太有损** ：只有 8 个质心，反量化权重丢失了 torchao 的 INT4 量化器无法恢复的关键细粒度信息。最佳操作点要求 PolarQuant 预处理保留足够的信号供下游量化器利用。在 Q5（32 个质心）时，反量化权重保留足够信息供 torchao 产生高质量的 INT4 表示。

这建立了级联量化的原则：**预处理量化器必须以足够高的位宽操作，以保留下游量化器所需的信息。** 在我们的设置中，Q5 是 Q5→INT4 级联的最佳中间位宽。

### 5.3 计算开销

PolarQuant 反量化在 RTX PRO 6000 Blackwell 上增加约 8 秒的模型加载时间。这是一次性成本。在推理时，计算路径与标准 FP16 或 INT4 推理相同，零额外开销。反量化期间的主要成本是每块的批处理矩阵乘法 H₁₂₈b̂ᵢ；我们的实现使用 PyTorch 的 torch.matmul，通过利用优化的 cuBLAS GEMM 内核，比朴素的快速 Walsh-Hadamard 变换（FWHT）实现快 25 倍。

## 6 结论

我们提出了 PolarQuant，一种简单有效的权重量化方法，通过利用 Hadamard 旋转权重块的高斯结构实现接近无损的压缩。我们的消融表明，Hadamard 旋转是基本组件，在 Q5 贡献了 98% 的质量改进。我们的主要贡献是：

1. **接近无损压缩** 。PolarQuant Q5 在 Qwen3.5-9B 上达到 6.39 的困惑度（与 FP16 相比 Δ=+0.02），无需任何校准数据，每块仅需一次 Hadamard 矩阵乘法。
2. **旋转是关键** 。我们的消融表明，Hadamard 旋转单独贡献了 98% 的改进（PPL 6.90 → 6.40），而 Lloyd-Max 质心在 Q5 仅提供边际额外收益。这一发现将方法简化为其基本组件：确定性正交旋转。
3. **改进的 INT4 推理** 。当用作预处理步骤时，PolarQuant Q5 + torchao INT4 达到 6.56 的困惑度，而直接 absmax INT4 为 6.68，吞吐量相同（43.1 tok/s）和几乎相同的显存（6.5 GB）。
4. **跨平台部署** 。PolarQuant 在 NVIDIA GPU 和 Apple Silicon 上运行，在 Mac mini M4 上达到 19.7 tok/s，显存 4.8 GB。

PolarQuant 与任何下游量化器（torchao、GGUF、MLX）兼容，其核心算法不需要校准数据（AWQ 是唯一使用校准的组件）。该方法不增加运行时开销，仅在模型加载时增加几秒钟的一次性反量化成本。

 **局限性和未来工作** 。PolarQuant 假设 Hadamard 旋转权重块可以被很好地近似为独立同分布高斯分布，这可能不适用于所有架构。当前实现未利用块间相关性。未来工作可将 PolarQuant 扩展到激活量化，探索具有高斯码本的向量量化，并研究级联量化管道的理论极限。

## 致谢

作者感谢开源社区提供的工具和框架，包括 PyTorch、torchao、MLX，以及 Qwen 团队发布高质量的开放权重模型。

## 参考文献

[1] Apple Machine Learning Research (2024). MLX: an array framework for Apple silicon. [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)

[2] S. Ashkboos 等人 (2024). QuaRot: outlier-free 4-bit inference in rotated LLMs. NeurIPS.

[3] S. Ashkboos 等人 (2025). TurboQuant: online vector quantization with near-optimal distortion rate. arXiv:2504.19874.

[4] R. Banner, Y. Nahshan, D. Soudry (2019). Post training 4-bit quantization of convolutional networks for rapid-deployment.

[5] J. Chee 等人 (2023). QuIP: 2-bit quantization of large language models with guarantees. NeurIPS.

[6] J. Chee 等人 (2024). QuIP#: even better LLM quantization with hadamard incoherence and lattice codebooks. ICML.

[7] T. Dettmers 等人 (2023). QLoRA: efficient finetuning of quantized language models. NeurIPS.

[8] E. Frantar 等人 (2023). GPTQ: accurate post-training quantization for generative pre-trained transformers. ICLR.

[9] R. M. Gray, D. L. Neuhoff (1998). Quantization. IEEE Transactions on Information Theory.

[10] B. Jacob 等人 (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference.

[11] J. Lin 等人 (2024). AWQ: activation-aware weight quantization for on-device LLM compression and acceleration. MLSys.

[12] Z. Liu 等人 (2025). SpinQuant: LLM quantization with learned rotations. ICLR.

[13] S. P. Lloyd (1982). Least squares quantization in PCM. IEEE Transactions on Information Theory.

[14] J. Max (1960). Quantizing for minimum distortion. IRE Transactions on Information Theory.

[15] S. Merity 等人 (2016). Pointer sentinel mixture models. arXiv:1609.07843.

[16] PyTorch Team (2024). Torchao: PyTorch architecture optimization. [https://github.com/pytorch/ao](https://github.com/pytorch/ao)

[17] Qwen Team (2025). Qwen3 technical report. arXiv preprint.

---

## 附录 A Lloyd-Max 质心值

表 5 列出了位宽 2 到 5 的标准正态分布的预计算 Lloyd-Max 质心。这些值通过 100 次 Lloyd-Max 算法迭代计算，并根据命题 3.5 关于零对称。仅显示非负质心；负质心通过取反获得。

**表 5：N(0,1) 的 Lloyd-Max 质心。仅显示非负值；完整码本是对称的。**

Q2 质心 {−1.5104, −0.4528, +0.4528, +1.5104} 和 Q3 质心 {−2.1520, −1.3440, −0.7560, −0.2451, +0.2451, +0.7560, +1.3440, +2.1520} 直接在实现中使用。

## 附录 B 混合位分配策略

受 Unsloth Dynamic 2.0 启发，我们研究了一种基于量化敏感性为不同张量类型分配不同位宽的混合位策略：

**表 6：按张量类型的混合位分配。**

这实现了约 3.7 位平均，质量与统一 Q5 相当但存储更小。然而，如第 5.2 节所述，将此混合位方案用作 torchao INT4 的预处理会降低质量（PPL 7.25），因为 Q3 层丢失太多信息。因此，混合位策略仅推荐用于直接 PolarQuant 推理（无下游重新量化）。

## 附录 C 条件期望公式的推导

为完整起见，我们推导公式 (7) 中使用的条件期望公式。对于 X ∼ N(0,1) 和密度 φ(x) = (1/√(2π))e^(−x²/2)：

**E**[**X**∣**a**<**X**≤**b**]**=**∫**a**bϕ**(**x**)**d**x**∫**a**bx**ϕ**(**x**)**d**x

**=**Φ**(**b**)**−**Φ**(**a**)**∫**a**b****x**⋅**2**π1e**−**x**2**/2**d**x

**=**Φ**(**b**)**−**Φ**(**a**)**[**−**2**π1e**−**x**2**/2**]**a**b**

**=**Φ**(**b**)**−**Φ**(**a**)**2**π1e**−**a**2**/2**−**2**π1e**−**b**2**/2 **

**= \frac{\phi(a) - \phi(b)}{\Phi(b) - \Phi(a)} \tag{10}**

关键步骤使用恒等式 d/dx[−φ(x)] = x φ(x)，这来自对 φ(x) = (1/√(2π))e^(−x²/2) 的微分。

## 附录 D 压缩比

表 7 总结了 PolarQuant 在不同位宽下的存储需求和压缩比。

**表 7：Qwen3.5-9B（约 9×10⁹ 参数）的 PolarQuant 存储分析。**

开销列考虑每块范数（fp16，每 128 个元素一个 = 16/128 = 0.125 位每权重）。AWQ 尺度为每通道添加一个 fp16 值，对于大矩阵可忽略。在 PolarQuant Q5 + torchao INT4 管道中，最终模型使用 torchao 的原生 INT4 格式，因此推理时开销为零。
