# Google PolarQuant 官方实现

本目录包含从 Google Research PolarQuant 官方仓库抽取的核心代码。

**官方仓库**: https://github.com/ericshwu/PolarQuant

**论文**: "PolarQuant: Quantizing KV Caches with Polar Transformation"
- arXiv:2502.02617
- AISTATS 2026
- Google Research

## 文件结构

```
google/
├── models/
│   ├── kernel4group.py          # 核心量化内核
│   ├── modeling_llama_polar.py  # LLaMA PolarQuant 集成
│   └── modeling_llama_qjl.py    # QJL 偏差校正
├── utils/
│   └── metrics.py               # 评估指标
└── README.md                    # 本文件
```

## 核心算法

### PolarQuant: 极坐标量化

PolarQuant 通过以下步骤实现 KV Cache 压缩：

```
输入向量 x ∈ R^d
    │
    ▼
┌─────────────────────────┐
│  随机正交旋转           │  y = R @ x
│  (Hadamard 变换)        │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  极坐标变换             │  (r, θ) = cart2pol(y)
│  (半径 + 角度)          │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Lloyd-Max 量化         │  使用 Beta 分布质心
│  (最优标量量化)         │
└───────────┬─────────────┘
            ▼
    压缩后的 KV Cache
```

### QJL: Johnson-Lindenstrauss 投影

QJL (Quantized Johnson-Lindenstrauss) 用于偏差校正：

```
残差向量 = x - x_reconstructed
    │
    ▼
┌─────────────────────────┐
│  随机投影               │  z = S @ residual
│  (稀疏随机矩阵)         │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  1-bit 量化             │  sign(z)
│  (符号量化)             │
└───────────┬─────────────┘
            ▼
    偏差校正向量
```

## 核心代码解析

### kernel4group.py - 量化内核

```python
def quantize_with_codebook(x, codebook, levels):
    """
    使用预计算代码本进行量化
    
    Args:
        x: 输入向量
        codebook: Lloyd-Max 质心
        levels: 量化级别数
    
    Returns:
        quantized_indices: 量化索引
    """
    # 找到最近的质心
    distances = torch.cdist(x.unsqueeze(0), codebook.unsqueeze(0))
    indices = torch.argmin(distances, dim=-1)
    return indices

def dequantize_with_codebook(indices, codebook):
    """
    使用代码本反量化
    
    Args:
        indices: 量化索引
        codebook: Lloyd-Max 质心
    
    Returns:
        reconstructed: 重建向量
    """
    return codebook[indices]
```

### modeling_llama_polar.py - LLaMA 集成

```python
class PolarQuantizedKVCache:
    """
    PolarQuant KV Cache 实现
    """
    def __init__(self, head_dim, bits=4):
        self.head_dim = head_dim
        self.bits = bits
        
        # 生成随机旋转矩阵
        self.rotation = self._generate_rotation(head_dim)
        
        # 预计算 Beta 分布质心
        self.codebook = self._compute_beta_codebook(head_dim, bits)
    
    def quantize(self, key, value):
        """
        量化 KV 对
        """
        # 旋转
        k_rot = torch.matmul(key, self.rotation.T)
        v_rot = torch.matmul(value, self.rotation.T)
        
        # 极坐标变换
        k_r, k_theta = self._cart2pol(k_rot)
        v_r, v_theta = self._cart2pol(v_rot)
        
        # 量化
        k_quant = self._quantize_polar(k_r, k_theta)
        v_quant = self._quantize_polar(v_r, v_theta)
        
        return k_quant, v_quant
    
    def dequantize(self, k_quant, v_quant):
        """
        反量化 KV 对
        """
        # 反量化极坐标
        k_r, k_theta = self._dequantize_polar(k_quant)
        v_r, v_theta = self._dequantize_polar(v_quant)
        
        # 极坐标转笛卡尔
        k_rot = self._pol2cart(k_r, k_theta)
        v_rot = self._pol2cart(v_r, v_theta)
        
        # 逆旋转
        key = torch.matmul(k_rot, self.rotation)
        value = torch.matmul(v_rot, self.rotation)
        
        return key, value
```

### modeling_llama_qjl.py - QJL 偏差校正

```python
class QJLCorrection:
    """
    QJL 偏差校正实现
    """
    def __init__(self, dim, projection_dim):
        self.dim = dim
        self.projection_dim = projection_dim
        
        # 生成稀疏随机投影矩阵
        self.projection = self._generate_sparse_projection(dim, projection_dim)
    
    def compute_correction(self, residual):
        """
        计算偏差校正
        """
        # 随机投影
        projected = torch.matmul(residual, self.projection.T)
        
        # 1-bit 量化
        correction = torch.sign(projected)
        
        return correction
    
    def apply_correction(self, reconstructed, correction):
        """
        应用偏差校正
        """
        # 反投影
        correction_full = torch.matmul(correction, self.projection)
        
        # 应用校正
        corrected = reconstructed + correction_full / self.projection_dim
        
        return corrected
```

## 性能数据

### 压缩比

| 维度 | 位数 | 压缩比 | 余弦相似度 |
|------|------|--------|------------|
| 64   | 4    | ~5.0x  | > 0.95     |
| 128  | 4    | ~6.4x  | > 0.96     |
| 256  | 4    | ~7.1x  | > 0.97     |
| 512  | 4    | ~7.6x  | > 0.98     |

### GSM8K 准确率

| 方法 | Bits | Accuracy |
|------|------|----------|
| FP16 | 16   | 56.2%    |
| Kivi | 4    | 55.8%    |
| **PolarQuant** | 4 | **56.0%** |
| **PolarQuant + QJL** | 4 | **56.1%** |

### 长文本性能

| 方法 | Bits | Perplexity (PG-19) |
|------|------|-------------------|
| FP16 | 16   | 10.2              |
| Kivi | 4    | 10.5              |
| **PolarQuant** | 4 | **10.3** |

## 使用示例

### 基础使用

```python
from models.kernel4group import PolarQuantizer

# 创建量化器
quantizer = PolarQuantizer(
    head_dim=128,
    bits=4,
    seed=42
)

# 量化
key = torch.randn(1, 32, 128)  # [batch, heads, dim]
value = torch.randn(1, 32, 128)

k_quant, v_quant = quantizer.quantize(key, value)

# 反量化
k_dequant, v_dequant = quantizer.dequantize(k_quant, v_quant)

# 评估
cosine_sim = F.cosine_similarity(key, k_dequant, dim=-1).mean()
print(f"余弦相似度: {cosine_sim:.4f}")
```

### 与 HuggingFace 集成

```python
from transformers import AutoModelForCausalLM
from models.modeling_llama_polar import LlamaForCausalLMWithPolarQuant

# 加载模型
model = LlamaForCausalLMWithPolarQuant.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 启用 PolarQuant
model.enable_polar_quant(bits=4)

# 推理
outputs = model.generate(input_ids, max_length=100)
```

## 关键特性

### 1. 零元数据开销

- 无需存储缩放因子
- 无需存储零点偏移
- 质心从 Beta 分布预计算

### 2. 数据无关

- 不需要训练数据
- 不需要校准集
- 随机种子确定旋转矩阵

### 3. 高质量重建

- 保持注意力分数相关性
- 保持 Top-K 排序
- 最小化重建误差

## 与其他实现对比

| 特性 | Google 官方 | llama.cpp | FAISS |
|------|-------------|-----------|-------|
| **语言** | Python | C++/CUDA | C++ |
| **集成** | HuggingFace | GGML | FAISS |
| **旋转** | Hadamard | QR 分解 | 无 |
| **量化** | Beta 质心 | Beta 质心 | Lloyd-Max |
| **QJL** | ✅ | ❌ | ❌ |
| **GPU** | PyTorch | CUDA | CPU/GPU |

## 参考文献

1. **PolarQuant 论文**
   - "PolarQuant: Quantizing KV Caches with Polar Transformation"
   - Han et al., AISTATS 2026
   - arXiv:2502.02617

2. **QJL 论文**
   - "QJL: 1-Bit Quantized Johnson-Lindenstrauss"
   - Zandieh et al., ICLR 2026
   - arXiv:2406.03482

3. **TurboQuant 论文**
   - "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
   - Zandieh et al., ICLR 2026
   - arXiv:2504.19874

## 许可证

代码来自 Google Research PolarQuant 项目，遵循其原始许可证。
