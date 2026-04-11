import numpy as np
import faiss

# ======================
# 1. 构造数据
# ======================
d = 128  # 向量维度
nb = 100000  # 10万条数据（模拟真实规模）
nq = 5  # 5条查询

# 随机生成向量（faiss 必须 float32）
xb = np.random.rand(nb, d).astype("float32")
xq = np.random.rand(nq, d).astype("float32")

print(f"数据量：{nb} 条 {d} 维向量")
print("="*50)

# ======================
# 2. IndexIVFFlat 示例（推荐：速度快、精度高）
# ======================
print("=== IndexIVFFlat ===")

nlist = 100  # 把数据分成 100 个桶（聚类中心）
quantizer = faiss.IndexFlatL2(d)  # 基础索引（用于分桶）

# 创建 IVF 索引
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)

# 必须先训练！
index_ivf.train(xb)

# 添加数据
index_ivf.add(xb)

# 搜索（nprobe 越大越准，越慢；默认10）
index_ivf.nprobe = 10
D_ivf, I_ivf = index_ivf.search(xq, k=5)

print("IVF 搜索结果（前5条邻居ID）：")
print(I_ivf[:2])  # 只打印前2条查询结果

print("="*50)

# ======================
# 3. IndexIVFPQ 示例（压缩存储：超省内存）
# ======================
print("=== IndexIVFPQ ===")

m = 16  # 把 128 维向量切成 16 段
nbits = 8  # 每段用 8bit 存储（压缩 16倍！）

# 创建 PQ 压缩索引
index_pq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# 必须训练
index_pq.train(xb)

# 添加数据
index_pq.add(xb)

# 搜索
index_pq.nprobe = 10
D_pq, I_pq = index_pq.search(xq, k=5)

print("IVFPQ 搜索结果（前5条邻居ID）：")
print(I_pq[:2])

print("\n✅ 全部运行成功！")
