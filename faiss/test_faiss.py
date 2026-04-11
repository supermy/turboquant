import numpy as np
import faiss

# 向量维度
d = 128

# 造 1000 个随机向量
xb = np.random.rand(1000, d).astype("float32")

# 建立索引
index = faiss.IndexFlatL2(d)
index.add(xb)

# 搜索：拿前 5 个向量查最相似的 5 个邻居
k = 5
xq = xb[:5]
D, I = index.search(xq, k)

print("最近邻索引：")
print(I)
print(D)
print("最近邻索引:", I[0][1])
print("距离:", D[0][1])
