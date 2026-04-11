import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time

# ======================
# 0. 配置
# ======================
dim = 384  # 向量维度（all-MiniLM 固定 384）
nlist = 50  # 分桶数
k = 3  # 搜索 Top3

# ======================
# 1. 加载模型 + 构造文本数据
# ======================
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "uv 是目前最快的 Python 包管理器，比 pip 快 10~100 倍",
    "Faiss 用于海量向量相似度搜索，是 RAG 系统必备组件",
    "IndexIVFFlat 适合百万级数据，速度快、精度高",
    "IndexIVFPQ 支持向量压缩，内存占用降低 8~32 倍",
    "RAG 通过检索增强生成，解决大模型遗忘问题",
    "macOS 上使用 Faiss 推荐 conda 或 uv 安装",
    "向量检索是 AI 应用的核心基础设施",
    "大模型需要搭配向量数据库才能实时更新知识",
]

print("生成向量中...")
vectors = model.encode(documents).astype("float32")
print(f"向量 shape: {vectors.shape}")

# ======================
# 2. 【推荐】IndexIVFFlat 示例（速度 + 精度平衡）
# ======================
print("\n===== IndexIVFFlat（工业级首选）=====")

quantizer = faiss.IndexFlatL2(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
index_ivf.train(vectors)
index_ivf.add(vectors)
index_ivf.nprobe = 10  # 精度调节

# 查询
query = "Python 最快包管理器"
query_vec = model.encode([query]).astype("float32")
start = time.time()
D, I = index_ivf.search(query_vec, k)
print(f"耗时: {time.time() - start:.3f}s")

for i, (idx, dist) in enumerate(zip(I[0], D[0])):
    print(f"{i+1}. {documents[idx]} | 距离: {dist:.2f}")

# 保存索引
faiss.write_index(index_ivf, "ivf_flat.faiss")
print("✅ 已保存: ivf_flat.faiss")

# ======================
# 3. 【大数据】IndexIVFPQ 示例（超省内存）
# ======================
print("\n===== IndexIVFPQ（千万级数据）=====")

m = 16    # 必须整除 dim（384/16=24）
nbits = 8 # 压缩参数

index_pq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
index_pq.train(vectors)
index_pq.add(vectors)
index_pq.nprobe = 10

start = time.time()
D, I = index_pq.search(query_vec, k)
print(f"耗时: {time.time() - start:.3f}s")

for i, (idx, dist) in enumerate(zip(I[0], D[0])):
    print(f"{i+1}. {documents[idx]} | 距离: {dist:.2f}")

faiss.write_index(index_pq, "ivf_pq.faiss")
print("✅ 已保存: ivf_pq.faiss")

# ======================
# 4. 加载索引示例（真实项目必用）
# ======================
print("\n===== 加载已保存的索引 =====")
load_index = faiss.read_index("ivf_flat.faiss")
D, I = load_index.search(query_vec, k)
print("加载后搜索成功！")
print("结果 ID:", I[0])
