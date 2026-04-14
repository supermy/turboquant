#!/usr/bin/env python3
"""
测试 Homebrew 安装的 Faiss 是否支持 RaBitQ
"""

import sys
import numpy as np

def test_faiss_installation():
    """测试基础 Faiss 安装"""
    try:
        import faiss
        print(f"✅ Faiss 导入成功")
        print(f"   版本: {faiss.__version__}")
        print(f"   编译选项: {faiss.__compile_options__ if hasattr(faiss, '__compile_options__') else 'N/A'}")
        return faiss
    except ImportError as e:
        print(f"❌ Faiss 导入失败: {e}")
        sys.exit(1)

def test_basic_index(faiss):
    """测试基础索引功能"""
    print("\n--- 测试基础索引 ---")
    d = 128
    nb = 10000
    xb = np.random.random((nb, d)).astype('float32')
    
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    
    xq = np.random.random((5, d)).astype('float32')
    D, I = index.search(xq, 10)
    
    print(f"✅ IndexFlatL2 工作正常")
    print(f"   数据库大小: {index.ntotal}")
    print(f"   查询结果形状: D={D.shape}, I={I.shape}")
    return True

def test_rabitq_index(faiss):
    """测试 RaBitQ 索引（Faiss 1.11.0+）"""
    print("\n--- 测试 RaBitQ 索引 ---")
    
    # 检查版本是否支持 RaBitQ
    version_str = faiss.__version__
    major, minor, patch = map(int, version_str.split('.')[:3])
    
    if (major, minor) < (1, 11):
        print(f"⚠️  Faiss 版本 {version_str} 不支持 RaBitQ（需要 >= 1.11.0）")
        return False
    
    try:
        # 测试 IndexRaBitQ
        d = 128
        nb = 10000
        nq = 100
        
        np.random.seed(1234)
        xb = np.random.random((nb, d)).astype('float32')
        xq = np.random.random((nq, d)).astype('float32')
        
        # 创建 RaBitQ 索引
        index = faiss.IndexRaBitQ(d, 8)  # 8 个子向量
        print(f"✅ IndexRaBitQ 创建成功")
        
        # 训练并添加向量
        index.train(xb)
        index.add(xb)
        print(f"✅ 训练完成，已添加 {index.ntotal} 个向量")
        
        # 搜索
        k = 10
        D, I = index.search(xq, k)
        print(f"✅ 搜索完成")
        print(f"   距离形状: {D.shape}")
        print(f"   索引形状: {I.shape}")
        print(f"   示例结果（第一查询前3个）:")
        print(f"     索引: {I[0, :3]}")
        print(f"     距离: {D[0, :3]}")
        
        return True
        
    except AttributeError as e:
        print(f"❌ RaBitQ 不可用: {e}")
        return False
    except Exception as e:
        print(f"❌ RaBitQ 测试失败: {e}")
        return False

def test_ivfrabitq_index(faiss):
    """测试 IVF + RaBitQ 索引"""
    print("\n--- 测试 IVFRaBitQ 索引 ---")
    
    try:
        d = 128
        nb = 10000
        nq = 100
        
        np.random.seed(1234)
        xb = np.random.random((nb, d)).astype('float32')
        xq = np.random.random((nq, d)).astype('float32')
        
        # 创建 IVF + RaBitQ
        nlist = 10
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFRaBitQ(quantizer, d, nlist, 8)
        
        print(f"✅ IndexIVFRaBitQ 创建成功")
        print(f"   度量类型: {index.metric_type} (1=L2, 0=IP)")
        
        index.train(xb)
        index.add(xb)
        print(f"✅ 训练完成，已添加 {index.ntotal} 个向量")
        
        index.nprobe = 5
        D, I = index.search(xq, 10)
        print(f"✅ 搜索完成 (nprobe={index.nprobe})")
        print(f"   结果形状: D={D.shape}, I={I.shape}")
        
        return True
        
    except AttributeError as e:
        print(f"❌ IVFRaBitQ 不可用: {e}")
        return False
    except Exception as e:
        print(f"❌ IVFRaBitQ 测试失败: {e}")
        return False

def main():
    print("=" * 60)
    print("Faiss + RaBitQ 测试 (Homebrew 安装)")
    print("=" * 60)
    
    # 测试 1: 基础安装
    faiss = test_faiss_installation()
    
    # 测试 2: 基础索引
    test_basic_index(faiss)
    
    # 测试 3: RaBitQ
    rabitq_ok = test_rabitq_index(faiss)
    
    # 测试 4: IVFRaBitQ
    ivfrabitq_ok = test_ivfrabitq_index(faiss)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"Faiss 版本: {faiss.__version__}")
    print(f"RaBitQ 支持: {'✅ 是' if rabitq_ok else '❌ 否'}")
    print(f"IVFRaBitQ 支持: {'✅ 是' if ivfrabitq_ok else '❌ 否'}")
    
    if rabitq_ok and ivfrabitq_ok:
        print("\n🎉 所有测试通过！Homebrew + pip 安装的 Faiss 完全支持 RaBitQ")
        return 0
    else:
        print("\n⚠️  部分测试失败，建议升级 Faiss 版本")
        return 1

if __name__ == "__main__":
    sys.exit(main())
