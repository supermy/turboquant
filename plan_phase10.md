# Phase 10: GitHub Actions 多平台多CPU构建 + 代码性能优化计划

## 一、问题分析

### 当前 CI 配置问题

| 问题 | 严重程度 | 说明 |
|------|----------|------|
| build.rs `-std=c++17` `-O3` 在 MSVC 上不兼容 | 🔴 高 | MSVC 使用 `/std:c++17` `/O2` |
| rocksdb 0.22 Windows 编译需要 MSVC+CMake+Perl | 🔴 高 | CI 上可能编译失败 |
| nng 1.0 Windows 需要 CMake | 🟡 中 | CI 上可能编译失败 |
| CI 缺少 aarch64-linux 交叉编译 | 🟡 中 | 缺少 ARM 服务器目标 |
| CI 缺少 macOS x86_64 构建 | 🟡 中 | Intel Mac 用户 |
| sq8_distance_simd 缺少 x86_64 AVX2 实现 | 🟡 中 | x86_64 上 SQ8 走标量路径 |
| C++ `__AVX2__` 宏在 MSVC 上不定义 | 🟡 中 | MSVC 用 `_M_AVX2` |
| Cargo.toml 缺少 [features] 条件编译 | 🟢 低 | 无法按需禁用 nng/C++ 引擎 |

### 性能优化机会

| 优化项 | 预估提升 | 说明 |
|--------|----------|------|
| sq8_distance_simd AVX2 实现 | +10-15% QPS (x86_64) | SQ8 精排在 x86_64 上走标量 |
| C++ AVX2 路径在 MSVC 上激活 | +10-15% QPS (Windows) | `__AVX2__` → 兼容 MSVC 检测 |
| Cargo feature 按需编译 | 减少编译时间 | 可选禁用 nng/C++ 引擎 |
| Release profile 优化 | 减少二进制大小 | LTO + strip + panic=abort |

## 二、实施计划

### Step 1: build.rs 跨平台修复 (关键)

**文件**: `build.rs`

修改内容:
1. `-std=c++17` → 仅非 Windows 使用; Windows 用 `/std:c++17`
2. `-O3` → 仅非 Windows 使用; Windows 用 `/O2`
3. x86_64 Windows 添加 `/arch:AVX2` (可选) 或 `/arch:SSE2`
4. aarch64 Windows 添加 `/arch:ARMv8-A`
5. Windows 链接 `rpcrt4` 库 (rocksdb 需要)
6. `find_rocksdb_src()` 改用 `librocksdb-sys` 的 OUT_DIR 查找

```rust
// 伪代码
if target_os == "windows" {
    build.flag("/std:c++17").flag("/O2");
    if target_arch == "x86_64" {
        build.flag_if_supported("/arch:AVX2");
    }
    println!("cargo:rustc-link-lib=rpcrt4");
} else {
    build.flag("-std=c++17").flag("-O3");
    if target_arch == "x86_64" {
        build.flag_if_supported("-msse4.2");
    } else if target_arch == "aarch64" {
        build.flag_if_supported("-march=armv8-a+simd");
    }
}
```

### Step 2: C++ AVX2 检测兼容 MSVC

**文件**: `cpp/simd_distance.h`

修改内容:
```cpp
// 修改前
#elif defined(__AVX2__)

// 修改后
#elif defined(__AVX2__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(_M_ARM64)
```

### Step 3: sq8_distance_simd 添加 x86_64 AVX2 实现

**文件**: `src/utils.rs`

为 `sq8_distance_simd` 添加 x86_64 AVX2 分支:
```rust
#[cfg(target_arch = "x86_64")]
{
    #[cfg(target_feature = "avx2")]
    {
        use std::arch::x86_64::*;
        // 8-wide SQ8 解码 + 距离计算
        // 使用 _mm256_cvtepi8_ps 不存在, 需要分步:
        // 1. 加载 8 个 u8 → _mm_loadl_epi64
        // 2. 零扩展到 u16 → _mm256_cvtepu8_epi16
        // 3. 转换到 f32 → _mm256_cvtepi32_ps (分两半)
        // 4. FMA: decoded = vmin + code * scale
        // 5. diff = decoded - query, dist += diff * diff
    }
}
```

### Step 4: Cargo.toml 添加 features 条件编译

**文件**: `Cargo.toml`

```toml
[features]
default = ["nng", "cpp-engine"]
nng = ["dep:nng"]
cpp-engine = []  # C++ SIMD 引擎 (需要 C++17 编译器)
static-rocksdb = []  # 静态链接 RocksDB

[dependencies]
nng = { version = "1.0", optional = true }
# rocksdb 始终需要 (核心存储)
rocksdb = "0.22"
```

### Step 5: GitHub Actions CI 全面升级

**文件**: `.github/workflows/ci.yml`

构建矩阵:
```yaml
matrix:
  include:
    # Linux
    - os: ubuntu-latest
      target: x86_64-unknown-linux-gnu
    - os: ubuntu-latest
      target: aarch64-unknown-linux-gnu
      cross: true
    # macOS
    - os: macos-latest
      target: aarch64-apple-darwin
    - os: macos-13  # Intel Mac
      target: x86_64-apple-darwin
    # Windows
    - os: windows-latest
      target: x86_64-pc-windows-msvc
```

关键改进:
1. 添加 aarch64-linux 交叉编译 (使用 cross)
2. 添加 macOS x86_64 (macos-13 runner)
3. Windows 安装 CMake + Perl (rocksdb/nng 需要)
4. 分离 feature 测试 (default / no-default-features)
5. 添加 Release artifact 上传
6. 缓存优化: 按 target 分离缓存

### Step 6: Release profile 优化

**文件**: `Cargo.toml`

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"
```

### Step 7: server.rs nng 条件编译

**文件**: `src/server.rs`

```rust
#[cfg(feature = "nng")]
pub struct TurboQuantServer { ... }

#[cfg(not(feature = "nng"))]
pub struct TurboQuantServer;
```

## 三、实施顺序

| 步骤 | 优先级 | 预估时间 | 风险 |
|------|--------|----------|------|
| Step 1: build.rs 跨平台修复 | P0 | 30min | 低 |
| Step 2: C++ AVX2 MSVC 兼容 | P0 | 15min | 低 |
| Step 3: sq8_distance_simd AVX2 | P1 | 30min | 中 |
| Step 4: Cargo features | P1 | 30min | 中 |
| Step 5: CI 全面升级 | P0 | 45min | 中 |
| Step 6: Release profile | P2 | 10min | 低 |
| Step 7: nng 条件编译 | P2 | 20min | 低 |

## 四、验证标准

1. ✅ `cargo build` 在 macOS aarch64 上编译通过
2. ✅ CI 在 ubuntu-latest (x86_64) 上编译通过
3. ✅ CI 在 macos-latest (aarch64) 上编译通过
4. ✅ CI 在 windows-latest (x86_64) 上编译通过
5. ✅ CI 在 ubuntu-latest + aarch64 交叉编译通过
6. ✅ `cargo test --lib` 全部通过
7. ✅ `cargo build --no-default-features` 编译通过 (无 nng)
8. ✅ Release 二进制大小减少 > 20%
