//! Lloyd-Max 量化器
//!
//! 实现 Lloyd-Max 最优量化算法，针对 Beta 分布优化的标量量化。
//! 专为 Hadamard 旋转后的单位向量设计。
//!
//! # 核心思想
//! 1. Hadamard 旋转后的向量分量服从 Beta 分布
//! 2. Lloyd-Max 算法找到最优量化中心和边界
//! 3. 最小化量化误差的均方误差

/// Lloyd-Max 量化器
///
/// 针对单位向量分量 (Beta 分布) 优化的标量量化器。
#[derive(Clone)]
pub struct LloydMaxQuantizer {
    /// 向量维度
    pub d: usize,
    /// 每维量化位数
    pub nbits: usize,
    /// 量化中心数量 (k = 2^nbits)
    pub k: usize,
    /// 量化中心点
    pub centroids: Vec<f32>,
    /// 量化边界点
    pub boundaries: Vec<f32>,
}

impl LloydMaxQuantizer {
    /// 创建新的 Lloyd-Max 量化器
    ///
    /// # 参数
    /// - `d`: 向量维度
    /// - `nbits`: 每维量化位数 (1-8)
    ///
    /// # 返回值
    /// 初始化的量化器，已计算最优中心和边界
    pub fn new(d: usize, nbits: usize) -> Self {
        let k = 1usize << nbits;
        let mut q = Self {
            d,
            nbits,
            k,
            centroids: Vec::new(),
            boundaries: Vec::new(),
        };
        q.build_codebook();
        q
    }

    /// 构建量化码本
    ///
    /// 计算最优量化中心和边界。
    fn build_codebook(&mut self) {
        self.centroids = vec![0.0f32; self.k];
        self.boundaries = vec![0.0f32; self.k - 1];

        // 简单情况: 一维向量，使用均匀分布
        if self.d == 1 {
            for i in 0..self.k {
                self.centroids[i] = if i < self.k / 2 { -1.0 } else { 1.0 };
            }
        } else {
            // 一般情况: 使用 Lloyd-Max 迭代
            self.lloyd_max_iteration();
        }

        // 边界点为相邻中心的平均值
        for i in 0..self.k - 1 {
            self.boundaries[i] = 0.5 * (self.centroids[i] + self.centroids[i + 1]);
        }
    }

    /// Lloyd-Max 迭代算法
    ///
    /// 针对 Beta 分布优化的迭代算法:
    /// 1. 初始化: 均匀分割 [-1, 1]
    /// 2. 迭代:
    ///    a. 根据当前边界计算新的中心
    ///    b. 根据新中心更新边界
    ///    c. 收敛检查
    fn lloyd_max_iteration(&mut self) {
        // 网格化积分参数
        let ngrid = 32768usize;
        let step = 2.0 / ngrid as f64;
        // Beta 分布参数: α = (d - 3) / 2
        let alpha = 0.5 * (self.d as f64 - 3.0);

        // 预计算权重和累积和
        let mut xs = vec![0.0f64; ngrid];
        let mut prefix_w = vec![0.0f64; ngrid + 1];   // 权重累积和
        let mut prefix_wx = vec![0.0f64; ngrid + 1];  // 加权累积和

        for i in 0..ngrid {
            let x = -1.0 + (i as f64 + 0.5) * step;
            // Beta 分布权重: (1 - x²)^α
            let one_minus_x2 = (1.0 - x * x).max(0.0);
            let w = if alpha == 0.0 {
                1.0
            } else {
                one_minus_x2.powf(alpha)
            };
            let w = if w.is_finite() && w >= 0.0 { w } else { 0.0 };

            xs[i] = x;
            prefix_w[i + 1] = prefix_w[i] + w;
            prefix_wx[i + 1] = prefix_wx[i] + w * x;
        }

        // 计算区间 [i0, i1] 的加权均值
        let range_mean = |i0: usize, i1: usize, fallback: f64| -> f64 {
            let w = prefix_w[i1] - prefix_w[i0];
            if w <= 0.0 {
                return fallback;
            }
            (prefix_wx[i1] - prefix_wx[i0]) / w
        };

        // 初始分割点: 等权重分割
        let mut cuts = vec![0usize; self.k + 1];
        cuts[self.k] = ngrid;
        let total_w = *prefix_w.last().unwrap();

        for i in 1..self.k {
            let target = total_w * i as f64 / self.k as f64;
            cuts[i] = match prefix_w.binary_search_by(|v| v.partial_cmp(&target).unwrap()) {
                Ok(idx) => idx.min(ngrid),
                Err(idx) => idx.min(ngrid),
            };
        }

        // 初始中心
        let mut centroids_d: Vec<f64> = (0..self.k)
            .map(|i| {
                let left = -1.0 + 2.0 * i as f64 / self.k as f64;
                let right = -1.0 + 2.0 * (i + 1) as f64 / self.k as f64;
                range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right))
            })
            .collect();

        // Lloyd-Max 迭代
        let mut boundaries_d = vec![0.0f64; self.k - 1];
        for _ in 0..100 {
            // 更新边界
            for i in 0..self.k - 1 {
                boundaries_d[i] = 0.5 * (centroids_d[i] + centroids_d[i + 1]);
            }

            // 更新分割点
            cuts[0] = 0;
            cuts[self.k] = ngrid;
            for i in 1..self.k {
                cuts[i] = match xs.binary_search_by(|v| v.partial_cmp(&boundaries_d[i - 1]).unwrap()) {
                    Ok(idx) => idx,
                    Err(idx) => idx,
                };
            }

            // 更新中心并检查收敛
            let mut max_delta = 0.0f64;
            for i in 0..self.k {
                let left = if i == 0 { -1.0 } else { boundaries_d[i - 1] };
                let right = if i + 1 == self.k { 1.0 } else { boundaries_d[i] };
                let c = range_mean(cuts[i], cuts[i + 1], 0.5 * (left + right));
                let c = c.min(right).max(left);
                max_delta = max_delta.max((c - centroids_d[i]).abs());
                centroids_d[i] = c;
            }

            // 收敛检查
            if max_delta < 1e-8 {
                break;
            }
        }

        // 排序并转换为 f32
        centroids_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.centroids = centroids_d.iter().map(|&c| c as f32).collect();
    }

    /// 计算编码后的字节大小
    ///
    /// # 返回值
    /// 每个向量的编码字节数
    pub fn code_size(&self) -> usize {
        (self.d * self.nbits + 7) / 8
    }

    /// 选择量化索引
    ///
    /// 根据输入值选择最近的量化中心索引。
    ///
    /// # 参数
    /// - `x`: 输入值
    ///
    /// # 返回值
    /// 量化中心索引 (0..k-1)
    pub fn select_index(&self, x: f32) -> u8 {
        match self.boundaries.binary_search_by(|b| b.partial_cmp(&x).unwrap()) {
            Ok(idx) => idx as u8,
            Err(idx) => idx as u8,
        }
    }

    /// 编码单个索引到位流
    ///
    /// # 参数
    /// - `idx`: 量化索引
    /// - `code`: 编码缓冲区
    /// - `i`: 第几个分量
    pub fn encode_index(&self, idx: u8, code: &mut [u8], i: usize) {
        let bit_offset = i * self.nbits;
        let byte_offset = bit_offset >> 3;
        let bit_shift = bit_offset & 7;
        let mask = ((1u16 << self.nbits) - 1) as u16;
        let packed = ((idx as u16) & mask) << bit_shift;
        code[byte_offset] |= (packed & 0xff) as u8;
        if bit_shift + self.nbits > 8 {
            code[byte_offset + 1] |= (packed >> 8) as u8;
        }
    }

    /// 从位流解码单个索引
    ///
    /// # 参数
    /// - `code`: 编码缓冲区
    /// - `i`: 第几个分量
    ///
    /// # 返回值
    /// 量化索引
    pub fn decode_index(&self, code: &[u8], i: usize) -> u8 {
        let bit_offset = i * self.nbits;
        let byte_offset = bit_offset >> 3;
        let bit_shift = bit_offset & 7;
        let mask = ((1u16 << self.nbits) - 1) as u16;

        let mut packed = code[byte_offset] as u16;
        if bit_shift + self.nbits > 8 {
            packed |= (code[byte_offset + 1] as u16) << 8;
        }
        ((packed >> bit_shift) & mask) as u8
    }

    /// 编码整个向量
    ///
    /// # 参数
    /// - `x`: 输入向量
    /// - `code`: 输出编码缓冲区
    pub fn encode(&self, x: &[f32], code: &mut [u8]) {
        code.fill(0);
        for i in 0..self.d {
            let idx = self.select_index(x[i]);
            self.encode_index(idx, code, i);
        }
    }

    /// 解码整个向量
    ///
    /// # 参数
    /// - `code`: 编码缓冲区
    /// - `x`: 输出向量
    pub fn decode(&self, code: &[u8], x: &mut [f32]) {
        for i in 0..self.d {
            let idx = self.decode_index(code, i);
            x[i] = self.centroids[idx as usize];
        }
    }

    /// 计算编码向量与查询的距离
    ///
    /// # 参数
    /// - `code`: 编码向量
    /// - `query`: 查询向量 (已旋转)
    ///
    /// # 返回值
    /// L2 距离平方
    pub fn compute_distance(&self, code: &[u8], query: &[f32]) -> f32 {
        let mut dist = 0.0f32;
        for i in 0..self.d {
            let idx = self.decode_index(code, i);
            let diff = self.centroids[idx as usize] - query[i];
            dist += diff * diff;
        }
        dist
    }

    /// 构建查询向量的字节级查找表
    ///
    /// 对于 4-bit 量化，每字节包含 2 个索引。
    /// 预计算每个字节值对距离的贡献，查询时直接查表。
    ///
    /// # 返回值
    /// lookup[byte_idx][byte_val] = 该字节对距离平方的贡献
    pub fn build_distance_lut(&self, query: &[f32]) -> Vec<[f32; 256]> {
        let code_sz = self.code_size();
        let mut lookup = Vec::with_capacity(code_sz);

        if self.nbits == 4 {
            for byte_idx in 0..code_sz {
                let mut table = [0.0f32; 256];
                for byte_val in 0u32..256 {
                    let idx_lo = (byte_val & 0x0F) as usize;
                    let idx_hi = ((byte_val >> 4) & 0x0F) as usize;
                    let dim_lo = byte_idx * 2;
                    let dim_hi = byte_idx * 2 + 1;
                    let mut acc = 0.0f32;
                    if dim_lo < self.d {
                        let diff = self.centroids[idx_lo] - query[dim_lo];
                        acc += diff * diff;
                    }
                    if dim_hi < self.d {
                        let diff = self.centroids[idx_hi] - query[dim_hi];
                        acc += diff * diff;
                    }
                    table[byte_val as usize] = acc;
                }
                lookup.push(table);
            }
        } else {
            for byte_idx in 0..code_sz {
                let mut table = [0.0f32; 256];
                for byte_val in 0u32..256 {
                    let mut acc = 0.0f32;
                    for bit in 0..8 {
                        let dim = byte_idx * 8 + bit;
                        if dim * self.nbits / 8 >= code_sz {
                            break;
                        }
                        if dim < self.d {
                            let idx = self.decode_index_from_byte(byte_val as u8, byte_idx, bit);
                            let diff = self.centroids[idx as usize] - query[dim];
                            acc += diff * diff;
                        }
                    }
                    table[byte_val as usize] = acc;
                }
                lookup.push(table);
            }
        }

        lookup
    }

    pub fn build_distance_lut_into(&self, query: &[f32], buf: &mut Vec<[f32; 256]>) {
        let code_sz = self.code_size();
        if buf.len() < code_sz {
            buf.resize(code_sz, [0.0f32; 256]);
        }

        if self.nbits == 4 {
            for byte_idx in 0..code_sz {
                let table = &mut buf[byte_idx];
                for byte_val in 0u32..256 {
                    let idx_lo = (byte_val & 0x0F) as usize;
                    let idx_hi = ((byte_val >> 4) & 0x0F) as usize;
                    let dim_lo = byte_idx * 2;
                    let dim_hi = byte_idx * 2 + 1;
                    let mut acc = 0.0f32;
                    if dim_lo < self.d {
                        let diff = self.centroids[idx_lo] - query[dim_lo];
                        acc += diff * diff;
                    }
                    if dim_hi < self.d {
                        let diff = self.centroids[idx_hi] - query[dim_hi];
                        acc += diff * diff;
                    }
                    table[byte_val as usize] = acc;
                }
            }
        } else {
            for byte_idx in 0..code_sz {
                let table = &mut buf[byte_idx];
                for byte_val in 0u32..256 {
                    let mut acc = 0.0f32;
                    for bit in 0..8 {
                        let dim = byte_idx * 8 + bit;
                        if dim * self.nbits / 8 >= code_sz {
                            break;
                        }
                        if dim < self.d {
                            let idx = self.decode_index_from_byte(byte_val as u8, byte_idx, bit);
                            let diff = self.centroids[idx as usize] - query[dim];
                            acc += diff * diff;
                        }
                    }
                    table[byte_val as usize] = acc;
                }
            }
        }
    }

    pub fn build_distance_lut_range_into(&self, query: &[f32], range_start: usize, range_end: usize, buf: &mut Vec<[f32; 256]>) {
        let chunk_len = range_end - range_start;
        if buf.len() < chunk_len {
            buf.resize(chunk_len, [0.0f32; 256]);
        }

        if self.nbits == 4 {
            for j in 0..chunk_len {
                let byte_idx = range_start + j;
                let table = &mut buf[j];
                for byte_val in 0u32..256 {
                    let idx_lo = (byte_val & 0x0F) as usize;
                    let idx_hi = ((byte_val >> 4) & 0x0F) as usize;
                    let dim_lo = byte_idx * 2;
                    let dim_hi = byte_idx * 2 + 1;
                    let mut acc = 0.0f32;
                    if dim_lo < self.d {
                        let diff = self.centroids[idx_lo] - query[dim_lo];
                        acc += diff * diff;
                    }
                    if dim_hi < self.d {
                        let diff = self.centroids[idx_hi] - query[dim_hi];
                        acc += diff * diff;
                    }
                    table[byte_val as usize] = acc;
                }
            }
        } else {
            for j in 0..chunk_len {
                let byte_idx = range_start + j;
                let table = &mut buf[j];
                for byte_val in 0u32..256 {
                    let mut acc = 0.0f32;
                    for bit in 0..8 {
                        let dim = byte_idx * 8 + bit;
                        if dim * self.nbits / 8 >= self.code_size() {
                            break;
                        }
                        if dim < self.d {
                            let idx = self.decode_index_from_byte(byte_val as u8, byte_idx, bit);
                            let diff = self.centroids[idx as usize] - query[dim];
                            acc += diff * diff;
                        }
                    }
                    table[byte_val as usize] = acc;
                }
            }
        }
    }

    pub fn build_split_lut(&self, query: &[f32]) -> (Vec<[f32; 16]>, Vec<[f32; 16]>) {
        let code_sz = self.code_size();
        let mut lo_lut = vec![[0.0f32; 16]; code_sz];
        let mut hi_lut = vec![[0.0f32; 16]; code_sz];

        if self.nbits == 4 {
            for j in 0..code_sz {
                let dim_lo = j * 2;
                let dim_hi = j * 2 + 1;
                for idx in 0..16usize {
                    if dim_lo < self.d {
                        let diff = self.centroids[idx] - query[dim_lo];
                        lo_lut[j][idx] = diff * diff;
                    }
                    if dim_hi < self.d {
                        let diff = self.centroids[idx] - query[dim_hi];
                        hi_lut[j][idx] = diff * diff;
                    }
                }
            }
        }

        (lo_lut, hi_lut)
    }

    pub fn compute_distance_with_split_lut(&self, code: &[u8], lo_lut: &[[f32; 16]], hi_lut: &[[f32; 16]]) -> f32 {
        let mut dist = 0.0f32;
        for j in 0..code.len() {
            let byte = code[j] as usize;
            dist += lo_lut[j][byte & 0xF] + hi_lut[j][byte >> 4];
        }
        dist
    }

    pub fn compute_distance_with_lut(&self, code: &[u8], lut: &[[f32; 256]]) -> f32 {
        let mut dist = 0.0f32;
        for byte_idx in 0..code.len() {
            dist += lut[byte_idx][code[byte_idx] as usize];
        }
        dist
    }

    fn decode_index_from_byte(&self, byte_val: u8, _byte_idx: usize, bit: usize) -> u8 {
        let i = _byte_idx * 8 + bit;
        let bit_offset = i * self.nbits;
        let byte_offset = bit_offset >> 3;
        let bit_shift = bit_offset & 7;
        let mask = ((1u16 << self.nbits) - 1) as u16;
        let mut packed = byte_val as u16;
        if bit_shift + self.nbits > 8 {
            packed |= (byte_val as u16) << 8;
        }
        ((packed >> bit_shift) & mask) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试 4-bit 量化器初始化
    #[test]
    fn test_lloyd_max_4bit() {
        let q = LloydMaxQuantizer::new(128, 4);
        assert_eq!(q.k, 16);
        assert_eq!(q.centroids.len(), 16);
        assert_eq!(q.boundaries.len(), 15);
        // 中心应该单调递增
        assert!(q.centroids.windows(2).all(|w| w[0] <= w[1]));
    }

    /// 测试编解码往返
    #[test]
    fn test_encode_decode_roundtrip() {
        let q = LloydMaxQuantizer::new(128, 4);
        let x = vec![0.5f32; 128];
        let mut code = vec![0u8; q.code_size()];
        q.encode(&x, &mut code);
        let mut decoded = vec![0.0f32; 128];
        q.decode(&code, &mut decoded);
        for i in 0..128 {
            let idx = q.select_index(x[i]);
            assert!((decoded[i] - q.centroids[idx as usize]).abs() < 1e-5);
        }
    }

    /// 测试编码大小计算
    #[test]
    fn test_code_size() {
        let q4 = LloydMaxQuantizer::new(128, 4);
        assert_eq!(q4.code_size(), 64);  // 128 * 4 / 8 = 64
        let q6 = LloydMaxQuantizer::new(128, 6);
        assert_eq!(q6.code_size(), 96);  // 128 * 6 / 8 = 96
        let q8 = LloydMaxQuantizer::new(128, 8);
        assert_eq!(q8.code_size(), 128); // 128 * 8 / 8 = 128
    }
}
