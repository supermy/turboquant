//! SIFT 数据集读取模块
//!
//! fvecs 格式: 每个向量 = [维度(int32): 4B] + [d 个 float32: d×4B]
//! ivecs 格式: 每个向量 = [维度(int32): 4B] + [d 个 int32: d×4B]

use std::fs::File;
use std::io::{BufReader, Read, Result};
use std::path::Path;

/// 读取 fvecs 文件
///
/// 返回 (向量数, 维度, 数据)
pub fn read_fvecs(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len() as usize;
    let mut reader = BufReader::new(file);

    // 读取向量 0 的维度头，确定 d
    let mut hdr = [0u8; 4];
    reader.read_exact(&mut hdr)?;
    let d = i32::from_le_bytes(hdr) as usize;
    if !(1..=65536).contains(&d) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("无效维度: {}", d),
        ));
    }

    // 向量大小 = 4(维度头) + d*4(数据)
    let vec_sz = 4 + d * 4;
    let n = file_size / vec_sz;

    let mut data = Vec::with_capacity(n * d);

    // 第一个向量: 维度头已读，只读数据
    let mut buf = vec![0u8; d * 4];
    reader.read_exact(&mut buf)?;
    for chunk in buf.chunks(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    // 剩余向量: 每个先读维度头(校验)，再读数据
    for _ in 1..n {
        reader.read_exact(&mut hdr)?;
        let dim = i32::from_le_bytes(hdr) as usize;
        if dim != d {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("维度不一致: {} vs {}", dim, d),
            ));
        }
        reader.read_exact(&mut buf)?;
        for chunk in buf.chunks(4) {
            data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
    }

    Ok((n, d, data))
}

/// 读取 ivecs 文件
pub fn read_ivecs(path: &Path) -> Result<(usize, usize, Vec<i32>)> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len() as usize;
    let mut reader = BufReader::new(file);

    let mut hdr = [0u8; 4];
    reader.read_exact(&mut hdr)?;
    let k = i32::from_le_bytes(hdr) as usize;
    if !(1..=65536).contains(&k) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("无效维度: {}", k),
        ));
    }

    let vec_sz = 4 + k * 4;
    let n = file_size / vec_sz;

    let mut data = Vec::with_capacity(n * k);
    let mut buf = vec![0u8; k * 4];

    // 第一个向量
    reader.read_exact(&mut buf)?;
    for chunk in buf.chunks(4) {
        data.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    for _ in 1..n {
        reader.read_exact(&mut hdr)?;
        let dim = i32::from_le_bytes(hdr) as usize;
        if dim != k {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("维度不一致: {} vs {}", dim, k),
            ));
        }
        reader.read_exact(&mut buf)?;
        for chunk in buf.chunks(4) {
            data.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
    }

    Ok((n, k, data))
}

/// SIFT Small 数据集
pub struct SiftSmallDataset {
    pub base: Vec<f32>,
    pub nb: usize,
    pub query: Vec<f32>,
    pub nq: usize,
    pub groundtruth: Vec<i32>,
    pub d: usize,
    pub k: usize,
}

impl SiftSmallDataset {
    pub fn load(data_dir: &Path) -> Result<Self> {
        let (nb, d, base) = read_fvecs(&data_dir.join("siftsmall_base.fvecs"))?;
        let (nq, _dq, query) = read_fvecs(&data_dir.join("siftsmall_query.fvecs"))?;
        let (_, k, groundtruth) = read_ivecs(&data_dir.join("siftsmall_groundtruth.ivecs"))?;
        Ok(Self { base, nb, query, nq, groundtruth, d, k })
    }

    #[inline] pub fn get_base(&self, i: usize) -> &[f32] { &self.base[i * self.d..][..self.d] }
    #[inline] pub fn get_query(&self, i: usize) -> &[f32] { &self.query[i * self.d..][..self.d] }
    #[inline] pub fn get_groundtruth(&self, i: usize) -> &[i32] { &self.groundtruth[i * self.k..][..self.k] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_siftsmall() {
        let dir = Path::new("/Users/moyong/project/ai/models/data/siftsmall");
        if !dir.exists() { return; }
        let ds = SiftSmallDataset::load(dir).unwrap();
        println!("SIFT Small: {} base({}D) {} query({}D) gt({})",
                 ds.nb, ds.d, ds.nq, ds.d, ds.k);
        assert_eq!(ds.nb, 10000);
        assert_eq!(ds.nq, 100);
        assert_eq!(ds.d, 128);
        assert!(ds.k >= 10);
    }
}
