//! TurboQuant 命令行服务启动器
//!
//! 单文件启动向量搜索服务，读取 config.ini 配置。
//!
//! # 用法
//!
//! ```bash
//! # 使用默认配置启动
//! turboquant-server
//!
//! # 指定配置文件
//! turboquant-server -c /path/to/config.ini
//!
//! # 指定端口
//! turboquant-server -p 6000
//!
//! # 生成默认配置文件
//! turboquant-server --generate-config
//!
//! # 加载已有索引
//! turboquant-server --load-index /path/to/index.db
//! ```

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut config_path = PathBuf::from("config.ini");
    let mut port: Option<u16> = None;
    let mut generate_config = false;
    let mut load_index: Option<String> = None;
    let mut d: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-c" | "--config" => {
                if i + 1 < args.len() {
                    config_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "-p" | "--port" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "-d" | "--dimension" => {
                if i + 1 < args.len() {
                    d = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--generate-config" => {
                generate_config = true;
            }
            "--load-index" => {
                if i + 1 < args.len() {
                    load_index = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "-h" | "--help" => {
                print_help();
                return;
            }
            "-v" | "--version" => {
                println!("turboquant-server {}", env!("CARGO_PKG_VERSION"));
                return;
            }
            _ => {
                eprintln!("未知参数: {}", args[i]);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if generate_config {
        let ini_str = turboquant::TurboConfig::generate_default_ini();
        match std::fs::write(&config_path, &ini_str) {
            Ok(()) => println!("默认配置已写入: {}", config_path.display()),
            Err(e) => eprintln!("写入配置文件失败: {}", e),
        }
        return;
    }

    let mut config = if config_path.exists() {
        match turboquant::TurboConfig::load_from_file(&config_path) {
            Ok(c) => {
                println!("已加载配置: {}", config_path.display());
                c
            }
            Err(e) => {
                eprintln!("加载配置失败 ({}): {}", config_path.display(), e);
                eprintln!("使用默认配置...");
                turboquant::TurboConfig::default()
            }
        }
    } else {
        println!("配置文件不存在 ({}), 使用默认配置", config_path.display());
        turboquant::TurboConfig::default()
    };

    if let Some(dim) = d {
        config.index.d = dim;
    }

    if let Some(p) = port {
        let base_url = format!("tcp://127.0.0.1:{}", p);
        #[cfg(feature = "nng")]
        {
            config.server.query_url = base_url.clone();
            config.server.write_url = format!("tcp://127.0.0.1:{}", p + 1);
            config.server.notify_url = format!("tcp://127.0.0.1:{}", p + 2);
        }
        let _ = base_url;
    }

    if let Err(e) = config.validate() {
        eprintln!("配置验证失败: {}", e);
        std::process::exit(1);
    }

    config.storage.ensure_dirs().unwrap_or_else(|e| {
        eprintln!("创建数据目录失败: {}", e);
    });

    println!("{}", "=".repeat(60));
    println!("TurboQuant Server v{}", env!("CARGO_PKG_VERSION"));
    println!("{}", "=".repeat(60));
    println!("维度: {}", config.index.d);
    println!("聚类: {}", config.index.nlist);
    println!("量化: {}-bit", config.index.nbits);
    println!("SQ8: {}", config.index.use_sq8);
    println!("数据目录: {}", config.storage.data_dir.display());
    println!("数据库路径: {}", config.storage.db_path.display());
    #[cfg(feature = "nng")]
    println!("查询端口: {}", config.server.query_url);
    println!("{}", "=".repeat(60));

    #[cfg(feature = "nng")]
    {
        let engine = std::sync::Arc::new(turboquant::VectorEngineService::new(config.index.d));

        if let Some(ref index_path) = load_index {
            let t0 = Instant::now();
            match engine.load_index(index_path) {
                Ok(ntotal) => {
                    println!(
                        "已加载索引: {} ({} 向量, {:.1}ms)",
                        index_path,
                        ntotal,
                        t0.elapsed().as_secs_f64() * 1000.0
                    );
                }
                Err(e) => {
                    eprintln!("加载索引失败: {}", e);
                    std::process::exit(1);
                }
            }
        }

        let mut server = turboquant::TurboQuantServer::with_config(config.server.clone());
        if let Some(p) = port {
            server = server.with_query_url(&format!("tcp://127.0.0.1:{}", p));
        }

        println!("服务启动中...");
        match server.run() {
            Ok(()) => println!("服务已停止"),
            Err(e) => {
                eprintln!("服务错误: {}", e);
                std::process::exit(1);
            }
        }
    }

    #[cfg(not(feature = "nng"))]
    {
        eprintln!("NNG feature 未启用，无法启动网络服务");
        eprintln!("请使用 --features nng 编译");
        std::process::exit(1);
    }
}

fn print_help() {
    println!(
        r#"TurboQuant Server - 高性能向量量化搜索服务

用法: turboquant-server [选项]

选项:
  -c, --config <PATH>       配置文件路径 (默认: config.ini)
  -p, --port <PORT>         查询服务端口 (默认: 5555)
  -d, --dimension <DIM>     向量维度 (默认: 128)
  --generate-config         生成默认配置文件
  --load-index <PATH>       启动时加载已有索引
  -h, --help                显示帮助
  -v, --version             显示版本

示例:
  turboquant-server                           # 默认配置启动
  turboquant-server -p 6000                   # 指定端口
  turboquant-server -c /etc/tq/config.ini     # 指定配置
  turboquant-server --generate-config         # 生成配置文件
  turboquant-server --load-index data/tq.db   # 加载索引
"#
    );
}
