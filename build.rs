use std::env;
use std::path::PathBuf;

fn main() {
    let rocksdb_src = find_rocksdb_src();
    let rocksdb_include = rocksdb_src.join("include");

    cc::Build::new()
        .cpp(true)
        .file("cpp/vector_query_engine.cpp")
        .include(&rocksdb_include)
        .include("cpp")
        .flag("-std=c++17")
        .flag("-O3")
        .flag_if_supported("-march=native")
        .flag_if_supported("-D__STDC_FORMAT_MACROS")
        .compile("vector_engine");

    println!("cargo:rerun-if-changed=cpp/");
    println!("cargo:rustc-link-lib=static=vector_engine");
}

fn find_rocksdb_src() -> PathBuf {
    if let Ok(dir) = env::var("ROCKSDB_SRC_DIR") {
        return PathBuf::from(dir);
    }

    let registry = PathBuf::from(env::var("CARGO_HOME").unwrap_or_else(|_| "~/.cargo".into()))
        .join("registry/src");

    if let Ok(entries) = std::fs::read_dir(&registry) {
        for entry in entries.flatten() {
            let dir = entry.path();
            if let Ok(sub_entries) = std::fs::read_dir(&dir) {
                for sub in sub_entries.flatten() {
                    let name = sub.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with("librocksdb-sys") {
                        let rocksdb_dir = sub.path().join("rocksdb");
                        if rocksdb_dir.join("include/rocksdb/db.h").exists() {
                            return rocksdb_dir;
                        }
                    }
                }
            }
        }
    }

    let target_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default())
        .join("../../..");
    if let Ok(entries) = std::fs::read_dir(&target_dir) {
        for entry in entries.flatten() {
            let build_dir = entry.path().join("build");
            let inc = build_dir.join("rocksdb/include");
            if inc.join("rocksdb/db.h").exists() {
                return build_dir.join("rocksdb");
            }
        }
    }

    PathBuf::from("/usr/local")
}
