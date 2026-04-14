use std::env;
use std::path::PathBuf;

fn main() {
    let rocksdb_src = find_rocksdb_src();
    let rocksdb_include = rocksdb_src.join("include");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("cpp/vector_query_engine.cpp")
        .include(&rocksdb_include)
        .include("cpp")
        .flag_if_supported("-D__STDC_FORMAT_MACROS");

    if target_os == "windows" {
        build.flag("/std:c++17");
        build.flag("/O2");
        build.flag("/EHsc");
        if target_arch == "x86_64" {
            build.flag_if_supported("/arch:AVX2");
        }
    } else {
        build.flag("-std=c++17");
        build.flag("-O3");
        if target_arch == "x86_64" {
            build.flag_if_supported("-msse4.2");
        } else if target_arch == "aarch64" {
            build.flag_if_supported("-march=armv8-a+simd");
        }
    }

    build.compile("vector_engine");

    println!("cargo:rerun-if-changed=cpp/");
    println!("cargo:rustc-link-lib=static=vector_engine");

    if target_os == "windows" {
        println!("cargo:rustc-link-lib=shlwapi");
        println!("cargo:rustc-link-lib=rpcrt4");
    }
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

    let target_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default()).join("../../..");
    if let Ok(entries) = std::fs::read_dir(&target_dir) {
        for entry in entries.flatten() {
            let build_dir = entry.path().join("build");
            let inc = build_dir.join("rocksdb/include");
            if inc.join("rocksdb/db.h").exists() {
                return build_dir.join("rocksdb");
            }
        }
    }

    if cfg!(target_os = "windows") {
        PathBuf::from("C:/rocksdb")
    } else {
        PathBuf::from("/usr/local")
    }
}
