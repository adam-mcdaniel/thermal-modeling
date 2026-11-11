//! Generate rocBLAS wrappers from the installed ROCm headers, as tapir-rocm does, with one difference: the rocblas_handle is
//! captured (as its address, tag "handle"), so the hook can find the stream the call runs on.
//! Then compile the C that does the work: ../thermal_core.c (the decisions) and ../thermal_rocm.c (HIP, rocBLAS, rocm_smi).
//!
//! Build-time environment: ROCM_PATH (or ROCM_HOME, CRAY_ROCM_DIR), default /opt/rocm. Without ROCm headers, a stub header
//! with a few functions is used, which is only good for checking that everything compiles. CC picks the C compiler.

use std::path::{Path, PathBuf};
use std::process::Command;

use tapir_gen::{Generator, TypeMap};
use tapir_vocab::LINALG;

/// Setup, query and bookkeeping APIs: not work. The same list as tapir-rocm.
const EXCLUDE: &[&str] = &[
    "*_create*", "*_destroy*", "*_set_*", "*_get_*", "*_query*", "*_version*", "*_initialize",
    "*_is_*", "*_abort", "*status_to_string", "*buffer_size*", "*_start_*", "*_stop_*",
    "*_log_*", "*device_malloc*", "*_info_*", "*_extract_*",
];

const ENUMS: &[&str] = &[
    "rocblas_operation", "rocblas_fill", "rocblas_diagonal", "rocblas_side", "rocblas_datatype", "rocblas_computetype",
    "rocblas_gemm_algo", "rocblas_gemm_flags", "rocblas_layer_mode", "rocblas_pointer_mode", "rocblas_atomics_mode",
    "rocblas_math_mode", "rocblas_geam_ex_operation", "rocblas_status", "rocblas_check_numerics_mode",
];

fn rocm_root() -> PathBuf {
    for var in ["ROCM_PATH", "ROCM_HOME", "CRAY_ROCM_DIR"] {
        println!("cargo:rerun-if-env-changed={var}");
        if let Some(p) = std::env::var_os(var).filter(|v| !v.is_empty()) {
            return p.into();
        }
    }
    "/opt/rocm".into()
}

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Wrappers for every rocBLAS function
    let mut include = rocm_root().join("include");
    if !include.join("rocblas/internal/rocblas-functions.h").exists() {
        println!("cargo:warning=tapir-thermal: no ROCm headers under {}, so using the stub header: this build can't govern a real program", include.display());
        include = here.join("stub");
    }
    let mut types = TypeMap::c_basic();
    types.scalar("rocblas_int", "i32").scalar("rocblas_stride", "i64").scalar("rocblas_size_t", "usize").scalar("rocblas_layer_mode_flags", "u32");
    types.handle("rocblas_handle").handle("hipStream_t");
    for e in ENUMS {
        types.enumeration(e);
    }
    let mut g = Generator::new(types);
    g.rules.parse_text(LINALG.args).unwrap();
    g.rules.add(":handle", "val@handle").unwrap();
    for pat in EXCLUDE {
        g.exclude(pat);
    }
    for header in ["rocblas/internal/rocblas-functions.h", "rocblas/internal/rocblas-beta.h"] {
        let path = include.join(header);
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
            g.header(&path, "rocblas", &["ROCBLAS_EXPORT"]).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        }
    }
    std::fs::write(out.join("wrappers.rs"), g.render()).unwrap();
    std::fs::write(out.join("report.txt"), g.report()).unwrap();

    // The C part, as a static library
    let runtime = here.join("..");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let mut objects = Vec::new();
    for source in ["thermal_core.c", "thermal_rocm.c"] {
        let object = out.join(Path::new(source).with_extension("o"));
        let status = Command::new(&cc)
            .args(["-O2", "-fPIC", "-c", "-Wno-unused-function"])
            .arg(runtime.join(source))
            .arg("-o")
            .arg(&object)
            .status()
            .expect("can't run the C compiler (set CC)");
        assert!(status.success(), "compiling {source} failed");
        objects.push(object);
    }
    let library = out.join("libthermal.a");
    let _ = std::fs::remove_file(&library);
    assert!(Command::new("ar").arg("crs").arg(&library).args(&objects).status().unwrap().success());
    for file in ["thermal_core.c", "thermal_core.h", "thermal_rocm.c", "gpu_telemetry.h", "../gpu_model.h"] {
        println!("cargo:rerun-if-changed={}", runtime.join(file).display());
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=thermal");
    println!("cargo:rustc-link-lib=dylib=m");
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
    }
}
