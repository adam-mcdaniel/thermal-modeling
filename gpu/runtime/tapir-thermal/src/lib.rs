//! `libtapir_thermal.so`: keeps a GPU's junction temperature steady by pacing a program's rocBLAS calls.
//!
//! ```sh
//! TAPIR_PROFILERS="thermal(fit=gpu_fit.h,log=thermal.csv)" LD_PRELOAD=libtapir_thermal.so ./app
//! ```
//!
//! Before each rocBLAS call the program makes, the C core (../thermal_core.c) decides on a gap. While the GPU idles, it decides
//! on filler work (../thermal_rocm.c). This file only connects tapir's events to it.
//!
//! Options (all optional; see ../thermal_core.h for what they do):
//!
//! | option | default | |
//! |---|---|---|
//! | `fit` | `$THERMAL_FIT` or `gpu_fit.h` | the numbers from `fit_gpu` |
//! | `gpu` | the program's device | which GPU's numbers in `fit` |
//! | `horizon` | 5 | seconds the forecast looks ahead |
//! | `band` | 1 | °C either side of the target |
//! | `setpoint` | `follow` | a fixed target in °C, or `follow` |
//! | `rise`, `fall` | 1, 0.05 | °C/s a following target may rise and fall |
//! | `ceiling` | `none` | °C a following target never exceeds |
//! | `max_gap` | 0.005 | the longest gap before one call, in seconds |
//! | `fill` | 0.001 | seconds of filler per piece; 0 = never fill |
//! | `fill_size` | 2048 | filler DGEMMs are this size, n x n |
//! | `log`, `log_every` | none, 0.01 | a CSV of what it saw and did, every `log_every` seconds |
//! | `smi` | found from the PCI address | rocm_smi's number for the GPU |
//! | `dry_run` | 0 | 1 = decide and log, but don't apply |

use std::cell::Cell;
use std::ffi::{c_char, c_int, c_long, c_void, CString};

use tapir::{Call, Ctx, Event, Opts, Profiler, Sink, Value};

tapir::preload!(|b| {
    b.default_key(tapir_vocab::LINALG.key);
    b.factory("thermal", Thermal::from_opts);
    b.from_env("thermal");
});

include!(concat!(env!("OUT_DIR"), "/wrappers.rs"));

/// The same layout as `TgSettings` in thermal_core.h.
#[repr(C)]
struct TgSettings {
    horizon_s: f64,
    band_c: f64,
    setpoint_c: f64,
    rise_c_per_s: f64,
    fall_c_per_s: f64,
    ceiling_c: f64,
    max_gap_s: f64,
    fill_chunk_s: f64,
    prior_flops_per_s: f64,
}

/// The same layout as `ThermalOptions` in thermal_rocm.c.
#[repr(C)]
struct ThermalOptions {
    settings: TgSettings,
    fit_path: *const c_char,
    fit_gpu: c_int,
    log_path: *const c_char,
    log_every_s: f64,
    smi_index: c_int,
    fill_size: c_int,
    dry_run: c_int,
}

extern "C" {
    fn thermal_start(options: *const ThermalOptions) -> c_int;
    fn thermal_before(handle: *mut c_void, key: u64, flops: f64) -> c_long;
    fn thermal_after(handle: *mut c_void, token: c_long);
    fn thermal_stop();
}

thread_local! {
    /// The token from `thermal_before`, for the matching `thermal_after` on the same thread.
    static TOKEN: Cell<c_long> = const { Cell::new(-1) };
}

struct Thermal {
    name: String,
    options: ThermalOptions,
    _fit: CString,
    _log: Option<CString>,
    /// None until the first call, then whether governing started.
    started: Option<bool>,
}

// The raw pointers point into the CStrings this struct owns.
unsafe impl Send for Thermal {}

/// A number, or a word meaning "none" (NaN).
fn number_or_none(o: &Opts, key: &str, default: f64) -> Result<f64, String> {
    match o.get(key) {
        None => Ok(default),
        Some("follow" | "none") => Ok(f64::NAN),
        Some(v) => v.parse().map_err(|_| format!("thermal: {key}={v} is not a number")),
    }
}

impl Thermal {
    fn from_opts(o: &Opts) -> Result<Thermal, String> {
        let fit = o.get("fit").map(String::from).or_else(|| std::env::var("THERMAL_FIT").ok()).unwrap_or_else(|| "gpu_fit.h".into());
        let fit = CString::new(fit).map_err(|e| e.to_string())?;
        let log = o.get("log").map(|p| CString::new(p).map_err(|e| e.to_string())).transpose()?;
        let settings = TgSettings {
            horizon_s: o.parse_or("horizon", 5.0)?,
            band_c: o.parse_or("band", 1.0)?,
            setpoint_c: number_or_none(o, "setpoint", f64::NAN)?,
            rise_c_per_s: o.parse_or("rise", 1.0)?,
            fall_c_per_s: o.parse_or("fall", 0.05)?,
            ceiling_c: number_or_none(o, "ceiling", f64::NAN)?,
            max_gap_s: o.parse_or("max_gap", 0.005)?,
            fill_chunk_s: o.parse_or("fill", 0.001)?,
            prior_flops_per_s: o.parse_or("prior_flops", 20e12)?,
        };
        let options = ThermalOptions {
            settings,
            fit_path: fit.as_ptr(),
            fit_gpu: o.parse_or("gpu", -1)?,
            log_path: log.as_ref().map_or(std::ptr::null(), |l| l.as_ptr()),
            log_every_s: o.parse_or("log_every", 0.01)?,
            smi_index: o.parse_or("smi", -1)?,
            fill_size: o.parse_or("fill_size", 2048)?,
            dry_run: o.flag("dry_run", false) as c_int,
        };
        Ok(Thermal { name: o.name.clone(), options, _fit: fit, _log: log, started: None })
    }
}

/// The rocBLAS handle, captured as an address by the generated wrappers (rule `:handle val@handle`).
fn handle_of(c: &Call) -> Option<*mut c_void> {
    match c.tagged("handle").next() {
        Some(Value::Ptr(p)) if p != 0 => Some(p as *mut c_void),
        _ => None,
    }
}

/// Which kind of operation this is: the function, its sizes to the nearest power of two, and its modes. Calls with the same key
/// are expected to cost about the same.
fn kind_of(c: &Call) -> u64 {
    let mut key: u64 = 0xcbf29ce484222325;
    let mut mix = |x: u64| {
        key ^= x;
        key = key.wrapping_mul(0x100000001b3);
    };
    c.spec.name.bytes().for_each(|b| mix(b as u64));
    for (_, tag, value) in c.named_args() {
        match (tag, value.as_i64()) {
            ("dim" | "count", Some(v)) => mix(64 - (v.max(0) as u64).leading_zeros() as u64),
            ("flag", Some(v)) => mix(v as u64),
            _ => {}
        }
    }
    key
}

/// A size hint for a kind never seen before: 2mnk for a GEMM (x4 for complex, x batch), 0 for anything else.
fn flops_of(c: &Call) -> f64 {
    let name = c.spec.name;
    if !name.contains("gemm") {
        return 0.0;
    }
    let dim = |n| c.arg(n).and_then(|v| v.as_i64()).unwrap_or(0) as f64;
    let batch = c.tagged("count").filter_map(|v| v.as_i64()).next().unwrap_or(1) as f64;
    let complex = if name.starts_with("rocblas_c") || name.starts_with("rocblas_z") { 4.0 } else { 1.0 };
    2.0 * dim("m") * dim("n") * dim("k") * batch * complex
}

impl Profiler for Thermal {
    fn name(&self) -> &str {
        &self.name
    }

    fn needs_time(&self) -> bool {
        false
    }

    fn wants_enter(&self) -> bool {
        true
    }

    fn on_event(&mut self, ev: &Event, _: &Ctx) {
        let (c, before) = match ev {
            Event::Enter(c) => (c, true),
            Event::Call(c) => (c, false),
            _ => return,
        };
        // Only the program's own rocBLAS calls: not ones rocBLAS makes itself
        if c.depth > 0 || c.spec.lib != "rocblas" {
            return;
        }
        let Some(handle) = handle_of(c) else { return };
        if before {
            let on = *self.started.get_or_insert_with(|| unsafe { thermal_start(&self.options) } == 0);
            if on {
                TOKEN.with(|t| t.set(unsafe { thermal_before(handle, kind_of(c), flops_of(c)) }));
            }
        } else {
            let token = TOKEN.with(|t| t.replace(-1));
            if token >= 0 {
                unsafe { thermal_after(handle, token) };
            }
        }
    }

    fn finish(&mut self, _: &mut Sink) {
        if self.started == Some(true) {
            unsafe { thermal_stop() };
        }
    }
}
