# tapir hook

**Purpose:** get the [governor core](governor-core.md) in front of every rocBLAS call a program makes, without changing or recompiling the program.

**In one line:** `libtapir_thermal.so` is a tapir preload library that wraps every rocBLAS function. Before each call it asks the core for a gap and applies it. After each call it marks the call's end on the GPU stream, so its run time and energy can be measured.

## Background: what tapir does

[tapir](https://github.com/adam-mcdaniel/tapir) is an `LD_PRELOAD` interposition framework:
* It generates a wrapper for every function in a library's header.
* Each wrapper captures chosen arguments, hands them to "profilers", then calls the real function (found with `dlsym(RTLD_NEXT)`).
* A profiler can ask for an `Enter` event, delivered on the calling thread **just before** the real call runs. That's where a gap goes.
* Calls the hook makes itself (HIP and rocBLAS calls from inside the profiler) skip the wrappers, so there's no recursion.

## What `tapir-thermal` adds

* **Generated wrappers, the same way as tapir's own `tapir-rocm`,** with one extra rule: `:handle val@handle`. The `rocblas_handle` argument is then captured as a pointer, and the hook asks rocBLAS which GPU stream it uses (`rocblas_get_stream`). Without ROCm headers it falls back to a stub header with 4 functions, which is only good for checking that it compiles.
* **A profiler called `thermal`:**
  * **On `Enter`, for the program's own rocBLAS calls only** (not calls rocBLAS makes internally), it works out the kind key (function, sizes, modes) and a FLOP estimate (2mnk for GEMMs). It calls `thermal_before(handle, key, flops)`:
    1. ask the core for a gap;
    2. if there is one, wait for the stream to finish its queued work (`hipStreamSynchronize`), then wait the gap, so the GPU really sits idle that long;
    3. record a HIP event on the stream to mark the call's start.
  * **On return,** it calls `thermal_after`, which records an event marking the end. The call may still be running on the GPU; the event completes when it's done.

**A background thread** (`thermal_rocm.c`), every millisecond:
1. Reads the energy counter and junction ([telemetry](telemetry.md)), and updates the model.
2. Finds finished calls: when a call's end event completes, it reads the start and end times, takes the energy used between them from the counter history, and teaches the core that kind's cost.
3. Asks the core for filler. If yes, it runs DGEMMs on a **low-priority stream** with its own rocBLAS handle. They're called through `RTLD_NEXT`, so the filler isn't itself governed.
4. Writes a log row.

**GPU time vs host time:** HIP events give times relative to each other. One reference event, recorded at start-up while the host clock is read, converts them to host time, so they line up with the telemetry.

## Using it

```sh
TAPIR_PROFILERS="thermal(fit=gpu_fit.h,log=thermal.csv,rise=1)" LD_PRELOAD=$PWD/libtapir_thermal.so ./app
```

| Option | Default | |
|---|---|---|
| `fit`, `gpu` | `$THERMAL_FIT` or `gpu_fit.h`; the program's HIP device | The calibration, and which row of it |
| `horizon`, `band`, `rise`, `fall`, `setpoint`, `ceiling`, `max_gap`, `fill` | see [governor core](governor-core.md) | |
| `fill_size` | 2048 | Filler DGEMM size |
| `log`, `log_every` | none, 0.01 s | CSV columns: `time_s, power_W, junction_C, model_C, target_C, top_C, bottom_C, worst_case_C, gaps, gap_s, fills, fill_s` |
| `smi` | found by PCI address | rocm_smi's index, if matching fails |
| `dry_run` | 0 | 1 = decide and log, but don't apply |

On exit it prints a summary: calls, gaps (count and total seconds), filler pieces.

## Building

```sh
cd gpu/runtime/tapir-thermal && cargo build --release      # ROCM_PATH picks the headers; tapir comes from GitHub
```

`build.rs` generates the wrappers, then compiles `thermal_core.c` and `thermal_rocm.c` into a static library linked into the `.so`. Everything GPU-related is looked up at run time with `dlsym`, so no ROCm libraries are needed at link time.

## What could go wrong on real hardware

* **Each gap drains the stream,** so calls launched ahead lose their overlap whenever a gap is needed.
* **Filler competes with the program** for the GPU; the program may wait up to 1 ms.
* **Several processes on one GPU** each govern on their own, without knowing about each other.
* **The first rocBLAS call starts the governor.** If HIP isn't ready by then, set-up fails, and it prints why and stays out of the way.

## Code

`gpu/runtime/tapir-thermal/` (`Cargo.toml`, `build.rs`, `src/lib.rs`, `stub/`), `gpu/runtime/thermal_rocm.c`.
