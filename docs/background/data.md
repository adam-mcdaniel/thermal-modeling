# Data

**Purpose:** list every measurement this project uses, where it came from, and how it becomes the input to the model.

## GPU traces (power + junction)

Five Score-P traces of single-node jobs on Frontier, each with the rocm_smi plugin recording the energy counter and junction sensor for devices 0, 2, 4 and 6 (one per MI250X package).

| Trace | Node | Job | Length | Where it lives |
|---|---|---|---|---|
| `hpl-trace` | frontier05319 | rocHPL, 8 ranks | 82 s | `data/rochpl-frontier/` (symlinks) and `~/Documents/ampere/examples/hpl-trace` |
| `hpl-captured-params` | frontier05987 | rocHPL | 102 s | `~/Documents/ampere/examples/` |
| `hpl-captured-params-HIP` | frontier06446 | rocHPL (HIP build) | 103 s | same |
| `hpgmxp` | frontier03455 | HPG-MxP, 128³ per rank | 54 s | same |
| `hpg-mxp` | frontier00995 | HPG-MxP | 58 s | same |

The original runs are under `/lustre/orion/csc688/world-shared/scorep-amd/runs/results/` on Frontier.

**From trace to samples:** `gpu/make_samples.py` reads each trace with Ampere:
* **Power:** Ampere's built-in rate from the energy counter (`get_statistics_vectorized(...)['rate']`). One sample per counter read, every ~3–9 ms.
* **Junction:** sensor 1, every ~6.5–20 ms, in whole °C.

It writes `gpu/samples/gpuN.txt` (rocHPL) and `gpu/samples/<trace>/gpuN.txt` (the others). Each line is `power_W timestamp_s junction_C`, with `nan` on lines that only carry power. That's the format `fit_gpu` and `predict_gpu` read.

**Quirks:**
* Single power samples spike up to ~3,700 W. These are counter-timing artifacts; the energy is right on average.
* Devices 1, 3, 5, 7 (the second GCD in each package) report zero energy.

## Cooling group telemetry (no junction)

| File | What | Rate |
|---|---|---|
| `3Cab_POW.txt` | 3 cabinets + their CDU during a 30-minute rocHPL run: cabinet AC power, node power sums, inlet (`LIT_Cn`) and outlet (`LOT_Cn`) water temperatures, CDU flows and temperatures | ~1 s |
| `cabinet_power_recon_gpu_energy_1ms.csv` (287 MB) | GPU power per cabinet and per slot, reconstructed from energy counters | 1 ms |

**There's no junction temperature for this run,** which is why learning the coolant from junction and power hasn't been tested on real data (see [coolant](../model/coolant.md)).

## Moving to another machine

* The `data/rochpl-frontier` symlinks point at a Mac path, so they'll break. `make_samples.py` takes `AMPERE_EXAMPLES` to point at the traces' new location.
* The samples files (`gpu/samples/`, 25 MB) are enough to refit and rerun everything without the raw traces.
