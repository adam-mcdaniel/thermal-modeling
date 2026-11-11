# GPU thermal model

Calibrate any liquid-cooled GPU from its power and junction sensor. Then predict its junction temperature from power alone.

    cc -O2 fit_gpu.c -lm -o fit_gpu
    ./fit_gpu gpu_fit.h gpu0.txt gpu1.txt ...         # each line: power_W timestamp_s junction_C (nan when there's no reading)

    cc -O2 predict_gpu.c -lm -o predict_gpu
    ./predict_gpu 0 power.txt > predicted.csv         # GPU 0; only power_W and timestamp_s are used

`fit_gpu` needs no coolant data and has no machine-specific numbers. It reports the error on the last 30% of the run when fitted only on the rest, and warns if the run was too short or too gentle.

**How good it is** (`gpu_validation.ipynb`, 20 GPUs on 5 Frontier nodes, rocHPL and HPG-MxP):

| Test | Junction error |
|---|---|
| Unseen last 40% of each run, against raw 1 °C readings | 0.82 °C |
| The same, on half-second averages, which removes the sensor's flicker | 0.36 °C (worst GPU 0.75 °C) |
| Calibrated on rocHPL, predicting another job on a *different* GPU in the same slot | 0.95 °C (0.44-1.58 °C), with the coolant set once from idle |
| Group-wide cooling learned from junction and power, predicting from power (real inlet, model-built layer response) | 0.16-0.23 °C, against about 1.2 °C with the water off |

**Not yet tested on real data:** predicting another job on the same GPU, and learning the coolant from real junction readings in a group-wide run. Neither run exists in these traces.

Read `coolant_C` as a rough estimate only. From junction and power alone, the fit can't fully separate warmer water from slightly more resistance. The junction comes out right, but the learned coolant was about 1.8 °C off in the test.

## 1. Calibrate

Use a run with a big power swing: the junction should move at least 15 °C, and high power should hold long enough for the temperature to level off. A calibration on a gentle job doesn't carry over (3.7-4.7 °C on rocHPL in the test). The best recipe, with the rest of the machine quiet so the coolant holds steady:

| Time | Load |
|---|---|
| 0-20 s | Idle |
| 20-140 s | The hardest load you have, held |
| 140-190 s | Idle |
| 190-215 s | 1 s on, 1 s off |
| 215-240 s | 0.1 s on, 0.1 s off |

This recipe has only been tested on a simulated GPU. Real rocHPL runs of 82-103 s also calibrate well.

**For cooling on a shared loop:** a run of at least 5 minutes with the whole cooling group busy lets `fit_gpu` find 3 more numbers for how the coolant moves. It keeps them only if they predict the held-back part better. Then pass `group` to `predict_gpu` when a job fills the group. A single-node run can't show this. The Frontier numbers in `gpu_fit.h` now come from `fit_water.py`, which uses a measured inlet instead, because no group-wide run with junction readings exists yet. One GPU's power can't reveal other jobs sharing the group.

**If today's coolant differs from the calibration run's,** a power-only prediction is off by that difference. One junction reading corrects it.

## 2. Use it from your own code

    GpuState state;
    GpuForecast forecast;
    const GpuFit *fit = &GPU_FITS[gpu];
    gpu_start(&state, fit, first_power_W, 0.0, first_time_s);
    gpu_forecast_start(&forecast, fit, 5.0);                  /* look 5 s ahead */

    /* every sample; junction_C can be NAN */
    GpuPrediction now = gpu_predict(&state, fit, power_W, junction_C, time_s);
    double worst = gpu_worst_case(&state, fit, &forecast);    /* the hottest it could be in 5 s */

    /* every 1-2 s */
    double cap_W = gpu_allowed_power(&state, fit, &forecast, limit_C);

Keep `cap_W` between idle power and the GPU's own limit before applying it. Each sample takes about 25 ns.

`gpu_predict` also returns `junction_C`, `coolant_C`, `water_out_C`, `flux_generated`, `flux_die_exit` and `flux_to_water`.

## Holding the temperature steady

From `gpu_control.ipynb`, on the rocHPL trace, with a limit 6 C below the peak:

| Method | Runtime | Energy | Fatigue (q = 4, relative to no control) |
|---|---|---|---|
| Power cap, updated every 1-2 s, 5 s forecast (assumes power goes as speed^2) | +9% | -6% | x0.46 |
| Sleeps, a decision every 20 ms or less | +21% | +4% | x0.48-0.54 |
| Sleeps, a decision every 100 ms | +21% | +4% | x1.6 (worse than nothing) |

- **Measured:** the worst-case forecast (held 97-100% of the time on the rocHPL trace, largest miss 0.7 C).
- **Simulated:** all of the control results. Nothing has controlled a real GPU yet.
- **Sleeps must be short.** Keep the work-plus-sleep period at 20 ms or less. Longer sleeps become thermal cycles of their own.
- **Check the speed assumption.** The cap's advantage depends on how power and speed trade off. Run a short job at 2-3 power caps to measure it.

## Files

| File | What it is |
|---|---|
| `gpu_model.h` | The model, forecast and control decisions. Include this from your own code |
| `gpu_fit.h` | The fitted numbers, written by `fit_gpu` |
| `fit_gpu.c` | Calibration: finds every number from power and junction samples |
| `predict_gpu.c` | Command line: power trace in, junction, cooling response and (with `horizon=5`) forecasts out |
| `replay_gpu.c` | Simulates a controller on a power trace: sleeps, power cap or sensor-only |
| `thermal_metrics.py` | Scores a temperature trace: rainflow cycle counting and relative fatigue |
| `fit_water.py` | Adds the 3 coolant numbers to `gpu_fit.h` from a measured inlet, for when no group-wide run with junction readings exists |
| `make_samples.py` | Writes the samples files from the Score-P traces, using Ampere |
| `reproduce.sh` | Regenerates the samples and fits and builds everything |
| `runtime/` | The thermal governor: a tapir library that paces rocBLAS calls to keep the junction steady, and a stress workload. See `runtime/README.md` |
| `gpu_validation.ipynb`, `gpu_model.ipynb`, `gpu_control.ipynb` | How good the model is, how it works, and using it to hold temperature steady. Each is generated by its `make_*.py` |

`DIE_AREA_CM2` and `WATER_CAPACITY` default to the MI250X and Frontier. Override them with `-D` for another GPU. They only affect the flux and `water_out_C` outputs, so the junction doesn't depend on them.
