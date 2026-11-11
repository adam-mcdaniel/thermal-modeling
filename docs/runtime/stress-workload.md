# Stress workload

**Purpose:** make a GPU's temperature swing on purpose, in controlled patterns. That serves three jobs:
* **calibrating** the model on any GPU;
* **finding** which workload rhythms stress the cooling most;
* **measuring** what the governor does, before and after.

**In one line:** a small C program that runs DGEMMs back to back for "on" and sleeps for "off", following a schedule you choose. Optionally it logs power and junction every millisecond in `fit_gpu`'s format.

## Patterns

| Pattern | What it does | What it's for |
|---|---|---|
| `square --period 10 --duty 0.5` | on for 5 s, off for 5 s, repeated | The textbook case. A 30–60 s period should give the largest swing per cycle, since the cold plate (~16 s) has time to fully heat and cool. A 50–100 ms period works the die alone |
| `chirp --from 0.01 --to 60` | a square wave whose period sweeps from 10 ms to 60 s on a log scale | One run passes through every layer's response time (die ~30 ms, package ~1.5 s, cold plate ~16 s), showing which rhythm swings the junction most |
| `random --shortest 0.02 --longest 20` | on and off for random lengths | No rhythm to learn: the hardest case for anything predictive |
| `steps --levels 0,0.25,0.5,1 --hold 30` | held power levels | Checks the model is linear in power (each step's rise should scale) |
| `recipe` | `fit_gpu`'s 4-minute calibration run | Calibration (see [calibration](../model/calibration.md)) |

A level between 0 and 1 runs that fraction of each 10 ms frame. `--idle` adds idle time before and after (default 10 s), and `--size` sets the DGEMM size (default 4096).

## Two clocks

* **Default, time:** follows the schedule by the wall clock, timed against the schedule itself so small delays don't add up. Use this for calibration and stress.
* **`--work`, fixed work:** each "on" stretch runs as many DGEMMs as full speed would fit in, however long that takes. Use this to test the governor: its gaps then show up as a longer run.

## Using it

```sh
./thermal_stress recipe --device 0 --samples gpu0.txt                          # calibration data
./thermal_stress chirp --from 0.01 --to 60 --seconds 300 --samples chirp.txt   # stress the cooling
./thermal_stress random --work --samples off.txt                               # governor test, without it
```

It prints the DGEMM speed it measured at start-up, and at the end how long the run took against the schedule.

## Checked so far

Linked against a fake rocBLAS where each DGEMM sleeps 2 ms:
* **Time mode follows the schedule to the hundredth of a second:** 2.40 s for 2.40 s scheduled, and 3.71 s for 3.71 s.
* **Work mode runs exactly the planned work** (400 DGEMMs) and takes ~10% longer than the schedule from sleep overheads. That overhead is the same with or without the governor, so comparisons stay fair.
* **It hasn't run on a GPU.**

## Code

`gpu/runtime/thermal_stress.c`. It declares the few HIP and rocBLAS functions it uses, so no ROCm headers are needed. It links with `-lrocblas -lamdhip64` (see `build.sh`).
