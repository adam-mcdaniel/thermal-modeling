# Open questions

**Purpose:** what's still unproven, in one place, with the run that would settle each question.

## Unproven, by importance

| # | Question | Current evidence | What would settle it |
|---|---|---|---|
| 1 | Does the governor work on a real GPU? | Simulation only. GPU code compiles but has never run | The [first runs](development.md#first-runs-on-hardware-in-order) |
| 2 | Does calibrating once predict *other* jobs on the *same* GPU? | 0.95 °C on a *different* GPU in the same slot. The same GPU should do better | Two different jobs on one node, with the junction sensor on |
| 3 | Can the coolant response be learned from real junction readings? | Yes in a test built on the real inlet, but the layer response was synthetic | A job filling a cooling group for 10+ minutes, with junction logging on several nodes |
| 4 | Is a power cap cheaper than gaps, and by how much? | The simulation assumes power goes as speed² | A fixed job at 2–3 power caps |
| 5 | Does the 4-minute calibration recipe work on real hardware? | Tested on a simulated GPU only | `thermal_stress recipe`, then check `fit_gpu`'s held-back error |
| 6 | Does the model carry over to another GPU type? | Only MI250X data. The calibrator is shown to be power-scale independent | A calibration run on Lux |
| 7 | Why did low-power HPG-MxP get ~25% less rise per watt? | Possibly memory power not heating the junction | Log a memory-bound kernel and a compute-bound kernel at the same power |
| 8 | What wear exponent q applies to this chip? | Unknown, so results are shown for 2, 4 and 6 | Not measurable here; it needs reliability data |

## Ideas not built yet

* **A power-cap actuator in the governor:** the [control study](control/sleeps-and-caps.md) suggests it's cheaper than gaps, updated every 1–2 s.
* **Splitting large calls:** a 100 ms DGEMM is one burst long enough to cycle the die. The hook could split it into column blocks with gaps between.
* **Coordination between processes sharing a GPU,** and between GPUs sharing a cooling group.
* **Keeping the idle floor up between jobs, not just within one:** the biggest swing in rocHPL is the job starting and stopping, which the default setting lets through.
