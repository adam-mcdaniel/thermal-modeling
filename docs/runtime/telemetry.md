# Telemetry

**Purpose:** read a GPU's power and junction temperature from inside a program, about every millisecond, with no tools or libraries needed at build time.

**In one line:** `gpu_telemetry.h` loads rocm_smi at run time (`dlopen`), finds the GPU by PCI address, and reads the same two counters the Score-P traces used.

## What it reads

| Counter | rocm_smi call | Becomes |
|---|---|---|
| Energy used so far | `rsmi_dev_energy_count_get` (counts × resolution, in µJ) | Power = Δenergy / Δtime between reads |
| Junction temperature | `rsmi_dev_temp_metric_get(sensor 1 = junction, current)`, in m°C | °C |

## Finding the right GPU

HIP and rocm_smi number GPUs differently, especially when `HIP_VISIBLE_DEVICES` is set. So the caller passes the GPU's PCI address (from `hipDeviceGetPCIBusId`), and `tl_open` searches rocm_smi's devices for the same address. You can also pass an index directly.

## MI250X quirk

An MI250X package holds two GCDs, which HIP shows as two devices. They share one energy counter, which only the even-numbered device reports; the odd one reads 0. For an odd GCD, pass its partner's index (`--smi` for the stress workload, `smi=` for the governor). Its "power" is then the whole package's, which is also what the model was calibrated on.

## Using it

```c
#include "gpu_telemetry.h"
Telemetry tl;
tl_open(&tl, "0000:c1:00.0", -1);        // PCI address, or an index >= 0
double energy_J, junction_C;
tl_read(&tl, &energy_J, &junction_C);    // NAN for whichever couldn't be read
```

## Limits

* **Untested on hardware.** It follows the rocm_smi C API, but hasn't been run.
* **The cost of a read isn't known.** If a read takes longer than ~1 ms, the governor's and the stress workload's 1 ms loops will run slower than intended.

## Code

`gpu/runtime/gpu_telemetry.h`. Used by `thermal_rocm.c` and `thermal_stress.c`.
