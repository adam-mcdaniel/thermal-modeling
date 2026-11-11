# Background

Earlier work, and the data everything is built on. **The [model](../model/README.md) supersedes these where they overlap.** They're still worth knowing for what they found about the cooling system as a whole.

| Page | What it is | Where |
|---|---|---|
| [Data](data.md) | The traces and telemetry used, and how they're turned into samples | `data/`, `gpu/samples/`, `3Cab_POW.txt` |
| [Cabinet model](cabinet-model.md) | A model of Frontier's whole cooling group: cabinets, CDU, valve, down to the GPU | `frontier_cooling_group.ipynb` (copy in `cabinet/`) |
| [Workload stress study](workload-stress-study.md) | Which workload rhythms hurt the cooling and the chip most | `frontier_workload_stress.ipynb` |
| [Boundary layer](boundary-layer.md) | The first study: how the liquid at the chip's surface reacts to rocHPL's power | `liquid_bl_rochpl.ipynb` |
