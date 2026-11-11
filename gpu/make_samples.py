"""Write one samples file per GPU from each Frontier Score-P trace: power_W timestamp_s junction_C, using Ampere's power from the energy counter.

samples/gpu*.txt is the rocHPL trace that the fits in gpu_fit.h come from. samples/<trace>/gpu*.txt are the other jobs, which test the model.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from ampere import Ensemble, MetricConfig, MetricType, set_backend

set_backend('pandas')
HERE = Path(__file__).resolve().parent
EXAMPLES = Path(os.environ.get('AMPERE_EXAMPLES', Path.home() / 'Documents' / 'ampere' / 'examples'))
TRACES = {HERE / 'samples': HERE.parent / 'data' / 'rochpl-frontier'}
for name in ('hpl-captured-params', 'hpl-captured-params-HIP', 'hpgmxp', 'hpg-mxp'):
    TRACES[HERE / 'samples' / name] = EXAMPLES / name

configs = {re.compile(r".*rocm_smi.*energy.*"): MetricConfig(MetricType.CUMULATIVE, scale_factor=1e-6),   # uJ -> J
           re.compile(r".*rocm_smi.*temp.*"): MetricConfig(MetricType.INSTANTANEOUS, scale_factor=1e-3)}   # m°C -> °C

for folder, trace in TRACES.items():
    folder.mkdir(exist_ok=True)
    node = Ensemble.from_trace_paths([str(trace)], {"Node0": ["MPI Rank 0"]}, configs).runs[0].nodes[0]
    for gpu, device in enumerate((0, 2, 4, 6)):
        energy = node.metrics[f"A2rocm_smi:::energy_count:device={device}"]
        times = np.asarray(energy.times.to_ndarray(), float)
        power = np.asarray(energy.get_statistics_vectorized(times[:-1], times[1:])['rate'], float)   # power[i] holds from times[i] to times[i + 1]

        sensor = node.metrics[f"A2rocm_smi:::temp_current:device={device}:sensor=1"]
        reading_times = np.asarray(sensor.times.to_ndarray(), float)
        readings = np.asarray(sensor.values.to_ndarray(), float)
        inside = (reading_times >= times[0]) & (reading_times < times[-1])
        reading_times, readings = reading_times[inside], readings[inside]

        # Power samples have no reading (nan). Sensor readings carry the power holding at that moment.
        rows = pd.DataFrame({
            'power_W': np.r_[power, power[np.searchsorted(times, reading_times, side='right') - 1]],
            'timestamp_s': np.r_[times[:-1], reading_times] - times[0],
            'junction_C': np.r_[np.full(len(power), np.nan), readings],
        }).sort_values('timestamp_s', kind='stable')
        np.savetxt(folder / f'gpu{gpu}.txt', rows.values, fmt='%.4f %.6f %.0f')
        print(f"{trace.name} GPU {gpu}: {len(power)} power samples, {len(readings)} junction readings")
