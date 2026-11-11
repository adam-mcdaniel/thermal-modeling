"""Find how the coolant responds from a measured inlet, for when no group-wide run with junction readings exists yet, and add it to gpu_fit.h.

fit_gpu finds these same 3 numbers from junction and power alone, given a run with the whole cooling group busy. Frontier's traces with junction readings are all single-node,
so until one exists, this uses one run's GPU power every millisecond and the cabinets' measured inlet. R and C aren't touched.
Only the first 1400 s are used for fitting, and the rest tests the result.
"""
import re
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
TARGET_INLET = 28.64       # the inlet when things are settled: its average on the steady part of the run
GPU_AREA_CM2 = 14.48

recon = pd.read_csv(HERE.parent / 'cabinet_power_recon_gpu_energy_1ms.csv', usecols=['utc', 't_since_job_start_s', 'C1_gpu_energy_W', 'C2_gpu_energy_W', 'C3_gpu_energy_W'])
telemetry = pd.read_csv(HERE.parent / '3Cab_POW.txt', sep=r'\s+', skiprows=11)
start = pd.to_datetime(telemetry.TS.iloc[0])
t_telemetry = (pd.to_datetime(telemetry.TS) - start).dt.total_seconds().to_numpy()
t_offset = (pd.to_datetime(recon.utc.iloc[0], utc=True) - start).total_seconds() - recon.t_since_job_start_s.iloc[0]
t_gpu = recon.t_since_job_start_s.to_numpy() + t_offset

# The heat one average GPU puts into the water, from the GPU model
power = recon[['C1_gpu_energy_W', 'C2_gpu_energy_W', 'C3_gpu_energy_W']].to_numpy().mean(1) / 128 / 4
text = '\n'.join(f'{p:.4f} {t:.6f}' for p, t in zip(power, t_gpu))
result = subprocess.run(['./predict_gpu', '0', '/dev/stdin'], cwd=HERE, input=text, capture_output=True, text=True, check=True)
output = pd.read_csv(io.StringIO(result.stdout))

seconds = np.arange(np.ceil(t_gpu[0]), np.floor(t_gpu[-1]))
heat = np.interp(seconds, output.time_s, output.flux_to_water_W_cm2) * GPU_AREA_CM2
inlet = np.interp(seconds, t_telemetry, np.mean([telemetry[f'LIT_C{c}'].to_numpy(float) for c in (1, 2, 3)], 0))


def coolant_change(gain, rise_s, settle_s):
    rising = settling = gain * heat[0]
    keep_rising, keep_settling = np.exp(-1 / rise_s), np.exp(-1 / settle_s)
    change = np.empty(len(seconds))
    for i, h in enumerate(heat):
        change[i] = rising - settling
        rising = gain * h + (rising - gain * h) * keep_rising
        settling = rising + (settling - rising) * keep_settling
    return change


fit_part = (seconds > 200) & (seconds < 1400)
test_part = seconds >= 1400
errors = lambda n: (TARGET_INLET + coolant_change(*n) - inlet)[fit_part]
fit = least_squares(errors, [0.05, 60, 100], bounds=([0, 5, 5], [1, 2000, 2000]))
gain, rise_s, settle_s = fit.x

rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
predicted = TARGET_INLET + coolant_change(*fit.x)
print(f"gain {gain:.4f} °C per W, rise {rise_s:.1f} s, settle {settle_s:.1f} s")
print(f"inlet error: {rms((predicted - inlet)[fit_part]):.2f} °C on the part it fitted, {rms((predicted - inlet)[test_part]):.2f} °C on the part it never saw "
      f"(a fixed inlet: {rms((TARGET_INLET - inlet)[fit_part]):.2f} and {rms((TARGET_INLET - inlet)[test_part]):.2f} °C)")
header = HERE / 'gpu_fit.h'
water = f'{gain:.6g}, {rise_s:.6g}, {settle_s:.6g}'
header.write_text(re.sub(r'(\{\{.*?\}, \{.*?\}(?:, [^,}]+){4}), [^,}]+, [^,}]+, [^,}]+\}', lambda m: f'{m.group(1)}, {water}}}', header.read_text()))
