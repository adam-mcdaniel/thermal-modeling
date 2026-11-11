"""Turn the rocHPL Score-P trace and the cabinet telemetry into the CSV files fit_cooling_model reads. Writes them to fit_data/."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from ampere import Ensemble, MetricConfig, MetricType, set_backend

set_backend('pandas')
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'fit_data'; OUT.mkdir(exist_ok=True)

# --- One CSV per GPU: its power (Ampere's rate from the energy counter), and its junction sensor ---
configs = {re.compile(r".*rocm_smi.*energy.*"): MetricConfig(MetricType.CUMULATIVE, scale_factor=1e-6),    # uJ -> J
           re.compile(r".*rocm_smi.*temp.*"): MetricConfig(MetricType.INSTANTANEOUS, scale_factor=1e-3)}    # m°C -> °C
node = Ensemble.from_trace_paths([str(ROOT / 'data' / 'rochpl-frontier')], {"Node0": ["MPI Rank 0"]}, configs).runs[0].nodes[0]
print("By-hand check of each GPU's total resistance: (junction under load - at idle) / (power under load - at idle)")
for k, device in enumerate((0, 2, 4, 6)):
    energy = node.metrics[f"A2rocm_smi:::energy_count:device={device}"]
    t = np.asarray(energy.times.to_ndarray(), float)
    power = np.asarray(energy.get_statistics_vectorized(t[:-1], t[1:])['rate'], float)      # power[i] holds from t[i] to t[i + 1]
    sensor = node.metrics[f"A2rocm_smi:::temp_current:device={device}:sensor=1"]
    t_j = np.asarray(sensor.times.to_ndarray(), float); junction = np.asarray(sensor.values.to_ndarray(), float)
    keep = (t_j >= t[0]) & (t_j < t[-1]); t_j, junction = t_j[keep], junction[keep]

    # Rows for power samples have no reading; rows for junction readings carry the power holding at that moment
    rows = pd.DataFrame({'time_s': np.r_[t[:-1], t_j] - t[0], 'power_W': np.r_[power, power[np.searchsorted(t, t_j, side='right') - 1]],
                         'junction_C': np.r_[np.full(len(power), np.nan), junction]}).sort_values('time_s', kind='stable')
    rows.to_csv(OUT / f'gpu{k}.csv', index=False, na_rep='')

    idle = t[:-1] - t[0] < 14; busy = (t[:-1] - t[0] > 60) & (t[:-1] - t[0] < 75)
    j_idle, j_busy = junction[t_j - t[0] < 14].mean(), junction[(t_j - t[0] > 60) & (t_j - t[0] < 75)].mean()
    print(f"  GPU {k}: ({j_busy:.1f} - {j_idle:.1f} °C) / ({power[busy].mean():.0f} - {power[idle].mean():.0f} W) = {(j_busy - j_idle) / (power[busy].mean() - power[idle].mean()):.4f} K/W")

# --- The cooling group: 3 cabinets and their CDU ---
raw = pd.read_csv(ROOT / '3Cab_POW.txt', sep=r'\s+', skiprows=11)
group = pd.DataFrame({'time_s': raw.TS})
for i in (1, 2, 3):
    group[f'power_kW_{i}'] = raw[f'ACP_C{i}'] / 1e3; group[f'supply_C_{i}'] = raw[f'LIT_C{i}']; group[f'outlet_C_{i}'] = raw[f'LOT_C{i}']
group['facility_in_C'], group['facility_out_C'], group['facility_gpm'] = raw.CDU_PIT, raw.CDU_POT, raw.CDU_PLF
group.to_csv(OUT / 'frontier_group.csv', index=False)
