"""Run test_core on every workload, with and without the governor, and score the junction: how much it varies, how much fatigue
that causes, and what it costs in runtime and energy. Draws governor_test.png. Needs no GPU.

    python test_governor.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from thermal_metrics import fatigue, time_std

PATTERNS = {'square': '5 s on, 5 s off', 'hammer': '50 ms on, 50 ms off', 'random': 'random on and off, 20 ms to 10 s', 'hpl': 'rocHPL power trace'}
RUNS = {'off': [], 'on': [], 'gaps only': ['fill=0'], 'gentle': ['rise=5'], 'strict': ['rise=0.2']}
OFF, ON, STRICT = '#8a8a86', '#2a78d6', '#eb6834'

subprocess.run(['cc', '-O2', '-Wno-unused-function', 'test_core.c', 'thermal_core.c', '-lm', '-o', 'test_core'], cwd=HERE, check=True)
folder = Path(tempfile.mkdtemp())


def run(pattern, label):
    path = folder / f'{pattern}-{label.replace(" ", "_")}.csv'
    mode = 'off' if label == 'off' else 'on'
    subprocess.run(['./test_core', pattern, mode, str(path)] + RUNS[label], cwd=HERE, check=True, capture_output=True)
    return pd.read_csv(path)


def score(run_data):
    t, j, p = run_data.time_s.to_numpy(), run_data.junction_C.to_numpy(), run_data.power_W.to_numpy()
    busy = t > 5
    return dict(took=t[-1], energy=p.sum() * 1e-3, std=time_std(t[busy], j[busy]), peak=j.max(), q2=fatigue(j, 2, 1.0), q4=fatigue(j, 4), q6=fatigue(j, 6))


traces, rows = {}, []
for pattern, description in PATTERNS.items():
    base = None
    for label in RUNS:
        data = run(pattern, label)
        traces[pattern, label] = data
        s = score(data)
        base = base or s
        rows.append({'workload': description, 'governor': label, 'runtime': f"{100 * (s['took'] / base['took'] - 1):+.0f}%",
                     'energy': f"{100 * (s['energy'] / base['energy'] - 1):+.0f}%", 'spread (std)': f"{s['std']:.1f} °C", 'peak': f"{s['peak']:.1f} °C",
                     'fatigue q=2 (swings ≥ 1 °C)': f"x{s['q2'] / base['q2']:.2f}", 'q=4': f"x{s['q4'] / base['q4']:.2f}", 'q=6': f"x{s['q6'] / base['q6']:.2f}"})
table = pd.DataFrame(rows)
print(table.to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex='col', gridspec_kw={'height_ratios': [1.6, 1]})
for column, pattern in enumerate(['square', 'hpl']):
    for label, color, width, name in (('off', OFF, 1.4, 'No governor'), ('on', ON, 1.1, 'Balanced (the default, rise=1)'), ('strict', STRICT, 1.1, 'Strict (rise=0.2)')):
        data = traces[pattern, label]
        tenth = (data.time_s // 0.1).astype(int)
        mean = data.groupby(tenth).mean()
        axes[0, column].plot(mean.time_s, mean.junction_C, color=color, linewidth=width, label=name)
        axes[1, column].plot(mean.time_s, mean.power_W, color=color, linewidth=width)
    axes[0, column].set_title(PATTERNS[pattern], loc='left', fontsize=10)
    axes[1, column].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Junction (°C), 0.1 s averages'); axes[1, 0].set_ylabel('Power (W), 0.1 s averages')
axes[0, 0].legend(frameon=False, fontsize=8.5, loc='lower right')
fig.suptitle('Simulated GPU: the same work with and without the governor (the governor has deliberately wrong numbers)', fontsize=11)
plt.tight_layout()
plt.savefig(HERE / 'governor_test.png', dpi=130, bbox_inches='tight')
print(f"wrote {HERE / 'governor_test.png'}")
