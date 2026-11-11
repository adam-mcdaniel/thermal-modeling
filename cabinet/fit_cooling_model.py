"""Estimate the cooling-model parameters for a machine from its telemetry, and write the C constants for predict_cooling.c.

    python fit_cooling_model.py telemetry.csv --name lux
    cc -DCOOLING_CONSTANTS='"cooling_lux.h"' -DPREDICT_COOLING_MAIN predict_cooling.c -lm -o predict_cooling_lux

The CSV needs one row per sample (about 1 Hz is ideal) for ONE group of cabinets that share a CDU:

    time_s                         seconds, or a date-time such as 2026-02-24T20:14:13Z
    power_kW_1, power_kW_2, ...    electrical power of each cabinet; the number of these sets the cabinet count
    supply_C_1, supply_C_2, ...    water entering each cabinet (or one shared supply_C column)
    outlet_C_1, outlet_C_2, ...    water leaving each cabinet
    facility_in_C, facility_out_C  building water entering and leaving the CDU
    facility_gpm                   building-water flow through the CDU

The run must contain a big power step, ideally a job starting and stopping across the whole group. Without one, nothing in the
data shows how the system responds, and the script warns you. The first half of the run is used for fitting and the rest is
held out to test the result (change with --train-seconds).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter, lfilter_zi

GPM_WK = 6.309e-5 * 998.0 * 4180.0   # W/K carried by 1 gpm of water


def load(path):
    '''Read the telemetry CSV onto a uniform 1 s grid. Returns a dict of arrays; power in W, each per-cabinet array (N, n).'''
    d = pd.read_csv(path)
    missing = [c for c in ('time_s', 'facility_in_C', 'facility_out_C', 'facility_gpm', 'power_kW_1', 'outlet_C_1') if c not in d]
    if missing:
        sys.exit(f"{path} is missing columns: {missing}. See the top of this script for the format.")
    if pd.api.types.is_numeric_dtype(d.time_s):
        t = d.time_s.to_numpy(float)
    else:
        ts = pd.to_datetime(d.time_s, utc=True); t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    t = t - t[0]
    g = np.arange(0.0, np.floor(t[-1]) + 1)
    col = lambda c: np.interp(g, t, d[c].to_numpy(float))
    N = sum(c.startswith('power_kW_') for c in d.columns)
    sup = [col(f'supply_C_{i + 1}') if f'supply_C_{i + 1}' in d else col('supply_C') for i in range(N)]
    return dict(t=g, N=N, power=np.array([col(f'power_kW_{i + 1}') for i in range(N)]) * 1e3, supply=np.array(sup),
                outlet=np.array([col(f'outlet_C_{i + 1}') for i in range(N)]), pit=col('facility_in_C'), pot=col('facility_out_C'), plf=col('facility_gpm'))


rms = lambda r: float(np.sqrt(np.mean(np.square(r))))
delay = lambda x, d: np.interp(np.arange(len(x)) - d, np.arange(len(x)), x)


def lag1(x, tau):
    a = np.exp(-1 / tau)
    return lfilter([0, 1 - a], [1, -a], x, zi=lfilter_zi([0, 1 - a], [1, -a]) * x[0])[0]


def hx_eff(UA, Cs, Cp):
    Cmin, Cmax = np.minimum(Cs, Cp), np.maximum(Cs, Cp)
    cr = Cmin / Cmax; ex = np.exp(-UA / Cmin * (1 - cr))
    return (1 - ex) / (1 - cr * ex) * Cmin


def cabinet_outlet(ac, tin, Ch, UA, mcp, d):
    tin = delay(tin, d); eps = 1 - np.exp(-UA / mcp); a = np.exp(-eps * mcp / Ch)
    th = np.empty_like(tin); th[0] = tin[0] + ac[0] / (eps * mcp)
    for k in range(1, len(tin)):
        ss = tin[k - 1] + ac[k - 1] / (eps * mcp); th[k] = ss + (th[k - 1] - ss) * a
    return tin + eps * (th - tin)


def fit_cabinet(ac, tin, out, train):
    '''Stage 1. Inputs: measured power and measured inlet. Fits heat capacity, conductance, stream capacity, delay, sensor offset.'''
    guess = ac.mean() / max((out - tin).mean(), 0.5)                                                # stream capacity from power / temperature rise
    best = None
    for tau_guess in (40.0, 120.0):                                                                 # two starts, keep the better fit
        q0 = [tau_guess * 0.5, 1.0, 1.0, 3.0, 0.0]                                                  # all relative to `guess`, so any machine size works
        lo, hi = [1.0, 0.03, 0.2, 0.0, -3.0], [2000.0, 20.0, 5.0, 60.0, 3.0]
        def res(q, m=train):
            return (cabinet_outlet(ac, tin, q[0] * guess, q[1] * guess, q[2] * guess, q[3]) + q[4] - out)[m]
        s = least_squares(res, q0, bounds=(lo, hi))
        if best is None or s.cost < best[0].cost:
            best = (s, res)
    s, res = best
    q = s.x; p = dict(Ch=q[0] * guess, UA=q[1] * guess, mcp=q[2] * guess, d=q[3], b=q[4])
    p['tau'] = p['Ch'] / ((1 - np.exp(-p['UA'] / p['mcp'])) * p['mcp'])
    p['limit_hit'] = [n for n, v, a, b in zip(('Ch', 'UA', 'mcp'), q, lo, hi) if v <= a * 1.01 or v >= b * 0.99]
    return p, rms(s.fun), rms(res(q, ~train))


def fit_cdu(tret, pit, plf, lit, pot, Cs, train):
    '''Stage 2. Inputs: modelled return water, measured building-water temperature and flow. Fits the heat exchanger.'''
    def model(q):
        Cp = plf * GPM_WK; tr = delay(tret, q[1])
        Q = hx_eff(q[0] * Cs, Cs, Cp) * (tr - pit)
        return lag1(tr - Q / Cs, q[2]), lag1(pit + q[4] * Q / Cp, q[3]) + q[5]
    def res(q, m=train):
        a, b = model(q); return np.r_[(a - lit)[m], (b - pot)[m]]
    s = least_squares(res, [3.5, 10, 10, 10, 0.9, 0], bounds=([0.1, 0, 0.5, 0.5, 0.6, -5], [60, 90, 300, 300, 1.4, 5]))
    q = s.x; a, b = res(q, train).reshape(2, -1), res(q, ~train).reshape(2, -1)
    return dict(UA_hx=q[0] * Cs, d_s=q[1], tau_s=q[2], tau_p=q[3], kappa=q[4], b_p=q[5]), (rms(a[0]), rms(b[0])), (rms(a[1]), rms(b[1]))


def fit_valve(lit, plf, train):
    '''Stage 3. Input: measured supply temperature. Fits the valve controller; flow limits are the lowest and highest flow seen.'''
    rng = (plf.min(), plf.max())
    def model(q):
        e = delay(lit, q[3]) - q[0]; u = np.empty_like(e); u[0] = plf[0]
        for k in range(1, len(e)):
            u[k] = np.clip(u[k - 1] + q[1] * (e[k] - e[k - 1]) + q[2] * e[k], *rng)
        return u
    def res(q, m=train):
        return (model(q) - plf)[m]
    best = None
    for d0 in (10.0, 60.0):
        s = least_squares(res, [np.median(lit), 2, 0.1, d0], bounds=([lit.min() - 1, 0, 0, 0], [lit.max() + 1, 50, 5, 120]))
        if best is None or s.cost < best.cost:
            best = s
    q = best.x
    return dict(T_set=q[0], Kp=q[1], Ki=q[2], d_c=q[3], flow_min=rng[0], flow_max=rng[1]), rms(best.fun), rms(res(q, ~train))


def closed_loop(power, pit, cab, cdu, ctl, lit0, plf0, tpo0, th0):
    '''The whole group running on its own: inputs are only power and building-water temperature.'''
    N, n = power.shape
    mcp = np.array([p['mcp'] for p in cab]); Ch = np.array([p['Ch'] for p in cab]); eps = 1 - np.exp(-np.array([p['UA'] for p in cab]) / mcp)
    a_h = np.exp(-eps * mcp / Ch); a_s = np.exp(-1 / cdu['tau_s']); a_p = np.exp(-1 / cdu['tau_p']); Cs = mcp.sum()
    dcab = [int(round(p['d'])) for p in cab]; ds, dc = int(round(cdu['d_s'])), int(round(ctl['d_c'])); H = max(*dcab, ds, dc) + 1
    th = th0.copy(); lit = [lit0] * H; ret = [(mcp * (lit0 + eps * (th - lit0))).sum() / Cs] * H; plf, tpo, ts = plf0, tpo0, lit0
    e_prev = lit0 - ctl['T_set']; out = dict(LIT=np.empty(n), LOT=np.empty((N, n)), POT=np.empty(n), PLF=np.empty(n))
    for k in range(n):
        tin = np.array([lit[-1 - d] for d in dcab]); tout = tin + eps * (th - tin)
        out['LIT'][k], out['LOT'][:, k], out['PLF'][k], out['POT'][k] = lit[-1], tout, plf, tpo + cdu['b_p']
        ss = tin + power[:, k] / (eps * mcp); th = ss + (th - ss) * a_h
        ret.append((mcp * tout).sum() / Cs); ret.pop(0); tr = ret[-1 - ds]; Cp = plf * GPM_WK
        Q = hx_eff(cdu['UA_hx'], Cs, Cp) * (tr - pit[k])
        ts = tr - Q / Cs + (ts - (tr - Q / Cs)) * a_s
        tpo = pit[k] + cdu['kappa'] * Q / Cp + (tpo - (pit[k] + cdu['kappa'] * Q / Cp)) * a_p
        lit.append(ts); lit.pop(0)
        e = lit[-1 - dc] - ctl['T_set']
        plf = float(np.clip(plf + ctl['Kp'] * (e - e_prev) + ctl['Ki'] * e, ctl['flow_min'], ctl['flow_max'])); e_prev = e
    return out


def checks(D, train_s, cab, cdu, ctl, heat_fraction, worst_error):
    '''Things that make the fitted numbers untrustworthy, in plain words.'''
    w = []; total = D['power'].sum(0); span = np.ptp(uniform(total)) / max(np.percentile(total, 95), 1)
    if span < 0.3:
        w.append(f"Power only moved {span * 100:.0f} % of its peak. Use a run with a big job start or stop, or the time-response numbers are guesses.")
    if D['t'][-1] < 1200:
        w.append(f"The run is only {D['t'][-1] / 60:.0f} minutes. Aim for at least 30, so the system can settle before and after a step.")
    if D['t'][-1] <= train_s:
        w.append("Nothing is left to test on, so the test numbers mean nothing.")
    if np.ptp(D['plf']) < 0.1 * np.median(D['plf']):
        w.append("Building-water flow barely moved, so the valve controller can't be estimated from this run.")
    for i, p in enumerate(cab):
        if p['limit_hit']:
            w.append(f"Cabinet {i + 1}: {', '.join(p['limit_hit'])} hit the edge of its allowed range, so don't trust that cabinet's fit.")
    if cdu['kappa'] < 0.65 or cdu['kappa'] > 1.35:
        w.append(f"kappa = {cdu['kappa']:.2f} is far from 1: check the flow units (gpm expected) and building-water sensors.")
    q_p = np.mean(D['plf'] * GPM_WK * (D['pot'] - D['pit'])); ratio = q_p / np.mean(D['power'].sum(0))
    if abs(ratio - heat_fraction) > 0.07:
        w.append(f"Heat in the building water is {ratio:.2f} x the electrical power, but the model assumes {heat_fraction:.2f}. If some heat leaves by "
                 f"air, try --heat-fraction {ratio:.2f}. Otherwise check the flow units (gpm expected) and sensors.")
    if worst_error > 1.0:
        w.append(f"Running on its own, the model is off by up to {worst_error:.1f} °C (RMS), so it doesn't describe this machine well yet. Fix the warnings above first.")
    w.append(f"Flow limits ({ctl['flow_min']:.0f}-{ctl['flow_max']:.0f} gpm) are only the lowest and highest flow in this run, not known hardware limits.")
    return w, ratio


def uniform(x, n=30):
    return lfilter(np.ones(n) / n, [1], x)[n:]


def c_block(name, cab, cdu, ctl, heat_fraction):
    '''The constants, in the form predict_cooling.c includes.'''
    f = lambda v: f"{v:.6g}"; arr = lambda key: ', '.join(f(p[key]) for p in cab)
    H = max(max(int(round(p['d'])) for p in cab), int(round(cdu['d_s'])), int(round(ctl['d_c']))) + 1
    return f"""/* Cooling-model constants for {name}, written by fit_cooling_model.py. See predict_cooling.c for what each one means. */
#define N_CAB {len(cab)}
#define HISTORY {H}
static const double Ch[N_CAB] = {{{arr('Ch')}}};
static const double UA[N_CAB] = {{{arr('UA')}}};
static const double mcp[N_CAB] = {{{arr('mcp')}}};
static const int inlet_delay[N_CAB] = {{{', '.join(str(int(round(p['d']))) for p in cab)}}};
static const double offset[N_CAB] = {{{arr('b')}}};
static const double UA_hx = {f(cdu['UA_hx'])}, tau_s = {f(cdu['tau_s'])}, tau_p = {f(cdu['tau_p'])}, kappa = {f(cdu['kappa'])}, b_p = {f(cdu['b_p'])};
static const int hx_delay = {int(round(cdu['d_s']))}, sensor_delay = {int(round(ctl['d_c']))};
static const double T_set = {f(ctl['T_set'])}, Kp = {f(ctl['Kp'])}, Ki = {f(ctl['Ki'])}, flow_min = {f(ctl['flow_min'])}, flow_max = {f(ctl['flow_max'])};
static const double heat_to_water = {f(heat_fraction)};
static const double GPM = {f(GPM_WK)};
"""


def fit(path, name, train_seconds=None, heat_fraction=1.0, outdir='.'):
    D = load(path); t = D['t']; N = D['N']
    train_s = train_seconds if train_seconds is not None else t[-1] / 2
    train = t < train_s
    print(f"{name}: {N} cabinets, {t[-1] / 60:.0f} minutes; fitting on the first {train_s:.0f} s, testing on the rest\n")
    power = D['power'] * heat_fraction
    lit = D['supply'].mean(0)

    print("Stage 1: each cabinet (measured power and inlet -> outlet)")
    cab = []
    for i in range(N):
        p, tr, te = fit_cabinet(power[i], D['supply'][i], D['outlet'][i], train); cab.append(p)
        print(f"  cabinet {i + 1}: Ch {p['Ch'] / 1e6:.2f} MJ/K, UA {p['UA'] / 1e3:.1f} kW/K, mcp {p['mcp'] / 1e3:.1f} kW/K, delay {p['d']:.1f} s, "
              f"offset {p['b']:+.2f} °C, time constant {p['tau']:.0f} s | error {tr:.2f} / {te:.2f} °C (fit / test)")

    print("\nStage 2: heat exchanger (return water and building water -> supply and building-water return)")
    ms = np.array([p['mcp'] for p in cab])
    tout = np.array([cabinet_outlet(power[i], lit, cab[i]['Ch'], cab[i]['UA'], cab[i]['mcp'], cab[i]['d']) for i in range(N)])
    tret = (ms[:, None] * tout).sum(0) / ms.sum()
    cdu, e_l, e_p = fit_cdu(tret, D['pit'], D['plf'], lit, D['pot'], ms.sum(), train)
    print(f"  UA_hx {cdu['UA_hx'] / 1e3:.0f} kW/K, delay {cdu['d_s']:.1f} s, tau_s {cdu['tau_s']:.1f} s, tau_p {cdu['tau_p']:.1f} s, kappa {cdu['kappa']:.3f}, "
          f"b_p {cdu['b_p']:+.2f} °C | supply error {e_l[0]:.2f} / {e_l[1]:.2f}, return error {e_p[0]:.2f} / {e_p[1]:.2f} °C")

    print("\nStage 3: valve controller (supply temperature -> flow)")
    ctl, tr, te = fit_valve(lit, D['plf'], train)
    print(f"  target {ctl['T_set']:.2f} °C, Kp {ctl['Kp']:.2f}, Ki {ctl['Ki']:.3f}, delay {ctl['d_c']:.0f} s, flow {ctl['flow_min']:.0f}-{ctl['flow_max']:.0f} gpm | error {tr:.1f} / {te:.1f} gpm")

    eps = np.array([1 - np.exp(-p['UA'] / p['mcp']) for p in cab])
    th0 = lit[0] + (D['outlet'][:, 0] - np.array([p['b'] for p in cab]) - lit[0]) / eps
    cl = closed_loop(power, D['pit'], cab, cdu, ctl, lit[0], D['plf'][0], D['pot'][0] - cdu['b_p'], th0)
    print("\nFinal test: the whole model on its own, from power and building-water temperature only (fit / test)")
    res = {'supply': (cl['LIT'] - lit, '°C'), 'building-water return': (cl['POT'] - D['pot'], '°C'), 'valve flow': (cl['PLF'] - D['plf'], 'gpm')}
    for i in range(N):
        res[f'outlet {i + 1}'] = (cl['LOT'][i] + cab[i]['b'] - D['outlet'][i], '°C')
    for k, (r, u) in res.items():
        print(f"  {k:22s} {rms(r[train]):5.2f} / {rms(r[~train]) if (~train).any() else float('nan'):5.2f} {u}")

    worst = max(rms(r) for k, (r, u) in res.items() if u == '°C')
    warn, ratio = checks(D, train_s, cab, cdu, ctl, heat_fraction, worst)
    print(f"\nHeat in the building water / electrical power: {ratio:.2f}")
    print("\nCheck these:"); [print(f"  - {x}") for x in warn]

    out = Path(outdir); h = out / f'cooling_{name}.h'; j = out / f'cooling_{name}.json'
    h.write_text(c_block(name, cab, cdu, ctl, heat_fraction))
    j.write_text(json.dumps(dict(cabinets=cab, cdu=cdu, controller=ctl, heat_to_water=heat_fraction), indent=2, default=float))
    print(f"\nWrote {h} and {j}\nBuild with:  cc -DCOOLING_CONSTANTS='\"{h.name}\"' -DPREDICT_COOLING_MAIN predict_cooling.c -lm -o predict_cooling_{name}")
    return cab, cdu, ctl


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('csv'); a.add_argument('--name', default='machine')
    a.add_argument('--train-seconds', type=float, help='fit on this many seconds from the start (default: half the run)')
    a.add_argument('--heat-fraction', type=float, default=1.0, help='fraction of electrical power that ends up in the water (default 1)')
    a.add_argument('--outdir', default='.')
    x = a.parse_args()
    fit(x.csv, x.name, x.train_seconds, x.heat_fraction, x.outdir)
