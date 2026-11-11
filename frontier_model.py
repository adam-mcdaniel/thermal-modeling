"""The fitted Frontier cooling-group and GPU models from frontier_cooling_group.ipynb, as importable functions.

The fits themselves live in that notebook, which writes results/cooling_group_params.json and
results/gpu_junction_params.csv. This module only loads those numbers and runs the models.
"""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parent
GPM_WK = 6.309e-5 * 998.0 * 4180.0      # W/K carried by 1 gpm of water (~263)
GPM_TO_LPM = 3.78541
N_NODES = 128                           # nodes per cabinet
A_DIE_CM2 = 2 * 7.24                    # two 724 mm^2 GCDs per MI250X package
DT_CG, DT_GPU = 1.0, 1e-3               # cooling-group and GPU time steps [s]


def load_model():
    '''Fitted parameters: dict with cab, cdu, ctl, ac_poly, ac_range, node_poly and jnet (one entry per GPU package).'''
    p = json.loads((ROOT / 'results' / 'cooling_group_params.json').read_text())
    ctl = dict(p['controller']); ctl['plf_range'] = tuple(ctl['plf_range'])
    j = pd.read_csv(ROOT / 'results' / 'gpu_junction_params.csv')
    jnet = [dict(R=np.array([r.R1_K_per_W, r.R2_K_per_W, r.R3_K_per_W]), C=np.array([r.C1_J_per_K, r.C2_J_per_K, r.C3_J_per_K]),
                 Tc=r.Tc_fit_C, dTc=r.Tc_fit_C - ctl['T_set'], rms=r.rms_K) for r in j.itertuples()]
    return dict(cab=p['cabinets'], cdu=p['cdu'], ctl=ctl, ac_poly=np.array(p['ac_from_node_sum_poly']),
                ac_range=tuple(p['ac_from_node_sum_range_W']), node_poly=np.array(p['node_from_packages_poly']), jnet=jnet)


def cabinet_ac_from_nodes(M, node_sum_W):
    '''Cabinet AC [W] from the cabinet's node-power sum [W]: the fitted curve, straight-line (RAPS slope) outside the data.'''
    x = np.asarray(node_sum_W, float)
    xc = np.clip(x, *M['ac_range'])
    return np.polyval(M['ac_poly'], xc) + (x - xc) / (0.98 * 0.96)


def package_from_node(M, node_W):
    '''Average GPU package power [W] for a per-node power [W] (4 packages per node).'''
    return np.maximum((np.asarray(node_W, float) - M['node_poly'][1]) / M['node_poly'][0] / 4, 0)


def hx_eff(UA, Cs, Cp):
    '''Counterflow heat-exchanger effectiveness times C_min [W/K] (epsilon-NTU).'''
    Cmin, Cmax = min(Cs, Cp), max(Cs, Cp)
    cr, ntu = Cmin / Cmax, UA / Cmin
    if abs(1 - cr) < 1e-6:
        return ntu / (1 + ntu) * Cmin
    ex = np.exp(-ntu * (1 - cr))
    return (1 - ex) / (1 - cr * ex) * Cmin


def simulate_cooling_group(M, ac, pit, state=None):
    '''Closed-loop cooling group, one step per second. ac: (3, n) cabinet AC [W]; pit: facility supply [°C], scalar or (n,).

    Optional what-if settings in M['ctl']: Kff (feed-forward, gpm per W of change in total AC power), ff_tau (delay the power
    signal by this lag [s], so the valve moves when the heat arrives) and rate (largest valve movement, gpm/s). Returns dict of arrays LIT, LOT (3, n), POT, PLF, QHX and 'state' to continue from.
    '''
    cab, cdu, ctl = M['cab'], M['cdu'], M['ctl']
    ac = np.atleast_2d(ac); n = ac.shape[1]; pit = np.broadcast_to(np.asarray(pit, float), (n,))
    mcp = np.array([p['mcp'] for p in cab]); Ch = np.array([p['Ch'] for p in cab])
    eps = 1 - np.exp(-np.array([p['UA'] for p in cab]) / mcp); Cs = mcp.sum()
    a_h = np.exp(-eps * mcp / Ch); a_s = np.exp(-1 / cdu['tau_s']); a_p = np.exp(-1 / cdu['tau_p'])
    dcab = [int(round(p['d'])) for p in cab]; ds = int(round(cdu['d_s'])); dc = int(round(ctl['d_c']))
    H = max(*dcab, ds, dc) + 1
    Kff, rate, ff_tau = ctl.get('Kff', 0.0), ctl.get('rate', np.inf), ctl.get('ff_tau', 0.0); lo, hi = ctl['plf_range']
    a_f = np.exp(-1 / ff_tau) if ff_tau > 0 else 0.0
    if state is None:
        s0 = ctl['T_set']
        state = dict(th=s0 + ac[:, 0] / (eps * mcp), lit_hist=np.full(H, s0), tret_hist=np.full(H, s0 + ac[:, 0].sum() / Cs),
                     plf=float(np.clip(ac[:, 0].sum() / (GPM_WK * max(s0 - pit[0], 1.0)), lo, hi)),
                     tpo=pit[0] + cdu['kappa'] * ac[:, 0].sum() / (GPM_WK * 150), e_prev=0.0, ac_prev=ac[:, 0].sum())
    th = state['th'].copy(); lit_h = list(state['lit_hist']); tret_h = list(state['tret_hist'])
    plf, tpo, e_prev, tsup = state['plf'], state['tpo'], state['e_prev'], state['lit_hist'][-1]
    ac_prev, cmd = state.get('ac_prev', ac[:, 0].sum()), state.get('cmd', plf)
    out = {k: np.empty(n) for k in ('LIT', 'POT', 'PLF', 'QHX')}; out['LOT'] = np.empty((3, n))
    for k in range(n):
        tin = np.array([lit_h[-1 - d] for d in dcab])
        tout = tin + eps * (th - tin)
        out['LIT'][k], out['LOT'][:, k], out['PLF'][k] = lit_h[-1], tout, plf
        ss = tin + ac[:, k] / (eps * mcp); th = ss + (th - ss) * a_h
        tret_h.append((mcp * tout).sum() / Cs); tret_h.pop(0)
        tr = tret_h[-1 - ds]; Cp = plf * GPM_WK
        Q = hx_eff(cdu['UA_hx'], Cs, Cp) * (tr - pit[k]); out['QHX'][k] = Q
        tss, pss = tr - Q / Cs, pit[k] + cdu['kappa'] * Q / Cp
        out['POT'][k] = tpo + cdu['b_p']
        tsup = tss + (tsup - tss) * a_s; tpo = pss + (tpo - pss) * a_p
        lit_h.append(tsup); lit_h.pop(0)
        e = lit_h[-1 - dc] - ctl['T_set']; total = ac[:, k].sum() + (ac_prev - ac[:, k].sum()) * a_f
        cmd = float(np.clip(cmd + ctl['Kp'] * (e - e_prev) + ctl['Ki'] * e + Kff * (total - ac_prev), lo, hi))
        plf = plf + float(np.clip(cmd - plf, -rate, rate)); e_prev, ac_prev = e, total
    out['state'] = dict(th=th, lit_hist=np.array(lit_h), tret_hist=np.array(tret_h), plf=plf, tpo=tpo, e_prev=e_prev, ac_prev=ac_prev, cmd=cmd)
    return out


def run_cooling(M, ac, pit, settle_s=6000):
    '''simulate_cooling_group, starting from the settled state for the first power sample.'''
    ac = np.atleast_2d(ac); pit0 = np.broadcast_to(np.asarray(pit, float), (ac.shape[1],))[0]
    st = simulate_cooling_group(M, np.repeat(ac[:, :1], settle_s, 1), pit0)['state']
    return simulate_cooling_group(M, ac, pit, state=st)


def cauer_matrices(R, C):
    '''State-space form x' = A x + B [P, T_c] of the 3-layer thermal ladder.'''
    n = len(R); A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -1 / (R[i] * C[i]) - (1 / (R[i - 1] * C[i]) if i else 0)
        if i: A[i, i - 1] = 1 / (R[i - 1] * C[i])
        if i < n - 1: A[i, i + 1] = 1 / (R[i] * C[i])
    B = np.zeros((n, 2)); B[0, 0] = 1 / C[0]; B[-1, 1] = 1 / (R[-1] * C[-1])
    return A, B


def junction_response(P, Tc, R, C, dt=DT_GPU, area_cm2=A_DIE_CM2):
    '''Layer temperatures T (3, N; T[0] = junction) and heat fluxes q_gen, q_die, q_wall [W/cm^2] for power P [W], coolant Tc [°C].'''
    P = np.asarray(P, float); Tc = np.broadcast_to(np.asarray(Tc, float), P.shape)
    A, B = cauer_matrices(R, C)
    lam, V = np.linalg.eig(A); lam, V = lam.real, V.real; W = np.linalg.inv(V)
    u = np.stack([P, Tc])
    u0 = u[:, :max(1, int(round(1.0 / dt)))].mean(1, keepdims=True)
    x0 = -np.linalg.solve(A, B @ u0[:, 0])
    drive = W @ B @ (u - u0)
    z = np.empty_like(drive)
    for i, l in enumerate(lam):
        a = np.exp(l * dt)
        z[i] = lfilter([0, (a - 1) / l], [1, -a], drive[i])
    T = x0[:, None] + V @ z
    return dict(T=T, q_gen=P / area_cm2, q_die=(T[0] - T[1]) / R[0] / area_cm2, q_wall=(T[-1] - Tc) / R[-1] / area_cm2)


def rainflow(x, dt, min_range=0.25):
    '''Rainflow cycle count (ASTM E1049) of a temperature history x sampled every dt seconds.

    Each loop of "heat up, cool back down" becomes one cycle, even when small loops sit inside big ones.
    Wiggles smaller than min_range are ignored. Returns arrays: range [K], count (1 or 0.5), peak [°C], period [s].
    '''
    x = np.asarray(x, float)
    d = np.diff(x); keep = np.flatnonzero(d != 0)
    if len(keep) == 0:
        return (np.array([]),) * 4
    turn = keep[np.flatnonzero(np.sign(d[keep][1:]) != np.sign(d[keep][:-1]))] + 1
    idx = np.r_[0, turn, len(x) - 1]
    # Hysteresis: drop reversals smaller than min_range
    pts = [idx[0]]; direction = 0
    for i in idx[1:]:
        delta = x[i] - x[pts[-1]]
        if direction == 0:
            if abs(delta) >= min_range:
                direction = np.sign(delta); pts.append(i)
        elif np.sign(delta) == direction:
            pts[-1] = i
        elif abs(delta) >= min_range:
            direction = -direction; pts.append(i)
    rng, cnt, pk, per, st = [], [], [], [], []
    for i in pts:
        st.append(i)
        while len(st) >= 3:
            a, b, c = st[-3], st[-2], st[-1]
            X, Y = abs(x[c] - x[b]), abs(x[b] - x[a])
            if X < Y:
                break
            if len(st) == 3:
                rng.append(Y); cnt.append(0.5); pk.append(max(x[a], x[b])); per.append(2 * (b - a) * dt)
                st.pop(0)
            else:
                rng.append(Y); cnt.append(1.0); pk.append(max(x[a], x[b])); per.append(abs(c - a) * dt)
                st[-3:] = [c]
    for a, b in zip(st[:-1], st[1:]):
        rng.append(abs(x[b] - x[a])); cnt.append(0.5); pk.append(max(x[a], x[b])); per.append(2 * (b - a) * dt)
    return np.array(rng), np.array(cnt), np.array(pk), np.array(per)


def cycling_damage_rate(cycles, duration_s, n=2.65, m=0.136, Ea_over_k=2185.0, min_dT=2.0):
    '''Relative fatigue damage per hour from rainflow cycles (Norris-Landzberg form; defaults are the SAC solder fit).

    Each cycle counts as dT^n * f^(-m) * exp(-Ea/(k*T_peak)). Only ratios between workloads mean anything, and the same
    exponents must be used for both. Cycles smaller than min_dT are treated as harmless (an assumption).
    '''
    r, c, pk, per = cycles
    keep = r >= min_dT
    if not keep.any():
        return 0.0
    f = 1 / np.maximum(per[keep], 1e-3)
    return float(np.sum(c[keep] * r[keep] ** n * f ** (-m) * np.exp(-Ea_over_k / (pk[keep] + 273.15))) * 3600 / duration_s)


def wear_rate(T_C, Ea_eV=0.9, T_ref_C=60.0):
    '''Time-average Arrhenius wear rate relative to sitting at T_ref_C (e.g. electromigration; Ea 0.9 eV is a typical Cu value).'''
    k = 8.617e-5
    return float(np.mean(np.exp(-Ea_eV / k * (1 / (np.asarray(T_C) + 273.15) - 1 / (T_ref_C + 273.15)))))


def single_gpu_flow_ladder(R, flow_lpm, flow_ref_lpm, n=0.33, serial_gpus=4):
    '''The fitted ladder's resistances at a different coolant flow.

    Only the last resistance depends on flow. It is split into a water-warming part (temperature rise of the coolant across the
    plate, proportional to 1/flow) and a convective part (proportional to 1/flow^n). flow_ref_lpm is the flow the fit was made at.
    Returns (R array, advective part at this flow).
    '''
    ref_cap = flow_ref_lpm / GPM_TO_LPM * GPM_WK                # W/K of the stream at the reference flow
    R_adv_ref = 1 / (2 * ref_cap)
    R_conv_ref = max(R[2] - R_adv_ref, 0.0)
    ratio = flow_ref_lpm / np.asarray(flow_lpm, float)
    return np.array([R[0], R[1], R_conv_ref * ratio ** n + R_adv_ref * ratio]), R_adv_ref * ratio


def single_gpu(P_W, flow_lpm, T_in_C, jn, flow_ref_lpm, n=0.33, dt=0.01):
    '''Junction and layer temperatures of ONE GPU on its own cold plate, with coolant flow that may change over time.

    P_W: power [W]; flow_lpm: coolant flow [L/min], number or array; T_in_C: coolant inlet [°C], number or array; jn: a fitted
    package from load_model()['jnet']; flow_ref_lpm: the flow that package was fitted at; n: flow exponent of the convective
    part (assumed, not measured). All arrays on a uniform dt grid. Starts settled at the first sample.
    Exact while flow is constant; flow is rounded to 1 % steps to reuse matrices. Returns dict T (3, N), R3 (N,), q_wall.
    '''
    P = np.asarray(P_W, float); N = len(P)
    F = np.broadcast_to(np.asarray(flow_lpm, float), (N,)); Tin = np.broadcast_to(np.asarray(T_in_C, float), (N,))
    key = np.round(np.log(F) / 0.01).astype(int)
    cache = {}
    def step(k):
        if k not in cache:
            A, B = cauer_matrices(single_gpu_flow_ladder(jn['R'], np.exp(k * 0.01), flow_ref_lpm, n)[0], jn['C'])
            M = np.zeros((5, 5)); M[:3, :3] = A; M[:3, 3:] = B
            E = expm(M * dt); cache[k] = (E[:3, :3], E[:3, 3:], A, B)
        return cache[k]
    Ad, Bd, A, B = step(key[0])
    x = -np.linalg.solve(A, B @ np.array([P[0], Tin[0]]))
    T = np.empty((3, N)); R3 = np.empty(N)
    for i in range(N):
        T[:, i] = x
        R3[i] = single_gpu_flow_ladder(jn['R'], np.exp(key[i] * 0.01), flow_ref_lpm, n)[0][2]
        Ad, Bd, _, _ = step(key[i])
        x = Ad @ x + Bd @ np.array([P[i], Tin[i]])
    return dict(T=T, R3=R3, q_wall=(T[2] - Tin) / R3 / A_DIE_CM2)


def with_changes(M, **sections):
    '''A copy of model M with what-if changes, e.g. with_changes(M, ctl=dict(d_c=10), cdu=dict(UA_hx=5e5)).'''
    out = deepcopy(M)
    for sec, kv in sections.items():
        out[sec].update(kv)
    return out
