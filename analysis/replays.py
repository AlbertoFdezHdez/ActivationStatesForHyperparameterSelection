"""Paired offline promotion replays (Section 5.4, Table 5, Appendix C.5).

Default mode aggregates the supplied promotion cells. ``recompute(mode)``
replays every paired repetition from the run-level records:
ASHA (100 arrival permutations x 4/8/12 workers, single runs), Hyperband
(200 random streams) and BOHB-style (200 frozen TPE proposal streams), both
on three-seed configurations. Only the promotion score differs within a pair.
"""
import heapq
import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .common import DATA, PRIMARY_WEIGHT, ALL_TAUS, cvar
from . import data

ASHA_RUNGS = (0.05, 0.15, 0.45, 1.0)
ASHA_REPS = 100
BRACKET_REPS = 200
G_LABEL = 'G 1/2'
EXTRAP_LABEL = 'Eval. + extrapolation 1/2'


def rank(values):
    """Normalized rank; missing values receive the worst rank."""
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    out = np.ones(len(values), float)
    if valid.any():
        out[valid] = (rankdata(values[valid], method='average')-1)/max(1, valid.sum()-1)
    return out


# ----------------------------------------------------------------------------- ASHA
def asha(arrays, ids, order, workers, weight):
    """Event-driven ASHA on recorded incremental durations; returns (regret, cost, makespan)."""
    events, worker_times = [], np.zeros(workers)
    arrived = [[] for _ in ASHA_RUNGS]
    promoted = [set() for _ in ASHA_RUNGS]
    finished, queue, work = [], list(order), 0.0

    def schedule(worker, index, rung, start):
        now = float(arrays[ASHA_RUNGS[rung]]['wall'][index])
        prev = float(arrays[ASHA_RUNGS[rung-1]]['wall'][index]) if rung else 0.0
        heapq.heappush(events, (start+max(now-prev, 1e-9), worker, index, rung))

    for worker in range(min(workers, len(queue))):
        schedule(worker, queue.pop(0), 0, 0.0)
    while events:
        finish, worker, index, rung = heapq.heappop(events)
        worker_times[worker] = finish
        tau = ASHA_RUNGS[rung]
        arrived[rung].append(index)
        now = float(arrays[tau]['steps'][index])
        prev = float(arrays[ASHA_RUNGS[rung-1]]['steps'][index]) if rung else 0.0
        work += max(now-prev, 0.0)
        if rung == len(ASHA_RUNGS)-1:
            finished.append(index)
            if queue:
                schedule(worker, queue.pop(0), 0, finish)
            continue
        observed = np.asarray(arrived[rung], dtype=int)
        score = rank(arrays[tau]['external'][observed])
        if weight:
            score += weight*rank(arrays[tau]['aux'][observed])
        positions = np.lexsort((ids[observed], score))      # identity ties shared by both policies
        top = set(observed[positions[:max(1, int(np.ceil(len(observed)/3)))]])
        if (len(observed) < 3 or index in top) and index not in promoted[rung]:
            promoted[rung].add(index)
            schedule(worker, index, rung+1, finish)
        elif queue:
            schedule(worker, queue.pop(0), 0, finish)
    final = arrays[1.0]['final']
    lo, hi = float(final.min()), float(final.max())
    incumbent = float(final[finished].min()) if finished else hi
    return (incumbent-lo)/(hi-lo+1e-12), work/float(arrays[1.0]['steps'].sum()), float(worker_times.max())


def asha_cells():
    r = data.runs()
    r = r[r.tau.isin(ASHA_RUNGS)]
    complete = r.groupby('run_id').tau.nunique()
    r = r[r.run_id.isin(complete[complete.eq(len(ASHA_RUNGS))].index)]
    rows = []
    for exp, g in r.groupby('experiment_id', sort=True):
        ids = np.sort(g.run_id.unique().astype(str))
        arrays = {}
        for tau in ASHA_RUNGS:
            f = g[g.tau.eq(tau)].set_index('run_id').loc[ids]
            arrays[tau] = dict(external=f.external_signal.to_numpy(float), aux=(-f.G).to_numpy(float),
                               steps=f.cumulative_steps.to_numpy(float), wall=f.cumulative_wall_time.to_numpy(float),
                               final=f.final_objective_oriented.to_numpy(float))
        for rep in range(ASHA_REPS):
            order = np.random.default_rng(rep).permutation(len(ids))
            for workers in (4, 8, 12):
                for label, weight in (('Evaluation', None), (G_LABEL, PRIMARY_WEIGHT)):
                    regret, cost, wall = asha(arrays, ids, order, workers, weight)
                    rows.append(dict(experiment_id=exp, family=g.family.iloc[0], repetition=rep, workers=workers,
                                     method=label, regret=regret, cost_normalized=cost, recorded_makespan=wall,
                                     n_runs=len(ids)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ Hyperband / BOHB-style
def brackets():
    """Three brackets (eta=3, minimum 0.05) mapped down to the recorded fidelity grid."""
    eta, minimum, maximum = 3, 0.05, 1.0
    smax = int(math.floor(math.log(maximum/minimum, eta)))
    budget = (smax+1)*maximum
    available = np.asarray([0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.50, 0.60, 0.75, 1.0])
    out = []
    for s in range(smax, -1, -1):
        n = int(math.ceil((budget/maximum)*eta**s/(s+1)))
        theoretical = [min(maximum, maximum*eta**(-s)*eta**i) for i in range(s+1)]
        mapped = [float(available[available <= v+1e-12].max()) for v in theoretical]
        out.append(dict(s=s, n=n, theoretical=theoretical, mapped=mapped))
    return out


def random_streams(n_pool, repetition):
    rng = np.random.default_rng(1_000_000+repetition)
    return [rng.choice(n_pool, size=min(b['n'], n_pool), replace=False).tolist() for b in brackets()]


def tpe_choice(frame, observed, targets, candidates, rng):
    """Factorized categorical TPE over observed LR/WD levels with add-one smoothing."""
    if len(observed) < 5:
        return int(rng.choice(candidates))
    ranked = sorted(observed, key=lambda idx: targets[idx])
    n_good = max(1, math.ceil(0.20*len(ranked)))
    good, bad = ranked[:n_good], ranked[n_good:]
    if not bad:
        return int(rng.choice(candidates))
    lr_levels = sorted(frame.lr.unique().tolist())
    wd_values = frame.wd.fillna(-1.0)
    wd_levels = sorted(wd_values.unique().tolist())

    def probability(indices, value, values, levels):
        count = sum(float(values[idx]) == float(value) for idx in indices)
        return (count+1.0)/(len(indices)+len(levels))

    lr, wd = frame.lr.to_numpy(float), wd_values.to_numpy(float)
    ratios = []
    for idx in candidates:
        l = probability(good, lr[idx], lr, lr_levels)*probability(good, wd[idx], wd, wd_levels)
        b = probability(bad, lr[idx], lr, lr_levels)*probability(bad, wd[idx], wd, wd_levels)
        ratios.append(l/max(b, 1e-12))
    best = np.flatnonzero(np.isclose(ratios, np.max(ratios)))
    return int(candidates[int(rng.choice(best))])


def proposal_streams(frames, repetition):
    """Frozen BOHB-style proposer: uniform until six observations, then TPE w.p. 2/3."""
    rng = np.random.default_rng(repetition)
    n_pool = len(next(iter(frames.values())))
    history, streams = {}, []
    for bracket in brackets():
        low = frames[bracket['mapped'][0]]
        selected = []
        for _ in range(min(bracket['n'], n_pool)):
            candidates = [i for i in range(n_pool) if i not in selected]
            target = low.external_signal.to_numpy(float).copy()
            for i, value in history.items():
                target[i] = value
            if len(history) < 6 or rng.random() < 1/3:
                choice = int(rng.choice(candidates))
            else:
                # The historical helper draws its own exploration coin with probability 0.
                rng.random()
                choice = tpe_choice(low, list(history), target, candidates, rng)
            selected.append(choice)
            history[choice] = float(low.external_signal.iloc[choice])
        streams.append(selected)
    return streams


def bracket_scores(f, method):
    score = rank(f.external_signal.to_numpy(float))
    if method == G_LABEL:
        score = score+PRIMARY_WEIGHT*rank(-f.G.to_numpy(float))
    elif method == EXTRAP_LABEL:
        score = score+0.5*rank(f.linear_prediction.to_numpy(float))
    return score


def bracket_replay(frames, streams, method):
    incumbent, work, previous = np.inf, 0.0, {}
    for b_id, (bracket, selected) in enumerate(zip(brackets(), streams)):
        survivors = selected[:]
        for rung_id, tau in enumerate(bracket['mapped']):
            full = frames[tau]
            current = full.iloc[survivors].reset_index(drop=True)
            order = np.lexsort((current.config_id.astype(str).to_numpy(), bracket_scores(current, method)))
            for idx in survivors:
                step = float(full.cumulative_steps.iloc[idx])
                work += max(step-previous.get((b_id, idx), 0.0), 0.0)
                previous[(b_id, idx)] = step
            if rung_id < len(bracket['mapped'])-1:
                survivors = [survivors[i] for i in order[:max(1, len(survivors)//3)]]
            elif np.isclose(tau, 1.0):
                incumbent = min(incumbent, float(current.final_objective_mean_seed.min()))
    final = frames[1.0]
    lo, hi = float(final.final_objective_mean_seed.min()), float(final.final_objective_mean_seed.max())
    return (incumbent-lo)/(hi-lo+1e-12), work/float(final.cumulative_steps.sum())


def bracket_cells(mode):
    taus = sorted({t for b in brackets() for t in b['mapped']})
    d = data.configs(ALL_TAUS)
    methods = ['Evaluation', G_LABEL] + ([EXTRAP_LABEL] if mode == 'bohb' else [])
    rows = []
    for exp, g in d[d.tau.isin(taus)].groupby('experiment_id', sort=True):
        frames = {tau: g[g.tau.eq(tau)].sort_values('config_id').reset_index(drop=True) for tau in taus}
        assert all(f.config_id.tolist() == frames[taus[0]].config_id.tolist() for f in frames.values())
        for rep in range(BRACKET_REPS):
            n = len(frames[taus[0]])
            streams = proposal_streams(frames, rep) if mode == 'bohb' else random_streams(n, rep)
            for method in methods:
                regret, cost = bracket_replay(frames, streams, method)
                rows.append(dict(experiment_id=exp, family=g.family.iloc[0], repetition=rep, method=method,
                                 regret=regret, cost_normalized=cost, n_configs=n))
    return pd.DataFrame(rows)


def recompute(mode):
    return asha_cells() if mode == 'asha' else bracket_cells(mode)


# ---------------------------------------------------------------------- aggregation
def supplied(mode):
    return pd.read_csv(DATA/'promotion_cells'/f'{mode}_cells.csv')


def summarize(cells_by_mode):
    """Table 5: repetitions averaged within tasks, then mean/CVaR/max over 23 task means."""
    rows = []
    for mode, label in (('hyperband', 'Hyperband'), ('bohb', 'BOHB-style'), ('asha', 'ASHA')):
        c = cells_by_mode[mode]
        blocks = c.groupby('workers') if mode == 'asha' else [(None, c)]
        for workers, block in blocks:
            t = block.groupby(['experiment_id', 'method']).agg(regret=('regret', 'mean'), cost=('cost_normalized', 'mean')).reset_index()
            for method in ('Evaluation', G_LABEL, EXTRAP_LABEL):
                x = t[t.method.eq(method)]
                if x.empty:
                    continue
                assert x.experiment_id.nunique() == 23
                v = x.regret.to_numpy()
                rows.append(dict(replay=label+(f' ({int(workers)})' if workers else ''), method=method,
                                 mean=v.mean(), cvar25=cvar(v, .25), cvar5=cvar(v, .05), worst=v.max(),
                                 cost=x.cost.mean()))
    return pd.DataFrame(rows)
