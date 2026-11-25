"""Additional one-shot controls reported in Sections 4.1 and 4.3.

* learning-rate prior: Evaluation plus the rank of log learning rate, in either orientation,
  with the coefficient chosen on the common grid (in-sample and held out by task), and the same
  prior with the half-weight G rank added;
* Spearman correlation of G with log learning rate and log weight decay within each
  experiment and prefix;
* Evaluation plus the LC-PFN rank on the common grid;
* median-normalized regret: the loss gap divided by the gap between the median and the best
  candidate instead of the full candidate range.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .common import KS, PRIMARY_WEIGHT, rank, order, stats
from . import data, oneshot as O

EXTRA = {'LR (smaller)': ('log_lr', 1), 'LR (larger)': ('log_lr', -1), 'LC-PFN': ('lcpfn_prediction', 1)}


def table():
    d = data.primary().copy()
    d['log_lr'] = np.log(d.lr.astype(float))
    d['log_wd'] = np.log(d.wd.astype(float).clip(lower=1e-12))   # rank-based use only
    return d


def combined(name, weight, plus_G=0.0):
    col, sign = {**O.DIAGNOSTICS, **EXTRA}[name]
    return lambda g: (rank(g.external_signal)+weight*rank(sign*g[col].to_numpy(float))
                      + plus_G*rank(-g.G.to_numpy(float)))


@lru_cache(maxsize=None)
def run():
    d = table()
    from .common import GRID
    names = list(EXTRA)
    cells = O.decide(d, {(n, w): combined(n, w) for n in names for w in GRID})
    chosen = {n: O.best_weight(cells[cells.method.eq(n)]) for n in names}
    best = {n: stats(cells[cells.method.eq(n) & cells.weight.eq(chosen[n])].regret) for n in names}
    lr = min(['LR (smaller)', 'LR (larger)'], key=lambda n: best[n]['mean'])
    held = []
    pool = cells[cells.method.eq(lr)]
    for t in sorted(pool.experiment_id.unique()):
        w = O.best_weight(pool[pool.experiment_id.ne(t)])
        held.append(pool[pool.experiment_id.eq(t) & pool.weight.eq(w)])
    plus = O.decide(d, {('LR+G', chosen[lr]): combined(lr, chosen[lr], PRIMARY_WEIGHT)})

    rows = []
    for (exp, tau), g in d.groupby(['experiment_id', 'tau']):
        rho_wd = spearmanr(g.G, g.log_wd).statistic if g.log_wd.nunique() > 1 else np.nan
        rows.append(dict(experiment_id=exp, tau=tau, rho_lr=spearmanr(g.G, g.log_lr).statistic, rho_wd=rho_wd))
    rho = pd.DataFrame(rows)
    by_task = rho.groupby('experiment_id').rho_lr.mean()

    med = []
    for (exp, tau), g in d.groupby(['experiment_id', 'tau']):
        g = g.reset_index(drop=True)
        ids, y = g.config_id.to_numpy(str), g.final_objective_mean_seed.to_numpy(float)
        den = max(float(np.median(y)-y.min()), 1e-12)
        for (method, _), fn in O.primary_policies().items():
            o = order(ids, fn(g))
            for k in KS:
                med.append(dict(experiment_id=exp, tau=tau, k=k, method=method, regret=(y[o[:k]].min()-y.min())/den))
    med = pd.DataFrame(med)
    pm = med.pivot_table(index=['experiment_id', 'tau', 'k'], columns='method', values='regret')
    task_gain = (pm.Evaluation-pm.G).groupby('experiment_id').mean()
    return dict(
        chosen=chosen, best=best, lr_orientation=lr,
        lr_heldout=stats(pd.concat(held).regret), lr_plus_G=stats(plus.regret),
        rho_lr_abs_median=float(rho.rho_lr.abs().median()),
        rho_wd_abs_median=float(rho.rho_wd.abs().median()),
        wd_tasks=int(rho.dropna(subset=['rho_wd']).experiment_id.nunique()),
        lr_negative_tasks=int((by_task < 0).sum()), lr_positive_tasks=int((by_task > 0).sum()),
        median_norm=dict(E=stats(med[med.method.eq('Evaluation')].regret), G=stats(med[med.method.eq('G')].regret),
                         improved=int((task_gain > 1e-12).sum()), worsened=int((task_gain < -1e-12).sum()),
                         tied=int((task_gain.abs() <= 1e-12).sum())),
    )
