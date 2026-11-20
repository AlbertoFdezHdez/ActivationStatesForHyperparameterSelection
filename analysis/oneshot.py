"""One-shot early selection: 23 experiments x 4 prefixes x 4 continuation counts."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from .common import (EPSILON, GRID, KS, PRIMARY_WEIGHT, SWEEP, PROBE_TASKS, TEST_SPLIT_TASKS,
                     ALL_TAUS, order, rank, regret, summarize, stats, profile_G)
from . import data

C1_C2 = {'cnn:c1_resnet34_imagenet100', 'cnn:c2_densenet121_tinyimagenet'}

# Auxiliary diagnostics: column and orientation (smaller oriented value ranks better).
DIAGNOSTICS = {
    'G': ('G', -1), 'Arithmetic mean': ('M', -1), 'Log AM/GM': ('LUD', 1),
    'Grad norm': ('grad_norm', 1), 'Parameter norm': ('param_norm', 1),
    'Update norm': ('update_norm', 1), 'Update/weight': ('update_to_weight', 1),
    'Best prefix': ('best_external_signal', 1), 'Extrapolation': ('linear_prediction', 1),
    'Sparsity': ('activation_sparsity', 1), 'Train loss': ('train_loss', 1),
    'Eff. rank': ('rankme', -1),
}
STANDALONE = {'Best prefix (standalone)': 'best_external_signal',
              'Linear extrapolation': 'linear_prediction', 'LC-PFN': 'lcpfn_prediction'}


def decide(d, policies, objective='final_objective_mean_seed', ks=KS):
    """Apply score policies to every (experiment, prefix) group.

    ``policies`` maps (method, weight) -> function(group) -> score. Returns one
    row per decision with regret, the retained set and the best retained ID.
    """
    rows = []
    for (exp, tau), g in d.groupby(['experiment_id', 'tau'], sort=True):
        g = g.reset_index(drop=True)
        ids = g.config_id.to_numpy(str)
        y = g[objective].to_numpy(float)
        assert np.isfinite(y).all()
        for (method, weight), fn in policies.items():
            o = order(ids, fn(g))
            for k in ks:
                rows.append(dict(experiment_id=exp, family=g.family.iloc[0], tau=tau, k=k,
                                 method=method, weight=weight, regret=regret(y, o[:k]),
                                 retained=frozenset(ids[o[:k]])))
    return pd.DataFrame(rows)


def evaluation(g):
    return rank(g.external_signal)


def combined(name, weight):
    col, sign = DIAGNOSTICS[name]
    return lambda g: rank(g.external_signal)+weight*rank(sign*g[col].to_numpy(float))


def standalone(col):
    return lambda g: rank(g[col])


def primary_policies():
    return {('Evaluation', 0.0): evaluation, ('G', PRIMARY_WEIGHT): combined('G', PRIMARY_WEIGHT)}


def best_weight(c):
    """Coefficient minimizing mean regret; ties resolved toward the smaller coefficient."""
    m = c.groupby('weight').regret.mean()
    return min(m.index, key=lambda w: (round(float(m[w]), 15), w))


def optimized(d, names, grid=GRID):
    """Grid cells for each diagnostic and its selected coefficient on ``d``'s support."""
    cells = decide(d, {(n, w): combined(n, w) for n in names for w in grid})
    chosen = {n: best_weight(cells[cells.method.eq(n)]) for n in names}
    return cells, chosen


def support(d, col):
    ok = d.groupby('experiment_id')[col].apply(lambda s: np.isfinite(s.to_numpy(float)).all())
    return d[d.experiment_id.isin(ok[ok].index)]


def pair(cells):
    """Evaluation and G regrets side by side for every primary decision."""
    e = cells[cells.method.eq('Evaluation')].set_index(['experiment_id', 'family', 'tau', 'k'])
    g = cells[cells.method.eq('G')].set_index(['experiment_id', 'family', 'tau', 'k'])
    p = pd.DataFrame({'E': e.regret, 'G': g.regret, 'set_E': e.retained, 'set_G': g.retained})
    p['gain'] = p.E-p.G
    return p.reset_index()


def run():
    d = data.primary()
    out = {}
    base = decide(d, primary_policies())
    out['primary_cells'] = base
    p = pair(base)
    out['pairs'] = p

    # Table 3: complete-support controls with coefficients optimized on the common grid.
    full = ['G', 'Grad norm', 'Parameter norm', 'Update norm', 'Update/weight', 'Best prefix', 'Extrapolation',
            'Arithmetic mean', 'Log AM/GM']
    grid_cells, chosen = optimized(d, full)
    out['grid_cells'] = grid_cells
    out['chosen_weights'] = chosen
    alone = decide(d, {(n, 0.0): standalone(c) for n, c in STANDALONE.items()})
    rows = [('Evaluation', None, base[base.method.eq('Evaluation')]),
            ('Eval. + G', PRIMARY_WEIGHT, base[base.method.eq('G')])]
    for label, name in [('Eval. + Grad norm', 'Grad norm'), ('Eval. + Parameter norm', 'Parameter norm'),
                        ('Eval. + Update norm', 'Update norm')]:
        rows.append((label, chosen[name], grid_cells[grid_cells.method.eq(name) & grid_cells.weight.eq(chosen[name])]))
    rows.append(('Best prefix', None, alone[alone.method.eq('Best prefix (standalone)')]))
    w = chosen['Best prefix']
    rows.append(('Eval. + best-prefix rank', w, grid_cells[grid_cells.method.eq('Best prefix') & grid_cells.weight.eq(w)]))
    rows.append(('Linear extrapolation', None, alone[alone.method.eq('Linear extrapolation')]))
    w = chosen['Extrapolation']
    rows.append(('Eval. + extrapolation', w, grid_cells[grid_cells.method.eq('Extrapolation') & grid_cells.weight.eq(w)]))
    rows.append(('LC-PFN', None, alone[alone.method.eq('LC-PFN')]))
    out['table3'] = pd.DataFrame([dict(score=s, weight=w, **stats(c.regret)) for s, w, c in rows])

    # Table 4: matched support for partially recorded diagnostics.
    t4 = []
    for subset, cols in [('Non-RL', ['activation_sparsity', 'train_loss']), ('Eff. rank', ['rankme'])]:
        s = d
        for col in cols:
            s = support(s, col)
        b = decide(s, primary_policies())
        t4.append(dict(subset=subset, tasks=s.experiment_id.nunique(), score='Evaluation', weight=None,
                       **stats(b[b.method.eq('Evaluation')].regret)))
        t4.append(dict(subset=subset, tasks=s.experiment_id.nunique(), score='G', weight=PRIMARY_WEIGHT,
                       **stats(b[b.method.eq('G')].regret)))
        names = ['Sparsity', 'Train loss'] if subset == 'Non-RL' else ['Eff. rank']
        cells, ch = optimized(s, names)
        for n in names:
            c = cells[cells.method.eq(n) & cells.weight.eq(ch[n])]
            t4.append(dict(subset=subset, tasks=s.experiment_id.nunique(), score=n, weight=ch[n], **stats(c.regret)))
    out['table4'] = pd.DataFrame(t4)

    # Table 12, Figure 2 and influence summaries: per-task means over 16 decisions.
    task = p.groupby(['experiment_id', 'family']).agg(E=('E', 'mean'), G=('G', 'mean'), gain=('gain', 'mean')).reset_index()
    task['configs'] = d.groupby('experiment_id').config_id.nunique().reindex(task.experiment_id).to_numpy()
    out['task'] = task
    out['family'] = task.groupby('family')[['E', 'G', 'gain']].mean().reset_index()

    # Table 13 and Figure 3.
    rows = []
    for (tau, k), g in p.groupby(['tau', 'k']):
        rows.append(dict(tau=tau, k=k, **{f'E_{s}': v for s, v in stats(g.E).items()},
                         **{f'G_{s}': v for s, v in stats(g.G).items()}, diff=g.E.mean()-g.G.mean()))
    out['tau_k'] = pd.DataFrame(rows)
    out['by_k'] = p.groupby('k')[['E', 'G']].mean().reset_index()
    out['by_tau'] = p.groupby('tau')[['E', 'G']].mean().reset_index()

    # Table 15: coefficient sweep with changed sets, wins and losses.
    sweep = decide(d, {('G', w): combined('G', w) for w in SWEEP})
    e = p.set_index(['experiment_id', 'tau', 'k'])
    rows = []
    for w, g in sweep.groupby('weight'):
        g = g.set_index(['experiment_id', 'tau', 'k']).loc[e.index]
        rows.append(dict(weight=w, **stats(g.regret), sets=int((g.retained != e.set_E).sum()),
                         wins=int((g.regret < e.E-1e-12).sum()), losses=int((g.regret > e.E+1e-12).sum())))
    out['table15'] = pd.DataFrame(rows)

    # Table 16: aggregation at coefficient 1/2.
    agg = grid_cells[grid_cells.weight.eq(PRIMARY_WEIGHT) & grid_cells.method.isin(['Arithmetic mean', 'G', 'Log AM/GM'])]
    rows = [dict(score='Evaluation', **stats(p.E))]
    for name, label in [('Arithmetic mean', 'Arithmetic mean'), ('G', 'Geometric mean'), ('Log AM/GM', 'Log AM/GM')]:
        rows.append(dict(score=label, **stats(agg[agg.method.eq(name)].regret)))
    out['table16'] = pd.DataFrame(rows)

    # Table 17: held-out coefficient selection (task and family folds).
    rows, choices = [], []
    names = ['Arithmetic mean', 'Best prefix', 'Extrapolation', 'G', 'Grad norm', 'Parameter norm',
             'Update norm', 'Update/weight']
    for unit in ('experiment_id', 'family'):
        for n in names:
            pool = grid_cells[grid_cells.method.eq(n)]
            held = []
            for target in sorted(pool[unit].unique()):
                w = best_weight(pool[pool[unit].ne(target)])
                held.append(pool[pool[unit].eq(target) & pool.weight.eq(w)])
                choices.append(dict(unit=unit, diagnostic=n, held_out=target, weight=w))
            rows.append(dict(unit=unit, diagnostic=n, **stats(pd.concat(held).regret)))
    out['table17'] = pd.DataFrame(rows)
    out['transfer_choices'] = pd.DataFrame(choices)

    # Table 18 and seed-summary objectives: fixed early decisions, new terminal objectives.
    rows = []
    objectives = [('Best within budget', 'final_objective_mean_seed'),
                  ('Last within budget', 'last_objective_within_budget'),
                  ('Final checkpoint', 'final_checkpoint_objective'),
                  ('Median seed', 'final_objective_median_seed'),
                  ('Worst seed', 'final_objective_worst_seed')]
    pol = dict(primary_policies())
    pol[('LC-PFN', 0.0)] = standalone('lcpfn_prediction')
    for label, col in objectives:
        c = decide(d, pol, objective=col)
        for m in ('Evaluation', 'G', 'LC-PFN'):
            rows.append(dict(objective=label, score=m, **stats(c[c.method.eq(m)].regret)))
    rows += global_incumbent(d)
    out['table18'] = pd.DataFrame(rows)

    # Text: restriction to the 19 tasks with their own validation data.
    q = p[~p.experiment_id.isin(TEST_SPLIT_TASKS)]
    out['own_validation'] = dict(tasks=q.experiment_id.nunique(), E=stats(q.E), G=stats(q.G))
    # Text: further restriction that also leaves out C1 and C2 (17 tasks).
    q = q[~q.experiment_id.isin(C1_C2)]
    out['own_validation_no_c1c2'] = dict(tasks=q.experiment_id.nunique(), E=stats(q.E), G=stats(q.G))

    # Text: agreement of the complete ordering with the terminal objective.
    out['rank_agreement'] = rank_agreement(d)

    # LC-PFN fallback count among eligible configuration-prefix rows.
    out['lcpfn_rows'] = dict(rows=len(d), fallback_rows=int(d.fallback_seeds.gt(0).sum()))
    out['table6'], out['epsilon_counts'] = epsilon_table(d)
    out['table7'] = cnn_hooks(d, task)
    out['table14'] = temporal()
    return out


def global_incumbent(d):
    """Return the better of the retained sets' oracle and any oracle-metric checkpoint seen so far."""
    rows = []
    cells = []
    for (exp, tau), g in d.groupby(['experiment_id', 'tau'], sort=True):
        g = g.reset_index(drop=True)
        ids = g.config_id.to_numpy(str)
        y = g.final_objective_mean_seed.to_numpy(float)
        incumbent = float(g.prefix_incumbent.min())
        assert incumbent >= y.min()-1e-6
        for (m, w), fn in primary_policies().items():
            o = order(ids, fn(g))
            for k in KS:
                value = min(float(y[o[:k]].min()), incumbent)
                cells.append(dict(method=m, regret=(value-y.min())/(y.max()-y.min()+1e-12)))
    cells = pd.DataFrame(cells)
    for m in ('Evaluation', 'G'):
        rows.append(dict(objective='Global grid incumbent', score=m, **stats(cells[cells.method.eq(m)].regret)))
    return rows


def rank_agreement(d):
    rows = []
    for (exp, tau), g in d.groupby(['experiment_id', 'tau']):
        y = g.final_objective_mean_seed.to_numpy(float)
        for (m, w), fn in primary_policies().items():
            s = fn(g)
            ok = np.std(s) > 0 and np.std(y) > 0
            rows.append(dict(method=m, spearman=spearmanr(s, y).statistic if ok else np.nan,
                             kendall=kendalltau(s, y).statistic if ok else np.nan))
    return pd.DataFrame(rows).groupby('method')[['spearman', 'kendall']].mean()


def layer_G(exclude=None, epsilons=(EPSILON,)):
    """Recompute run-level G from module profiles, optionally dropping named hooks."""
    lp = pd.read_parquet(data.DATA/'layer_profiles.parquet')
    if exclude is not None:
        lp = lp[~lp.apply(lambda r: exclude(r.experiment_id, r.layer_id), axis=1)]
    lp['config_id'] = lp.run_id.str.rsplit('|seed=', n=1).str[0]
    lp = lp[lp.config_id.isin(data.eligible_configs())]
    rows, zero_layers, all_zero = [], 0, 0
    for (exp, cfg, run, tau), g in lp.groupby(['experiment_id', 'config_id', 'run_id', 'tau']):
        u = g.U_l.to_numpy(float)
        zero_layers += int((u == 0).sum())
        all_zero += int((u == 0).all())
        rows.append(dict(experiment_id=exp, config_id=cfg, run_id=run, tau=tau,
                         **{str(e): profile_G(u, e) for e in epsilons}))
    r = pd.DataFrame(rows)
    counts = dict(profiles=len(r), zero_layers=zero_layers, all_zero_profiles=all_zero)
    return r.groupby(['config_id', 'tau'])[[str(e) for e in epsilons]].mean().reset_index(), counts


def epsilon_table(d):
    eps = (0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
    g, counts = layer_G(epsilons=eps)
    x = d.drop(columns=['G']).merge(g, on=['config_id', 'tau'], validate='one_to_one')
    # The recomputed default must coincide with the recorded selector input.
    assert np.max(np.abs(x[str(EPSILON)].to_numpy()-d.G.to_numpy())) < 1e-8
    rows = []
    for e in eps:
        c = decide(x, {('G', PRIMARY_WEIGHT): lambda h, e=e: rank(h.external_signal)+PRIMARY_WEIGHT*rank(-h[str(e)])})
        rows.append(dict(epsilon=e, **stats(c.regret)))
    return pd.DataFrame(rows), counts


def cnn_hooks(d, task):
    g, _ = layer_G(exclude=lambda exp, layer: exp.startswith('cnn:') and 'avgpool' in str(layer))
    x = d.drop(columns=['G']).merge(g, on=['config_id', 'tau'], validate='one_to_one').rename(columns={str(EPSILON): 'G'})
    x = x[x.family.eq('cnn')]
    c = decide(x, {('G', PRIMARY_WEIGHT): combined('G', PRIMARY_WEIGHT)})
    without = c.groupby('experiment_id').regret.mean()
    t = task[task.family.eq('cnn')].set_index('experiment_id')
    return pd.DataFrame(dict(E=t.E, G=t.G, G_without_avgpool=without)).reset_index()


def temporal():
    """Additional prefixes, averaged over K; tasks need finite early signals for every candidate."""
    d = data.configs(tuple(t for t in ALL_TAUS if t < 1))
    rows = []
    for tau, g in d.groupby('tau'):
        ok = g.groupby('experiment_id')[['external_signal', 'G']].apply(lambda s: np.isfinite(s.to_numpy(float)).all())
        g = g[g.experiment_id.isin(ok[ok].index)]
        c = decide(g, primary_policies())
        rows.append(dict(tau=tau, tasks=g.experiment_id.nunique(),
                         E=c[c.method.eq('Evaluation')].regret.mean(), G=c[c.method.eq('G')].regret.mean()))
    return pd.DataFrame(rows)
