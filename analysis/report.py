"""Assemble every table, figure series and in-text number, and compare with the article.

Comparison rule: a recomputed number, rounded to the decimals printed in the
article, must reproduce the printed string exactly (signs of zero ignored).
Label cells are compared after removing LaTeX markup. Bold face is not checked.
"""
import json
import re
import numpy as np
import pandas as pd
from .common import ROOT, LABELS, PRIMARY_TAUS, KS

EXPECTED = ROOT/'expected'
FRACTIONS = {0.0: '0', 1/8: '1/8', 1/4: '1/4', 1/3: '1/3', 1/2: '1/2', 2/3: '2/3', 3/4: '3/4', 1.0: '1', 3/2: '3/2', 2.0: '2'}
P = 100.0  # tables print regrets as percentages


def clean(s):
    s = re.sub(r'\\textbf\{([^{}]*)\}', r'\1', str(s))
    s = re.sub(r'\\acsp?\{([^{}]*)\}|\\acp?\{([^{}]*)\}', lambda m: m.group(1) or m.group(2), s)
    s = s.replace(r'\(', '').replace(r'\)', '').replace('$', '').replace(r'\times', 'x').replace(r'\%', '%')
    s = s.replace('{', '').replace('}', '').replace('^*', '').replace(r'\,', '')
    return re.sub(r'\s+', ' ', s).strip()


def decimals(s):
    return len(s.split('.')[1]) if '.' in s else 0


def number_matches(value, published):
    p = clean(published).replace(',', '').replace('+', '').rstrip('%')
    if p.startswith('-') and float(p) == 0:
        p = p[1:]
    d = decimals(p)
    v = f'{float(value):.{d}f}'
    if float(v) == 0:
        v = v.lstrip('-')
    return v == p


def compare_rows(name, generated, published):
    problems = []
    if len(generated) != len(published):
        return [f'{name}: {len(generated)} generated rows vs {len(published)} published']
    for i, (g, p) in enumerate(zip(generated, published)):
        if len(g) != len(p):
            problems.append(f'{name} row {i+1}: {len(g)} vs {len(p)} cells')
            continue
        for j, (a, b) in enumerate(zip(g, p)):
            if a is None:
                continue
            ok = clean(a) == clean(b) if isinstance(a, str) else number_matches(a, b)
            if not ok:
                problems.append(f'{name} row {i+1} col {j+1}: reproduced {a!r} vs published {b!r}')
    return problems


def pct(s):
    return [P*s['mean'], P*s['cvar25'], P*s['cvar5'], P*s['worst']]


def frac(w):
    return FRACTIONS[min(FRACTIONS, key=lambda x: abs(x-w))]


def tables(o, u, rep, mech, grids, rl_grid, catalog):
    T = {}
    m = mech.set_index('activation')
    T['tab:mechanism-new'] = [['GELU', int(m.loc['GELU', 'checkpoints']), m.loc['GELU', 'correlation'],
                               P*m.loc['GELU', 'median_error'], f"{m.loc['GELU', 'positive_gain']}/{m.loc['GELU', 'checkpoints']}"],
                              ['ReLU', int(m.loc['ReLU', 'checkpoints']), m.loc['ReLU', 'correlation'],
                               P*m.loc['ReLU', 'median_error'], f"{m.loc['ReLU', 'positive_gain']}/{m.loc['ReLU', 'checkpoints']}"]]
    t3 = o['table3']
    rows = []
    for r in t3.itertuples():
        label = {'Eval. + G': 'Eval. + G', 'LC-PFN': 'LC-PFN'}.get(r.score, r.score)
        rows.append([label, '--' if pd.isna(r.weight) else frac(r.weight)] + pct(r._asdict()))
    T['tab:main-new'] = rows
    rows = []
    for r in o['table4'].itertuples():
        subset = 'Non-RL' if r.subset == 'Non-RL' else 'Eff. rank'
        if r.score == 'Evaluation':
            label = 'Evaluation'
        elif r.score == 'G':
            label = 'Eval. + G 1/2'
        else:
            label = f'Eval. + {r.score} {frac(r.weight)}'
        rows.append([subset, r.tasks, label] + pct(r._asdict()))
    T['tab:matched-new'] = rows
    rows = []
    names = {'Evaluation': 'Evaluation', 'G 1/2': 'G', 'Eval. + extrapolation 1/2': 'Eval. + extrapolation'}
    for r in rep.itertuples():
        rows.append([r.replay, names[r.method]] + pct(r._asdict()) + [P*r.cost])
    T['tab:schedulers-new'] = rows
    eps_label = {0: '0', 1e-12: '10^-12', 1e-10: '10^-10', 1e-8: '10^-8', 1e-6: '10^-6', 1e-4: '10^-4', 1e-2: '10^-2'}
    T['tab:epsilon-new'] = [[eps_label[r.epsilon]] + pct(r._asdict()) for r in o['table6'].itertuples()]
    arch = {'C1': 'ResNet-34', 'C2': 'DenseNet-121', 'C3': 'EfficientNet-B0', 'C4': 'MobileNetV2'}
    T['tab:cnn-hooks-new'] = [[f'{LABELS[r.experiment_id][0]}: {arch[LABELS[r.experiment_id][0]]}', P*r.E, P*r.G, P*r.G_without_avgpool]
                              for r in o['table7'].itertuples()]
    c = catalog.set_index('id')
    vision = ['C1', 'C2', 'C3', 'C4', 'V1', 'V2', 'V3', 'M1', 'M2', 'M3', 'M4']
    rows = []
    for i in vision:
        r = c.loc[i]
        inp = f'{int(r.input)}^2' if i[0] in 'CV' else None   # MLP input widths are dataset facts
        rows.append([i, None, None, int(r.epochs), int(r.bs), inp, int(r.L)])
    T['tab:experiment-catalog'] = rows
    T['tab:nlp-catalog'] = [[i, None, None, f'{int(c.loc[i, "steps"])//1000}k', f'{int(c.loc[i, "bs"])}x{int(c.loc[i, "ga"])}',
                             c.loc[i, 'length'], int(c.loc[i, 'L'])] for i in ['P1', 'P2', 'P3', 'F1', 'F2', 'F3']]
    def k(n):
        return f'{n//1_000_000}M' if n >= 1_000_000 else f'{n//1000}k'
    T['tab:rl-catalog'] = [[i, c.loc[i, 'algorithm'], c.loc[i, 'env'], k(int(c.loc[i, 'timesteps'])),
                            f"{int(c.loc[i, 'probe']):,}", int(c.loc[i, 'episodes'])] for i in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']]
    rows = []
    for r in grids.sort_values('id', key=lambda s: s.map(_order)).itertuples():
        rl = r.id.startswith('R')
        rows.append([r.id, 'common RL grid' if rl else r.lrs, r.wds, str(r.budget)])
    T['tab:observed-grids-new'] = rows
    task = o['task'].copy()
    task['id'] = task.experiment_id.map(lambda e: LABELS[e][0])
    T['tab:all-experiments-new'] = [[f'{r.id}: {LABELS[r.experiment_id][1]}', int(r.configs), P*r.E, P*r.G, P*r.gain]
                                    for r in task.sort_values('id', key=lambda s: s.map(_order)).itertuples()]
    rows = []
    for r in o['tau_k'].itertuples():
        rows.append([f'{round(100*r.tau)}%', r.k, P*r.E_mean, P*r.G_mean, P*r.E_cvar25, P*r.G_cvar25, P*r.E_cvar5, P*r.G_cvar5])
    T['tab:tau-k-new'] = rows
    tm = o['table14'].set_index('tau')
    left, right = [.01, .02, .05, .10, .15], [.20, .30, .45, .60, .75]   # the 50% row is not reported
    T['tab:temporal-new'] = [[f'{round(100*a)}%', int(tm.loc[a, 'tasks']), P*tm.loc[a, 'E'], P*tm.loc[a, 'G'],
                              f'{round(100*b)}%', int(tm.loc[b, 'tasks']), P*tm.loc[b, 'E'], P*tm.loc[b, 'G']]
                             for a, b in zip(left, right)]
    T['tab:weights-new'] = [[frac(r.weight)] + pct(r._asdict()) + [r.sets, r.wins, r.losses] for r in o['table15'].itertuples()]
    T['tab:level-ablation-new'] = [[r.score] + pct(r._asdict()) for r in o['table16'].itertuples()]
    dn = {'Arithmetic mean': 'Arithmetic mean of U', 'G': 'G (geometric mean)'}
    T['tab:transfer-new'] = [['Task' if r.unit == 'experiment_id' else 'Family', dn.get(r.diagnostic, r.diagnostic)] + pct(r._asdict())
                             for r in o['table17'].itertuples()]
    t18 = o['table18']
    rows = []
    for obj in ['Best within budget', 'Last within budget', 'Final checkpoint', 'Global grid incumbent']:
        for score in ['Evaluation', 'G']:
            r = t18[t18.objective.eq(obj) & t18.score.eq(score)].iloc[0]
            rows.append([obj, score] + pct(r))
    T['tab:objective-robustness-new'] = rows
    T['tab:uncertainty-new'] = [
        ['Percentile bootstrap'] + [P*x for x in u['percentile_abs'] + u['percentile_rel']],
        ['BCa bootstrap'] + [P*x for x in u['bca_abs'] + u['bca_rel']],
        ['Student / delta method'] + [P*x for x in u['student_abs'] + u['student_rel']]]
    return T


def _order(i):
    return 'CMFPRV'.index(i[0])*10+int(i[1])


def text_values(o, u, rep, mech, cost, scope, eps_counts, theory):
    t3 = o['table3'].set_index('score')
    E, G = t3.loc['Evaluation'], t3.loc['Eval. + G']
    red = lambda c: P*(E[c]-G[c])/E[c]
    t4 = o['table4']
    m4 = lambda subset, score: P*t4[t4.subset.eq(subset) & t4.score.eq(score)].iloc[0]['mean']
    t17 = o['table17'].set_index(['unit', 'diagnostic'])
    t18 = o['table18'].set_index(['objective', 'score'])
    r = rep.set_index(['replay', 'method'])
    fam = o['family'].set_index('family')
    mech = mech.set_index('activation')
    t6 = o['table6'].set_index('epsilon')
    t16 = o['table16'].set_index('score')
    t7 = o['table7']
    k = o['by_k'].set_index('k')
    tau = o['by_tau'].set_index('tau')
    p = o['pairs']
    v = {
        'abstract.runs': scope['runs'], 'abstract.mean_reduction': red('mean'), 'abstract.cvar25_reduction': red('cvar25'),
        'abstract.cvar5_reduction': red('cvar5'),
        'abstract.heldout_G': P*t17.loc[('experiment_id', 'G'), 'mean'], 'abstract.heldout_E': P*E['mean'],
        'sec2.sparsity_17': m4('Non-RL', 'Sparsity'), 'sec2.G_17': m4('Non-RL', 'G'),
        'sec3.observation_tasks': 23-4, 'sec3.cost_median': P*cost['median'], 'sec3.cost_p90': P*cost['p90'],
        'sec3.cost_runs': cost['runs'], 'sec3.cost_largest': P*cost['largest_task_median'],
        'sec5.configurations': scope['eligible_configurations'],
        'sec5.own_validation_E': P*o['own_validation']['E']['mean'], 'sec5.own_validation_G': P*o['own_validation']['G']['mean'],
        'sec5.own_validation_cvar_E': P*o['own_validation']['E']['cvar25'],
        'sec5.own_validation_cvar_G': P*o['own_validation']['G']['cvar25'],
        'sec5.mech_median_relu': P*mech.loc['ReLU', 'median_error'], 'sec5.mech_median_gelu': P*mech.loc['GELU', 'median_error'],
        'sec5.mech_bound_checks': mech.loc['ReLU', 'bound_checks'], 'sec5.mech_module_checks': mech.loc['ReLU', 'module_checks'],
        'sec5.main_mean_E': P*E['mean'], 'sec5.main_mean_G': P*G['mean'], 'sec5.main_mean_reduction': red('mean'),
        'sec5.main_cvar25_reduction': red('cvar25'), 'sec5.main_cvar5_reduction': red('cvar5'),
        'sec5.changed_sets': int((p.set_E != p.set_G).sum()), 'sec5.improved_cells': int((p.gain > 1e-12).sum()),
        'sec5.worsened_cells': int((p.gain < -1e-12).sum()), 'sec5.tied_cells': int((p.gain.abs() <= 1e-12).sum()),
        'sec5.max_regret': P*max(E['worst'], G['worst']),
        'sec5.spearman_E': o['rank_agreement'].loc['Evaluation', 'spearman'], 'sec5.kendall_E': o['rank_agreement'].loc['Evaluation', 'kendall'],
        'sec5.spearman_G': o['rank_agreement'].loc['G', 'spearman'], 'sec5.kendall_G': o['rank_agreement'].loc['G', 'kendall'],
        'sec5.top_two_share': P*u['top_two_fraction'], 'sec5.rest_gain': P*u['rest_mean_gain'],
        'sec5.rest_improved': u['rest_improved'], 'sec5.rest_worsened': u['rest_worsened'], 'sec5.rest_tied': u['rest_tied'],
        'sec5.vit_gap_below': 0.01 if -P*fam.loc['vit', 'gain'] < 0.01 and fam.loc['vit', 'gain'] < 0 else np.nan,
        'sec5.cnn_with_pool': P*t7.G.mean(), 'sec5.cnn_without_pool': P*t7.G_without_avgpool.mean(),
        'sec5.k1_E': P*k.loc[1, 'E'], 'sec5.k1_G': P*k.loc[1, 'G'], 'sec5.k3_E': P*k.loc[3, 'E'], 'sec5.k3_G': P*k.loc[3, 'G'],
        'sec5.tau20_E': P*tau.loc[.20, 'E'], 'sec5.tau20_G': P*tau.loc[.20, 'G'],
        'sec5.k10_E': P*k.loc[10, 'E'], 'sec5.k10_G': P*k.loc[10, 'G'],
        'sec5.matched_sparsity': m4('Non-RL', 'Sparsity'), 'sec5.matched_G17': m4('Non-RL', 'G'),
        'sec5.matched_trainloss': m4('Non-RL', 'Train loss'), 'sec5.matched_effrank': m4('Eff. rank', 'Eff. rank'),
        'sec5.matched_G11': m4('Eff. rank', 'G'),
        'sec5.hb_streams': rep.attrs.get('bracket_reps', 200), 'sec5.asha_arrivals': rep.attrs.get('asha_reps', 100),
        'sec5.bohb_G': P*r.loc[('BOHB-style', 'G 1/2'), 'mean'], 'sec5.bohb_E': P*r.loc[('BOHB-style', 'Evaluation'), 'mean'],
        'sec5.asha4_G': P*r.loc[('ASHA (4)', 'G 1/2'), 'mean'], 'sec5.asha8_G': P*r.loc[('ASHA (8)', 'G 1/2'), 'mean'],
        'sec5.asha12_G': P*r.loc[('ASHA (12)', 'G 1/2'), 'mean'], 'sec5.asha4_E': P*r.loc[('ASHA (4)', 'Evaluation'), 'mean'],
        'sec5.asha8_E': P*r.loc[('ASHA (8)', 'Evaluation'), 'mean'], 'sec5.asha12_E': P*r.loc[('ASHA (12)', 'Evaluation'), 'mean'],
        'sec5.hb_G': P*r.loc[('Hyperband', 'G 1/2'), 'mean'], 'sec5.hb_E': P*r.loc[('Hyperband', 'Evaluation'), 'mean'],
        'sec5.bohb_extrap': P*r.loc[('BOHB-style', 'Eval. + extrapolation 1/2'), 'mean'],
        'sec5.bootstrap_draws': 50_000, 'sec5.mean_gain': P*u['mean_gain'],
        'sec5.pct_abs_low': P*u['percentile_abs'][0], 'sec5.pct_abs_high': P*u['percentile_abs'][1],
        'sec5.pct_rel_low': P*u['percentile_rel'][0], 'sec5.pct_rel_high': P*u['percentile_rel'][1],
        'sec5.tasks_improved': u['improved'], 'sec5.tasks_worsened': u['worsened'],
        'sec5.tasks_worsened_tiny': u['worsened_below_001pp'], 'sec5.tasks_tied': u['tied'],
        'sec5.sign_test': u['sign_test_p'], 'sec5.wilcoxon': u['wilcoxon_p'], 'sec5.sign_flip': u['sign_flip_p'],
        'sec5.loo_abs_low': P*u['loo_abs'][0], 'sec5.loo_abs_high': P*u['loo_abs'][1],
        'sec5.loo_rel_low': P*u['loo_rel'][0], 'sec5.loo_rel_high': P*u['loo_rel'][1],
        'sec5.heldout_G': P*t17.loc[('experiment_id', 'G'), 'mean'], 'sec5.heldout_E': P*E['mean'],
        'sec5.heldout_cvar_G': P*t17.loc[('experiment_id', 'G'), 'cvar25'], 'sec5.heldout_cvar_E': P*E['cvar25'],
        'sec5.family_G': P*t17.loc[('family', 'G'), 'mean'], 'sec5.family_worst_G': P*t17.loc[('family', 'G'), 'worst'],
        'sec5.family_worst_E': P*E['worst'],
        'sec5.last_E': P*t18.loc[('Last within budget', 'Evaluation'), 'mean'], 'sec5.last_G': P*t18.loc[('Last within budget', 'G'), 'mean'],
        'sec5.final_E': P*t18.loc[('Final checkpoint', 'Evaluation'), 'mean'], 'sec5.final_G': P*t18.loc[('Final checkpoint', 'G'), 'mean'],
        'sec6.later_prefix': _first_evaluation_win(o['table14']),
        'appB.eps_profiles': eps_counts['profiles'], 'appB.eps_zero_layers': eps_counts['zero_layers'],
        'appB.eps_all_zero': eps_counts['all_zero_profiles'],
        'appB.eps_range_low': P*t6.loc[[1e-12, 1e-10, 1e-8, 1e-6], 'mean'].min(),
        'appB.eps_range_high': P*t6.loc[[1e-12, 1e-10, 1e-8, 1e-6], 'mean'].max(),
        'appB.eps_zero': P*t6.loc[0, 'mean'], 'appB.eps_default': P*t6.loc[1e-8, 'mean'],
        'appC.language_configs': scope['language_configurations'], 'appC.lcpfn_rows': o['lcpfn_rows']['rows'],
        'appC.lcpfn_fallback': o['lcpfn_rows']['fallback_rows'], 'appC.asha_runs': rep.attrs.get('asha_runs', scope['runs']),
        'appD.mech_p90_relu': P*mech.loc['ReLU', 'p90_error'], 'appD.mech_p90_gelu': P*mech.loc['GELU', 'p90_error'],
        'appD.mech_gate_relu': P*mech.loc['ReLU', 'median_gate_change'], 'appD.mech_gate_gelu': P*mech.loc['GELU', 'median_gate_change'],
        'appD.mech_checks': mech.loc['ReLU', 'bound_checks'],
        'appE.weights_best': P*o['table15'].set_index('weight').loc[.5, 'mean'],
        'appE.agg_AM': P*t16.loc['Arithmetic mean', 'mean'], 'appE.agg_GM': P*t16.loc['Geometric mean', 'mean'],
        'appE.agg_LUD': P*t16.loc['Log AM/GM', 'mean'],
        'appE.family_G': P*t17.loc[('family', 'G'), 'mean'], 'appE.family_extrap': P*t17.loc[('family', 'Extrapolation'), 'mean'],
        'appE.median_E': P*t18.loc[('Median seed', 'Evaluation'), 'mean'], 'appE.median_G': P*t18.loc[('Median seed', 'G'), 'mean'],
        'appE.worst_seed_E': P*t18.loc[('Worst seed', 'Evaluation'), 'mean'], 'appE.worst_seed_G': P*t18.loc[('Worst seed', 'G'), 'mean'],
        'appE.worst_seed_max_G': P*t18.loc[('Worst seed', 'G'), 'worst'], 'appE.worst_seed_max_E': P*t18.loc[('Worst seed', 'Evaluation'), 'worst'],
        'appE.incumbent_E': P*t18.loc[('Global grid incumbent', 'Evaluation'), 'mean'],
        'appE.incumbent_G': P*t18.loc[('Global grid incumbent', 'G'), 'mean'],
        'appE.lcpfn_best': P*t18.loc[('Best within budget', 'LC-PFN'), 'mean'],
        'appE.lcpfn_last': P*t18.loc[('Last within budget', 'LC-PFN'), 'mean'],
        'appE.lcpfn_last_G': P*t18.loc[('Last within budget', 'G'), 'mean'],
        'appE.top_two_share': P*u['top_two_fraction'], 'appE.rest_gain': P*u['rest_mean_gain'],
        'appE.rest_relative': P*u['rest_relative'], 'appE.sign_flip_assignments': 100_000,
    }
    for key, value in theory.items():
        v['theory.'+key] = value
    v['theory.regret_example'] = P*theory['regret_example']
    v['theory.relative_reduction_example'] = P*theory['relative_reduction_example']
    v['theory.floor_fraction'] = P*theory['floor_fraction']
    return v


def _first_evaluation_win(t):
    """Smallest reported prefix from which Evaluation has lower mean regret at every later reported prefix."""
    t = t[t.tau.ne(.50)].sort_values('tau')
    later = [row.tau for row in t.itertuples() if all(x.E < x.G for x in t[t.tau >= row.tau].itertuples())]
    return round(100*min(later))


def figures(o):
    task = o['task'].copy()
    task['id'] = task.experiment_id.map(lambda e: LABELS[e][0])
    task = task.sort_values('id', key=lambda s: s.map(_fig_order))
    fam = o['family'].set_index('family').loc[['cnn', 'mlp', 'nlp', 'rl', 'vit']]
    heat = []
    tk = o['tau_k'].set_index(['tau', 'k'])
    biggest = (P*tk['diff']).abs().max()
    for ti, tau in enumerate(PRIMARY_TAUS):
        for ki, kk in enumerate(KS):
            d = P*tk.loc[(tau, kk), 'diff']
            heat.append(dict(k_index=ki, tau_index=ti, colour='blue' if d >= 0 else 'warm',
                             intensity=7+75*abs(d)/biggest, value=d))
    return dict(fig2_task_gains=(P*task.gain).tolist(), fig2_family_marks=(P*fam.gain).tolist(),
                fig2_family_means=[[P*a, P*b] for a, b in zip(fam.E, fam.G)], fig3_cells=heat)


def _fig_order(i):
    return 'CMFPRV'.index(i[0])*10+int(i[1])


def compare_figures(gen, exp, errata, known):
    problems = []
    for key in ('fig2_task_gains', 'fig2_family_marks'):
        if len(gen[key]) != len(exp[key]):
            problems.append(f'{key}: length {len(gen[key])} vs {len(exp[key])}')
        for i, (a, b) in enumerate(zip(gen[key], exp[key])):
            name = f'{key}[{i}]'
            if number_matches(a, b):
                continue
            if name in errata and number_matches(a, errata[name]['reproduced']):
                known.append(dict(key=name, published=b, reproduced=errata[name]['reproduced'], note=errata[name]['note']))
            else:
                problems.append(f'{name}: reproduced {a:.4f} vs published {b}')
    for i, (a, b) in enumerate(zip(gen['fig2_family_means'], exp['fig2_family_means'])):
        for x, y in zip(a, b):
            if not number_matches(x, y):
                problems.append(f'fig2_family_means[{i}]: reproduced {x:.4f} vs published {y}')
    published = {(c['tau_index'], c['k_index']): c for c in exp['fig3_cells']}
    for c in gen['fig3_cells']:
        p = published[(c['tau_index'], c['k_index'])]
        if not number_matches(c['value'], p['label']):
            problems.append(f"fig3 cell {c['tau_index']},{c['k_index']}: value {c['value']:.4f} vs label {p['label']}")
        if abs(c['intensity']-p['intensity']) > 1e-3 or (c['colour'] != p['colour'] and abs(c['value']) > 5e-3):
            problems.append(f"fig3 cell {c['tau_index']},{c['k_index']}: shade {c['colour']}!{c['intensity']:.4f} vs {p['colour']}!{p['intensity']}")
    return problems


def verify(T, text, figs):
    exp_t = json.loads((EXPECTED/'tables.json').read_text(encoding='utf8'))['tables']
    exp_x = json.loads((EXPECTED/'text.json').read_text(encoding='utf8'))['values']
    exp_f = json.loads((EXPECTED/'figures.json').read_text(encoding='utf8'))
    errata = {e['key']: e for e in json.loads((EXPECTED/'errata.json').read_text(encoding='utf8'))['errata']}
    problems, checked = [], dict(tables=0, table_cells=0, text=0, figure_values=0)
    for label, rows in exp_t.items():
        problems += compare_rows(label, T[label], rows)
        checked['tables'] += 1
        checked['table_cells'] += sum(1 for r in T[label] for c in r if c is not None)
    known = []
    for e in exp_x:
        value = text[e['key']]
        ok = number_matches(value, e['published'])
        if not ok and e['key'] in errata and number_matches(value, errata[e['key']]['reproduced']):
            known.append(dict(key=e['key'], published=e['published'], reproduced=errata[e['key']]['reproduced'],
                              note=errata[e['key']]['note']))
        elif not ok:
            problems.append(f"text {e['key']}: reproduced {value!r} vs published {e['published']!r} ({e['context']})")
        checked['text'] += 1
    problems += compare_figures(figs, exp_f, errata, known)
    checked['figure_values'] = len(exp_f['fig2_task_gains'])+len(exp_f['fig2_family_marks'])+2*len(exp_f['fig2_family_means'])+len(exp_f['fig3_cells'])
    return problems, known, checked
