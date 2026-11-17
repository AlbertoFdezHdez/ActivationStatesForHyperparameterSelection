"""Benchmark inventory: grids (Table 11), catalogs (Tables 8-10), scope counts and costs."""
import gzip
import json
import numpy as np
import pandas as pd
import yaml
from .common import DATA, LABELS
from . import data


def inventory():
    return pd.read_csv(DATA/'run_inventory.csv.gz')


def configurations():
    rows = []
    with gzip.open(DATA/'run_configurations.jsonl.gz', 'rt', encoding='utf8') as f:
        for line in f:
            z = json.loads(line)
            text = z['configuration_text']
            c = yaml.safe_load(text)  # JSON is valid YAML
            rows.append(dict(run_id=z['run_id'], experiment_id=z['run_id'].split('|')[0], config=c,
                             sha256=z['sha256'], text=text))
    return rows


def _value(v):
    return float(v) if not isinstance(v, (int, float)) else v


def grids():
    """Observed LR/WD levels and nominal budget per experiment."""
    inv = inventory()
    rows = []
    for exp, g in inv.groupby('experiment_id'):
        rl = exp.startswith('rl:')
        rows.append(dict(id=LABELS[exp][0], experiment_id=exp,
                         lrs=', '.join(str(x) for x in sorted(g.lr.unique())),
                         wds='N/A' if rl else ', '.join(str(x).rstrip('0').rstrip('.') if x else '0' for x in sorted(g.wd.unique())),
                         budget=int(g.total_budget.iloc[0]), budgets=g.total_budget.nunique(),
                         seeds=sorted(g.seed.unique().tolist()), runs=len(g)))
    t = pd.DataFrame(rows).sort_values('id').reset_index(drop=True)
    rl_grid = sorted(inv[inv.experiment_id.str.startswith('rl:')].lr.unique())
    return t, rl_grid


def monitored_layers():
    lp = pd.read_parquet(DATA/'layer_profiles.parquet')
    n = lp.groupby(['experiment_id', 'run_id', 'tau']).layer_id.nunique()
    per = n.groupby(level=0).agg(['min', 'max'])
    assert (per['min'] == per['max']).all(), 'monitored-layer count must be constant within a task'
    return per['min']


def catalog():
    """Fields of Tables 8-10 read from the resolved configurations of all 1,104 runs."""
    L = monitored_layers()
    rows = {}
    for z in configurations():
        exp, c = z['experiment_id'], z['config']
        fam = exp.split(':')[0]
        if fam in ('cnn', 'vit', 'mlp'):
            k, t = c['config'], c['config']['training']
            row = dict(epochs=t['epochs'], bs=t['effective_batch_size'])
            if fam == 'mlp':
                row.update(hidden=tuple(k['model']['hidden_dims']), dropout=tuple(k['model']['dropouts']))
            else:
                row.update(input=k['data']['image_size'])
        elif fam == 'nlp':
            t = c['training']
            length = t.get('sequence_length', t.get('max_sequence_length'))
            if length is None:
                length = f"{t['max_source_length']}/{t['max_target_length']}"
            row = dict(steps=t['max_steps'], bs=t['batch_size'], ga=t['gradient_accumulation_steps'],
                       length=str(length), betas=tuple(t['betas']), clip=t['gradient_clip'],
                       warmup=t['warmup_ratio'], scheduler=t['scheduler'])
        else:
            k = c['config']
            row = dict(algorithm=k['algorithm'], env=k['env']['id'], timesteps=k['training']['total_timesteps'],
                       probe=k['tracking']['probe_states'], episodes=k['tracking']['eval_episodes'])
        row = {key: _value(v) if key in ('epochs', 'bs', 'steps', 'ga', 'timesteps', 'probe', 'episodes') else v
               for key, v in row.items()}
        rows.setdefault(exp, []).append(row)
    out = []
    for exp, values in rows.items():
        first = values[0]
        assert all(v == first for v in values), f'catalog fields vary within {exp}'
        out.append(dict(id=LABELS[exp][0], experiment_id=exp, runs=len(values),
                        L=int(L[exp]) if exp in L.index else None, **first))
    return pd.DataFrame(out).sort_values('id').reset_index(drop=True)


def scope():
    r = data.runs()
    inv = inventory()
    return dict(experiments=r.experiment_id.nunique(), runs=r.run_id.nunique(),
                nominal_configurations=r.config_id.nunique(), eligible_configurations=len(data.eligible_configs()),
                complete_runs=int(inv.complete.sum()), failed_runs=int(inv.failed.sum()),
                language_configurations=len([z for z in configurations() if z['experiment_id'].startswith('nlp:')]))


def costs():
    """Measurement time as a fraction of elapsed training time at the 5% cutoff."""
    r = data.runs()
    x = r[r.tau.eq(.05)].dropna(subset=['probe_oui_wall_fraction'])
    task = x.groupby('experiment_id').probe_oui_wall_fraction.median()
    return dict(runs=len(x), median=float(x.probe_oui_wall_fraction.median()),
                p90=float(x.probe_oui_wall_fraction.quantile(.9)),
                largest_task=task.idxmax(), largest_task_median=float(task.max()))
