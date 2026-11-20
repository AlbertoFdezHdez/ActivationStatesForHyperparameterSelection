"""Load the supplied run-level records and rebuild the candidate tables.

Every selector sees one row per configuration and prefix: seed means of the
latest record at or before the cutoff, with G computed within each run first.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from .common import DATA, EPSILON, PROBE_TASKS, PRIMARY_TAUS

SEED_MEAN = ['external_signal', 'best_external_signal', 'linear_prediction', 'train_loss',
             'grad_norm', 'param_norm', 'update_norm', 'update_to_weight', 'rankme',
             'activation_sparsity', 'cumulative_steps', 'cumulative_wall_time',
             'final_objective_oriented', 'G', 'M', 'LUD']


@lru_cache(maxsize=None)
def runs():
    r = pd.read_parquet(DATA/'runs.parquet')
    probe = r.experiment_id.isin(PROBE_TASKS)
    # Recorded log(AM/GM) gives G = (AM + eps) exp(-log(AM/GM)) for probe profiles.
    probe_G = (r.mean_probe_oui+EPSILON)*np.exp(-r.neg_log_probe_C)
    r['G'] = np.where(probe, probe_G, r.geometric_mean_oui)
    r['M'] = np.where(probe, r.mean_probe_oui, r.mean_oui)            # arithmetic mean of U
    r['LUD'] = np.where(probe, r.neg_log_probe_C, r.neg_log_C)        # log AM/GM
    sign = np.where(r.objective_direction.eq('maximize'), -1.0, 1.0)  # orient terminal fields
    r['last_objective_within_budget'] = r.last_objective_within_budget*sign
    r['final_checkpoint_objective'] = r.final_checkpoint_objective*sign
    return r


@lru_cache(maxsize=None)
def eligible_configs():
    """Complete three-seed groups with probe records at every primary prefix."""
    r = runs()
    p = r[r.tau.isin(PRIMARY_TAUS)]
    ok = p.groupby('config_id').apply(
        lambda g: g.seed.nunique() == 3 and len(g) == 12 and np.isfinite(g.mean_probe_oui).all(),
        include_groups=False)
    return frozenset(ok[ok].index)


@lru_cache(maxsize=None)
def configs(taus=PRIMARY_TAUS):
    """Configuration x prefix candidate table on the strict eligible pool."""
    r = runs()
    r = r[r.config_id.isin(eligible_configs()) & r.tau.isin(taus)]
    g = r.groupby(['experiment_id', 'family', 'config_id', 'lr', 'wd', 'tau'], dropna=False)
    d = g[SEED_MEAN].mean()
    d['seeds'] = g.seed.nunique()
    d['final_objective_mean_seed'] = d.pop('final_objective_oriented')
    d['final_objective_median_seed'] = g.final_objective_oriented.median()
    d['final_objective_worst_seed'] = g.final_objective_oriented.max()
    d['last_objective_within_budget'] = g.last_objective_within_budget.mean()
    d['final_checkpoint_objective'] = g.final_checkpoint_objective.mean()
    d = d.reset_index()
    # Oracle-metric incumbent observed up to each prefix (global-incumbent variant).
    a = runs().sort_values(['run_id', 'tau']).copy()
    a['prefix_incumbent'] = a.groupby('run_id').objective_oriented.cummin()
    inc = a.groupby(['config_id', 'tau']).prefix_incumbent.mean().rename('prefix_incumbent')
    d = d.merge(inc.reset_index(), on=['config_id', 'tau'], how='left', validate='one_to_one')
    assert (d.seeds == 3).all()
    return d.sort_values(['experiment_id', 'tau', 'config_id']).reset_index(drop=True)


@lru_cache(maxsize=None)
def lcpfn():
    p = pd.read_csv(DATA/'lcpfn_predictions.csv')
    p['config_id'] = p.run_id.str.rsplit('|seed=', n=1).str[0]
    return p.groupby(['config_id', 'tau']).agg(
        lcpfn_prediction=('lcpfn_prediction', 'mean'), fallback_seeds=('fallback', 'sum'),
        prediction_seeds=('run_id', 'nunique')).reset_index()


def primary():
    """Primary candidate table with the LC-PFN forecasts attached."""
    d = configs().merge(lcpfn(), on=['config_id', 'tau'], how='left', validate='one_to_one')
    assert len(d) == 1472 and d.config_id.nunique() == 368 and d.experiment_id.nunique() == 23
    assert d.prediction_seeds.eq(3).all()
    return d
