"""Experiment-level uncertainty, task tests and influence (Section 5.5, Table 19)."""
import numpy as np
from scipy.stats import bootstrap, t, wilcoxon, binomtest

SEED = 20260926
DRAWS = 50_000
FLIPS = 100_000


def relative(e, g, axis=-1):
    return (e.mean(axis)-g.mean(axis))/e.mean(axis)


def run(task):
    """``task`` holds one row per experiment with mean regrets E and G over its 16 decisions."""
    e, g = task.E.to_numpy(float), task.G.to_numpy(float)
    gains = e-g
    n = len(gains)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, (DRAWS, n))
    abs_draws = gains[idx].mean(1)
    rel_draws = relative(e[idx], g[idx], 1)
    boot = lambda data, stat, paired=False: bootstrap(
        data, stat, paired=paired, vectorized=True, method='BCa', confidence_level=.95,
        n_resamples=DRAWS, rng=np.random.default_rng(SEED)).confidence_interval
    bca_abs = boot((gains,), lambda x, axis: x.mean(axis))
    bca_rel = boot((e, g), relative, paired=True)
    se = gains.std(ddof=1)/np.sqrt(n)
    tc = t.ppf(.975, n-1)
    # Delta method for (mean E - mean G)/mean E, including the covariance with mean E.
    ratio = relative(e, g)
    grad = np.array([g.mean()/e.mean()**2, -1/e.mean()])
    cov = np.cov(np.vstack([e, g]), ddof=1)/n
    se_rel = float(np.sqrt(grad@cov@grad))
    signs = rng.choice([-1, 1], size=(FLIPS, n))
    flip_p = (1+np.count_nonzero(np.abs((signs*gains).mean(1)) >= abs(gains.mean())))/(FLIPS+1)
    loo = np.array([np.delete(gains, i).mean() for i in range(n)])
    loo_rel = np.array([relative(np.delete(e, i), np.delete(g, i)) for i in range(n)])
    order = np.argsort(-gains, kind='stable')
    top = order[:2]
    rest = np.setdiff1d(np.arange(n), top)
    return dict(
        tasks=n, mean_gain=float(gains.mean()), relative_gain=float(ratio),
        percentile_abs=np.quantile(abs_draws, [.025, .975]).tolist(),
        percentile_rel=np.quantile(rel_draws, [.025, .975]).tolist(),
        bca_abs=[float(bca_abs.low), float(bca_abs.high)],
        bca_rel=[float(bca_rel.low), float(bca_rel.high)],
        student_abs=[float(gains.mean()-tc*se), float(gains.mean()+tc*se)],
        student_rel=[float(ratio-tc*se_rel), float(ratio+tc*se_rel)],
        improved=int((gains > 0).sum()), worsened=int((gains < 0).sum()), tied=int((gains == 0).sum()),
        worsened_below_001pp=int(((gains < 0) & (gains > -1e-4)).sum()),
        sign_test_p=float(binomtest(int((gains > 0).sum()), int((gains != 0).sum()), .5).pvalue),
        wilcoxon_p=float(wilcoxon(gains).pvalue), sign_flip_p=float(flip_p),
        loo_abs=[float(loo.min()), float(loo.max())], loo_rel=[float(loo_rel.min()), float(loo_rel.max())],
        top_two=task.experiment_id.to_numpy()[top].tolist(),
        top_two_fraction=float(gains[top].sum()/gains.sum()),
        rest_mean_gain=float(gains[rest].mean()), rest_relative=float(relative(e[rest], g[rest])),
        rest_improved=int((gains[rest] > 0).sum()), rest_worsened=int((gains[rest] < 0).sum()),
        rest_tied=int((gains[rest] == 0).sum()))
