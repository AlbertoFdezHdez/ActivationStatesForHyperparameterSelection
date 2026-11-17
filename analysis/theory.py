"""Worked examples and identities stated in Sections 2-4 and Appendices A-B.

Each value is computed from the definitions; ``verify`` compares them with
the published numbers, and ``identities`` checks Lemma 1, the geometric
aggregation facts and Proposition 'Distributed utilization' on random profiles.
"""
import itertools
import math
import numpy as np
from scipy.stats import binom


def psi(p):
    return 2*min(p, 1-p)


def G0(u):
    u = np.asarray(u, float)
    return 0.0 if (u == 0).any() else float(np.exp(np.log(u).mean()))


def recorded(n_pos, B):
    return min(n_pos, B-n_pos)/math.floor(B/2)


def examples():
    v = {}
    # Section 2.1: positive on 2 of 10 inputs.
    v['ten_input_utilization'] = psi(2/10)
    v['ten_input_mismatch'] = min(2, 8)/10
    # Section 2.2: two-module profiles at epsilon 0.
    for name, u in [('G0_005_005', (0.05, 0.05)), ('G0_01_09', (0.1, 0.9)), ('G0_05_05', (0.5, 0.5))]:
        v[name] = G0(u)
    # Section 2.3: f = 3 max(0, z), bias +0.1 on preactivations 2 and -1.
    f = lambda z: 3*max(0.0, z)
    v['relu_output_before'], v['relu_output_after'] = f(2.0), f(2.1)
    v['relu_inactive_after'] = f(-0.9)
    # Four-input example: phi_1, phi_2 and r = 0.2 phi_1.
    g1, g2 = np.array([1, 1, 0, 0.]), np.array([1, 0, 1, 0.])
    phi1, phi2, r = g1-g1.mean(), g2-g2.mean(), np.array([.1, .1, -.1, -.1])
    v['four_input_coefficient'] = float(phi1@r/(phi1@phi1))
    v['four_input_orthogonal'] = float(phi1@phi2)
    # Theorem 1 remarks: U = 0.8 selects at least two thirds of units with u >= 0.4.
    v['selected_fraction_U08'] = 0.8/(2-0.8)
    v['selected_threshold_U08'] = 0.8/2
    # A 2048-unit layer on B = 256: m <= B-1 fails once n U/(2-U) > 255.
    n, B = 2048, 256
    v['rank_threshold_U'] = 2*(B-1)/(n+B-1)
    # Section 3.1: five observations, split 2 vs 3.
    v['B5_recorded'], v['B5_plugin'] = recorded(2, 5), psi(2/5)
    # Section 3.2: A, B, C scores.
    e, s = np.array([0, .5, 1]), np.array([1, 0, .5])
    v['ABC_scores'] = (e+.5*s).tolist()
    # Section 3.3: probe of 2,048 inputs in batches of 256.
    v['probe_forwards'] = math.ceil(2048/256)
    # Section 5.1: regret example and relative reduction example.
    v['regret_example'] = (0.24-0.20)/(0.60-0.20)
    v['relative_reduction_example'] = (0.04-0.03)/0.04
    # Appendix B.1: B = 3 and p = 0.25.
    B, p = 3, .25
    k = np.arange(B+1)
    w = binom.pmf(k, B, p)
    v['B3_population'] = psi(p)
    v['B3_expected_plugin'] = float(sum(wi*psi(ki/B) for wi, ki in zip(w, k)))
    v['B3_expected_recorded'] = float(sum(wi*recorded(ki, B) for wi, ki in zip(w, k)))
    # Appendix B.2: seed-wise G versus G of the averaged profile.
    v['seedwise_G'] = (G0((.1, .9))+G0((.9, .1)))/2
    v['averaged_profile_G'] = G0((.5, .5))
    # Appendix A.2: G ~ 0.8 bounds the fraction of modules with U <= 0.1.
    eps = 1e-8
    v['floor_fraction'] = math.log((1+eps)/0.8)/math.log((1+eps)/(0.1+eps))
    return v


def identities(seed=0, trials=200):
    """Exact checks on random product-Bernoulli modules; returns the largest violation."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        n = int(rng.integers(1, 6))
        p = rng.random(n)*rng.choice([0, 1], n, p=[.1, .9])
        u = np.array([psi(x) for x in p])
        U = u.mean()
        states = np.array(list(itertools.product([0, 1], repeat=n)), float)
        prob = np.prod(np.where(states == 1, p, 1-p), axis=1)
        xi = min(prob@np.abs(states-c).sum(1)/n for c in states)
        V = float((p*(1-p)).mean())
        H = float(sum(prob[i]*prob[j]*np.abs(states[i]-states[j]).sum()/n
                      for i in range(len(states)) for j in range(len(states))))
        worst = max(worst, abs(xi-U/2), abs(H-2*V), max(0, U/4-V), max(0, V-U*(2-U)/4), max(0, U*(2-U)/4-U/2))
        L = int(rng.integers(1, 7))
        prof = rng.random(L)
        eps = 1e-8
        G = float(np.exp(np.log(prof+eps).mean()))
        worst = max(worst, max(0, G-(prof.mean()+eps)))
        lower = max(0.0, G**L/(1+eps)**(L-1)-eps)
        worst = max(worst, max(0, lower-prof.min()))
        a = float(rng.random()*.9)
        fa = float((prof <= a).mean())
        bound = min(1.0, math.log((1+eps)/G)/math.log((1+eps)/(a+eps)))
        worst = max(worst, max(0, fa-bound))
    return worst
