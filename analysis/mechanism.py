"""Summary of the controlled functional-mechanism study (Table 2, Appendix D).

``data/mechanism/measurements.csv`` holds the 144 checkpoint measurements
written by ``mechanism_study/local_mechanism.py`` (CPU, about 30 s; requires torch).
"""
import pandas as pd
from .common import DATA


def summarize(measurements=None):
    m = pd.read_csv(measurements or DATA/'mechanism/measurements.csv')
    rows = []
    for activation, g in m.groupby('activation'):
        positive = g[g.predicted_gain > 1e-10]
        relative = (positive.actual_gain-positive.predicted_gain).abs()/positive.predicted_gain
        rows.append(dict(
            activation=activation.upper().replace('RELU', 'ReLU'), checkpoints=len(g),
            correlation=float(g.predicted_gain.corr(g.actual_gain)),
            median_error=float(relative.median()), p90_error=float(relative.quantile(.9)),
            positive_gain=int(g.actual_gain.gt(0).sum()), median_gate_change=float(g.gate_change_fraction.median()),
            bound_checks=int(g.relu_bound_coordinates.sum()), bound_violations=int(g.relu_bound_violations.sum()),
            module_checks=int(g.module_certificate_checks.sum()),
            module_violations=int(g.module_certificate_violations.sum()),
            max_output_weight_scaling_error=float(g.transmission_scaling_error.max())))
    return pd.DataFrame(rows), len(m)
