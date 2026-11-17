"""Fixed protocol constants and the primitives shared by every analysis.

Historical field names are kept in the data files: ``geometric_mean_oui``,
``mean_oui`` and ``*_probe_oui`` hold the module utilizations U and their
geometric mean G of the article (OUI was the name of the earlier indicator).
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
OUT = ROOT/'outputs'

EPSILON = 1e-8                      # numerical constant inside G
PRIMARY_WEIGHT = 0.5                # lambda of Eq. (primary)
PRIMARY_TAUS = (0.05, 0.10, 0.15, 0.20)
ALL_TAUS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.50, 0.60, 0.75, 1.0)
KS = (1, 3, 5, 10)
GRID = (0.0, 1/8, 1/4, 1/3, 1/2, 2/3, 3/4, 1.0)   # common coefficient grid
SWEEP = GRID + (3/2, 2.0)                          # Table 15
FAMILIES = ('cnn', 'mlp', 'nlp', 'rl', 'vit')
# R3-R6 measure utilization on a fixed probe collected before training.
PROBE_TASKS = {'rl:r3_dqn_minatar_breakout', 'rl:r4_dqn_minatar_seaquest',
               'rl:r5_sac_halfcheetah', 'rl:r6_sac_hopper'}
# C3, C4, V1 and V3 use the official test split in the early-evaluation role.
TEST_SPLIT_TASKS = {'cnn:c3_efficientnet_b0_stanford_cars', 'cnn:c4_mobilenetv2_food101',
                    'vit:v1_vit_tiny16_cifar100', 'vit:v3_mobilevit_xs_pets'}
LABELS = {  # experiment ID -> article label (Tables 8-12)
    'cnn:c1_resnet34_imagenet100': ('C1', 'ResNet-34 / ImageNet-100'),
    'cnn:c2_densenet121_tinyimagenet': ('C2', 'DenseNet-121 / Tiny ImageNet'),
    'cnn:c3_efficientnet_b0_stanford_cars': ('C3', 'EfficientNet-B0 / Stanford Cars'),
    'cnn:c4_mobilenetv2_food101': ('C4', 'MobileNetV2 / Food-101'),
    'mlp:m1_fashionmnist_mlp': ('M1', 'MLP / Fashion-MNIST'),
    'mlp:m2_cifar10_mlp': ('M2', 'MLP / CIFAR-10'),
    'mlp:m3_uci_har_mlp': ('M3', 'MLP / UCI HAR'),
    'mlp:m4_covertype_mlp': ('M4', 'MLP / Covertype'),
    'nlp:f1_roberta_mnli': ('F1', 'RoBERTa / MNLI'),
    'nlp:f2_bart_xsum': ('F2', 'BART / XSum'),
    'nlp:f3_pythia_pubmed': ('F3', 'Pythia / PubMed'),
    'nlp:p1_llama_c4': ('P1', 'LLaMA / C4'),
    'nlp:p2_nanogpt_wikitext103': ('P2', 'nanoGPT / WikiText-103'),
    'nlp:p3_t5_bookcorpus': ('P3', 'T5 / BookCorpus'),
    'rl:r1_ppo_lunarlander': ('R1', 'PPO / LunarLander'),
    'rl:r2_ppo_minigrid_doorkey': ('R2', 'PPO / MiniGrid DoorKey'),
    'rl:r3_dqn_minatar_breakout': ('R3', 'DQN / MinAtar Breakout'),
    'rl:r4_dqn_minatar_seaquest': ('R4', 'DQN / MinAtar Seaquest'),
    'rl:r5_sac_halfcheetah': ('R5', 'SAC / HalfCheetah'),
    'rl:r6_sac_hopper': ('R6', 'SAC / Hopper'),
    'vit:v1_vit_tiny16_cifar100': ('V1', 'ViT-Tiny/16 / CIFAR-100'),
    'vit:v2_swin_tiny_flowers102': ('V2', 'Swin-Tiny / Flowers-102'),
    'vit:v3_mobilevit_xs_pets': ('V3', 'MobileViT-XS / Oxford Pets'),
}
REGRET_STABILITY = 1e-12            # denominator constant of range regret


def rank(x):
    """Normalized rank in [0, 1]; ties share their average rank."""
    x = np.asarray(x, float)
    return (rankdata(x)-1)/max(1, len(x)-1)


def order(ids, score):
    """Ascending score, ties broken by stable configuration ID."""
    return np.lexsort((np.asarray(ids, str), np.asarray(score, float)))


def regret(y, chosen):
    """Range-normalized regret of the best retained candidate."""
    y = np.asarray(y, float)
    return (float(y[chosen].min())-float(y.min()))/(float(y.max()-y.min())+REGRET_STABILITY)


def cvar(values, q):
    v = np.sort(np.asarray(values, float))
    return float(v[-math.ceil(len(v)*q):].mean())


def stats(values):
    v = np.asarray(values, float)
    return dict(n=len(v), mean=float(v.mean()), cvar25=cvar(v, .25), cvar5=cvar(v, .05), worst=float(v.max()))


def summarize(cells, keys):
    rows = []
    for key, g in cells.groupby(keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(keys, key), **stats(g.regret), tasks=g.experiment_id.nunique()))
    return pd.DataFrame(rows)


def profile_G(u, epsilon=EPSILON):
    """G of one run's module profile; the zero-epsilon value is 0 if a module is 0."""
    u = np.asarray(u, float)
    assert u.size and np.isfinite(u).all() and ((u >= 0) & (u <= 1)).all()
    if epsilon == 0 and (u == 0).any():
        return 0.0
    return float(np.exp(np.log(u+epsilon).mean()))
