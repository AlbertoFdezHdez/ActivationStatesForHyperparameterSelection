# Reading Activation States for Early Hyperparameter Selection — reproducibility package

This repository accompanies the article *Reading Activation States for Early Hyperparameter
Selection* (Fernández-Hernández, Mestre, Pérez-Corral, Dolz and Quintana-Ortí; submitted to
*Neural Networks*). It contains the benchmark records, the analysis programs and the expected
values needed to regenerate **every table, figure series and in-text number of the article**,
and the training programs and resolved configurations of all 1,104 benchmark runs.

For each monitored hidden coordinate, the article counts how often its response is positive
across inputs, measures its utilization `u = 2 min(p, 1 - p)`, averages it within each monitored
module (`U`) and takes the geometric mean across modules (`G`). The fixed selector ranks
candidates by early evaluation and adds, with weight 1/2, their rank by `G` (higher `G` ranks
better). The benchmark compares early choices with the best configuration available after full
training over 23 experiments, five learning families and 368 continuation decisions.

## Quick start (CPU, about one minute)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
.venv/bin/python reproduce.py                    # Windows: .venv\Scripts\python
```

`reproduce.py` checks the SHA-256 digests of the inputs, rebuilds the candidate tables available
at each cutoff, the seed groups and every selector decision, recomputes all controls, coefficient,
epsilon, aggregation, transfer, objective and uncertainty analyses, and compares each value with
the number printed in the article. It ends with `ALL CHECKS PASSED` (exit status 0) and writes
`outputs/verification.json`, the regenerated tables in `outputs/tables/` and the full-precision
values in `outputs/text_values.json`, `outputs/figures.json` and `outputs/uncertainty.json`.

Comparison rule: each recomputed value, rounded to the decimals printed in the article, must equal
the printed string (for example `.2f` for regrets, `.3f` for the mechanism correlations). The
expected values in `expected/` were extracted from the article source; every in-text number is
stored with the sentence that prints it. Bold face and layout are not compared.

Optional modes:

| Command | What it adds | Time |
|---|---|---|
| `python reproduce.py --replay` | Recomputes all paired promotion repetitions (ASHA: 100 arrival orders × 4/8/12 workers; Hyperband and BOHB-style: 200 streams each) from the run records instead of aggregating the supplied cells, and checks they are identical | ~8 min |
| `python reproduce.py --mechanism` | Runs the controlled mechanism study of Appendix D (needs `torch`, CPU) | ~30 s |

## What reproduces what

| Article item | Content | Program | Inputs |
|---|---|---|---|
| Table 1 | Monitored tensors by family | descriptive; hook patterns are in the resolved configurations | `data/run_configurations.jsonl.gz` |
| Table 2, App. D | Controlled local adaptation (Pearson 0.998/0.996, errors, checks) | `analysis/mechanism.py` | `data/mechanism/measurements.csv` (written by `mechanism_study/local_mechanism.py`) |
| Table 3 | Complete-support selection, 368 decisions | `analysis/oneshot.py` | `data/runs.parquet`, `data/lcpfn_predictions.csv` |
| Table 4 | Matched-support controls (17 and 11 tasks) | `analysis/oneshot.py` | `data/runs.parquet` |
| Table 5 | Paired Hyperband, BOHB-style (incl. Eval. + extrapolation) and ASHA replays | `analysis/replays.py` | `data/promotion_cells/*.csv`; with `--replay`, `data/runs.parquet` |
| Figure 1 | Pipeline schematic | — (no data) | — |
| Figure 2 | Per-task gain and family means | `analysis/oneshot.py`, `analysis/report.py` | `data/runs.parquet` |
| Figure 3 | Prefix × K mean-regret differences | `analysis/oneshot.py`, `analysis/report.py` | `data/runs.parquet` |
| Table 6 | Epsilon sensitivity | `analysis/oneshot.py` (`epsilon_table`) | `data/layer_profiles.parquet` |
| Table 7 | CNN global-pooling hook check | `analysis/oneshot.py` (`cnn_hooks`) | `data/layer_profiles.parquet` |
| Tables 8–10 | Experiment catalogs (epochs, batch, length, steps, monitored layers, probes) | `analysis/benchmark.py` (`catalog`) | `data/run_configurations.jsonl.gz`, `data/layer_profiles.parquet` |
| Table 11 | Observed grids and budgets | `analysis/benchmark.py` (`grids`) | `data/run_inventory.csv.gz` |
| Table 12 | Experiment-level regret | `analysis/oneshot.py` | `data/runs.parquet` |
| Table 13 | All prefix–K cells | `analysis/oneshot.py` | `data/runs.parquet` |
| Table 14 | Additional training prefixes | `analysis/oneshot.py` (`temporal`) | `data/runs.parquet` |
| Table 15 | Coefficient sweep, changed sets, wins, losses | `analysis/oneshot.py` | `data/runs.parquet` |
| Table 16 | Aggregation sensitivity | `analysis/oneshot.py` | `data/runs.parquet` |
| Table 17 | Held-out coefficient transfer (task, family) | `analysis/oneshot.py` | `data/runs.parquet` |
| Table 18 | Objective robustness, global incumbent | `analysis/oneshot.py` | `data/runs.parquet` |
| Table 19, §5.5 | Bootstrap (percentile, BCa), Student/delta, sign, Wilcoxon, sign-flip, leave-one-out, top-two influence | `analysis/uncertainty.py` | per-task means from `analysis/oneshot.py` |
| §4.1, §4.3 | Learning-rate prior control, correlation of $G$ with learning rate and weight decay, Evaluation + LC-PFN, median-normalized regret | `analysis/controls.py` | `data/runs.parquet`, `data/lcpfn_predictions.csv` |
| In-text numbers | 19-task validation-only check, rank correlations, costs, counts, worked examples | `analysis/report.py`, `analysis/benchmark.py`, `analysis/theory.py` | as above |

## Data

| File | Rows | Content |
|---|---|---|
| `data/runs.parquet` | 13,248 | One row per run (1,104) and requested fidelity (12): latest signal at or before the cutoff, module-utilization summaries, optimization diagnostics, cumulative steps and wall time, terminal objectives, probe summaries and measurement-time ratio |
| `data/layer_profiles.parquet` | — | Module utilizations `U_l` per run at the four primary prefixes (training observations; fixed probe for R3–R6) |
| `data/lcpfn_predictions.csv` | 4,416 | LC-PFN forecasts per run and primary prefix, with fallback flags (precomputed with the pretrained LC-PFN model) |
| `data/promotion_cells/` | — | Paired promotion outcomes for every repetition (regenerated exactly by `--replay`) |
| `data/mechanism/` | 144 | Mechanism-study measurements and protocol |
| `data/run_inventory.csv.gz` | 1,104 | Run inventory: grid values, budgets, completion, objective fields, recorded checkpoint arrays, configuration digests |
| `data/run_configurations.jsonl.gz` | 1,104 | Exact recorded configuration text of every run, its original relative path and SHA-256 |

Objectives are oriented for minimization (rewards are negated). The 1,104 runs are 368 nominal
configurations × three seeds; all 368 three-seed configurations have utilization records at every
primary prefix and form the strict comparison pool.

Historical field names: columns such as `mean_oui`, `geometric_mean_oui` and `*_probe_oui`, and the
training package `code/oui`, keep the name of the earlier Overfitting–Underfitting Indicator from
which the batch statistic derives; they hold the module utilization `U` and its geometric mean `G`.

## Training

`code/` contains the executed training programs of the five families and the shared activation
statistic (`code/oui`). `train_run.py` prepares any archived run from its recorded configuration:

```bash
python train_run.py --list
python train_run.py --run-id "mlp:m2_cifar10_mlp|lr=0.0001|wd=0|seed=0"            # print only
python train_run.py --run-id "mlp:m2_cifar10_mlp|lr=0.0001|wd=0|seed=0" --execute  # train
```

Training needs the packages in `requirements-training.txt`, a suitable accelerator and the datasets
named in each configuration (not redistributed here). Offline verification does not require training.

## Layout

```
reproduce.py            entry point of the offline verification
train_run.py            prepares or runs one archived training run
analysis/               analysis package (selection, replays, uncertainty, catalog, report)
mechanism_study/        CPU study of Appendix D
data/                   tabular inputs (hashed in MANIFEST.sha256.json)
expected/               numbers extracted from the article
code/                   training programs per family and the shared activation statistic
outputs/                written by reproduce.py
```

## Citation and license

See `CITATION.cff`. Code is released under the MIT License and data under CC BY 4.0 (`LICENSE`).
