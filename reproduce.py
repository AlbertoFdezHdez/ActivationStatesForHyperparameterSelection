"""Rebuild every table, figure series and in-text number of the article and check it.

    python reproduce.py               # offline analyses from the supplied tables (~1 min, CPU)
    python reproduce.py --replay      # also recompute all paired promotion repetitions (~8 min)
    python reproduce.py --mechanism   # also run the small CPU mechanism study (needs torch)

Exit status 0 means every compared value matches the article at its printed precision.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analysis import benchmark, mechanism, oneshot, replays, report, theory, uncertainty  # noqa: E402
from analysis.common import OUT  # noqa: E402


def check_inputs():
    manifest = json.loads((ROOT/'MANIFEST.sha256.json').read_text(encoding='utf8'))
    bad = [name for name, digest in manifest.items()
           if name.startswith(('data/', 'expected/')) and
           hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest]
    missing = [name for name in manifest if not (ROOT/name).is_file()]
    return bad, missing


def jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (frozenset, set)):
        return sorted(x)
    raise TypeError(type(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--replay', action='store_true', help='recompute the 100/200 paired promotion repetitions')
    parser.add_argument('--mechanism', action='store_true', help='run the CPU mechanism study (requires torch)')
    args = parser.parse_args()
    start = time.time()
    (OUT/'tables').mkdir(parents=True, exist_ok=True)
    status = {}

    bad, missing = check_inputs()
    status['input_hashes'] = dict(ok=not bad and not missing, mismatched=bad, missing=missing)
    print('[1/6] input hashes', 'ok' if status['input_hashes']['ok'] else f'FAILED {bad+missing}', flush=True)

    o = oneshot.run()
    u = uncertainty.run(o['task'])
    print('[2/6] one-shot selection, controls, robustness and uncertainty', flush=True)

    cells = {mode: replays.supplied(mode) for mode in ('asha', 'hyperband', 'bohb')}
    if args.replay:
        status['replay'] = {}
        for mode in ('hyperband', 'bohb', 'asha'):
            t = time.time()
            new = replays.recompute(mode)
            keys = [c for c in ('experiment_id', 'repetition', 'workers', 'method') if c in new]
            m = new.merge(cells[mode], on=keys, suffixes=('', '_supplied'), validate='one_to_one')
            diff = float(max(np.abs(m.regret-m.regret_supplied).max(), np.abs(m.cost_normalized-m.cost_normalized_supplied).max()))
            status['replay'][mode] = dict(rows=len(new), matched=len(m), max_abs_difference=diff, seconds=round(time.time()-t, 1))
            new.to_csv(OUT/f'replayed_{mode}_cells.csv', index=False)
            cells[mode] = new
            print(f'      {mode}: {len(new)} replays, max |difference| vs supplied cells {diff:.2e}', flush=True)
    rep = replays.summarize(cells)
    rep.attrs.update(bracket_reps=int(cells['hyperband'].repetition.nunique()),
                     asha_reps=int(cells['asha'].repetition.nunique()),
                     asha_runs=int(cells['asha'][cells['asha'].repetition.eq(0) & cells['asha'].workers.eq(4) &
                                                cells['asha'].method.eq('Evaluation')].n_runs.sum()))
    print('[3/6] promotion replays', 'recomputed' if args.replay else 'aggregated from supplied cells', flush=True)

    measurements = None
    if args.mechanism:
        import subprocess
        target = OUT/'mechanism_run'
        subprocess.run([sys.executable, str(ROOT/'mechanism_study/local_mechanism.py'), '--output', str(target)], check=True)
        measurements = target/'measurements.csv'
    mech, n_mech = mechanism.summarize(measurements)
    print('[4/6] mechanism study', 'recomputed' if args.mechanism else 'summarized from supplied measurements', flush=True)

    grids, rl_grid = benchmark.grids()
    catalog = benchmark.catalog()
    scope = benchmark.scope()
    cost = benchmark.costs()
    ex = theory.examples()
    worst_identity = theory.identities()
    status['theory_identities'] = dict(ok=worst_identity < 1e-9, largest_violation=worst_identity)
    print('[5/6] benchmark catalog, measurement costs and worked examples', flush=True)

    T = report.tables(o, u, rep, mech, grids, rl_grid, catalog)
    text = report.text_values(o, u, rep, mech, cost, scope, o['epsilon_counts'], ex)
    figs = report.figures(o)
    problems, checked = report.verify(T, text, figs)
    # RL learning-rate grid is printed in the text of Appendix C.2.
    rl_text = ', '.join(str(x) for x in rl_grid)
    if rl_text != '1e-05, 1.5e-05, 2.2e-05, 3.2e-05, 4.6e-05, 6.8e-05, 0.0001, 0.00015, 0.00022, 0.00032, 0.00046, 0.00068, 0.001, 0.0015, 0.0022, 0.003':
        problems.append('RL learning-rate grid differs from Appendix C.2')
    if scope['experiments'] != 23 or scope['runs'] != 1104 or scope['eligible_configurations'] != 368 or len(o['pairs']) != 368:
        problems.append(f'scope differs: {scope}')
    if not status['input_hashes']['ok']:
        problems.append('input hashes differ from MANIFEST.sha256.json')
    if not status['theory_identities']['ok']:
        problems.append('a theoretical identity check failed')
    if args.replay and any(v['max_abs_difference'] > 1e-10 or v['rows'] != v['matched'] for v in status['replay'].values()):
        problems.append('replayed promotion cells differ from the supplied cells')

    for label, rows in T.items():
        pd.DataFrame(rows).to_csv(OUT/'tables'/(label.replace(':', '_')+'.csv'), index=False, header=False)
    for name in ('table3', 'table4', 'table6', 'table7', 'table14', 'table15', 'table16', 'table17', 'table18',
                 'task', 'family', 'tau_k', 'transfer_choices'):
        o[name].to_csv(OUT/'tables'/f'{name}_full_precision.csv', index=False)
    rep.to_csv(OUT/'tables'/'table5_full_precision.csv', index=False)
    mech.to_csv(OUT/'tables'/'table2_full_precision.csv', index=False)
    o['primary_cells'].drop(columns='retained').to_csv(OUT/'primary_decisions.csv', index=False)
    (OUT/'text_values.json').write_text(json.dumps(text, indent=1, default=jsonable), encoding='utf8')
    (OUT/'figures.json').write_text(json.dumps(figs, indent=1, default=jsonable), encoding='utf8')
    (OUT/'uncertainty.json').write_text(json.dumps(u, indent=1, default=jsonable), encoding='utf8')
    report_ = dict(ok=not problems, checked=checked, problems=problems, status=status,
                   options=vars(args), seconds=round(time.time()-start, 1),
                   environment=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__))
    (OUT/'verification.json').write_text(json.dumps(report_, indent=2, default=jsonable), encoding='utf8')
    print(f"[6/6] compared {checked['tables']} tables ({checked['table_cells']} cells), {checked['text']} in-text numbers "
          f"and {checked['figure_values']} figure values in {report_['seconds']} s", flush=True)
    if problems:
        print(f'FAILED: {len(problems)} differences')
        for p in problems[:50]:
            print('  -', p)
        raise SystemExit(1)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
