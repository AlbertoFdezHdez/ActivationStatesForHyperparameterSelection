"""Prepare (and with --execute, run) one of the 1,104 archived training runs.

    python train_run.py --list
    python train_run.py --run-id "cnn:c1_resnet34_imagenet100|lr=0.0001|wd=0|seed=0"
    python train_run.py --run-id "..." --execute --gpu 0

Without --execute the command, the resolved configuration and its SHA-256 are
printed and nothing is trained. Training needs the family dependencies in
requirements-training.txt and the datasets named in the configuration.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FAMILIES = {'cnn': ('cnns_supervised', 'train_cnns.py'), 'mlp': ('mlps_supervised', 'train_mlps.py'),
            'vit': ('vision_transformers_supervised', 'train_vits.py'), 'rl': ('reinforcement_learning', 'train_rl.py')}


def records():
    with gzip.open(ROOT/'data/run_configurations.jsonl.gz', 'rt', encoding='utf8') as f:
        return [json.loads(line) for line in f]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-id')
    p.add_argument('--list', action='store_true')
    p.add_argument('--output-root', type=Path, default=ROOT/'training_outputs')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--gpu')
    args = p.parse_args()
    runs = records()
    if args.list:
        print('\n'.join(x['run_id'] for x in runs))
        return
    if not args.run_id:
        p.error('Specify --list or --run-id.')
    record = next((x for x in runs if x['run_id'] == args.run_id), None)
    if record is None:
        p.error('Unknown run ID; see --list.')
    assert hashlib.sha256(record['configuration_text'].encode('utf8')).hexdigest() == record['sha256']
    import yaml  # training dependency
    raw = yaml.safe_load(record['configuration_text'])
    cfg = raw.get('config', raw)
    experiment, *parts = args.run_id.split('|')
    family = experiment.split(':')[0]
    params = dict(x.split('=', 1) for x in parts)
    if family == 'nlp':
        folder = 'transformers_nlp_supervised'
        script = 'train_'+('pretraining' if cfg['runner'] == 'pretraining' else 'finetuning')+'.py'
    else:
        folder, script = FAMILIES[family]
    token = hashlib.sha256(args.run_id.encode()).hexdigest()[:16]
    output = args.output_root.resolve()/token
    config_path = args.output_root.resolve()/'resolved_configs'/(token+'.yaml')
    command = [sys.executable, str(ROOT/'code'/folder/'experiments'/script), '--config', str(config_path),
               '--lr', params['lr'], '--seed', params['seed'], '--output_dir', str(output)]
    if family != 'rl':
        command += ['--weight_decay', params['wd']]
    print(json.dumps(dict(run_id=args.run_id, source=record['configuration_file'], sha256=record['sha256'],
                          command=command, config=cfg, execute=args.execute), indent=2))
    if args.execute:
        if output.exists():
            raise FileExistsError('Choose a fresh output root; existing runs are preserved.')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf8')
        env = os.environ.copy()
        if args.gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = args.gpu
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
