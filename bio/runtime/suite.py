"""Run dataset-appropriate tasks in isolated subprocesses through the same CLI."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from bio.io.paths import PROJECT_ROOT

STANDARD = ('embedding_extraction', 'clustering', 'annotation', 'reconstruction')
PERTURBATION = {'adamson', 'norman', 'replogleK562', 'replogleRPE1'}
BATCHED = {'hPancreas', 'ms', 'immune', 'uc_epi', 'uc_imm', 'uc_imm1', 'rgc', 'celegan'}
DATASETS = ('pbmc12k', 'hPancreas', 'cortex', 'ms', 'liver', 'Zheng68k',
            'lung', 'immune', 'adamson', 'norman', 'replogleK562',
            'replogleRPE1', 'rgc', 'celegan', 'uc_epi', 'uc_imm', 'uc_imm1')

from bio.data.catalog import CURATED
DATASETS = tuple(dict.fromkeys((*DATASETS, *CURATED)))
BATCHED.update(name for name, spec in CURATED.items() if spec.get('batch'))
PERTURBATION.update(name for name, spec in CURATED.items() if spec.get('perturbation'))


def task_plan(dataset, cfg):
    if dataset not in DATASETS:
        raise ValueError(f'Unsupported suite dataset {dataset!r}; choose from {", ".join(DATASETS)}')
    skipped = {}
    if dataset in PERTURBATION:
        tasks = ['embedding_extraction', 'perturbation']
        if cfg['data_cfg'].get('evaluation_expression_scale') == 'log1p_normalized':
            tasks.append('perturbation_vcc')
        else:
            skipped['perturbation_vcc'] = 'Requires data_cfg.evaluation_expression_scale=log1p_normalized; verify the actual expression scale first.'
        for task in ('clustering', 'annotation', 'reconstruction', 'correction'):
            skipped[task] = 'Not part of the perturbation dataset evaluation protocol.'
    else:
        tasks = list(STANDARD)
        batch_key = cfg['data_cfg'].get('dataset_batch_keys', {}).get(dataset)
        if dataset in BATCHED or batch_key:
            tasks.insert(2, 'correction')
        else:
            skipped['correction'] = 'No batch mapping declared for this dataset; configure data_cfg.dataset_batch_keys to enable it.'
        skipped['perturbation'] = 'Not a perturbation dataset.'
        skipped['perturbation_vcc'] = 'Not a perturbation dataset.'
    return tasks, skipped


def selected_configs(opt, cfg=None):
    from bio.config import config_loader
    from bio.models import list_models
    from bio.io.paths import project_path
    if cfg is not None:
        return [(project_path(opt.cfgs), cfg)]
    if opt.cfgs:
        path = project_path(opt.cfgs)
        return [(path, config_loader(path))]
    result = []
    for model in list_models():
        name = 'pca' if model == 'PCA_model' else model.lower()
        path = project_path(f'configs/{name}.yaml')
        config = config_loader(path)
        if config['model_cfg']['model'] != model:
            raise ValueError(f'{path} does not configure registered model {model}')
        result.append((path, config))
    return result


def run_suite(opt, cfg=None):
    configs = selected_configs(opt, cfg)
    if not opt.dataset_name and not opt.cfgs:
        raise ValueError('Select a dataset with --dataset or --dataset_name for an all-model run')
    dataset = opt.dataset_name or configs[0][1]['data_cfg']['dataset_name']
    commands = []
    skipped = {}
    model_names = []
    for config_path, config in configs:
        model = config['model_cfg']['model']
        model_names.append(model)
        tasks, skipped[model] = task_plan(dataset, config)
        for task in tasks:
            command = [sys.executable, str(PROJECT_ROOT/'main.py'), '--cfgs', str(config_path),
                       '--dataset_name', dataset, '--task', 'perturbation' if task == 'perturbation_vcc' else task,
                       '--phase', opt.phase, '--seed', str(opt.seed), '--topk', str(opt.topk)]
            for flag in ('gpu', 'hvg'):
                if getattr(opt, flag) is not None:
                    command.extend(['--'+flag, str(getattr(opt, flag))])
            if task == 'perturbation_vcc':
                command.append('--vcc-eval')
            commands.append((model, task, command))
    print(f'Dataset: {dataset}; models: {len(model_names)}; planned tasks: {len(commands)}', flush=True)
    for model, task, command in commands:
        print(f'[{model}/{task}] {shlex.join(command)}', flush=True)
    for model, reasons in skipped.items():
        for task, reason in reasons.items():
            print(f'[skip {model}/{task}] {reason}', flush=True)
    if opt.dry_run:
        return 0
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    folder = PROJECT_ROOT/'output'/'suites'/run_id
    folder.mkdir(parents=True)
    report = dict(dataset=dataset, models=model_names, skipped=skipped, tasks=[])
    def save():
        path = folder/'summary.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        temp.replace(path)
    failed = set()
    stop = False
    for model, task, command in commands:
        dependency_failed = (model, 'embedding_extraction') in failed or (
            task == 'perturbation_vcc' and (model, 'perturbation') in failed)
        if stop or dependency_failed:
            reason = 'Stopped by --fail-fast.' if stop else 'Required task failed for this model.'
            report['tasks'].append(dict(model=model, task=task, command=command, status='skipped', reason=reason))
            save()
            continue
        log = folder/model/f'{task}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(model=model, task=task, command=command, log=str(log), status='running')
        report['tasks'].append(entry)
        save()
        print(f'Running {model}/{task}; log: {log}', flush=True)
        try:
            with log.open('w') as handle:
                result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT)
            entry.update(status='passed' if result.returncode == 0 else 'failed', returncode=result.returncode)
        except OSError as exc:
            entry.update(status='failed', error=str(exc))
        except KeyboardInterrupt:
            entry.update(status='interrupted')
            save()
            print(f'Interrupted. Summary: {folder / "summary.json"}', flush=True)
            return 130
        if entry['status'] == 'failed':
            failed.add((model, task))
            stop = opt.fail_fast
        save()
        print(f'{model}/{task}: {entry["status"]}', flush=True)
    print(f'Summary: {folder / "summary.json"}', flush=True)
    return int(bool(failed))
