"""Verify task eligibility, CLI dispatch, and failure dependencies without training."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bio.runtime.suite import task_plan, run_suite, selected_configs


class SuiteTests(unittest.TestCase):
    def test_eligibility_and_vcc_scale_guard(self):
        cfg = {'data_cfg': {}}
        self.assertNotIn('perturbation', task_plan('pbmc12k', cfg)[0])
        self.assertNotIn('correction', task_plan('pbmc12k', cfg)[0])
        self.assertIn('correction', task_plan('uc_epi', cfg)[0])
        self.assertNotIn('perturbation_vcc', task_plan('norman', cfg)[0])
        cfg['data_cfg']['evaluation_expression_scale'] = 'log1p_normalized'
        self.assertEqual(task_plan('norman', cfg)[0][-2:], ['perturbation', 'perturbation_vcc'])
        with self.assertRaises(ValueError): task_plan('unknown', cfg)

    def test_dataset_cli_dispatches_suite(self):
        from bio.cli import main
        with patch('bio.runtime.suite.run_suite', return_value=0) as suite:
            self.assertEqual(main(['--dataset', 'pbmc12k', '--dry-run']), 0)
        opt, = suite.call_args.args
        self.assertEqual(opt.task, 'all')
        self.assertIsNone(opt.cfgs)

    def test_failed_perturbation_skips_vcc_and_reports_failure(self):
        opt = Namespace(dataset_name='norman', cfgs='configs/pca.yaml', phase='adapter_finetuning',
                        seed=41, topk=1, gpu=None, hvg=None, dry_run=False, fail_fast=False)
        cfg = {'data_cfg': {'evaluation_expression_scale': 'log1p_normalized'},
               'model_cfg': {'model': 'PCA_model'}}
        with tempfile.TemporaryDirectory() as tmp, patch('bio.runtime.suite.PROJECT_ROOT', Path(tmp)), \
             patch('bio.runtime.suite.subprocess.run', side_effect=[Namespace(returncode=0), Namespace(returncode=1)]) as run:
            self.assertEqual(run_suite(opt, cfg), 1)
            self.assertEqual(run.call_count, 2)
            report = json.loads(next(Path(tmp).rglob('summary.json')).read_text())
            self.assertEqual([r['status'] for r in report['tasks']], ['passed', 'failed', 'skipped'])

    def test_dry_run_does_not_launch_processes(self):
        opt = Namespace(dataset_name='pbmc12k', cfgs='configs/pca.yaml', phase='adapter_finetuning',
                        seed=41, topk=1, gpu=None, hvg=None, dry_run=True, fail_fast=False)
        with patch('bio.runtime.suite.subprocess.run') as run:
            self.assertEqual(run_suite(opt, {'data_cfg': {}, 'model_cfg': {'model': 'PCA_model'}}), 0)
            run.assert_not_called()

    def test_default_configs_cover_all_registered_models(self):
        from bio.models import list_models
        configs = selected_configs(Namespace(cfgs=None))
        self.assertEqual([cfg['model_cfg']['model'] for _, cfg in configs], list(list_models()))
        self.assertEqual(len(configs), 17)
        for _, cfg in configs:
            self.assertEqual(len(task_plan('pbmc12k', cfg)[0]), 4)

    def test_failure_is_scoped_to_model_and_single_config_is_respected(self):
        opt = Namespace(dataset_name='pbmc12k', cfgs=None, phase='adapter_finetuning',
                        seed=41, topk=1, gpu=None, hvg=None, dry_run=False, fail_fast=False)
        configs = [(Path('configs/pca.yaml'), {'data_cfg': {}, 'model_cfg': {'model': name}})
                   for name in ('First', 'Second')]
        with tempfile.TemporaryDirectory() as tmp, patch('bio.runtime.suite.PROJECT_ROOT', Path(tmp)), \
             patch('bio.runtime.suite.selected_configs', return_value=configs), \
             patch('bio.runtime.suite.subprocess.run', side_effect=[Namespace(returncode=1)]+[Namespace(returncode=0)]*4) as run:
            self.assertEqual(run_suite(opt), 1)
            self.assertEqual(run.call_count, 5)
            report = json.loads(next(Path(tmp).rglob('summary.json')).read_text())
            self.assertEqual([r['status'] for r in report['tasks']],
                             ['failed', 'skipped', 'skipped', 'skipped']+['passed']*4)
            self.assertTrue((next(Path(tmp).glob('output/suites/*'))/'Second'/'clustering.log').is_file())
        selected = selected_configs(Namespace(cfgs='configs/scgpt.yaml'))
        self.assertEqual([cfg['model_cfg']['model'] for _, cfg in selected], ['scGPT'])
