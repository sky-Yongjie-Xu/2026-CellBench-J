import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bio.io.results import load_runs, log_metrics, start_run
from bio.runtime.batch import commands


class LocalResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = dict(output_dir=self.temp.name, seed=41, topk=1,
                        data_cfg={'dataset_name': 'uc_epi'},
                        model_cfg={'model': 'PCA_model', 'tasks': 'annotation'})

    def test_local_metrics_merge_and_export_preserve_metadata(self):
        path = log_metrics(self.cfg, {'acc': np.float32(.5), 'invalid': np.nan})
        self.assertEqual(path, log_metrics(self.cfg, {'f1_m': .75}))
        record = json.loads(path.read_text())
        self.assertEqual(record['summary'], {'acc': .5, 'invalid': None, 'f1_m': .75})
        run = load_runs('annotation', self.temp.name)[0]
        self.assertEqual((run.dataset, run.model, run.seed), ('uc_epi', 'PCA_model', 41))
        self.assertEqual(run.summary['f1_m'], .75)

    def test_repeated_experiments_get_distinct_files_and_vcc_is_separate(self):
        first = log_metrics(self.cfg, {'acc': 1.})
        start_run(self.cfg)
        second = log_metrics(self.cfg, {'acc': .5})
        self.assertNotEqual(first, second)
        vcc = log_metrics(self.cfg, {'DES': .5}, task='perturbation_vcc')
        self.assertNotEqual(first.parent, vcc.parent)
        self.assertEqual(len(load_runs('annotation', self.temp.name)), 2)

    def test_nonprimary_rank_does_not_write(self):
        with patch.dict(os.environ, {'RANK': '1'}):
            self.assertIsNone(log_metrics(self.cfg, {'acc': .5}))
        self.assertFalse(list(Path(self.temp.name).rglob('*.json')))

    def test_batch_product_passes_arguments_and_vcc_flag(self):
        config = {'parameters': {'seed': {'values': [41, 42]},
                                 'cfgs': {'values': ['path with spaces.yaml', 'b.yaml']}},
                  'extra_args': ['--vcc-eval']}
        planned = list(commands(config, 'python'))
        self.assertEqual(len(planned), 4)
        self.assertIn('path with spaces.yaml', planned[0])
        self.assertEqual(planned[0][-1], '--vcc-eval')
        self.assertEqual(planned[-1][-3:], ['--cfgs', 'b.yaml', '--vcc-eval'])


if __name__ == '__main__':
    unittest.main()
