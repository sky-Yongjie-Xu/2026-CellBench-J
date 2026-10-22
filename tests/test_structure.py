"""Behavioral checks for package boundaries and migrated execution paths."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


class StructureTests(unittest.TestCase):
    def test_registry_config_and_cli_import_without_torch_or_adapters(self):
        code='''
import sys
sys.argv=['external-tool','--unrelated-flag']
import main
import bio.config
from bio.models import list_models
assert len(list_models()) == 17
assert 'torch' not in sys.modules
assert not any(k.startswith('bio.models.adapters.') for k in sys.modules)
'''
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))

    def test_migrated_script_entries_work_outside_project(self):
        for script in ('download_weights', 'download_frontier_weights',
                       'download_public_code', 'check_resources',
                       'check_adapter_inference', 'check_frontier_models',
                       'extract_frontier_embeddings', 'run_batch'):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(ROOT/'scripts'/f'{script}.py'), '--help'],
                    cwd='/tmp', capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_selecting_pca_does_not_import_other_adapters(self):
        code='''
import sys
from bio.models import get_model
assert get_model('PCA_model').__name__ == 'PCA_model'
assert 'bio.models.adapters.scgpt' not in sys.modules
assert 'bio.models.adapters.geneformer' not in sys.modules
assert 'bio.vendor.gears.heads' not in sys.modules
'''
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))

    def test_config_paths_independent_of_working_directory(self):
        from bio.config import config_loader
        from bio.io.paths import PROJECT_ROOT
        cwd=Path.cwd()
        try:
            os.chdir('/tmp')
            cfg=config_loader('configs/pca.yaml')
        finally:
            os.chdir(cwd)
        self.assertEqual(Path(cfg['data_cfg']['root_path']),ROOT/'datasets')
        self.assertEqual(PROJECT_ROOT,ROOT)

    def test_sparse_perturbation_dataset_and_collation_survive_move(self):
        import anndata as ad
        import numpy as np
        import pandas as pd
        from scipy.sparse import csr_matrix
        from bio.data.perturbation import create_perturbation_adata, PerturbationMatchingDataset, collate_fn
        raw=np.arange(12,dtype=np.float32).reshape(4,3)
        data=create_perturbation_adata([2,3],{2:0,3:1},raw,np.ones((4,2)),np.zeros(4),np.arange(4),np.array(['ctrl','ctrl','p1','p2']),np.array([1,1,0,0]))
        data.X=csr_matrix(data.X)
        dataset=PerturbationMatchingDataset(data)
        batch=collate_fn([dataset[0],dataset[1]])
        self.assertEqual(tuple(batch[0].shape),(2,2))
        self.assertEqual(tuple(batch[3].shape),(2,3))

    def test_all_shared_heads_forward_and_gradient(self):
        import torch
        from bio.models.heads import Pooler_classify, Decoder, PerturbDecoder
        x=torch.randn(3,5,requires_grad=True)
        classifier=Pooler_classify(5,2)
        reconstruction=Decoder(5,7)
        perturbation=PerturbDecoder(5,2,7)
        a=classifier(x); b=reconstruction(x); c=perturbation(x,torch.tensor([0,1,0]))
        self.assertEqual(tuple(a.shape),(3,2))
        self.assertEqual(tuple(b.shape),(3,7))
        self.assertEqual(tuple(c.shape),(3,7))
        (a.sum()+b.sum()+c.sum()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_cli_preserves_config_defaults_and_explicit_overrides(self):
        from unittest.mock import patch
        from bio.cli import main
        with patch('bio.tasks.runner.run_model') as runner:
            main(['--cfgs','configs/pca.yaml','--task','clustering','--dataset_name','pbmc12k','--seed','42'])
        cfg=runner.call_args.args[0]
        self.assertEqual(cfg['model_cfg']['model'],'PCA_model')
        self.assertEqual(cfg['model_cfg']['tasks'],'clustering')
        self.assertEqual(cfg['seed'],42)


if __name__ == '__main__': unittest.main()
