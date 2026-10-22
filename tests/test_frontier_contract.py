"""Verify that external embeddings cannot silently change cell identities or count inputs."""
import copy
from pathlib import Path
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from bio.data.foundation import align_embeddings, prepare_input


class IdentityTests(unittest.TestCase):
    def test_reorders_backend_rows(self):
        result = align_embeddings([[2, 3], [0, 1]], ['b', 'a'], ['a', 'b'])
        np.testing.assert_array_equal(result, [[0, 1], [2, 3]])

    def test_rejects_lost_duplicate_and_nonfinite_cells(self):
        for x, observed, expected in [([[1]], ['a'], ['a', 'b']),
                                      ([[1], [2]], ['a', 'a'], ['a', 'b']),
                                      ([[np.nan]], ['a'], ['a'])]:
            with self.subTest(observed=observed, expected=expected), self.assertRaises(ValueError):
                align_embeddings(x, observed, expected)


class CountTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.path = Path(self.work.name)
        self.genes = [f'GENE{i:03d}' for i in range(100)] + ['GENE000']
        self.counts = np.ones((2, 101), dtype=np.float32)
        self.counts[1] *= 3
        self.data = ad.AnnData(sp.csr_matrix(np.log1p(self.counts)),
                              obs=pd.DataFrame({'celltype': ['T', 'B']}, index=['a', 'b']),
                              var=pd.DataFrame({'gene_symbols': self.genes}, index=[f'id{i}' for i in range(101)]))
        self.data.layers['counts'] = sp.csr_matrix(self.counts)
        self.data.write_h5ad(self.path / 'source.h5ad')
        self.cfg = {'data_cfg': {'dataset_name': 'pbmc12k', 'foundation_input_path': str(self.path / 'source.h5ad')},
                    'model_cfg': {'model': 'UCE', 'expression_layer': 'auto'}}

    def tearDown(self):
        self.work.cleanup()

    def test_uses_full_counts_merges_duplicate_genes_and_hides_labels(self):
        evaluation = self.data[:, :20].copy()
        prepare_input(self.cfg, evaluation, self.path / 'out.h5ad')
        out = ad.read_h5ad(self.path / 'out.h5ad')
        self.assertEqual(out.shape, (2, 100))
        np.testing.assert_allclose(np.asarray(out.X.sum(axis=1)).ravel(), [101, 303])
        self.assertNotIn('celltype', out.obs)
        self.assertEqual(list(out.obs_names), ['cellbench_0', 'cellbench_1'])
        self.assertEqual(list(out.obs['_cellbench_cell_id']), ['cellbench_0', 'cellbench_1'])
        np.testing.assert_array_equal(out[:, 'GENE000'].X.toarray().ravel(), [2, 6])

    def test_reorders_source_to_evaluation(self):
        prepare_input(self.cfg, self.data[::-1].copy(), self.path / 'out.h5ad')
        out = ad.read_h5ad(self.path / 'out.h5ad')
        np.testing.assert_allclose(np.asarray(out.X.sum(axis=1)).ravel(), [303, 101])

    def test_rejects_normalized_input_instead_of_rounding(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['model_cfg']['expression_layer'] = 'X'
        with self.assertRaisesRegex(ValueError, 'raw integer counts'):
            prepare_input(cfg, self.data, self.path / 'out.h5ad')

    def test_rejects_unknown_cell_alignment(self):
        evaluation = self.data.copy()
        evaluation.obs_names = ['unknown_a', 'unknown_b']
        with self.assertRaisesRegex(ValueError, 'cell order'):
            prepare_input(self.cfg, evaluation, self.path / 'out.h5ad')


if __name__ == '__main__':
    unittest.main()
