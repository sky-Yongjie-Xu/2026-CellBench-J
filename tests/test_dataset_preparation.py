"""Check reproducible selection and the strict RNA counts contract."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from bio.assets.prepare_datasets import prepare, stratified_rows


class DatasetPreparationTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_covers_strata(self):
        obs=pd.DataFrame({'donor':['a']*10+['b']*10,'celltype':['x','y']*10})
        rows=stratified_rows(obs,['donor','celltype'],8,41)
        np.testing.assert_array_equal(rows,stratified_rows(obs,['donor','celltype'],8,41))
        self.assertEqual(len(rows),len(set(rows)))
        self.assertEqual(len(obs.iloc[rows].groupby(['donor','celltype'])),4)

    def fixture(self,root,counts=True):
        (root/'datasets').mkdir()
        data=ad.AnnData(sparse.csr_matrix(np.full((8,3),0.5,dtype=np.float32)),
            obs=pd.DataFrame({'label':['T','B']*4,'donor':['a']*4+['b']*4},index=[f'c{i}' for i in range(8)]),
            var=pd.DataFrame({'feature_types':['GEX','GEX','ADT'],'gene_id':['ENSG1','ENSG2','Ab1']},index=['G1','G2','Ab']))
        if counts:data.layers['counts']=sparse.csr_matrix(np.full((8,3),2,dtype=np.float32))
        path=root/'source.h5ad';data.write_h5ad(path);path.with_suffix('.download.json').write_text('{}')
        item=dict(id='fixture',path='source.h5ad',source='test',url='test')
        recipe=dict(label=['label'],donor=['donor'],batch=['donor'],rna_feature_columns=['feature_types'],rna_feature_values=['GEX'])
        return item,recipe

    def test_selects_counts_excludes_protein_and_preserves_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);item,recipe=self.fixture(root)
            with patch('bio.assets.prepare_datasets.PROJECT_ROOT',root):report=prepare(item,recipe,4,41)
            data=ad.read_h5ad(root/'datasets/fixture.h5ad')
            self.assertEqual(data.shape,(4,2));self.assertEqual(report['matrix'],'layers/counts')
            self.assertTrue(np.all(data.X.data==2));self.assertEqual(set(data.obs['batch']),{'a','b'})
            self.assertEqual(list(data.var['ensembl_id']),['ENSG1','ENSG2'])

    def test_normalized_only_source_does_not_replace_existing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);item,recipe=self.fixture(root,False)
            target=root/'datasets/fixture.h5ad';target.write_bytes(b'keep existing')
            with patch('bio.assets.prepare_datasets.PROJECT_ROOT',root),self.assertRaisesRegex(ValueError,'integer count'):
                prepare(item,recipe,4,41)
            self.assertEqual(target.read_bytes(),b'keep existing')

    def test_csc_row_selection_matches_scipy_including_empty_columns(self):
        import h5py
        from bio.assets.prepare_datasets import matrix_slice
        with tempfile.TemporaryDirectory() as tmp:
            x=sparse.csc_matrix([[0,1,0,2],[0,0,0,3],[0,4,0,0],[0,5,0,6]])
            path=Path(tmp)/'csc.h5ad';ad.AnnData(x).write_h5ad(path)
            with h5py.File(path) as f: actual=matrix_slice(f['X'],np.array([0,3]))
            np.testing.assert_array_equal(actual.toarray(),x[[0,3]].toarray())

    def test_perturbation_controls_are_mapped_before_guide_suffix_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);item,recipe=self.fixture(root)
            data=ad.read_h5ad(root/'source.h5ad')
            data.obs['perturbation']=['negative_vector']*4+['GENE_guide1','GENE_guide2']*2
            data.write_h5ad(root/'source.h5ad')
            recipe.update(perturbation=True,condition=['perturbation'],
                          controls=['negative_vector'],condition_split='_',control_fraction=0.5)
            with patch('bio.assets.prepare_datasets.PROJECT_ROOT',root):prepare(item,recipe,4,41)
            result=ad.read_h5ad(root/'datasets/fixture.h5ad')
            self.assertEqual(int(result.obs['control'].sum()),2)
            self.assertEqual(set(result.obs['condition']),{'ctrl','GENE'})
            self.assertIn('negative_vector',set(result.obs['perturbation']))

    def test_csr_sparse_row_windows_preserve_order(self):
        import h5py
        from bio.assets.prepare_datasets import matrix_slice
        with tempfile.TemporaryDirectory() as tmp:
            x=sparse.random(2500,7,density=0.2,format='csr',random_state=41)
            path=Path(tmp)/'csr.h5ad';ad.AnnData(x).write_h5ad(path)
            rows=np.array([0,2,1023,1024,2200,2499])
            with h5py.File(path) as f:actual=matrix_slice(f['X'],rows)
            np.testing.assert_array_equal(actual.toarray(),x[rows].toarray())
