import unittest
from unittest.mock import patch
import numpy as np
from sklearn import metrics
from bio.metrics import annotation, reconstruction, perturbation, correction, clustering


class EvaluationTests(unittest.TestCase):
    def test_precision_recall_order_and_probabilities(self):
        y = np.array([0, 0, 0, 1, 1])
        p = np.array([0, 1, 1, 1, 1])
        prob = np.array([[.9,.1],[.4,.6],[.3,.7],[.1,.9],[.2,.8]])
        r = annotation(p, y, prob, [0, 1])
        self.assertAlmostEqual(r['precision_macro'], metrics.precision_score(y,p,average='macro'))
        self.assertAlmostEqual(r['recall_macro'], metrics.recall_score(y,p,average='macro'))
        self.assertNotEqual(r['precision_macro'], r['recall_macro'])
        self.assertAlmostEqual(r['AUROC_macro'],1.)
        self.assertAlmostEqual(r['AUPRC_macro'],1.)

    def test_absent_classes_and_invalid_probabilities(self):
        r = annotation([0,0], [0,0], [[1,0],[1,0]], [0,1])
        self.assertIsNone(r['AUROC_macro'])
        self.assertEqual(r['absent_test_classes'], ['1'])
        with self.assertRaises(ValueError): annotation([0],[0],[[1,1]],[0,1])

    def test_constant_correlations_are_missing_not_perfect(self):
        r = reconstruction(np.ones((3,4)), np.ones((3,4)), 50)
        self.assertIsNone(r['Pearson_expr'])
        self.assertEqual(r['Pearson_expr_valid_n'],0)
        self.assertEqual(r['top_k'],4)
        self.assertEqual(r['MSE_expr'],0)
        self.assertNotIn('Sign_agree',r)
        self.assertNotIn('AvgGR',r)

    def test_correlations_have_explicit_axes(self):
        x = np.array([[1,2,4],[2,4,8],[3,6,12]])
        r=reconstruction(x,x)
        self.assertAlmostEqual(r['Pearson_expr'],1)
        self.assertAlmostEqual(r['Pearson_gene'],1)
        with self.assertRaises(ValueError): reconstruction([[np.nan]],[[1]])

    def test_condition_means_equal_weight_not_cell_weight(self):
        t=np.array([[1,2],[1,2],[1,2],[3,4]])
        p=np.array([[2,3],[2,3],[2,3],[6,7]])
        r=perturbation(p,t,np.zeros_like(t),['a','a','a','b'])
        self.assertAlmostEqual(r['MAE_expr'],2.)
        self.assertEqual(r['n_groups'],2)
        self.assertNotIn('MSE_LFC',r)
        self.assertNotIn('AvgPP',r)

    def test_no_effect_not_counted_as_direction_success(self):
        x=np.ones((3,4))
        r=perturbation(x,x,x,['a','a','b'])
        self.assertIsNone(r['Sign_agree_changed'])
        self.assertIsNone(r['TopK_directional_recall'])
        self.assertIsNone(r['MSE_skill_vs_control'])

    def test_direction_recall_requires_actual_downregulation(self):
        r=perturbation([[3,3,3]],[[1,3,2]],[[2,2,2]],['a'],top_k=50)
        self.assertAlmostEqual(r['Sign_agree_changed'],.5)
        self.assertAlmostEqual(r['TopK_directional_recall'],.5)

    def test_fixed_clustering_does_not_optimize_reference_labels(self):
        rng=np.random.default_rng(41)
        x=np.r_[rng.normal(-4,.1,(20,3)),rng.normal(4,.1,(20,3))]
        with patch('scib.metrics.clustering.opt_louvain',side_effect=AssertionError('oracle used')):
            r,_=clustering(x,np.repeat([0,1],20))
        self.assertEqual(r['clustering_mode'],'fixed')
        self.assertGreater(r['ari'],.9)

    def test_batch_asw_uses_cell_type_strata(self):
        # Each cell type has overlapping batches; types are far apart.
        rng=np.random.default_rng(2)
        x=np.r_[rng.normal(-4,.1,(20,3)),rng.normal(4,.1,(20,3))]
        y=np.repeat(['A','B'],20); b=np.tile(['a','b'],20)
        with patch('scib.metrics.ilisi_graph',return_value=.8), patch('scib.metrics.clisi_graph',return_value=.9):
            r=correction(x,b,y,{'compute_kbet':False})
        self.assertGreater(r['Batch_ASW'],.8)
        self.assertEqual(r['iLISI_scaled'],.8)
        self.assertEqual(r['cLISI_scaled'],.9)
        self.assertIsNone(r['kBET'])
        self.assertNotIn('AvgBC',r)

    def test_optional_failure_is_not_a_numeric_score(self):
        rng=np.random.default_rng(2)
        x=rng.normal(size=(40,3)); y=np.repeat(['A','B'],20); b=np.tile(['a','b'],20)
        with patch('scib.metrics.ilisi_graph',side_effect=RuntimeError('missing binary')), patch('scib.metrics.clisi_graph',return_value=.9):
            r=correction(x,b,y,{'compute_kbet':False})
        self.assertIsNone(r['iLISI_scaled'])
        self.assertIn('missing binary',r['metric_status']['iLISI_scaled'])

    def test_perturbation_head_can_predict_downregulation(self):
        import torch
        from bio.models.heads import PerturbDecoder
        from bio.models.adapters.cellplm import CellPLM
        from types import SimpleNamespace
        head=PerturbDecoder(3,2,4).eval()
        with torch.no_grad():
            head.proj_out.weight.zero_(); head.proj_out.bias.fill_(-.5)
            out=CellPLM.forward_perturbation(SimpleNamespace(pert_decoder=head),torch.zeros(2,3),torch.zeros(2,dtype=torch.long),torch.ones(2,4))
        torch.testing.assert_close(out,torch.full((2,4),.5))



class ExportAndVersionTests(unittest.TestCase):
    def test_ranking_is_scale_invariant_and_excludes_incomplete_blocks(self):
        import pandas as pd
        from bio.io.export import paired_ranks
        records=[]
        for dataset in ['d1','d2']:
            for model, corr, mse in [('a',.8,1.),('b',.7,2.)]:
                records.append(dict(dataset=dataset,model=model,seed=41,topk=100,phase='adapter',protocol='v2',Pearson_expr=corr,MSE_expr=mse))
        frame=pd.DataFrame(records)
        primary={'Pearson_expr':True,'MSE_expr':False}
        original,_=paired_ranks(frame,primary)
        frame['MSE_expr']*=10000
        rescaled,_=paired_ranks(frame,primary)
        pd.testing.assert_frame_equal(original,rescaled)
        frame.loc[frame.dataset=='d2','MSE_expr']=np.nan
        rank, coverage=paired_ranks(frame,primary)
        self.assertEqual(int(coverage.complete.sum()),1)
        self.assertEqual(rank.iloc[0]['model'],'a')

    def test_legacy_records_excluded_by_default(self):
        import json, tempfile
        from pathlib import Path
        from bio.io.results import log_metrics, load_runs
        with tempfile.TemporaryDirectory() as root:
            cfg={'output_dir':root,'data_cfg':{'dataset_name':'d'},'model_cfg':{'model':'m','tasks':'annotation'}}
            path=log_metrics(cfg,{'acc':1})
            legacy=json.loads(path.read_text()); legacy.pop('metric_version')
            (Path(path).parent/'old.json').write_text(json.dumps(legacy))
            self.assertEqual(len(load_runs('annotation',root)),1)
            self.assertEqual(len(load_runs('annotation',root,metric_version=1)),1)

    def test_vcc_rejects_unverified_expression_scale(self):
        from bio.metrics.vcc import evaluate_vcc
        with self.assertRaisesRegex(ValueError,'declared log1p'):
            evaluate_vcc(None,None,'dataset_defined')

    def test_vcc_per_condition_mean_not_describe_row_position(self):
        import anndata as ad
        import pandas as pd
        from bio.metrics.vcc import evaluate_vcc
        from types import SimpleNamespace
        obs=pd.DataFrame({'condition':['ctrl','p1','p2']*2,'celltype':['a']*3+['b']*3},index=list('abcdef'))
        data=ad.AnnData(np.ones((6,3))*.5,obs=obs)
        rows=[{'perturbation':'p1','discrimination_score_l1':.8,'overlap_at_N':.6,'mae':.2},
              {'perturbation':'p2','discrimination_score_l1':.4,'overlap_at_N':.2,'mae':.6}]
        result=SimpleNamespace(to_dicts=lambda:rows)
        fake=SimpleNamespace(compute=lambda **kwargs:(result,None))
        with patch('cell_eval.MetricsEvaluator',return_value=fake):
            r=evaluate_vcc(data,data.copy(),'log1p_normalized')
        self.assertEqual(r['n_groups'],4)
        self.assertAlmostEqual(r['PDS'],.6)
        self.assertAlmostEqual(r['DES'],.4)
        self.assertAlmostEqual(r['MAE'],.4)


class PerturbationArtifactTests(unittest.TestCase):
    def test_evaluation_keeps_all_same_type_controls_and_prediction_order(self):
        from bio.data.perturbation import create_evaluation_adata
        raw=np.arange(12,dtype=float).reshape(6,2)
        labels=np.array([0,0,0,0,1,1])
        conditions=np.array(['ctrl','ctrl','p1','p2','ctrl','p1'])
        controls=np.array([1,1,0,0,1,0])
        pred,truth=create_evaluation_adata([3,2],{3:0,2:0},raw,np.array([[30,31],[20,21]]),labels,np.arange(6),conditions,controls)
        self.assertEqual(truth.obs.cell_id.tolist(),[0,1,2,3])
        np.testing.assert_array_equal(pred.X[2:4],[[20,21],[30,31]])
        np.testing.assert_array_equal(pred.X[:2],raw[:2])


if __name__ == '__main__': unittest.main()
