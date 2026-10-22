import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bio.io.embeddings import fingerprint, validate_result


class EmbeddingContractTests(unittest.TestCase):
    def test_reorders_every_output_by_source_identity(self):
        result = validate_result([[20], [10]], [[2, 3], [0, 1]], [1, 0], [1, 0], 2)
        np.testing.assert_array_equal(result[0], [[10], [20]])
        np.testing.assert_array_equal(result[1], [[0, 1], [2, 3]])
        np.testing.assert_array_equal(result[2], [0, 1])

    def test_rejects_missing_duplicate_and_nonfinite_results(self):
        for ids, embedding in [([0, 0], [[1], [2]]), ([0], [[1]]),
                               ([0, 1], [[np.nan], [2]]), ([0, 1], [[np.inf], [2]])]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_result([[1], [2]], embedding, [0, 1], ids, 2)

    def test_cache_invalidates_on_data_weights_seed_and_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'input.h5ad'; data.write_bytes(b'data')
            weights = root / 'weights'; weights.mkdir()
            checkpoint = weights / 'model.bin'; checkpoint.write_bytes(b'one')
            cfg = {'seed': 41, 'data_cfg': {'dataset_name': 'pbmc12k', 'foundation_input_path': str(data)},
                   'model_cfg': {'model': 'GeneCompass', 'tasks': 'clustering', 'weights_dir': str(weights)}}
            first = fingerprint(cfg)[0]
            cfg['model_cfg']['tasks'] = 'annotation'
            self.assertEqual(first, fingerprint(cfg)[0])
            for change in ('seed', 'weights', 'data', 'config'):
                altered = copy.deepcopy(cfg)
                if change == 'seed': altered['seed'] = 42
                if change == 'weights': checkpoint.write_bytes(b'changed weights')
                if change == 'data': data.write_bytes(b'changed source')
                if change == 'config': altered['model_cfg']['max_length'] = 10
                self.assertNotEqual(first, fingerprint(altered)[0])


class NativeAttentionTests(unittest.TestCase):
    def test_nicheformer_low_integer_counts_are_not_exponentiated(self):
        import anndata as ad
        import pandas as pd
        from types import SimpleNamespace
        from bio.models.adapters.nicheformer import Nicheformer
        genes = [f'ENSG{i:011d}' for i in range(100)]
        counts = np.zeros((1, 100), dtype=np.float32)
        counts[0, :3] = [1, 2, 3]
        data = ad.AnnData(counts, var=pd.DataFrame(index=genes))
        mean = np.ones(100, dtype=np.float32)
        mean[2] = 2
        adapter = SimpleNamespace(gene_ids=genes, technology_mean=mean, context_length=6,
                                  gene_context_length=3, aux_tokens=30, vocab={'[PAD]': 0},
                                  default_specie_token=5, default_assay_token=15, default_modality_token=3)
        tokens, mask = Nicheformer._tokenize_adata(adapter, data)
        self.assertEqual(tokens.tolist(), [[5, 15, 3, 31, 32, 30]])
        self.assertTrue(mask.all())

    def test_all_seventeen_adapters_share_pipeline_and_cache(self):
        from bio import models as modeling
        from bio.models.adapter import FoundationAdapter
        models = ['CellPLM', 'Geneformer', 'LangCell', 'scFoundation', 'scGPT', 'scMulan',
                  'Nicheformer', 'Tabula', 'UCE', 'scPRINT2', 'TranscriptFormer', 'CellFM',
                  'GeneCompass', 'scLong', 'PCA_model', 'UMAP', 'scVI']
        for name in models:
            with self.subTest(model=name):
                model = getattr(modeling, name)
                self.assertTrue(issubclass(model, FoundationAdapter))
                for method in ('extract_cell_embeddings', 'save_embeddings', 'load_embeddings'):
                    self.assertIs(getattr(model, method), getattr(FoundationAdapter, method))
                for method in ('prepare_inputs', 'load_pretrained', 'encode_batch'):
                    self.assertIsNot(getattr(model, method), getattr(FoundationAdapter, method))

    def test_task_loader_is_prepared_even_outside_extraction(self):
        from types import SimpleNamespace
        from bio.models.adapter import FoundationAdapter
        calls = []
        fake = SimpleNamespace(prepare_inputs=lambda config, train: calls.append(train))
        FoundationAdapter.get_loader(fake, {}, train=False)
        self.assertEqual(calls, [True])

    def test_tabula_attention_matches_pytorch_reference(self):
        import torch
        from bio.models.compat.tabula_attention import NativeMHA
        torch.manual_seed(41)
        model = NativeMHA(12, 3, 0., True).double().eval()
        reference = torch.nn.MultiheadAttention(12, 3, dropout=0., batch_first=True).double().eval()
        with torch.no_grad():
            reference.in_proj_weight.copy_(model.Wqkv.weight)
            reference.in_proj_bias.copy_(model.Wqkv.bias)
            reference.out_proj.load_state_dict(model.out_proj.state_dict())
        x = torch.randn(2, 7, 12, dtype=torch.float64)
        expected, _ = reference(x, x, x, need_weights=False)
        torch.testing.assert_close(model(x), expected, rtol=1e-10, atol=1e-10)

    def test_rejects_missing_backbone_even_when_head_is_optional(self):
        import torch
        from bio.models.loading import verify_backbone
        model = torch.nn.Sequential(torch.nn.Linear(3, 2))
        with self.assertRaises(ValueError):
            verify_backbone(model, {'0.bias': model[0].bias}, ('0.',))
        self.assertEqual(verify_backbone(model, model.state_dict(), ('0.',))['checked_parameters'], 2)


if __name__ == '__main__':
    unittest.main()
