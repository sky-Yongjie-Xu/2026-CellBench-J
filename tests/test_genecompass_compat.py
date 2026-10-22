"""Check the reconstructed value path and explicit concatenation contract."""
import importlib.util
from pathlib import Path
import unittest

import torch
from transformers import BertConfig

from bio.models.compat.genecompass import checkpoint_embeddings


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'genecompass_upstream_embeddings',
    ROOT / 'public_code/GeneCompass/genecompass/knowledge_embeddings.py')
UPSTREAM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPSTREAM)


class CheckpointEmbeddingTests(unittest.TestCase):
    def make_model(self, first=False):
        config = BertConfig(vocab_size=5, hidden_size=4, max_position_embeddings=8,
                            hidden_dropout_prob=0., use_values=True, use_value_emb=True,
                            concat_values_first=first, use_promoter=True, use_co_exp=True,
                            use_gene_family=True, use_peca_grn=True)
        priors = {name: torch.arange(15).reshape(5, 3).float() for name in
                  ('promoter', 'co_exp', 'gene_family', 'peca_grn')}
        priors['homologous_gene_human2mouse'] = {2: 3}
        return checkpoint_embeddings(UPSTREAM)(config, priors).eval()

    def test_value_block_order_and_cpu_support(self):
        for first in (False, True):
            with self.subTest(first=first):
                model = self.make_model(first)
                captured = []
                hook = model.concat_embeddings.cat_fc.register_forward_pre_hook(
                    lambda module, args: captured.append(args[0].detach()))
                values = torch.tensor([[1., 2.]])
                result = model(input_ids=torch.tensor([[2, 3]]), values=values)
                hook.remove()
                expected = model.values_embeddings(values)
                actual = captured[0][..., 4:8] if first else captured[0][..., -4:]
                torch.testing.assert_close(actual, expected)
                self.assertEqual(result.shape, (1, 2, 4))
                self.assertTrue(torch.isfinite(result).all())
                # Six blocks: gene, four priors, learned value embedding.
                self.assertEqual(captured[0].shape[-1], 24)

    def test_expression_values_affect_output(self):
        torch.manual_seed(41)
        model = self.make_model()
        ids = torch.tensor([[2, 3]])
        first = model(input_ids=ids, values=torch.tensor([[0., 0.]]))
        second = model(input_ids=ids, values=torch.tensor([[3., 6.]]))
        self.assertFalse(torch.allclose(first, second))

    def test_strict_load_rejects_missing_value_encoder(self):
        model = self.make_model()
        state = model.state_dict()
        del state['values_embeddings.linear1.weight']
        with self.assertRaisesRegex(RuntimeError, 'values_embeddings.linear1.weight'):
            model.load_state_dict(state, strict=True)


if __name__ == '__main__':
    unittest.main()
