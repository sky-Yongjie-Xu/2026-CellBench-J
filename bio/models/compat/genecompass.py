"""Reconstruct the released value-embedding variant from its saved configuration.

The public main branch defines ContinuousValueEncoder but does not connect it to
KnowledgeBertEmbeddings. Keep that checkout intact and adapt the class at runtime.
concat_values_first controls whether values precede or follow the prior blocks.
This ordering follows the saved flag; bitwise agreement with the unpublished
training implementation has not been established.
"""
import copy

import torch
from torch import nn


def checkpoint_embeddings(upstream):
    class DeviceValueEncoder(upstream.ContinuousValueEncoder):
        def forward(self, values):
            # The upstream implementation forces CUDA. Preserve its operations
            # while following the model's device on both CPU and GPU.
            x = values.unsqueeze(-1).to(self.linear1.weight)
            x = torch.clamp(x, max=self.max_value)
            x = self.activation(self.linear1(x))
            return self.dropout(self.norm(self.linear2(x)))

    class CheckpointKnowledgeEmbeddings(upstream.KnowledgeBertEmbeddings):
        def __init__(self, config, knowledges):
            if not config.use_values or not getattr(config, 'use_value_emb', False):
                raise ValueError('Checkpoint adapter requires use_values/use_value_emb=true')
            if not isinstance(getattr(config, 'concat_values_first', None), bool):
                raise ValueError('Checkpoint must specify concat_values_first explicitly')
            # Build the unchanged gene/prior projections without the raw scalar.
            base = copy.deepcopy(config)
            base.use_values = False
            super().__init__(base, knowledges)
            self.use_values = True
            self.concat_values_first = config.concat_values_first
            self.values_embeddings = DeviceValueEncoder(config.hidden_size)
            width = self.concat_embeddings.cat_fc.in_features + config.hidden_size
            self.concat_embeddings.cat_fc = nn.Linear(width, config.hidden_size)

        def forward(self, input_ids=None, values=None, species=None,
                    token_type_ids=None, position_ids=None, inputs_embeds=None,
                    past_key_values_length=0, emb_warmup_alpha=1.):
            if input_ids is None or values is None:
                raise ValueError('GeneCompass requires gene IDs and expression values')
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            length = inputs_embeds.shape[1]
            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length:past_key_values_length + length]
            if token_type_ids is None:
                token_type_ids = self.token_type_ids[:, :length].expand(input_ids.shape[0], length)
            shifted = input_ids.clone()
            if species is not None:
                mouse = species.squeeze(1) == 1
                shifted[mouse] = self.homologous_index[shifted[mouse]]
            blocks = [inputs_embeds]
            value_block = self.values_embeddings(values)
            if self.concat_values_first:
                blocks.append(value_block)
            for name in ('promoter', 'co_exp', 'gene_family', 'peca_grn'):
                if getattr(self, 'use_' + name):
                    prior = getattr(self, name + '_knowledge')[shifted]
                    blocks.append(emb_warmup_alpha * getattr(self, name + '_embeddings')(prior))
            if not self.concat_values_first:
                blocks.append(value_block)
            embeddings = self.concat_embeddings(torch.cat(blocks, dim=-1))
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
            if self.position_embedding_type == 'absolute':
                embeddings = embeddings + self.position_embeddings(position_ids)
            return self.dropout(self.LayerNorm(embeddings))

    return CheckpointKnowledgeEmbeddings
