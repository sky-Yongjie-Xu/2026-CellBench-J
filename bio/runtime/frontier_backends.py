"""Model-specific inference following the checked-out official source APIs.

No backend may substitute random weights, a different architecture, or a PCA embedding.
"""
from argparse import Namespace
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import shutil

import numpy as np


def device(config):
    import torch
    chosen = config.get('backend_device', 'auto')
    return ('cuda' if torch.cuda.is_available() else 'cpu') if chosen == 'auto' else chosen


def dense(x):
    return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)


def uce(args, config):
    import torch
    import anndata as ad
    from accelerate import Accelerator
    # The reference implementation resolves model_files relative to the working directory.
    os.chdir(args.weights)
    sys.path.insert(0, str(args.source))
    import evaluate
    from data_proc import gene_embeddings
    species = config.get('species', 'human')
    root = args.weights / 'model_files'
    mapping = gene_embeddings.MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH['ESM2']
    mapping = {k: args.weights / v for k, v in mapping.items()}
    gene_embeddings.MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH['ESM2'] = mapping
    # The upstream helper eagerly loads every species; only the selected species is needed.
    def selected_species(_):
        return {species: {k.upper(): v for k, v in torch.load(mapping[species], map_location='cpu').items()}}
    evaluate.get_species_to_pe = selected_species
    # Preserve the official forward/data pipeline, but map checkpoint tensors into
    # memory and assign them to a meta model to avoid allocating a second 3GB ESM table.
    def memory_efficient_eval(adata, name, pe_idx_path, chroms_path, starts_path,
                              shapes_dict, accelerator, settings):
        from torch import nn
        from torch.utils.data import DataLoader
        from model import TransformerModel
        from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
        state = torch.load(settings.model_loc, map_location='cpu', mmap=True)
        with torch.device('meta'):
            model = TransformerModel(token_dim=5120, d_model=1280, nhead=20,
                                     d_hid=5120, nlayers=4, dropout=0.05, output_dim=1280)
            model.pe_embedding = nn.Embedding(145469, 5120)
        model.load_state_dict(state, strict=True, assign=True)
        del state
        # For the in-distribution vocabulary the reference runner retains the ESM
        # table from the checkpoint; its external all_tokens file is not substituted.
        pe = torch.load(settings.token_file, map_location='cpu', mmap=True)
        if pe.shape[0] not in (143574, 145469):
            raise ValueError('This UCE adapter supports the released in-distribution vocabulary only')
        del pe
        model.to(accelerator.device).eval()
        dataset = MultiDatasetSentences(sorted_dataset_names=[name], shapes_dict=shapes_dict,
                                        args=settings, npzs_dir=settings.dir,
                                        dataset_to_protein_embeddings_path=pe_idx_path,
                                        datasets_to_chroms_path=chroms_path,
                                        datasets_to_starts_path=starts_path)
        loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False,
                            num_workers=0, collate_fn=MultiDatasetSentenceCollator(settings))
        result = []
        with torch.inference_mode():
            for batch in loader:
                sentences = batch[0].permute(1, 0).long().to(accelerator.device)
                mask = batch[1].to(accelerator.device)
                pe = nn.functional.normalize(model.pe_embedding(sentences), dim=2)
                _, embedding = model(pe, mask=mask)
                result.append(embedding.float().cpu().numpy())
        adata.obsm['X_uce'] = np.concatenate(result)
        adata.write_h5ad(settings.dir + name + '_uce_adata.h5ad')
    evaluate.run_eval = memory_efficient_eval
    with tempfile.TemporaryDirectory(prefix='uce-', dir=args.output.parent) as work:
        settings = Namespace(adata_path=str(args.input), dir=work + '/', species=species,
                             filter=False, skip=False, model_loc=str(root / '4layer_model.torch'),
                             batch_size=config.get('inference_batch_size', 1), pad_length=1536,
                             pad_token_idx=0, chrom_token_left_idx=1, chrom_token_right_idx=2,
                             cls_token_idx=3, CHROM_TOKEN_OFFSET=143574, sample_size=1024,
                             CXG=True, nlayers=4, output_dim=1280, d_hid=5120, token_dim=5120,
                             multi_gpu=False, spec_chrom_csv_path=str(root / 'species_chrom.csv'),
                             token_file=str(root / 'all_tokens.torch'),
                             protein_embeddings_dir=str(root / 'protein_embeddings'),
                             offset_pkl_path=str(root / 'species_offsets.pkl'))
        accelerator = Accelerator(cpu=device(config) == 'cpu', project_dir=work)
        p = evaluate.AnndataProcessor(settings, accelerator)
        p.preprocess_anndata()
        if p.adata.n_vars == 0:
            raise ValueError('No UCE vocabulary overlap')
        p.generate_idxs()
        p.run_evaluation()
        out = ad.read_h5ad(Path(work) / f'{args.input.stem}_uce_adata.h5ad')
        return np.asarray(out.obsm['X_uce']), out.obs_names.to_numpy(dtype=str)


def transcriptformer(args, config):
    import anndata as ad
    import psutil
    length = int(config.get('max_length', 1023))
    if not 1 <= length <= 2047:
        raise ValueError('TF-Sapiens max_length must be between 1 and its pretrained limit of 2047')
    estimate = ((args.weights / 'tf_sapiens/model_weights.pt').stat().st_size * 2
                + 3 * 2**30 * (length / 2047)**2 + 2**29)
    if device(config) == 'cpu' and psutil.virtual_memory().available < estimate:
        raise MemoryError(f'TF-Sapiens CPU inference is estimated to need at least {estimate / 2**30:.1f} GiB '
                          'of available host RAM; use a larger-memory host or an appropriate GPU environment.')
    with tempfile.TemporaryDirectory(prefix='tf-', dir=args.output.parent) as work:
        # Call the public entry point in a child interpreter; package requires Python 3.11+.
        command = [sys.executable, '-c', 'from transcriptformer.cli import main; main()',
                   'inference', '--checkpoint-path', str(args.weights / 'tf_sapiens'),
                   '--data-file', str(args.input), '--output-path', work,
                   '--output-filename', 'embeddings.h5ad', '--gene-col-name', 'ensembl_id',
                   '--batch-size', str(config.get('inference_batch_size', 1)),
                   '--precision', '32', '--use-raw', 'False', '--device', device(config),
                   '--disable-compile-block-mask', '--oom-dataloader',
                   '--config-override', f'model.model_config.seq_len={length}']
        # This worker is already isolated. Call the CLI in-process so the configured
        # seed and CPU thread limits apply to tokenization as well as inference.
        from transcriptformer.cli import main
        previous = sys.argv
        try:
            sys.argv = ['transcriptformer', *command[3:]]
            main()
        finally:
            sys.argv = previous
        out = ad.read_h5ad(Path(work) / 'embeddings.h5ad')
        return np.asarray(out.obsm['embeddings']), out.obs['_cellbench_cell_id'].to_numpy(dtype=str)


def scprint2(args, config):
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    import torch
    from scprint2 import scPRINT2
    from scprint2.tasks import Embedder
    # Ontology setup belongs to this environment, not to the benchmark interpreter.
    os.chdir(args.output.parent)
    # Use the official constructor and compatibility hook with memory-mapped
    # tensors, avoiding a second full copy of the multi-species embedding table.
    checkpoint = torch.load(str(args.weights / 'small-v2.ckpt'), map_location='cpu',
                            mmap=True, weights_only=False)
    parameters = dict(checkpoint['hyper_parameters'])
    parameters.pop('_instantiator', None)
    parameters.update(precpt_gene_emb=None, gene_pos_file=None)
    model = scPRINT2(**parameters)
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=True, assign=True)
    del checkpoint
    if device(config) == 'cpu':
        model.float()
    model.to(device(config)).eval()
    data = ad.read_h5ad(args.input)
    organism = config.get('organism', 'NCBITaxon:9606')
    # Reindex counts into the checkpoint's complete organism vocabulary. This is
    # the gene table expected by the official Embedder/Collator offset checks.
    genes = model._genes[organism]
    mapping = {g: i for i, g in enumerate(genes)}
    source_idx = [i for i, g in enumerate(data.var_names) if g in mapping]
    if len(source_idx) < 100:
        raise ValueError('Insufficient scPRINT-2 vocabulary overlap')
    target_idx = [mapping[data.var_names[i]] for i in source_idx]
    transform = sp.csr_matrix((np.ones(len(source_idx)), (source_idx, target_idx)),
                              shape=(data.n_vars, len(genes)))
    data = ad.AnnData((sp.csr_matrix(data.X) @ transform).astype(np.float32), obs=data.obs.copy(),
                     var=pd.DataFrame({'organism': organism}, index=pd.Index(genes)))
    embedder = Embedder(batch_size=config.get('inference_batch_size', 1), num_workers=0,
                        max_len=config.get('max_length', 2000), doclass=False, doplot=False,
                        pred_embedding=['all'], use_knn=config.get('use_knn', False))
    chunks, identities = [], []
    chunk_size = int(config.get('inference_chunk_size', 256))
    if chunk_size < 1:
        raise ValueError('inference_chunk_size must be positive')
    for start in range(0, data.n_obs, chunk_size):
        # The official SimpleAnnDataset densifies its input. Bound that allocation
        # rather than materializing a whole atlas by the full species vocabulary.
        out, _ = embedder(model, data[start:start + chunk_size].copy())
        chunks.append(np.asarray(out.obsm['scprint_emb']))
        identities.append(out.obs_names.to_numpy(dtype=str))
    return np.concatenate(chunks), np.concatenate(identities)


def genecompass(args, config):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='genecompass-', dir=args.output.parent) as folder:
        stage = Path(folder)
        try:
            return _genecompass(args, config, stage)
        finally:
            if str(stage) in sys.path:
                sys.path.remove(str(stage))


def _genecompass(args, config, stage):
    import anndata as ad
    import torch
    # Upstream imports a dictionary at module import time via a source-relative path.
    # Stage the unchanged Python package with a link to weights; keep public_code clean.
    shutil.copytree(args.source / 'genecompass', stage / 'genecompass',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (stage / 'prior_knowledge').symlink_to(args.weights / 'prior_knowledge', target_is_directory=True)
    sys.path.insert(0, str(stage))
    from genecompass import BertForMaskedLM
    from genecompass import modeling_bert, knowledge_embeddings
    from bio.models.compat.genecompass import checkpoint_embeddings
    from transformers import BertConfig
    prior = args.weights / 'prior_knowledge'
    with (prior / 'human_mouse_tokens.pickle').open('rb') as f:
        tokens = pickle.load(f)
    # Use the priors serialized with this exact checkpoint, preserving token order.
    # Do not reconstruct them from a different vocabulary's external pickles.
    state = torch.load(str(args.weights / 'GeneCompass_Small/pytorch_model.bin'),
                       map_location='cpu', mmap=True, weights_only=True)
    names = ('promoter', 'co_exp', 'gene_family', 'peca_grn')
    knowledge = {name: state[f'bert.embeddings.{name}_knowledge'] for name in names}
    homologous = state['bert.embeddings.homologous_index']
    knowledge['homologous_gene_human2mouse'] = {
        i: int(v) for i, v in enumerate(homologous) if i != int(v)}
    model_config = BertConfig.from_pretrained(str(args.weights / 'GeneCompass_Small'),
                                               local_files_only=True)
    if len(tokens) != model_config.vocab_size or set(tokens.values()) != set(range(model_config.vocab_size)):
        raise ValueError('GeneCompass token dictionary does not match checkpoint vocabulary')
    # The release config enables the learned value encoder absent from main.
    # Adapt only the embedding class; retain the upstream BERT and task heads.
    modeling_bert.KnowledgeBertEmbeddings = checkpoint_embeddings(knowledge_embeddings)
    model = BertForMaskedLM(model_config, knowledges=knowledge)
    # Check every key and shape before copying any parameter.
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = {k: {'checkpoint': list(state[k].shape), 'source': list(expected[k].shape)}
                  for k in set(state) & set(expected) if state[k].shape != expected[k].shape}
    if missing or unexpected or mismatched:
        raise ValueError('GeneCompass official source/checkpoint architecture mismatch; '
                         'matching model implementation is required. '
                         + json.dumps(dict(missing_keys=missing, unexpected_keys=unexpected,
                                           mismatched_shapes=mismatched), sort_keys=True))
    # PyTorch 2.2 assign=True wraps plain tensors with requires_grad=True.
    # Preserve nontrainable integer counters without converting their dtype.
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point() and not parameter.is_complex():
            state[name] = torch.nn.Parameter(state[name], requires_grad=False)
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False)
    del state, expected, knowledge
    model.to(device(config)).eval()
    data = ad.read_h5ad(args.input)
    with (args.weights / 'scdata/dict/human_gene_median_after_filter.pickle').open('rb') as f:
        medians = pickle.load(f)
    keep = np.array([g in tokens for g in data.var_names])
    if keep.sum() < 100:
        raise ValueError('Insufficient GeneCompass vocabulary overlap')
    genes = data.var_names[keep]
    token_ids = np.array([tokens[g] for g in genes])
    scale = np.array([medians.get(g, 1.) for g in genes], dtype=np.float32)
    if np.any(scale <= 0):
        raise ValueError('Invalid GeneCompass median dictionary')
    x = data.X[:, keep]
    length = min(config.get('max_length', 2048), model.config.max_position_embeddings)
    batch_size = config.get('inference_batch_size', 1)
    result = []
    with torch.inference_mode():
        for start in range(0, data.n_obs, batch_size):
            block = np.log2(1 + dense(x[start:start + batch_size]) / scale)
            ids = np.zeros((len(block), length), dtype=np.int64)
            values = np.zeros((len(block), length), dtype=np.float32)
            mask = np.zeros_like(ids)
            for j, row in enumerate(block):
                nz = np.flatnonzero(row)
                order = nz[np.argsort(-row[nz], kind='stable')[:length]]
                if len(order) == 0:
                    raise ValueError('Cell has no GeneCompass tokens')
                ids[j, :len(order)], values[j, :len(order)] = token_ids[order], row[order]
                mask[j, :len(order)] = 1
            ids_t, values_t, mask_t = [torch.as_tensor(v, device=device(config)) for v in (ids, values, mask)]
            species = torch.zeros((len(block), 1), dtype=torch.long, device=device(config))
            encoded = model.bert(input_ids=ids_t, values=values_t, attention_mask=mask_t,
                                  species=species, return_dict=True).last_hidden_state
            if model.use_cls_token:
                encoded = encoded[:, 1:]
            pooled = (encoded * mask_t.unsqueeze(-1)).sum(1) / mask_t.sum(1, keepdim=True)
            result.append(pooled.float().cpu().numpy())
    return np.concatenate(result), data.obs_names.to_numpy(dtype=str)


def sclong(args, config):
    import anndata as ad
    import torch
    sys.path.insert(0, str(args.source))
    from performer_pytorch_cont.ding_models import DualEncoderSCFM
    with (args.weights / 'gocont_4096_48m_pretrain_1b_mix.pkl').open('rb') as f:
        params = pickle.load(f)
    params['gene2vec_file'] = str(args.weights / 'selected_gene2vec_27k.npy')
    params['device'] = device(config)
    model = DualEncoderSCFM(**params)
    checkpoint = torch.load(str(args.weights / 'gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth'), map_location='cpu', mmap=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True, assign=True)
    del checkpoint
    model.to(device(config)).eval()
    model.fix_projection_matrices_()
    data = ad.read_h5ad(args.input)
    genes = (args.weights / 'selected_genes_27k.txt').read_text().splitlines()
    mapping = {gene: i for i, gene in enumerate(genes)}
    pairs = [(i, mapping[g]) for i, g in enumerate(data.var_names) if g in mapping]
    if len(pairs) < 100:
        raise ValueError('Insufficient scLong vocabulary overlap')
    source_idx, target_idx = np.array(pairs).T
    batch_size = config.get('inference_batch_size', 1)
    result = []
    with torch.inference_mode():
        for start in range(0, data.n_obs, batch_size):
            block = dense(data.X[start:start + batch_size])
            full = np.zeros((len(block), len(genes) + 1), dtype=np.float32)
            full[:, target_idx] = block[:, source_idx]
            encoded = model(torch.as_tensor(full, device=device(config)), return_encodings=True)['merged_decodings']
            # Official embed.py returns per-gene decoder representations. Pool only matched
            # dataset genes, excluding the appended PAD, to obtain a cell-level benchmark input.
            pooled = encoded[:, target_idx].mean(1)
            result.append(pooled.float().cpu().numpy())
    return np.concatenate(result), data.obs_names.to_numpy(dtype=str)


def cellfm(args, config):
    import anndata as ad
    import mindspore as ms
    import pandas as pd
    import psutil
    checkpoint = args.weights / 'base_weight.ckpt'
    estimate = checkpoint.stat().st_size * 2
    if psutil.virtual_memory().available < estimate:
        raise MemoryError(f'CellFM checkpoint parsing is estimated to need at least {estimate / 2**30:.1f} GiB '
                          'of available host RAM. Run this backend on a larger-memory host; '
                          'do not load the 9.66 GB checkpoint into the current process.')
    sys.path.insert(0, str(args.source))
    from model import CellFM
    from config import Config
    target = config.get('backend_device', 'CPU')
    target = {'cpu': 'CPU', 'cuda': 'GPU', 'auto': 'CPU'}.get(target, target)
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target)
    cfg = Config()
    cfg.recompute = False
    genes = pd.read_csv(args.weights / 'gene_info.csv', index_col=0).index
    model = CellFM(len(genes), cfg)
    params = ms.load_checkpoint(str(checkpoint))
    missing, unused = ms.load_param_into_net(model, params, strict_load=True)
    if missing:
        raise ValueError(f'CellFM missing pretrained parameters: {missing}')
    model.set_train(False)
    data = ad.read_h5ad(args.input)
    mapping = {gene: i + 1 for i, gene in enumerate(genes)}
    keep = np.array([g in mapping for g in data.var_names])
    if keep.sum() < 100:
        raise ValueError('Insufficient CellFM vocabulary overlap')
    gene_ids = np.array([mapping[g] for g in data.var_names[keep]], dtype=np.int32)
    x = data.X[:, keep]
    result = []
    length = config.get('max_length', 2048)
    # Reference inference uses the unmasked expression and CLS representation.
    for i in range(data.n_obs):
        row = dense(x[i:i + 1]).ravel()
        nz = np.flatnonzero(row)
        if len(nz) < 2:
            raise ValueError('CellFM requires at least two expressed matching genes')
        if len(nz) > length:
            probability = np.log1p(row[nz])
            nz = np.sort(np.random.choice(nz, length, replace=False, p=probability / probability.sum()))
        expr = np.zeros((1, length, 2), dtype=np.float32)
        expr[0, :len(nz), 0] = 1
        expr[0, :len(nz), 1] = np.log1p(row[nz] / row.sum() * 1e4)
        tokens = np.zeros((1, length), dtype=np.int32)
        tokens[0, :len(nz)] = gene_ids[nz]
        mask = np.zeros((1, length + 1), dtype=np.float32)
        mask[0, :len(nz) + 1] = 1
        _, _, cls = model.forward(ms.Tensor(expr), ms.Tensor(tokens), ms.Tensor(mask))
        result.append(cls.asnumpy().astype(np.float32))
    return np.concatenate(result), data.obs_names.to_numpy(dtype=str)


BACKENDS = {'UCE': uce, 'scPRINT2': scprint2, 'TranscriptFormer': transcriptformer,
            'CellFM': cellfm, 'GeneCompass': genecompass, 'scLong': sclong}
