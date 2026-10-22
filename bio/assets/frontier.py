#!/usr/bin/env python3
"""Download the six additional model families from their authors' resource links.

Training datasets are excluded. Downloads resume through download_weights.py.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import tarfile

from bio.assets.weights import ROOT, WEIGHTS, download, digest

MODELS = ('UCE', 'scPRINT2', 'TranscriptFormer', 'CellFM', 'GeneCompass', 'scLong')
HF_REVISIONS = {
    'CellFM': '69361b50d1814ade0e3749c9d8eb829590c2ce3c',
    'scPRINT2': '8e1ad1a1bc40c1fc2f4f1700fe27923eea9a2bed',
}


def assets(model):
    def item(path, url, **kw):
        return dict(model=model, path=f'{model}/{path}', url=url, **kw)
    sources = {r['model']: r['commit'] for r in json.loads((ROOT / 'public_code/sources.json').read_text()) if r.get('commit')}
    if model == 'UCE':
        result = [item(path, f'https://ndownloader.figshare.com/files/{fid}')
                for path, fid in (
                    ('model_files/species_chrom.csv', 42706558),
                    ('model_files/species_offsets.pkl', 42706555),
                    ('model_files/protein_embeddings.tar.gz', 42715213),
                    ('model_files/all_tokens.torch', 42706585),
                    ('model_files/4layer_model.torch', 42706576))]
        result.append(item('model_files/new_species_protein_embeddings.csv',
                           f"https://raw.githubusercontent.com/snap-stanford/UCE/{sources[model]}/model_files/new_species_protein_embeddings.csv"))
        return result
    if model == 'TranscriptFormer':
        return [item('tf_sapiens.tar.gz', 'https://czi-transcriptformer.s3.amazonaws.com/weights/tf_sapiens.tar.gz')]
    if model == 'scPRINT2':
        return [item('small-v2.ckpt', f'https://huggingface.co/jkobject/scPRINT/resolve/{HF_REVISIONS[model]}/small-v2.ckpt')]
    if model == 'CellFM':
        return [item('base_weight.ckpt', f'https://huggingface.co/ShangguanNingyuan/CellFM/resolve/{HF_REVISIONS[model]}/base_weight.ckpt')] + [
            item(name, f"https://raw.githubusercontent.com/biomed-AI/CellFM/{sources[model]}/csv/{name}")
            for name in ('gene_info.csv', 'updated_hgcn.tsv')]
    if model == 'scLong':
        return sclong_assets()
    if model == 'GeneCompass':
        return genecompass_assets()
    raise ValueError(model)


def sclong_assets():
    """Resolve the public anonymous OneDrive folder linked by the official README."""
    import re
    import requests
    share = ('https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/'
             'EpvKzQW4hI5Bnb88-iM7vE0B_e2_U5r_ZGXb_FILCLTw3Q?e=TAmKk5')
    response = requests.get(share, timeout=60)
    response.raise_for_status()
    page = response.text.replace('\\"', '"')
    drive = re.search(r'"\.driveUrl":"([^"]+)"', page).group(1)
    token = re.search(r'"\.driveAccessToken":"([^"]+)"', page).group(1)
    headers = {'Authorization': 'Bearer ' + token.removeprefix('access_token=')}
    def children(path):
        r = requests.get(drive + path + '/children', headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()['value']
    folder = next(x for x in children('/root') if x['name'] == 'scLong')
    files = children('/items/' + folder['id'])
    wanted = {'selected_gene2vec_27k.npy', 'selected_genes_27k.txt',
              'human_ens_gene2go_graph.csv', 'gocont_4096_48m_pretrain_1b_mix.pkl',
              'gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth'}
    result = []
    for f in files:
        if f['name'] not in wanted:
            continue
        # Signed public download URLs expire; retain the stable source in the manifest.
        result.append(dict(model='scLong', path='scLong/' + f['name'],
                           url=f['@content.downloadUrl'] if '@content.downloadUrl' in f else f['@microsoft.graph.downloadUrl'],
                           source_url=share, ephemeral_url=True,
                           expected_bytes=f['size']))
    if {Path(x['path']).name for x in result} != wanted:
        raise RuntimeError('Official scLong share is missing expected model resources')
    return result


def genecompass_assets():
    # Filled from the author's ScienceDB resource manifest; no model mirrors.
    manifest = ROOT / 'resources/frontier_genecompass_assets.json'
    if not manifest.is_file():
        raise RuntimeError('GeneCompass resource manifest is not available')
    return json.loads(manifest.read_text())


def extract_archive(path, destination):
    """Extract regular files only, reject traversal and links, and resume safely."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted = []
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f'Unsafe archive path: {member.name}')
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f'Unsupported archive entry: {member.name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.stat().st_size != member.size:
                partial = target.with_name(target.name + '.part')
                with archive.extractfile(member) as src, partial.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                partial.replace(target)
            extracted.append(dict(path=str(target.relative_to(WEIGHTS)), bytes=member.size,
                                  sha256=digest(target)))
    return extracted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--manifest', type=Path, default=WEIGHTS / 'frontier_download_manifest.json')
    args = parser.parse_args()
    WEIGHTS.mkdir(exist_ok=True)
    manifest = args.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    records = {r['path']: r for r in json.loads(manifest.read_text())} if manifest.exists() else {}
    selected = []
    failed = False
    for model in args.models:
        try:
            selected.extend(assets(model))
        except Exception as exc:
            key = f'{model}/RESOURCE_DISCOVERY'
            records[key] = dict(model=model, path=key, status='failed', error=str(exc))
            print(f'FAILED {model} discovery: {exc}', flush=True)
            failed = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(download, selected):
            if record.get('ephemeral_url'):
                record['url'] = record.pop('source_url')
            if record['status'] == 'complete':
                try:
                    target = WEIGHTS / record['path']
                    if record.get('expected_bytes') and target.stat().st_size != record['expected_bytes']:
                        raise ValueError('Downloaded size differs from official resource metadata')
                    if target.name in ('tf_sapiens.tar.gz', 'protein_embeddings.tar.gz'):
                        record['extracted'] = extract_archive(target, target.parent)
                    key = f"{record['model']}/RESOURCE_DISCOVERY"
                    records.pop(key, None)
                except Exception as exc:
                    record.update(status='failed', error=str(exc))
            records[record['path']] = record
            tmp = manifest.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(list(records.values()), ensure_ascii=False, indent=2) + '\n')
            tmp.replace(manifest)
            failed |= record['status'] != 'complete'
    # Preserve discovery failures even when no downloads could start.
    manifest.write_text(json.dumps(list(records.values()), ensure_ascii=False, indent=2) + '\n')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
