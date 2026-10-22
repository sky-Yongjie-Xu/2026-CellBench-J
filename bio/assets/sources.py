#!/usr/bin/env python3
"""Fetch upstream source trees without downloading bundled binary datasets/weights."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from bio.io.paths import PROJECT_ROOT as ROOT
DEST = ROOT / 'public_code'
REPOSITORIES = {
    'CellPLM': ('https://github.com/OmicsML/CellPLM.git', None),
    'Geneformer': ('https://huggingface.co/ctheodoris/Geneformer',
                   '31bf641d0f173a18a5e06d9a6574bb0d3fe5edc1'),
    'LangCell': ('https://github.com/PharMolix/LangCell.git', None),
    'scFoundation': ('https://github.com/biomap-research/scFoundation.git', None),
    'scGPT': ('https://github.com/bowang-lab/scGPT.git', None),
    'scMulan': ('https://github.com/SuperBianC/scMulan.git', None),
    'Nicheformer': ('https://github.com/theislab/nicheformer.git', None),
    'Tabula': ('https://github.com/aristoteleo/tabula.git', None),
    'UCE': ('https://github.com/snap-stanford/UCE.git', '9c416007be15ad6753dc84af4468c1dc10421ab9'),
    'scPRINT2': ('https://github.com/cantinilab/scPRINT-2.git', 'db0ef5f1fd120084a4323572bd54f8f2a2af2566'),
    'TranscriptFormer': ('https://github.com/czi-ai/transcriptformer.git', 'c943a89a4de9511a8ab1010715bf1cfa8827cc99'),
    'CellFM': ('https://github.com/biomed-AI/CellFM.git', '72c9f4a9580a3716058c184900ed14a65151ed8f'),
    'GeneCompass': ('https://github.com/xCompass-AI/GeneCompass.git', '59e5e485cb940038a7db040c2f31c8a2ce76ba53'),
    'scLong': ('https://github.com/BaiDing1234/scLong.git', '41b72021540918e4386c4f2264351d7a6cbeed99'),
}
EXCLUDED = ('*.h5ad', '*.h5', '*.hdf5', '*.loom', '*.zip', '*.tar', '*.gz',
            '*.pt', '*.pth', '*.ckpt', '*.bin', '*.safetensors', '*.npy', '*.npz',
            '*.arrow', '*.parquet', '*.pkl', '*.pickle', '*.rds', '*.RData')


def git(directory, *args):
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')
    result = subprocess.run(['git', '-C', str(directory), *args], env=env,
                            capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    return result.stdout.strip()


def fetch(entry):
    name, (url, revision) = entry
    target = DEST / name
    record = {'model': name, 'url': url, 'path': f'public_code/{name}',
              'requested_revision': revision, 'excluded_binary_patterns': list(EXCLUDED)}
    try:
        initialized_only = (target / '.git').is_dir() and not any(p.name != '.git' for p in target.iterdir())
        if target.exists() and any(target.iterdir()) and not initialized_only:
            if not (target / '.git').is_dir():
                raise RuntimeError('Destination is nonempty and is not a Git repository; left unchanged')
            if git(target, 'remote', 'get-url', 'origin') != url:
                raise RuntimeError('Existing origin differs from requested source; left unchanged')
            if git(target, 'status', '--porcelain'):
                raise RuntimeError('Existing repository has local changes; left unchanged')
            # Reuse the checked-out revision, without resetting an existing checkout.
            commit = git(target, 'rev-parse', 'HEAD')
            if revision and commit != revision:
                raise RuntimeError('Existing checkout differs from pinned version; left unchanged')
        else:
            target.mkdir(parents=True, exist_ok=True)
            if not initialized_only:
                git(target, 'init', '-q')
                git(target, 'remote', 'add', 'origin', url)
            elif git(target, 'remote', 'get-url', 'origin') != url:
                raise RuntimeError('Existing origin differs from requested source; left unchanged')
            # Blob filtering avoids downloading binary files excluded by sparse checkout.
            git(target, 'config', 'remote.origin.promisor', 'true')
            git(target, 'config', 'remote.origin.partialclonefilter', 'blob:none')
            print(f'Downloading {name}', flush=True)
            git(target, 'fetch', '--depth=1', '--filter=blob:none', 'origin', revision or 'HEAD')
            git(target, 'sparse-checkout', 'init', '--no-cone')
            patterns = ['/*'] + ['!' + pattern for pattern in EXCLUDED]
            (target / '.git/info/sparse-checkout').write_text('\n'.join(patterns) + '\n')
            git(target, 'checkout', '--detach', 'FETCH_HEAD')
            commit = git(target, 'rev-parse', 'HEAD')
        python_files = list(target.rglob('*.py'))
        if not python_files or not any(p.is_file() for p in (target / 'README.md', target / 'readme.md')):
            raise RuntimeError('Checkout is missing expected Python source or README')
        record.update(status='complete', commit=commit, python_files=len(python_files),
                      downloaded_at=datetime.now(timezone.utc).isoformat())
        print(f'Complete {name}: {commit[:12]}, {len(python_files)} Python files', flush=True)
    except Exception as exc:
        record.update(status='failed', error=str(exc))
        print(f'FAILED {name}: {exc}', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=sorted(REPOSITORIES))
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    selected = {k: v for k, v in REPOSITORIES.items() if not args.models or k in args.models}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        fetched = list(pool.map(fetch, selected.items()))
    old = json.loads((DEST / 'sources.json').read_text()) if (DEST / 'sources.json').exists() else []
    by_name = {r['model']: r for r in old}
    by_name.update({r['model']: r for r in fetched})
    records = list(by_name.values())
    (DEST / 'sources.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 公开模型原始代码', '',
             '下载自作者公开仓库；Geneformer 固定到与本项目已下载权重匹配的官方历史版本。',
             '其他模型记录本次下载的具体 commit。Git 使用浅克隆，仅保留当前版本历史。', '',
             '此目录用于阅读和对照原始实现。二进制数据、权重和 pickle 等资源未检出，'
             '模型运行资源仍位于项目的 weights/ 中。所有源码、文档和源码目录结构保留。', '',
             '| 模型 | 来源 | Commit | 状态 |', '| --- | --- | --- | --- |']
    for r in records:
        web_url = r['url'].removesuffix('.git')
        lines.append(f"| {r['model']} | [{web_url}]({web_url}) | {r.get('commit', '')} | {r['status']} |")
    lines += ['', '精确来源、版本和排除的文件模式见 sources.json。',
              '重新下载或检查已有仓库：`python3 scripts/download_public_code.py`。', '']
    (DEST / 'README.md').write_text('\n'.join(lines))
    return int(any(r['status'] != 'complete' for r in fetched))


if __name__ == '__main__':
    raise SystemExit(main())
