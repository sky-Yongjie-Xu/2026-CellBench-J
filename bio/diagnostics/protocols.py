#!/usr/bin/env python3
"""Check the 14 model contracts and render the source protocol review.

Protocol assessments are explicit human-reviewed records, not inferred from a
successful import or inference run. Smoke tests and source status stay separate.
"""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from bio.io.paths import PROJECT_ROOT as ROOT


def main():
    rows = json.loads((ROOT / 'resources/model_protocols.json').read_text())
    sources = {r['model']: r for r in json.loads((ROOT / 'public_code/sources.json').read_text())}
    previous_path = ROOT / 'output/frontier_status.json'
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    results = []
    for row in rows:
        source = ROOT / 'public_code' / row['model']
        tree = ast.parse((ROOT / row['adapter']).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name in (row['model'], 'FrozenExternalModel'))
        # Concrete remote classes inherit all three implementations from the
        # shared FrozenExternalModel; local classes implement the same hooks.
        if not any(isinstance(n, ast.FunctionDef) for n in cls.body):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FrozenExternalModel')
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        hooks = all(k in methods for k in ('prepare_inputs', 'load_pretrained', 'encode_batch'))
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
        key = Path(row['config']).stem
        smoke_path = ROOT / 'output/adapter_smoke' / key / 'status.json'
        smoke = json.loads(smoke_path.read_text()) if smoke_path.is_file() else None
        results.append(dict(**row, reference_files_present=all((ROOT / p).is_file() for p in row['references']),
                            common_hooks=hooks, source_commit=commit,
                            source_matches_pin=commit == sources[row['model']]['commit'],
                            source_clean=not bool(dirty), adapter_smoke=smoke,
                            prior_backend_smoke=previous.get(row['model'], {}).get('inference_verified', False)))
    report = dict(generated_at=datetime.now(timezone.utc).isoformat(),
                  scope='14 foundation models and 3 statistical baselines share the adapter/cache contract',
                  note='Protocol port/native API is not a claim of byte-for-byte or numerical equivalence.',
                  models=results)
    (ROOT / 'output/model_adapter_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 模型适配器源码审查', '',
             '范围：14 个基础模型的数据、加载、embedding 三阶段；PCA/UMAP/scVI 三个基线也统一接口与缓存。', '',
             '本表区分官方 API、协议移植和兼容重建，不把“可以运行”当作“数值等价”。官方源码保持原样；基准缓存属于项目自有代码。', '',
             '| 模型 | 数据与基因匹配 | 模型加载 | embedding | 本次完整适配器小样本 |',
             '|---|---|---|---|---|']
    for row in results:
        smoke = row['adapter_smoke']
        status = ('通过 ' + str(smoke['shape'])) if smoke and smoke['ok'] else ('未通过，见 JSON 详情' if smoke else '未重跑；见已有后端记录')
        stages = row['stages']
        lines.append(f"| {row['model']} | {stages['data']} | {stages['loading']} | {stages['embedding']} | {status} |")
    lines += ['', '## 统一结构与缓存', '',
              '`FoundationAdapter` 统一 `prepare_inputs`、`load_pretrained`、`encode_batch`；兼容原来的训练/评测入口。',
              '14 个基础模型与 3 个基线共享源行 ID 校验、有限数校验、缓存读取和原子写入。六个独立后端在子进程中实际加载权重。',
              '模型输入从全基因计数重新读取；下游表达目标继续使用评测基因空间，二者不混用。标签不会传入预训练编码器。',
              '缓存位置为 `output/embeddings/<dataset>_<model>/v2/<signature>/all.pkl`；签名包括输入/侧文件、权重文件状态、配置、seed、本地与上游 Python 源码。',
              '旧缓存不删除，但不自动复用；非有限数、重复/丢失细胞及标签错位均明确失败。', '',
              'PCA/UMAP/scVI 使用 `StatisticalAdapter`，在当前数据上拟合，不加载基础模型检查点。scVI 必须使用真实计数。PCA 已通过两细胞流程；UMAP/scVI 本次未执行完整拟合。', '',
              '复现：`python scripts/check_adapter_inference.py --model Geneformer --input datasets/pbmc_12k.h5ad --cells 2`。',
              '两细胞测试验证加载、输出结构、细胞顺序和缓存复读，不代表全数据评测或与官方实现的逐数值对照。', '', '## 逐模型差异与依据', '']
    for row in results:
        lines += ['### ' + row['model'], '', '**审查类型：** `' + row['consistency'] + '`。', '', row['protocol'], '']
        lines += ['- ' + note for note in row['differences']]
        lines += ['', '源码：' + '、'.join(f'[{Path(p).name}](../{p})' for p in row['references']),
                  '', f"固定提交：`{row['source_commit']}`；源码干净：`{row['source_clean']}`。", '']
    (ROOT / 'docs/model_adapter_audit.md').write_text('\n'.join(lines) + '\n')
    ok = all(r['common_hooks'] and r['source_matches_pin'] and r['source_clean'] and r['reference_files_present'] for r in results)
    print(f'{len(results)} models reviewed; structure/source checks={ok}')
    return int(not ok)


if __name__ == '__main__':
    raise SystemExit(main())
