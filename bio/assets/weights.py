#!/usr/bin/env python3
"""Download CellBench model resources; record sources, sizes and SHA-256 hashes.

Install the small download dependencies with: pip install requests gdown
Run from any directory: python scripts/download_weights.py [--models MODEL ...]
Interrupted HTTP downloads are retained as .part files and resumed on retry.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

import requests

from bio.io.paths import PROJECT_ROOT as ROOT
WEIGHTS = ROOT / "weights"
GF_REV = "31bf641d0f173a18a5e06d9a6574bb0d3fe5edc1"
NICHE_REV = "0ba4ba162f9ad3d6c8b713761954c31be2fa69ed"


def hf(repo, path, revision="main"):
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


ASSETS = []


def asset(model, path, url, **kwargs):
    ASSETS.append(dict(model=model, path=path, url=url, **kwargs))


for name in ("config.json", "pytorch_model.bin"):
    asset("Geneformer", f"Geneformer/gf-12L-30M-i2048/{name}",
          hf("ctheodoris/Geneformer", f"gf-12L-30M-i2048/{name}", GF_REV))
for name in ("token_dictionary_gc30M.pkl", "gene_median_dictionary_gc30M.pkl"):
    asset("Geneformer", f"Geneformer/{name}",
          hf("ctheodoris/Geneformer", f"geneformer/gene_dictionaries_30m/{name}", GF_REV))
    asset("LangCell", f"LangCell/{name}",
          hf("ctheodoris/Geneformer", f"geneformer/gene_dictionaries_30m/{name}", GF_REV))

for name in ("config.json", "vocab.json", "model.safetensors", "configuration_nicheformer.py",
             "modeling_nicheformer.py", "tokenization_nicheformer.py", "masking.py", "__init__.py", "LICENSE"):
    asset("Nicheformer", f"Nicheformer/{name}", hf("theislab/Nicheformer", name, NICHE_REV))
asset("Nicheformer", "Nicheformer/dissociated_mean_script.npy",
      "https://raw.githubusercontent.com/theislab/nicheformer/main/data/model_means/dissociated_mean_script.npy")

for path, file_id in (
    ("cell_bert/config.json", "1F8rIavbUzF3xqLjGDMhZTBHNl01mhc-q"),
    ("cell_bert/pytorch_model.bin", "119ic1oiZr6Lxa5Bguv240L-OL5lPGEAq"),
    ("cell_proj.bin", "1euTzbl3jHirjwDyTLhprdmjD25w7YwzC"),
):
    asset("LangCell", f"LangCell/{path}", f"https://drive.google.com/uc?id={file_id}", gdrive=True)

for name, file_id in (
    ("args.json", "1hh2zGKyWAx3DyovD30GStZ3QlzmSqdk1"),
    ("best_model.pt", "14AebJfGOUF047Eg40hk57HCtrb0fyDTm"),
    ("vocab.json", "1H3E_MJ-Dl36AQV6jLbna2EdvgPaqvqcC"),
):
    asset("scGPT", f"scGPT/human/{name}", f"https://drive.google.com/uc?id={file_id}", gdrive=True)

asset("Tabula", "Tabula/pancreas.pth",
      "https://drive.google.com/uc?id=1xvUCfbi3K0Gn7iCniYg2S4_FwzwWk6St", gdrive=True)
for name in ("vocab.json", "finetune_framework_annotation.yaml"):
    asset("Tabula", f"Tabula/{name}", f"https://raw.githubusercontent.com/aristoteleo/tabula/main/resource/{name}")

asset("CellPLM", "CellPLM/20230926_85M.config.json",
      "https://raw.githubusercontent.com/OmicsML/CellPLM/main/ckpt/20230926_85M.config.json")
asset("CellPLM", "CellPLM/checkpoints.zip",
      "https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h?rlkey=o8hi0xads9ol07o48jdityzv1&dl=1",
      extract="20230926_85M.best.ckpt")
asset("scMulan", "scMulan/ckpt_scMulan.pt",
      "https://cloud.tsinghua.edu.cn/f/2250c5df51034b2e9a85/?dl=1")
asset("scMulan", "scMulan/meta_info.pt",
      "https://raw.githubusercontent.com/SuperBianC/scMulan/main/scMulan/utils/meta_info.pt")
for name in ("GeneSymbolRef_upd0731.csv", "total_gene_list_42117.txt"):
    asset("scMulan", f"scMulan/{name}",
          f"https://raw.githubusercontent.com/SuperBianC/scMulan/main/scMulan/reference/GeneSymbolUniform/{name}")
asset("scFoundation", "scFoundation/models.ckpt",
      "https://hopebio2020.sharepoint.com/sites/PublicSharedfiles/_api/web/"
      "GetFileByServerRelativeUrl('/sites/PublicSharedfiles/Shared Documents/Public Shared files/models.ckpt')/$value",
      share_url="https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk",
      expected_sha256="9f40bf324d3d0084c4b288d06f5af4fddd12206e2a3f022551d12e89e33a0ea9")
asset("scFoundation", "scFoundation/OS_scRNA_gene_index.19264.tsv",
      "https://raw.githubusercontent.com/biomap-research/scFoundation/main/model/OS_scRNA_gene_index.19264.tsv")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate(path):
    if not path.stat().st_size:
        raise ValueError("Empty download")
    with path.open("rb") as f:
        head = f.read(256).lstrip()
    if head.lower().startswith((b"<!doctype html", b"<html", b"<?xml")):
        raise ValueError("Server returned an HTML/XML page instead of a model resource")
    if head.startswith(b"version https://git-lfs.github.com/spec"):
        raise ValueError("Downloaded a Git LFS pointer instead of file contents")
    if path.suffix == ".json":
        json.loads(path.read_text())


def http_download(url, target, share_url=None):
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    session = requests.Session()
    if share_url:
        session.get(share_url, timeout=30).raise_for_status()
    with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        if response.status_code == 416:
            partial.unlink(missing_ok=True)
            return http_download(url, target, share_url)
        response.raise_for_status()
        if response.status_code == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise ValueError(f"Unexpected resume range: {content_range}")
        mode = "ab" if offset and response.status_code == 206 else "wb"
        expected = response.headers.get("Content-Length")
        received = 0
        with partial.open(mode) as f:
            for block in response.iter_content(4 * 1024 * 1024):
                f.write(block)
                received += len(block)
        if expected and not response.headers.get("Content-Encoding") and received != int(expected):
            raise ValueError(f"Incomplete response: {received}/{expected} bytes")
    validate(partial)
    # Validate JSON using the final filename before promoting the partial file.
    if target.suffix == ".json":
        json.loads(partial.read_text())
    partial.replace(target)


def download(item):
    target = WEIGHTS / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    record = dict(item)
    for attempt in range(3):
        try:
            if not target.exists():
                print(f"Downloading {item['path']}", flush=True)
                if item.get("gdrive"):
                    import gdown
                    partial = target.with_name(target.name + ".part")
                    result = gdown.download(item["url"], str(partial), quiet=True, resume=True)
                    if result is None:
                        raise ValueError("Google Drive download failed")
                    validate(partial)
                    partial.replace(target)
                else:
                    http_download(item["url"], target, item.get("share_url"))
            validate(target)
            if item.get("extract"):
                output = target.parent / item["extract"]
                with zipfile.ZipFile(target) as archive:
                    members = [n for n in archive.namelist() if Path(n).name == output.name]
                    if len(members) != 1:
                        raise ValueError(f"Expected one {output.name}, found {members}")
                    if not output.exists():
                        import shutil
                        partial = output.with_name(output.name + ".part")
                        with archive.open(members[0]) as src, partial.open("wb") as dst:
                            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                        partial.replace(output)
                record["extracted"] = dict(path=str(output.relative_to(WEIGHTS)),
                                           bytes=output.stat().st_size, sha256=digest(output))
            checksum = digest(target)
            if item.get("expected_sha256") and checksum != item["expected_sha256"]:
                raise ValueError(f"SHA-256 mismatch: {checksum}")
            record.update(status="complete", bytes=target.stat().st_size, sha256=checksum)
            print(f"Complete {item['path']} ({target.stat().st_size:,} bytes)", flush=True)
            return record
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            if attempt < 2:
                time.sleep(2)
    print(f"FAILED {item['path']}: {record['error']}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selected = [a for a in ASSETS if not args.models or a["model"] in args.models]
    if args.models and set(args.models) - {a["model"] for a in ASSETS}:
        parser.error("Unknown model name (model names are case sensitive)")
    WEIGHTS.mkdir(exist_ok=True)
    if any(a['model'] == 'Tabula' for a in selected):
        source = WEIGHTS / 'Tabula/source'
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--depth', '1',
                            'https://github.com/aristoteleo/tabula.git', str(source)], check=True)
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        (WEIGHTS / 'source_revisions.json').write_text(json.dumps({
            'Tabula': {'repository': 'https://github.com/aristoteleo/tabula', 'commit': revision}
        }, indent=2) + '\n')
    manifest = WEIGHTS / "download_manifest.json"
    old = json.loads(manifest.read_text()) if manifest.exists() else []
    records = {r["path"]: r for r in old}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(download, selected):
            # Preserve results written by a separate model-specific downloader.
            if manifest.exists():
                records.update({r['path']: r for r in json.loads(manifest.read_text())})
            records[record["path"]] = record
            temporary = manifest.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(list(records.values()), ensure_ascii=False, indent=2) + "\n")
            temporary.replace(manifest)
    return int(any(records[a["path"]]["status"] != "complete" for a in selected))


if __name__ == "__main__":
    raise SystemExit(main())
